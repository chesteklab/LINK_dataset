import os
import pickle
import sys
import pandas as pd
import random
import numpy as np
import torch
import time
import datetime
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from tqdm import tqdm

from ..bci_decoding.lstm import LSTMModel
from ..bci_decoding.data_utils import prep_data_and_split
from ..bci_decoding.nn_utils import FingerDataset, create_optimizer_and_scheduler, train_model, SequenceScaler
from ..bci_decoding.metrics import calculate_mse_per_dof, calculate_correlation_per_dof, calculate_r2_per_dof

"""
Continual Learning Script for LSTM Models

This script implements a continual learning strategy for LSTM models:
1. Train an initial LSTM model on a specified starting day
2. For each consecutive day in the dataset:
   - Fine-tune the model using only a limited number of samples from that day
   - Update only the input (neural) normalizer, keep output normalizer fixed
   - Evaluate on the full test set of that day
3. Track performance across days

Results are saved as CSV with performance metrics for each day.
"""

########################################################
# Configuration Parameters
########################################################

data_folder = '../../link_dataset_pkl'
results_folder = 'continual_learning_results'

STARTING_DAYS = [
    '2020-07-08', # 1
    '2020-10-02', # 2
    '2020-12-12', # 3
    '2021-02-27', # 4
    '2021-04-05', # 5
    '2021-05-05', # 6
    '2021-06-16', # 7
    '2021-07-09', # 8
    '2021-07-29', # 9
    '2021-08-19', # 10
    '2021-09-14', # 11
    '2021-10-27', # 12
    '2022-01-17', # 13
    '2022-02-21', # 14
    '2022-04-01', # 15
    '2022-04-21', # 16
    '2022-06-01', # 17
    '2022-08-05', # 18
    '2022-09-01', # 19
    '2022-09-30', # 20
]

TARGET_TYPE = 'BOTH'  # 'CO', 'RD', or 'BOTH' - specify target type (BOTH prioritizes CO over RD)

# Continual learning parameter sweeps
# NUM_FINETUNE_SAMPLES_LIST = [30*32, 60*32, 120*32, 300*32]  # Different numbers of samples to sweep over (OLD - asumming 32ms bins)
NUM_FINETUNE_SAMPLES_LIST = [30*50, 60*50, 120*50, 300*50]  # Different numbers of samples to sweep over (New - assuming 20ms bins)

MAX_DAYS_TO_TEST = 200       # Maximum number of days to test (set to None for all available days)

MAX_DAY_DIFFERENCE = 200    # If not None, will stop if day diff is greater than this

# Option to force a specific number of finetune iterations for a given number of finetune samples
# (using this after we've already run the sweep)
FORCE_FINETUNE_ITERATIONS = True
FORCE_FINETUNE_ITERATIONS_DICT = {
    30*50: 50,
    60*50: 250,
    120*50: 250,
    300*50: 500,
}

FINETUNE_ITERATIONS_LIST = [10, 20, 50, 250, 500]         # Different numbers of finetune iterations to sweep over

SAVE_EVERY_N_ITERATIONS = 100  # Save results every N parameter combinations

# Option to update the neural (input) scaler on each day's limited data
UPDATE_NEURAL_SCALER = True   # If False, reuse previous day's scaler for fine-tuning and evaluation

# Optional repetition sweep
REPETITIONS = 1

FINETUNE_SAMPLES_FROM_END_OF_TRAINING = True # True - take from end of training, False - take from start of training

# Standard parameters
NUM_TRAIN_TRIALS = 300
NUM_CHANNELS = 96
NUM_OUTPUTS = 4

# LSTM parameters
lstm_seq_len = 20
lstm_num_inputs = NUM_CHANNELS
lstm_num_outputs = NUM_OUTPUTS
lstm_hidden_size = 300
lstm_num_layers = 1
lstm_dropout = 0
lstm_batch_size = 64

# Initial training parameters
initial_num_iterations = 2000
initial_start_lr = 2e-4
initial_end_lr = 1e-5
initial_weight_decay = 0.001

# Fine-tuning parameters
# finetune_num_iterations is now handled by FINETUNE_ITERATIONS_LIST sweep
finetune_start_lr = 2e-4       # Lower learning rate for fine-tuning
finetune_end_lr = 1e-5
finetune_weight_decay = 0.001

# Noise parameters for training
neural_noise_std = 0.1
bias_noise_std = 0.2

########################################################
########################################################

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Create results folder if it doesn't exist
os.makedirs(results_folder, exist_ok=True)

########################################################
# Helper Functions
########################################################

def select_best_available_data(data_CO, data_RD, target_type):
    """
    Select the best available data based on target type preference.
    
    Args:
        data_CO: Center-out data (can be None)
        data_RD: Random data (can be None)
        target_type: 'CO', 'RD', or 'BOTH' - type of data to prefer
        
    Returns:
        tuple: (selected_data, actual_target_type_used)
    """
    if target_type == 'CO':
        if data_CO is not None:
            return data_CO, 'CO'
        else:
            return None, None
    elif target_type == 'RD':
        if data_RD is not None:
            return data_RD, 'RD'
        else:
            return None, None
    elif target_type == 'BOTH':
        # Prioritize CO over RD
        if data_CO is not None:
            return data_CO, 'CO'
        elif data_RD is not None:
            return data_RD, 'RD'
        else:
            return None, None
    else:
        raise ValueError(f"Invalid target_type: {target_type}. Must be 'CO', 'RD', or 'BOTH'")

def get_sorted_dates(data_folder):
    """
    Get all available dates from the data folder and sort them chronologically.
    
    Args:
        data_folder: Path to folder containing preprocessed data files
        
    Returns:
        list: Sorted list of date strings in 'YYYY-MM-DD' format
    """
    # Get all pickle files and extract dates
    files = [f for f in os.listdir(data_folder) if f.endswith('_preprocess.pkl')]
    dates = [f.split('_preprocess.pkl')[0] for f in files]
    
    # Convert to datetime objects for proper sorting
    date_objects = []
    for date_str in dates:
        try:
            date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            date_objects.append((date_obj, date_str))
        except ValueError:
            print(f"Warning: Could not parse date {date_str}, skipping...")
            continue
    
    # Sort by datetime and return date strings
    date_objects.sort(key=lambda x: x[0])
    sorted_dates = [date_str for _, date_str in date_objects]
    
    return sorted_dates

def get_consecutive_dates_with_data(sorted_dates, starting_day, data_folder, target_type, max_days=None):
    """
    Get consecutive dates starting from a specified day, filtering for days with actual data.
    
    Args:
        sorted_dates: List of all available dates sorted chronologically
        starting_day: Starting date string
        data_folder: Path to data folder
        target_type: 'CO', 'RD', or 'BOTH' - type of data to check for
        max_days: Maximum number of days with data to return (None for all)
        
    Returns:
        list: List of consecutive dates with data starting from starting_day
    """
    if starting_day not in sorted_dates:
        raise ValueError(f"Starting day {starting_day} not found in available dates")
    
    start_idx = sorted_dates.index(starting_day)
    consecutive_dates = sorted_dates[start_idx:]
    
    # Filter for dates that actually have the required data
    dates_with_data = []
    for date in consecutive_dates:
        try:
            data_file = os.path.join(data_folder, f'{date}_preprocess.pkl')
            if os.path.exists(data_file):
                with open(data_file, 'rb') as f:
                    data_CO, data_RD = pickle.load(f)
                
                # Check if the required target type data exists
                required_data, _ = select_best_available_data(data_CO, data_RD, target_type)
                if required_data is not None:
                    dates_with_data.append(date)
                    
                    # Stop if we've reached the max number of days
                    if max_days is not None and len(dates_with_data) >= max_days:
                        break
        except Exception as e:
            print(f"Warning: Could not check data for {date}: {e}")
            continue
    
    return dates_with_data

def train_initial_lstm(train_neural, train_kinematics, device=device):
    """
    Train the initial LSTM model on the starting day.
    
    Args:
        train_neural: numpy array of shape (n_samples, seq_len, n_channels)
        train_kinematics: numpy array of shape (n_samples, n_outputs)
        device: torch device
        
    Returns:
        tuple: (model, input_scaler, output_scaler)
    """
    print("Training initial LSTM model...")
    
    # Create scalers
    input_scaler = SequenceScaler()
    train_neural_scaled = input_scaler.fit_transform(train_neural)
    
    output_scaler = StandardScaler()
    train_kinematics_scaled = output_scaler.fit_transform(train_kinematics)
    
    # Create dataset and dataloader
    dataset_train = FingerDataset(train_neural_scaled, train_kinematics_scaled)
    dataloader_train = DataLoader(dataset_train, batch_size=lstm_batch_size, shuffle=True)
    
    # Create model
    model = LSTMModel(input_dim=lstm_num_inputs,
                      hidden_size=lstm_hidden_size,
                      num_states=lstm_num_outputs,
                      num_layers=lstm_num_layers,
                      drop_prob=lstm_dropout)
    model.to(device)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, 
        lr=initial_start_lr, 
        weight_decay=initial_weight_decay, 
        final_lr=initial_end_lr,
        total_steps=initial_num_iterations
    )
    
    # Train the model
    model, history = train_model(
        model, 
        dataloader_train, 
        optimizer, 
        scheduler, 
        loss_fn=nn.MSELoss(),
        num_iterations=initial_num_iterations,
        device=device,
        noise_neural_std=neural_noise_std,
        noise_bias_std=bias_noise_std
    )
    
    return model, input_scaler, output_scaler

def finetune_lstm(model, train_neural_limited, train_kinematics_limited, original_output_scaler, prev_input_scaler=None, update_scaler=True, num_iterations=200, device=device):
    """
    Fine-tune the LSTM model on limited data from a new day.
    Optionally updates the input normalizer, keeps the output normalizer fixed.
    
    Args:
        model: Pre-trained LSTM model
        train_neural_limited: Limited neural data for fine-tuning (n_samples, seq_len, n_channels)
        train_kinematics_limited: Limited kinematics data for fine-tuning (n_samples, n_outputs)
        original_output_scaler: Original output scaler to keep fixed
        prev_input_scaler: Previous day's input scaler (if not refitting)
        update_scaler: Whether to update the input scaler on this day's data
        num_iterations: Number of training iterations for fine-tuning
        device: torch device
        
    Returns:
        tuple: (finetuned_model, input_scaler_used, original_output_scaler)
    """
    # Decide which input scaler to use
    if update_scaler:
        input_scaler = SequenceScaler()
        train_neural_scaled = input_scaler.fit_transform(train_neural_limited)
    else:
        if prev_input_scaler is None:
            raise ValueError("prev_input_scaler must be provided if update_scaler is False")
        input_scaler = prev_input_scaler
        train_neural_scaled = input_scaler.transform(train_neural_limited)
    
    # Use original output scaler (don't update it)
    train_kinematics_scaled = original_output_scaler.transform(train_kinematics_limited)
    
    # Create dataset and dataloader
    dataset_train = FingerDataset(train_neural_scaled, train_kinematics_scaled)
    dataloader_train = DataLoader(dataset_train, batch_size=min(lstm_batch_size, len(dataset_train)), shuffle=True)
    
    # Create optimizer and scheduler for fine-tuning
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, 
        lr=finetune_start_lr, 
        weight_decay=finetune_weight_decay, 
        final_lr=finetune_end_lr,
        total_steps=num_iterations
    )
    
    # Fine-tune the model
    model, history = train_model(
        model, 
        dataloader_train, 
        optimizer, 
        scheduler, 
        loss_fn=nn.MSELoss(),
        num_iterations=num_iterations,
        device=device,
        noise_neural_std=neural_noise_std,
        noise_bias_std=bias_noise_std
    )
    
    return model, input_scaler, original_output_scaler

def predict_lstm(model, input_scaler, output_scaler, test_neural, device, batch_size=512):
    """
    Make predictions using an LSTM model with batching to avoid memory issues.
    """
    model.eval()
    model.to(device)
    model.device = device
    
    # Scale the input
    test_neural_scaled = input_scaler.transform(test_neural)
    
    # Create a dataset and dataloader for batching
    dummy_targets = np.zeros((test_neural_scaled.shape[0], model.fc.out_features))
    test_dataset = FingerDataset(test_neural_scaled, dummy_targets)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Make predictions in batches
    predictions_scaled_list = []
    
    with torch.no_grad():
        for inputs, _ in test_dataloader:
            inputs = inputs.to(device)
            batch_predictions = model(inputs, save_hidden_state=False).cpu().numpy()
            predictions_scaled_list.append(batch_predictions)
    
    # Concatenate all batch predictions
    predictions_scaled = np.vstack(predictions_scaled_list)
    
    # Inverse transform predictions
    predictions = output_scaler.inverse_transform(predictions_scaled)
    
    return predictions

def evaluate_model_on_day(model, input_scaler, output_scaler, test_neural, test_kinematics):
    """
    Evaluate model performance on a test set.
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    predictions = predict_lstm(model, input_scaler, output_scaler, test_neural, device)
    
    # Calculate metrics
    mse_values = calculate_mse_per_dof(predictions, test_kinematics)
    correlations = calculate_correlation_per_dof(predictions, test_kinematics)
    r2_scores = calculate_r2_per_dof(predictions, test_kinematics)
    
    # Create results dictionary
    results = {
        'MSE': np.mean(mse_values),
        'Correlation': np.mean(correlations),
        'R2': np.mean(r2_scores)
    }
    
    # Add per-DOF metrics
    for i in range(NUM_OUTPUTS):
        results[f'MSE_DOF{i}'] = mse_values[i]
        results[f'Correlation_DOF{i}'] = correlations[i]
        results[f'R2_DOF{i}'] = r2_scores[i]
    
    return results

def calculate_day_difference(train_day, test_day):
    """
    Calculate the number of days between the training and testing dates.
    Args:
        train_day: Date string in format YYYY-MM-DD
        test_day: Date string in format YYYY-MM-DD
    Returns:
        int: Number of days between dates (negative if test day is before train day)
    """
    try:
        train_date = datetime.datetime.strptime(train_day, '%Y-%m-%d')
        test_date = datetime.datetime.strptime(test_day, '%Y-%m-%d')
        day_diff = (test_date - train_date).days
        return day_diff
    except Exception as e:
        print(f"Warning: Could not calculate day difference between {train_day} and {test_day}: {e}")
        return None

def run_continual_learning_experiment(starting_day_str, num_finetune_samples, finetune_num_iterations, 
                                     consecutive_dates, all_results, combination_count_start):
    """
    Run a single continual learning experiment for given parameters.
    
    Args:
        starting_day_str: Starting day string
        num_finetune_samples: Number of samples for fine-tuning
        finetune_num_iterations: Number of iterations for fine-tuning
        consecutive_dates: List of consecutive dates to test on
        all_results: List to append results to
        combination_count_start: Starting combination count for this experiment
        
    Returns:
        int: Number of combinations processed
    """
    combinations_processed = 0
    
    for rep in range(REPETITIONS):
        combination_count = combination_count_start + combinations_processed
        combinations_processed += 1
        print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: Run {combination_count + 1}")
        print(f"Start day: {starting_day_str}, Samples: {num_finetune_samples}, Iterations: {finetune_num_iterations}, Repetition: {rep+1}")
        
        try:
            # Load and prepare initial training data
            initial_data_file = os.path.join(data_folder, f'{starting_day_str}_preprocess.pkl')
            with open(initial_data_file, 'rb') as f:
                data_CO, data_RD = pickle.load(f)
            
            # Select appropriate data based on target type
            initial_data, initial_target_type_used = select_best_available_data(data_CO, data_RD, TARGET_TYPE)
            if initial_data is None:
                print(f"No suitable data available for starting day {starting_day_str} with target type {TARGET_TYPE}, skipping...")
                continue
            
            # Prepare initial training data
            print(f"Preparing initial training data for {starting_day_str} using {initial_target_type_used} data...")
            train_neural, test_neural, train_kinematics, test_kinematics = prep_data_and_split(
                initial_data, lstm_seq_len, NUM_TRAIN_TRIALS
            )
            
            # Train initial model
            model, input_scaler, output_scaler = train_initial_lstm(train_neural, train_kinematics, device)
            
            # Evaluate on initial day
            print(f"Evaluating on initial day {starting_day_str}...")
            initial_results = evaluate_model_on_day(model, input_scaler, output_scaler, test_neural, test_kinematics)
            
            # Store initial results
            result_entry = {
                'Starting_day': starting_day_str,
                'Day': starting_day_str,
                'Day_index': 0,
                'Repetition': rep,
                'Is_initial': True,
                'Target_type': initial_target_type_used,
                'Rel_day': 0,
                'Finetune_samples': num_finetune_samples,
                'Finetune_iterations': finetune_num_iterations,
                'Update_neural_scaler': UPDATE_NEURAL_SCALER,
                **initial_results
            }
            all_results.append(result_entry)
            
            print(f"Initial day performance - Correlation: {initial_results['Correlation']:.4f}, R2: {initial_results['R2']:.4f}")
            
            # Fine-tune and evaluate on consecutive days
            prev_input_scaler = input_scaler  # Start with initial scaler
            for day_idx, current_day in enumerate(tqdm(consecutive_dates[1:], desc="Fine-tuning on consecutive days"), 1):
                try:
                    # Check if we've exceeded the maximum day difference
                    day_diff = calculate_day_difference(starting_day_str, current_day)
                    if MAX_DAY_DIFFERENCE is not None and day_diff > MAX_DAY_DIFFERENCE:
                        break
                    
                    # Load data for current day (we already verified this data exists)
                    current_data_file = os.path.join(data_folder, f'{current_day}_preprocess.pkl')
                    with open(current_data_file, 'rb') as f:
                        data_CO, data_RD = pickle.load(f)
                    
                    # Select appropriate data based on target type
                    current_data, current_target_type_used = select_best_available_data(data_CO, data_RD, TARGET_TYPE)
                    
                    # Prepare data for current day
                    train_neural_full, test_neural, train_kinematics_full, test_kinematics = prep_data_and_split(
                        current_data, lstm_seq_len, NUM_TRAIN_TRIALS
                    )
                    
                    # Take only limited samples for fine-tuning
                    if FINETUNE_SAMPLES_FROM_END_OF_TRAINING:
                        # make sure we don't overlap with test set by using the seq len as a buffer
                        train_neural_limited = train_neural_full[-(num_finetune_samples+lstm_seq_len):-lstm_seq_len]
                        train_kinematics_limited = train_kinematics_full[-(num_finetune_samples+lstm_seq_len):-lstm_seq_len]
                    else:
                        train_neural_limited = train_neural_full[:num_finetune_samples]
                        train_kinematics_limited = train_kinematics_full[:num_finetune_samples]
                    
                    # Fine-tune model on limited data
                    print(f"Fine-tuning on {current_day} with {len(train_neural_limited)} samples using {current_target_type_used} data...")
                    model, input_scaler_used, output_scaler = finetune_lstm(
                        model, train_neural_limited, train_kinematics_limited, output_scaler,
                        prev_input_scaler=prev_input_scaler, update_scaler=UPDATE_NEURAL_SCALER, num_iterations=finetune_num_iterations, device=device
                    )
                    prev_input_scaler = input_scaler_used  # Update for next day
                    
                    # Evaluate on current day's test set
                    current_results = evaluate_model_on_day(model, input_scaler_used, output_scaler, test_neural, test_kinematics)
                    
                    # Store results
                    result_entry = {
                        'Starting_day': starting_day_str,
                        'Day': current_day,
                        'Day_index': day_idx,
                        'Repetition': rep,
                        'Is_initial': False,
                        'Target_type': current_target_type_used,
                        'Finetune_samples': num_finetune_samples,
                        'Finetune_iterations': finetune_num_iterations,
                        'Update_neural_scaler': UPDATE_NEURAL_SCALER,
                        'Rel_day': day_diff,
                        **current_results
                    }
                    all_results.append(result_entry)
                    
                    print(f"Day {current_day} performance - Correlation: {current_results['Correlation']:.4f}, R2: {current_results['R2']:.4f}")
                    
                except Exception as e:
                    print(f"Error processing day {current_day}: {e}")
                    continue
            
            # Save results periodically
            if all_results and (combination_count + 1) % SAVE_EVERY_N_ITERATIONS == 0:
                print(f"\n*** Saving results after {combination_count + 1} parameter combinations... ***")
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                norm_mode = 'neuralnormupdated' if UPDATE_NEURAL_SCALER else 'neuralnormfixed'
                results_df = pd.DataFrame(all_results)
                results_csv_path = os.path.join(
                    results_folder,
                    f'continual_learning_sweep_{TARGET_TYPE}_{norm_mode}_partial_{timestamp}.csv'
                )
                results_df.to_csv(results_csv_path, index=False)
                print(f"Partial results saved to: {results_csv_path}")
                
                # Print current progress summary
                current_combination_results = [r for r in all_results if r['Starting_day'] == starting_day_str and 
                                              r['Finetune_samples'] == num_finetune_samples and 
                                              r['Finetune_iterations'] == finetune_num_iterations and
                                              r['Repetition'] == rep]
                correlations = [r['Correlation'] for r in current_combination_results]
                print(f"Current combination correlations: {' '.join(f'{c:.3f}' for c in correlations)}")
                
        except Exception as e:
            print(f"Error with parameter combination {combination_count + 1}: {e}")
            continue
    
    return combinations_processed

########################################################
# Main Function
########################################################

def main():
    print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: Starting continual learning experiment")
    print(f"Starting days: {STARTING_DAYS}")
    print(f"Target type: {TARGET_TYPE}")
    print(f"Fine-tuning samples per day: {NUM_FINETUNE_SAMPLES_LIST}")
    print(f"Fine-tuning iterations: {FINETUNE_ITERATIONS_LIST}")
    print(f"Max days to test: {MAX_DAYS_TO_TEST}")
    print(f"Max day difference: {MAX_DAY_DIFFERENCE}")
    
    # Calculate total number of parameter combinations
    if FORCE_FINETUNE_ITERATIONS:
        # Count only sample sizes that have forced iterations defined
        valid_sample_sizes = len([s for s in NUM_FINETUNE_SAMPLES_LIST if s in FORCE_FINETUNE_ITERATIONS_DICT])
        total_combinations = len(STARTING_DAYS) * valid_sample_sizes * REPETITIONS
        print(f"Using forced finetune iterations: {FORCE_FINETUNE_ITERATIONS_DICT}")
    else:
        total_combinations = len(STARTING_DAYS) * len(NUM_FINETUNE_SAMPLES_LIST) * len(FINETUNE_ITERATIONS_LIST) * REPETITIONS
    print(f"Total parameter combinations to test: {total_combinations}")
    
    # Initialize results list
    all_results = []
    combination_count = 0
    
    # Sweep over starting days
    for starting_day_idx, starting_day_str in enumerate(STARTING_DAYS):
        print(f"\n{'='*60}")
        print(f"Starting day {starting_day_idx+1}/{len(STARTING_DAYS)}: {starting_day_str}")
        print(f"{'='*60}")
        
        # Get sorted dates and consecutive dates for testing
        all_dates = get_sorted_dates(data_folder)
        consecutive_dates = get_consecutive_dates_with_data(all_dates, starting_day_str, data_folder, TARGET_TYPE, MAX_DAYS_TO_TEST)
        
        print(f"Found {len(all_dates)} total days in dataset")
        print(f"Will test on {len(consecutive_dates)} consecutive days with {TARGET_TYPE} data starting from {starting_day_str}")
        if MAX_DAYS_TO_TEST is not None:
            print(f"(Limited to {MAX_DAYS_TO_TEST} days with actual data)")
        print(f"Note: When TARGET_TYPE='BOTH', CO data will be prioritized over RD data when both are available")
        
        # Sweep over fine-tuning sample sizes
        for samples_idx, num_finetune_samples in enumerate(NUM_FINETUNE_SAMPLES_LIST):
            print(f"\n{'-'*40}")
            print(f"Fine-tuning samples {samples_idx+1}/{len(NUM_FINETUNE_SAMPLES_LIST)}: {num_finetune_samples}")
            print(f"{'-'*40}")
            
            # Handle forced iterations or sweep over finetune iterations
            if FORCE_FINETUNE_ITERATIONS:
                # Use forced iterations if available, otherwise skip this sample size
                if num_finetune_samples in FORCE_FINETUNE_ITERATIONS_DICT:
                    finetune_num_iterations = FORCE_FINETUNE_ITERATIONS_DICT[num_finetune_samples]
                    print(f"Using forced finetune iterations: {finetune_num_iterations} for {num_finetune_samples} samples.")
                    
                    # Run the experiment for this parameter combination
                    combinations_processed = run_continual_learning_experiment(starting_day_str, num_finetune_samples, finetune_num_iterations, 
                                                                              consecutive_dates, all_results, combination_count)
                    combination_count += combinations_processed
                    
                    # Print current progress summary for the current parameter combination
                    current_combination_results = [r for r in all_results if r['Starting_day'] == starting_day_str and 
                                                  r['Finetune_samples'] == num_finetune_samples and 
                                                  r['Finetune_iterations'] == finetune_num_iterations]
                    correlations = [r['Correlation'] for r in current_combination_results]
                    print(f"Current parameter combination correlations: {' '.join(f'{c:.3f}' for c in correlations)}")
                else:
                    print(f"No forced finetune iterations found for {num_finetune_samples} samples, skipping...")
                    continue
            else:
                # Sweep over finetune iterations normally
                for iter_idx, finetune_num_iterations in enumerate(FINETUNE_ITERATIONS_LIST):
                    # Run the experiment for this parameter combination
                    combinations_processed = run_continual_learning_experiment(starting_day_str, num_finetune_samples, finetune_num_iterations, 
                                                                              consecutive_dates, all_results, combination_count)
                    combination_count += combinations_processed
                    
                    # Print current progress summary for the current parameter combination
                    current_combination_results = [r for r in all_results if r['Starting_day'] == starting_day_str and 
                                                  r['Finetune_samples'] == num_finetune_samples and 
                                                  r['Finetune_iterations'] == finetune_num_iterations]
                    correlations = [r['Correlation'] for r in current_combination_results]
                    print(f"Current parameter combination correlations: {' '.join(f'{c:.3f}' for c in correlations)}")
                
    # Final saving of all results
    if all_results:
        print(f"\n*** Saving final results for all sweeps... ***")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        norm_mode = 'neuralnormupdated' if UPDATE_NEURAL_SCALER else 'neuralnormfixed'
        results_df = pd.DataFrame(all_results)
        results_csv_path = os.path.join(
            results_folder,
            f'continual_learning_sweep_{TARGET_TYPE}_{norm_mode}_final_{timestamp}.csv'
        )
        results_df.to_csv(results_csv_path, index=False)
        print(f"Final continual learning results saved to: {results_csv_path}")
    
    print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: Continual learning experiment complete")
    
    return results_df if all_results else None

if __name__ == "__main__":
    main() 