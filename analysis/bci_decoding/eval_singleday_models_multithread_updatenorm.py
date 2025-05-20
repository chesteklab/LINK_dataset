import os
import pickle
import sys
import pandas as pd
import numpy as np
import torch
import time
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import concurrent.futures
import argparse

from lstm import LSTMModel
from data_utils import prep_data_and_split
from nn_utils import SequenceScaler, FingerDataset
from metrics import calculate_mse_per_dof, calculate_correlation_per_dof, calculate_r2_per_dof

"""
Multithreaded script for evaluating all trained models (LSTM and Ridge Regression) on every day of the dataset.
Results are saved as a CSV file with the following columns:
- Train day
- Train dataset type (CO/RD)
- Test day
- Test dataset type (CO/RD)
- Model type (LSTM/RR)
- Day difference (negative if test day is before train day)
- MSE (average across DOFs)
- Correlation (average across DOFs)
- R2 (average across DOFs)
- Per-DOF metrics (MSE_DOF0, MSE_DOF1, etc., Correlation_DOF0, Correlation_DOF1, etc., R2_DOF0, R2_DOF1, etc.)

This version updates the input normalizer for each evaluation day while keeping the original models and output scalers.
"""



########################################################
# Parameters
########################################################

DO_EVAL_LSTM = True
DO_EVAL_RR = True
NUM_THREADS = 24  # Default number of threads
LIMIT_EVALS_TO_CLOSE_DAYS = True # (set to True to run a limited number of evaluations, only close days to the training day)
MAX_NUM_CLOSE_DAYS = 100


# data_folder = '/run/user/1000/gvfs/smb-share:server=cnpl-drmanhattan.engin.umich.edu,share=share/Student Folders/Hisham_Temmar/big_dataset/2_autotrimming_and_preprocessing/preprocessing_092024_no7822nofalcon'
data_folder = './data_test'

# model_folder = '/home/joey/University of Michigan Dropbox/Joseph Costello/Chestek Lab/Code/NeuralNet/Temmar2025BigDataAnalysis/LINK_dataset/analysis/bci_decoding/single_day_models'
model_folder = './analysis/bci_decoding/single_day_models_withnoise'

results_folder = './analysis/bci_decoding/single_day_model_results'

NUM_TRAIN_TRIALS = 300 # (test on what's left after 300 trials)
NUM_CHANNELS = 96
NUM_OUTPUTS = 4

# LSTM params
lstm_seq_len = 20

# RR params
rr_seq_len = 8

########################################################
########################################################

# Create results folder if it doesn't exist
os.makedirs(results_folder, exist_ok=True)

########################################################
########################################################

def calculate_day_difference(train_day, test_day):
    """
    Calculate the number of days between the training and testing dates.
    
    Args:
        train_day: Date string in format YYYYMMDD
        test_day: Date string in format YYYYMMDD
        
    Returns:
        int: Number of days between dates (negative if test day is before train day)
    """
    try:
        # Parse the date strings to datetime objects
        train_date = datetime.datetime.strptime(train_day, '%Y-%m-%d')
        test_date = datetime.datetime.strptime(test_day, '%Y-%m-%d')
        
        # Calculate difference in days
        day_diff = (test_date - train_date).days
        return day_diff
    except:
        # If the dates are not in the expected format, return None
        return None

def predict_lstm(model, input_scaler, output_scaler, test_neural, device, batch_size=512):
    """
    Make predictions using an LSTM model with batching to avoid memory issues.
    
    Args:
        model: Trained LSTM model
        input_scaler: SequenceScaler for neural data
        output_scaler: StandardScaler for kinematics data
        test_neural: numpy array of shape (n_samples, seq_len, n_channels)
        device: torch device
        batch_size: Size of batches for prediction
        
    Returns:
        numpy array: Predictions of shape (n_samples, n_outputs)
    """
    model.eval()
    model.to(device)
    model.device = device
    
    # Scale the input
    test_neural_scaled = input_scaler.transform(test_neural)
    
    # Create a dataset and dataloader for batching
    dummy_targets = np.zeros((test_neural_scaled.shape[0], model.fc.out_features))  # Dummy targets with correct shape
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

def predict_ridge(model, test_neural):
    """
    Make predictions using a Ridge Regression model.
    
    Args:
        model: Trained Ridge Regression model
        test_neural: numpy array of shape (n_samples, n_features)
        
    Returns:
        numpy array: Predictions of shape (n_samples, n_outputs)
    """
    predictions = model.predict(test_neural)
    return predictions

def adapt_test_data_distribution(test_neural, train_neural_orig):
    """
    Adapts the test data to match the statistics of the original training data.
    
    Ridge Regression models are sensitive to input distribution shifts. This function transforms
    the test data to have similar statistical properties to the original training data.
    
    Args:
        test_neural: numpy array of test neural data
        train_neural_orig: numpy array of original training neural data
        
    Returns:
        numpy array: Transformed test neural data matching the original train data statistics
    """
    # Compute statistics for each feature in both datasets
    mean_orig = np.mean(train_neural_orig, axis=0)
    std_orig = np.std(train_neural_orig, axis=0) + 1e-8
    mean_test = np.mean(test_neural, axis=0)
    std_test = np.std(test_neural, axis=0) + 1e-8
    
    # Transform test data to match the statistics of original training data
    # Formula: X_transformed = (X - mean_test) / std_test * std_orig + mean_orig
    test_neural_adapted = (test_neural - mean_test) / std_test * std_orig + mean_orig
    
    return test_neural_adapted

def evaluate_lstm_model_on_day(model_file, test_day, dates, device):
    """
    Evaluate a single LSTM model on a single test day.
    
    Args:
        model_file: Name of the model file
        test_day: Day to test on
        dates: List of all available dates
        device: Device to run evaluation on
        
    Returns:
        list: List of result dictionaries
    """
    results = []
    
    # Extract train info from filename
    train_day = model_file.split('_lstm_')[0]
    train_data_type = model_file.split('_lstm_')[1].split('.pkl')[0]
    
    # Calculate day difference
    day_diff = calculate_day_difference(train_day, test_day)
    
    if LIMIT_EVALS_TO_CLOSE_DAYS and np.abs(day_diff) > MAX_NUM_CLOSE_DAYS:
        return results
    
    # Load the model and original scalers
    with open(os.path.join(model_folder, model_file), 'rb') as f:
        lstm_model, original_input_scaler, output_scaler = pickle.load(f)
    
    # Load test data
    test_file = os.path.join(data_folder, f'{test_day}_preprocess.pkl')
    with open(test_file, 'rb') as f:
        data_CO, data_RD = pickle.load(f)
    
    for test_data, test_data_type in [(data_CO, 'CO'), (data_RD, 'RD')]:
        if test_data is not None:
            # Prepare data for the test day - getting both train and test sets
            train_neural, test_neural, train_kinematics, test_kinematics = prep_data_and_split(
                test_data, lstm_seq_len, num_train_trials=NUM_TRAIN_TRIALS
            )
            
            # Create and fit new input scaler on the test day's training data
            updated_input_scaler = SequenceScaler()
            updated_input_scaler.fit(train_neural)
            
            # Make predictions using updated input scaler but original model and output scaler
            predictions = predict_lstm(lstm_model, updated_input_scaler, output_scaler, test_neural, device)
            
            # Calculate metrics
            mse_values = calculate_mse_per_dof(predictions, test_kinematics)
            correlations = calculate_correlation_per_dof(predictions, test_kinematics)
            r2_scores = calculate_r2_per_dof(predictions, test_kinematics)
            
            # Store results
            result = {
                'Train_day': train_day,
                'Train_data_type': train_data_type,
                'Test_day': test_day,
                'Test_data_type': test_data_type,
                'Model_type': 'LSTM_UpdatedNorm',
                'Day_diff': day_diff,
                'MSE': np.mean(mse_values),
                'Correlation': np.mean(correlations),
                'R2': np.mean(r2_scores)
            }
            
            # Add per-DOF metrics
            for i in range(NUM_OUTPUTS):
                result[f'MSE_DOF{i}'] = mse_values[i]
                result[f'Correlation_DOF{i}'] = correlations[i]
                result[f'R2_DOF{i}'] = r2_scores[i]
            
            results.append(result)
    
    return results

def evaluate_rr_model_on_day(model_file, test_day, dates):
    """
    Evaluate a single Ridge Regression model on a single test day.

    We have to do an annoying (and probably not efficient) thing where we load the original training data
    and use it to adapt the test data to the original training data distribution (since we didn't save stats 
    about the original training data in the model file).
    
    Args:
        model_file: Name of the model file
        test_day: Day to test on
        dates: List of all available dates
        
    Returns:
        list: List of result dictionaries
    """
    results = []
    
    # Extract train info from filename
    train_day = model_file.split('_rr_')[0]
    train_data_type = model_file.split('_rr_')[1].split('.pkl')[0]
    
    # Calculate day difference
    day_diff = calculate_day_difference(train_day, test_day)
    
    if LIMIT_EVALS_TO_CLOSE_DAYS and np.abs(day_diff) > MAX_NUM_CLOSE_DAYS:
        return results
    
    # Load the model
    with open(os.path.join(model_folder, model_file), 'rb') as f:
        rr_model = pickle.load(f)
    
    # Load original training data to get its statistics
    original_train_file = os.path.join(data_folder, f'{train_day}_preprocess.pkl')
    try:
        with open(original_train_file, 'rb') as f:
            orig_data_CO, orig_data_RD = pickle.load(f)
        
        # Find the right original training data based on model type
        if train_data_type == 'CO':
            orig_train_data = orig_data_CO
        else:  # 'RD'
            orig_train_data = orig_data_RD
            
        # Only continue if we have the original training data
        if orig_train_data is not None:
            # Get the original training neural data
            orig_train_neural, _, _, _ = prep_data_and_split(
                orig_train_data, rr_seq_len, num_train_trials=NUM_TRAIN_TRIALS
            )
            orig_train_neural = orig_train_neural.reshape(-1, NUM_CHANNELS * rr_seq_len)
        else:
            # Skip if original training data is not available
            return results
    except Exception as e:
        print(f"Error loading original training data {train_day}: {e}")
        return results
    
    # Load test data
    test_file = os.path.join(data_folder, f'{test_day}_preprocess.pkl')
    with open(test_file, 'rb') as f:
        data_CO, data_RD = pickle.load(f)
    
    for test_data, test_data_type in [(data_CO, 'CO'), (data_RD, 'RD')]:
        if test_data is not None:
            # Prepare data for the test day - getting both train and test sets
            train_neural, test_neural, train_kinematics, test_kinematics = prep_data_and_split(
                test_data, rr_seq_len, num_train_trials=NUM_TRAIN_TRIALS
            )
            
            # Reshape neural data for RR model
            train_neural_reshaped = train_neural.reshape(-1, NUM_CHANNELS * rr_seq_len)
            test_neural_reshaped = test_neural.reshape(-1, NUM_CHANNELS * rr_seq_len)
            
            # Transform test data to match original training data statistics
            test_neural_adapted = adapt_test_data_distribution(
                test_neural_reshaped, orig_train_neural
            )
            
            # Make predictions using the adapted test data
            predictions = predict_ridge(rr_model, test_neural_adapted)
            
            # Calculate metrics
            mse_values = calculate_mse_per_dof(predictions, test_kinematics)
            correlations = calculate_correlation_per_dof(predictions, test_kinematics)
            r2_scores = calculate_r2_per_dof(predictions, test_kinematics)
            
            # Store results
            result = {
                'Train_day': train_day,
                'Train_data_type': train_data_type,
                'Test_day': test_day,
                'Test_data_type': test_data_type,
                'Model_type': 'RR_UpdatedNorm',
                'Day_diff': day_diff,
                'MSE': np.mean(mse_values),
                'Correlation': np.mean(correlations),
                'R2': np.mean(r2_scores)
            }
            
            # Add per-DOF metrics
            for i in range(NUM_OUTPUTS):
                result[f'MSE_DOF{i}'] = mse_values[i]
                result[f'Correlation_DOF{i}'] = correlations[i]
                result[f'R2_DOF{i}'] = r2_scores[i]
            
            results.append(result)
    
    return results

def main(num_threads=NUM_THREADS):
    if LIMIT_EVALS_TO_CLOSE_DAYS:
        print('\n\033[91m!!! WARNING: EVALUATIONS LIMITED TO CLOSE DAYS !!!\033[0m')
        print(f"\tRunning with {MAX_NUM_CLOSE_DAYS} days close to the training day\n")
    
    print(f"Using {num_threads} threads for evaluation")
    
    # set device for LSTM
    device = torch.device('cpu')  # Force CPU for better multi-threading
    print(f'Using device: {device}')
    
    # get the list of dates and remove bad days
    dates = [f.split('_preprocess.pkl')[0] for f in os.listdir(data_folder) if f.endswith('_preprocess.pkl')]
    # with open(os.path.join(data_folder, 'bad_days.txt'), 'r') as f:
    #     bad_days = [line.strip() for line in f.readlines()]
    # dates = [date for date in dates if date not in bad_days]
    
    # Initialize separate results lists for LSTM and RR
    lstm_results = []
    rr_results = []
    
    # Get list of model files
    model_files = os.listdir(model_folder)
    lstm_models = [f for f in model_files if 'lstm' in f]
    rr_models = [f for f in model_files if 'rr' in f]
    
    print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: Starting evaluation with updated normalizers")
    
    # Evaluate LSTM models
    if DO_EVAL_LSTM:
        print(f"\nEvaluating {len(lstm_models)} LSTM models on {len(dates)} days with updated normalizers")
        
        # Create a list of all evaluation tasks
        tasks = [(model_file, test_day) for model_file in lstm_models for test_day in dates]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Create a dictionary mapping futures to their task info for easier tracking
            future_to_task = {
                executor.submit(evaluate_lstm_model_on_day, model_file, test_day, dates, device): 
                (model_file, test_day) for model_file, test_day in tasks
            }
            
            # Process results as they complete
            for i, future in enumerate(tqdm(concurrent.futures.as_completed(future_to_task), 
                                          total=len(tasks), 
                                          desc="LSTM Evaluations")):
                model_file, test_day = future_to_task[future]
                try:
                    results = future.result()
                    lstm_results.extend(results)
                except Exception as e:
                    print(f"\nError evaluating LSTM model {model_file} on day {test_day}: {e}")
        
        # Save LSTM results to CSV
        if lstm_results:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            lstm_df = pd.DataFrame(lstm_results)
            lstm_csv_path = os.path.join(results_folder, f'lstm_evaluation_updatednorm_{timestamp}.csv')
            lstm_df.to_csv(lstm_csv_path, index=False)
            print(f"\nLSTM evaluation complete. Results saved to: {lstm_csv_path}")
    
    # Evaluate Ridge Regression models
    if DO_EVAL_RR:
        print(f"\nEvaluating {len(rr_models)} Ridge Regression models on {len(dates)} days with updated normalizers")
        
        # Create a list of all evaluation tasks
        tasks = [(model_file, test_day) for model_file in rr_models for test_day in dates]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Create a dictionary mapping futures to their task info for easier tracking
            future_to_task = {
                executor.submit(evaluate_rr_model_on_day, model_file, test_day, dates): 
                (model_file, test_day) for model_file, test_day in tasks
            }
            
            # Process results as they complete
            for i, future in enumerate(tqdm(concurrent.futures.as_completed(future_to_task), 
                                          total=len(tasks), 
                                          desc="RR Evaluations")):
                model_file, test_day = future_to_task[future]
                try:
                    results = future.result()
                    rr_results.extend(results)
                except Exception as e:
                    print(f"\nError evaluating RR model {model_file} on day {test_day}: {e}")
        
        # Save RR results to CSV
        if rr_results:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            rr_df = pd.DataFrame(rr_results)
            rr_csv_path = os.path.join(results_folder, f'rr_evaluation_updatednorm_{timestamp}.csv')
            rr_df.to_csv(rr_csv_path, index=False)
            print(f"\nRR evaluation complete. Results saved to: {rr_csv_path}")
    
    print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: All evaluations complete")
    
    # Return both dataframes
    lstm_df = pd.DataFrame(lstm_results) if lstm_results else None
    rr_df = pd.DataFrame(rr_results) if rr_results else None
    return lstm_df, rr_df

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate single-day models with updated normalizers')
    parser.add_argument('--threads', type=int, default=NUM_THREADS,
                        help=f'Number of threads to use (default: {NUM_THREADS})')
    parser.add_argument('--lstm', action='store_true', 
                        help='Evaluate LSTM models')
    parser.add_argument('--rr', action='store_true',
                        help='Evaluate Ridge Regression models')
    
    args = parser.parse_args()
    
    # Set global flags based on arguments
    if args.lstm:
        DO_EVAL_LSTM = True
    if args.rr:
        DO_EVAL_RR = True
    
    # If neither --lstm nor --rr is specified, use the default values
    if not args.lstm and not args.rr:
        pass  # Use the default values defined at the top of the script
    
    main(num_threads=args.threads) 