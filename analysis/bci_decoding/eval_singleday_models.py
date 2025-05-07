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

from lstm import LSTMModel
from data_utils import prep_data_and_split
from nn_utils import SequenceScaler, FingerDataset
from metrics import calculate_mse_per_dof, calculate_correlation_per_dof, calculate_r2_per_dof

"""
Script for evaluating all trained models (LSTM and Ridge Regression) on every day of the dataset.
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
"""

DO_EVAL_LSTM = False
DO_EVAL_RR = True

DEBUG = False

########################################################
# Parameters
########################################################

data_folder = '/run/user/1000/gvfs/smb-share:server=cnpl-drmanhattan.engin.umich.edu,share=share/Student Folders/Hisham_Temmar/big_dataset/2_autotrimming_and_preprocessing/preprocessing_092024_no7822nofalcon'

model_folder = '/home/joey/University of Michigan Dropbox/Joseph Costello/Chestek Lab/Code/NeuralNet/Temmar2025BigDataAnalysis/LINK_dataset/analysis/bci_decoding/single_day_models'

results_folder = '/home/joey/University of Michigan Dropbox/Joseph Costello/Chestek Lab/Code/NeuralNet/Temmar2025BigDataAnalysis/LINK_dataset/analysis/bci_decoding/single_day_model_results'

NUM_TRAIN_TRIALS = 300 # (test on what's left after 300 trials)
NUM_CHANNELS = 96
NUM_OUTPUTS = 4

# LSTM params
lstm_seq_len = 20

# RR params
rr_seq_len = 8

########################################################
########################################################

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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

########################################################
########################################################  

def main():
    # if debug, print out a warning in red
    if DEBUG:
        print('\n\033[91m!!! WARNING: DEBUG MODE IS ON !!!\033[0m\n')

    # get the list of dates and remove bad days
    dates = [f.split('_preprocess.pkl')[0] for f in os.listdir(data_folder) if f.endswith('_preprocess.pkl')]
    with open(os.path.join(data_folder, 'bad_days.txt'), 'r') as f:
        bad_days = [line.strip() for line in f.readlines()]
    dates = [date for date in dates if date not in bad_days]
    
    # Initialize separate results lists for LSTM and RR
    lstm_results = []
    rr_results = []
    
    # Get list of model files
    model_files = os.listdir(model_folder)
    lstm_models = [f for f in model_files if 'lstm' in f]
    rr_models = [f for f in model_files if 'rr' in f]
    
    print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: Starting evaluation")
    
    # Evaluate LSTM models
    if DO_EVAL_LSTM:
        for i, model_file in enumerate(lstm_models):
            # Print progress information
            print(f"\nEvaluating LSTM model {i+1} of {len(lstm_models)}: {model_file}")
            
            # Extract train info from filename
            train_day = model_file.split('_lstm_')[0]
            train_data_type = model_file.split('_lstm_')[1].split('.pkl')[0]
            
            # Load the model
            with open(os.path.join(model_folder, model_file), 'rb') as f:
                lstm_model, input_scaler, output_scaler = pickle.load(f)
            
            # Test on all datasets
            for test_day in tqdm(dates, desc="Testing on days"):
                # Calculate day difference
                day_diff = calculate_day_difference(train_day, test_day)

                if DEBUG:
                    if np.abs(day_diff) > 5:
                        continue
                
                # Load test data
                test_file = os.path.join(data_folder, f'{test_day}_preprocess.pkl')
                with open(test_file, 'rb') as f:
                    data_CO, data_RD = pickle.load(f)
                
                for test_data, test_data_type in [(data_CO, 'CO'), (data_RD, 'RD')]:
                    if test_data is not None:
                        # Prepare test data
                        _, test_neural, _, test_kinematics = prep_data_and_split(test_data, lstm_seq_len, num_train_trials=NUM_TRAIN_TRIALS)
                        
                        # Make predictions
                        predictions = predict_lstm(lstm_model, input_scaler, output_scaler, test_neural, device)
                        
                        # Calculate metrics
                        mse_values = calculate_mse_per_dof(predictions, test_kinematics)
                        correlations = calculate_correlation_per_dof(predictions, test_kinematics)
                        r2_scores = calculate_r2_per_dof(predictions, test_kinematics)

                        if DEBUG:
                            print(correlations)
                        
                        # Store results
                        result = {
                            'Train_day': train_day,
                            'Train_data_type': train_data_type,
                            'Test_day': test_day,
                            'Test_data_type': test_data_type,
                            'Model_type': 'LSTM',
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
                        
                        lstm_results.append(result)
        
        # Save LSTM results to CSV
        if lstm_results:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            lstm_df = pd.DataFrame(lstm_results)
            lstm_csv_path = os.path.join(results_folder, f'lstm_evaluation_{timestamp}.csv')
            lstm_df.to_csv(lstm_csv_path, index=False)
            print(f"\nLSTM evaluation complete. Results saved to: {lstm_csv_path}")
    
    # Evaluate Ridge Regression models
    if DO_EVAL_RR:
        for i, model_file in enumerate(rr_models):
            # Print progress information
            print(f"\nEvaluating RR model {i+1} of {len(rr_models)}: {model_file}")
            
            # Extract train info from filename
            train_day = model_file.split('_rr_')[0]
            train_data_type = model_file.split('_rr_')[1].split('.pkl')[0]
            
            # Load the model
            with open(os.path.join(model_folder, model_file), 'rb') as f:
                rr_model = pickle.load(f)
            
            # Test on all datasets
            for test_day in tqdm(dates, desc="Testing on days"):
                # Calculate day difference
                day_diff = calculate_day_difference(train_day, test_day)

                if DEBUG:
                    if np.abs(day_diff) > 5:
                        continue
                
                # Load test data
                test_file = os.path.join(data_folder, f'{test_day}_preprocess.pkl')
                with open(test_file, 'rb') as f:
                    data_CO, data_RD = pickle.load(f)
                
                for test_data, test_data_type in [(data_CO, 'CO'), (data_RD, 'RD')]:
                    if test_data is not None:
                        # Prepare test data
                        _, test_neural, _, test_kinematics = prep_data_and_split(test_data, rr_seq_len, num_train_trials=NUM_TRAIN_TRIALS)
                        test_neural = test_neural.reshape(-1, NUM_CHANNELS * rr_seq_len)
                        
                        # Make predictions
                        predictions = predict_ridge(rr_model, test_neural)
                        
                        # Calculate metrics
                        mse_values = calculate_mse_per_dof(predictions, test_kinematics)
                        correlations = calculate_correlation_per_dof(predictions, test_kinematics)
                        r2_scores = calculate_r2_per_dof(predictions, test_kinematics)

                        if DEBUG:
                            print(correlations)
                        
                        # Store results
                        result = {
                            'Train_day': train_day,
                            'Train_data_type': train_data_type,
                            'Test_day': test_day,
                            'Test_data_type': test_data_type,
                            'Model_type': 'RR',
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
                        
                        rr_results.append(result)
        
        # Save RR results to CSV
        if rr_results:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            rr_df = pd.DataFrame(rr_results)
            rr_csv_path = os.path.join(results_folder, f'rr_evaluation_{timestamp}.csv')
            rr_df.to_csv(rr_csv_path, index=False)
            print(f"\nRR evaluation complete. Results saved to: {rr_csv_path}")
    
    print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: All evaluations complete")
    
    # Return both dataframes
    lstm_df = pd.DataFrame(lstm_results) if lstm_results else None
    rr_df = pd.DataFrame(rr_results) if rr_results else None
    return lstm_df, rr_df

if __name__ == "__main__":
    main() 