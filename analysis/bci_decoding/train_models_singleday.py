import os
import pickle
import sys
import pandas as pd
import random
import numpy as np
import torch
import time
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

from lstm import LSTMModel
from data_utils import prep_data_and_split
from nn_utils import FingerDataset, create_optimizer_and_scheduler, train_model, SequenceScaler

"""
Script for training an LSTM and a Ridge Regression model on every day of the dataset.
Models are saved as .pkl files.

After running this, use the eval script to evaluate the performance of the models on all days.

"""

########################################################
# Parameters
########################################################

data_folder = '/run/user/1000/gvfs/smb-share:server=cnpl-drmanhattan.engin.umich.edu,share=share/Student Folders/Hisham_Temmar/big_dataset/2_autotrimming_and_preprocessing/preprocessing_092024_no7822nofalcon'

model_folder = '/home/joey/University of Michigan Dropbox/Joseph Costello/Chestek Lab/Code/NeuralNet/Temmar2025BigDataAnalysis/LINK_dataset/single_day_models'

NUM_TRAIN_TRIALS = 300
NUM_CHANNELS = 96
NUM_OUTPUTS = 4

# LSTM params
lstm_seq_len = 20
lstm_num_inputs = NUM_CHANNELS
lstm_num_outputs = NUM_OUTPUTS
lstm_hidden_size = 300
lstm_num_layers = 1
lstm_dropout = 0
lstm_batch_size = 64
lstm_num_iterations = 2000  # Number of training iterations
lstm_start_lr = 2e-4
lstm_end_lr = 1e-5
lstm_weight_decay = 0.001

# RR params
rr_seq_len = 8

########################################################
########################################################

# set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

########################################################
########################################################

def train_lstm(train_neural, train_kinematics, device=device):   
    """
    train_neural: numpy array of shape (n_samples, seq_len, n_channels)
    train_kinematics: numpy array of shape (n_samples, 2 * n_fingers)
    """
    
    # Use the sequence scaler for neural data
    input_scaler = SequenceScaler()
    train_neural_scaled = input_scaler.fit_transform(train_neural)
    
    # Fit kinematics scaler
    output_scaler = StandardScaler()
    train_kinematics_scaled = output_scaler.fit_transform(train_kinematics)
    
    # Create datasets and dataloaders
    dataset_train = FingerDataset(train_neural_scaled, train_kinematics_scaled)
    dataloader_train = DataLoader(dataset_train, batch_size=lstm_batch_size, shuffle=True)

    # Create the model
    model = LSTMModel(input_dim=lstm_num_inputs,
                      hidden_size=lstm_hidden_size,
                      num_states=lstm_num_outputs,
                      num_layers=lstm_num_layers,
                      drop_prob=lstm_dropout
                     )
    model.to(device)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, 
        lr=lstm_start_lr, 
        weight_decay=lstm_weight_decay, 
        final_lr=lstm_end_lr,  # Final learning rate is 10% of initial
        total_steps=lstm_num_iterations  # Linear decay over all training iterations
    )
    
    # Train the model
    model, losses = train_model(
        model, 
        dataloader_train, 
        optimizer, 
        scheduler, 
        loss_fn=nn.MSELoss(),
        num_iterations=lstm_num_iterations,
        device=device
    )

    return model, input_scaler, output_scaler


def train_ridge(neural_hist, kinematics):
    # train an sklearn ridge regression model
    model = Ridge(alpha=0.1, fit_intercept=True)
    model.fit(neural_hist, kinematics)
    return model

########################################################
########################################################  


# get the list of dates and remove bad days
dates = [f.split('_preprocess.pkl')[0] for f in os.listdir(data_folder) if f.endswith('_preprocess.pkl')]
with open(os.path.join(data_folder, 'bad_days.txt'), 'r') as f:
    bad_days = [line.strip() for line in f.readlines()]
dates = [date for date in dates if date not in bad_days]

# loop over each day
for date in dates:
    print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: Training models for {date}")

    # load data
    fpath = os.path.join(data_folder, f'{date}_preprocess.pkl')
    with open(fpath, 'rb') as f:
        data_CO, data_RD = pickle.load(f)

    for this_data, data_type in [(data_CO, 'CO'), (data_RD, 'RD')   ]:
        if this_data is not None:
            # train LSTM
            train_neural, test_neural, train_kinematics, test_kinematics = prep_data_and_split(this_data, lstm_seq_len, NUM_TRAIN_TRIALS)
            lstm_model, input_scaler, output_scaler = train_lstm(train_neural, train_kinematics, device)
            with open(os.path.join(model_folder, f'{date}_lstm_{data_type}.pkl'), 'wb') as f:
                pickle.dump((lstm_model, input_scaler, output_scaler), f)

            # train Ridge Regression
            train_neural, test_neural, train_kinematics, test_kinematics = prep_data_and_split(this_data, rr_seq_len, NUM_TRAIN_TRIALS)
            train_neural = train_neural.reshape(-1, NUM_CHANNELS * rr_seq_len)
            test_neural = test_neural.reshape(-1, NUM_CHANNELS * rr_seq_len)
            rr_model = train_ridge(train_neural, train_kinematics)
            with open(os.path.join(model_folder, f'{date}_rr_{data_type}.pkl'), 'wb') as f:
                pickle.dump(rr_model, f)

print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: Training complete")
