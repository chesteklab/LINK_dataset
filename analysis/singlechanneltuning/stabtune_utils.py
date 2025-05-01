import pandas as pd
import os
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import pickle
import glob
import tqdm
import pdb

def compute_channel_tuning(neural, full_behavior, velocity_tuning = False):

    if velocity_tuning:
        behavior = full_behavior[:,[2,3]]
    else:
        behavior = full_behavior[:,[0,1]]

    channel_tunings = np.zeros((neural.shape[1], behavior.shape[1]))
    for channel in range(neural.shape[1]):
        single_channel = neural[:, channel]
        single_channel = (single_channel - np.mean(single_channel))/np.std(single_channel)
        
        # calculate tuning to each dof at a variety of lags, pick the one which gives the largest l2 norm across the 2 dof
        lag_range = 11 # look up to 10 bins of lag
        tuning_vec = np.zeros((lag_range, behavior.shape[1]))
        for lag in range(lag_range):
            if lag != 0:
                lagged_channel = single_channel[:-lag]
                lagged_behavior = behavior[lag:,:]
            else:
                lagged_channel = single_channel
                lagged_behavior = behavior
            # calculate tuning for each dof
            lm= LinearRegression(fit_intercept=True)
            lm.fit(lagged_channel.reshape(-1,1), lagged_behavior)
            tuning_vec[lag, :] = lm.coef_[:,0]
        
        best_lag = np.argmax(np.linalg.norm(tuning_vec, 2, axis=1))
        channel_tunings[channel, :] = tuning_vec[best_lag, :]

    # calculate magnitudes and angles
    magnitudes = np.linalg.norm(channel_tunings, 2, 1)
    angles = np.degrees(np.arctan2(channel_tunings[:,1], channel_tunings[:,0]))
    
    channels = np.arange(neural.shape[1])
    tuning_df = pd.DataFrame(np.concat((channels.reshape(-1,1), channel_tunings, magnitudes.reshape(-1,1), angles.reshape(-1,1)), axis=1), columns=('channel', 'idx', 'mrs', 'magnitude', 'angle'))

    return tuning_df

def compute_tuning_data(load_dir, is_save = False, save_dir = None):
    # Path to the folder containing pkl files
    data_folder = load_dir

    # check if the directories exists
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Directory {data_folder} does not exist.")
    if is_save and save_dir is None:
        raise ValueError("save_dir must be provided if is_save is True.")
    elif is_save and not os.path.exists(save_dir):
        print(f"Directory {save_dir} does not exist. Creating.")
        os.makedirs(save_dir)

    # Get list of pkl files
    if not os.path.join(data_folder, '*.pkl'):
        raise ValueError(f"No pkl files found in {data_folder}.")

    pkl_files = sorted(glob.glob(os.path.join(data_folder, '*.pkl')))

    # Dictionary to store results
    df_list = []

    # Process each pkl file
    for file in tqdm.tqdm(pkl_files, desc=f"Processing files in {data_folder}"):
        # Extract date from filename (assuming format like 'YYYY-MM-DD_data.pkl')
        date = pd.to_datetime(os.path.basename(file).split('_')[0])

        # Load data
        with open(file, 'rb') as f:
            data_CO, data_RD = pickle.load(f)
            
        # Compute channel tuning
        try:
            if data_CO != None: 
                channel_tuning = compute_channel_tuning(data_CO['sbp'], data_CO['finger_kinematics'], velocity_tuning=True)
            elif data_RD != None:
                channel_tuning = compute_channel_tuning(data_RD['sbp'], data_RD['finger_kinematics'], velocity_tuning=True)
        except:
            print(f"Error processing file {file}")
            continue

        channel_tuning['date'] = date
        df_list.append(channel_tuning)
    
    df = pd.concat(df_list)
    pdb.set_trace()
    # Save the DataFrame to a CSV file
    if is_save:
        output_file = os.path.join(save_dir, 'channelwise_stability_tuning.csv')
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    return df

def load_tuning_data(dir, overwrite=False):
    if not os.path.exists(os.path.join(dir, 'channelwise_stability_tuning.csv')) or overwrite:
        print(f"Directory {dir} does not exist, computing tuning data. Note that user input is required to continue.")
        load_dir = input("Please enter the directory containing the data files: ")
        df = compute_tuning_data(load_dir, is_save=True, save_dir=dir)
    else:
        df = pd.read_csv(os.path.join(dir, 'channelwise_stability_tuning.csv'))
    return df
    