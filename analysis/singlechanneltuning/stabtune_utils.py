import pandas as pd
import os
import numpy as np
from scipy import stats
import pickle
import glob
import tqdm

def compute_single_channel_tuning(data):
    # Check if data is a tuple and extract the dictionary if necessary
    if isinstance(data, tuple):
        if data[0] is None and isinstance(data[1], dict):
            data = data[1]
        elif isinstance(data[0], dict):
            data = data[0]
        else:
            print(f"Unexpected data structure: {type(data)}")
            return None

    # Extract relevant data
    finger_kinematics = data['finger_kinematics']
    sbp = data['sbp']
    
    channel_tuning = {}
    
    for i in range(sbp.shape[1]):  # Iterate over channels
        channel_data = sbp[:, i]
        
        # Check if channel_data or finger_kinematics are constant
        if np.all(channel_data == channel_data[0]) or np.all(finger_kinematics[:, 0] == finger_kinematics[0, 0]) or np.all(finger_kinematics[:, 1] == finger_kinematics[0, 1]):
            # If any are constant, set correlation to 0
            channel_tuning[f'sbp_channel_{i}'] = {
                'magnitude': 0,
                'angle': 0,
                'corr_index': 0,
                'corr_mrp': 0
            }
        else:
            # Compute correlation with index and mrp positions
            corr_index = stats.pearsonr(channel_data, finger_kinematics[:, 0])[0]
            corr_mrp = stats.pearsonr(channel_data, finger_kinematics[:, 1])[0]
            
            # Treat correlations as vector components
            magnitude = np.sqrt(corr_index**2 + corr_mrp**2)
            angle = np.degrees(np.arctan2(corr_mrp, corr_index))
            
            channel_tuning[f'sbp_channel_{i}'] = {
                'magnitude': magnitude,
                'angle': angle,
                'corr_index': corr_index,
                'corr_mrp': corr_mrp
            }
    
    return channel_tuning

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
    results = {}

    # Process each pkl file
    for file in tqdm.tqdm(pkl_files, desc=f"Processing files in {data_folder}"):
        # Extract date from filename (assuming format like 'YYYY-MM-DD_data.pkl')
        date = pd.to_datetime(os.path.basename(file).split('_')[0])

        # Load data
        with open(file, 'rb') as f:
            data = pickle.load(f)
            
        # Compute channel tuning
        try:
            channel_tuning = compute_single_channel_tuning(data)
            if channel_tuning is not None:
                results[date] = channel_tuning
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue

        results[date] = channel_tuning

    data_rows = []

    #TODO: this way of conversion is inefficient, could be done in one go, but it is working for now so it is kept that way.

    # Iterate through the results dictionary
    for date, channels in results.items():
        for channel, metrics in channels.items():
            magnitude = metrics['magnitude']
            angle = metrics['angle']
            channel_value = int(channel.split('_')[-1])
            data_rows.append({'date': date, 'channel': channel_value, 'magnitude': magnitude, 'angle': angle})

    df = pd.DataFrame(data_rows)

    # Save the DataFrame to a CSV file
    if is_save:
        output_file = os.path.join(save_dir, 'channelwise_stability_tuning.csv')
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    return df

def load_tuning_data(dir):
    if not os.path.exists(os.path.join(dir, 'channelwise_stability_tuning.csv')):
        print(f"Directory {dir} does not exist, computing tuning data. Note that user input is required to continue.")
        load_dir = input("Please enter the directory containing the data files: ")
        df = compute_tuning_data(load_dir, is_save=True, save_dir=dir)
    else:
        df = pd.read_csv(os.path.join(dir, 'channelwise_stability_tuning.csv'))
    return df
    