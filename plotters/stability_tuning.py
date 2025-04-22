### IMPORTS ###
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from collections import defaultdict
import glob
import sys
import pdb
from dataset_characterization import dataset_characterization
import matplotlib.gridspec as gridspec
import config

# This script is used to calculate the stability of the model performance over multiple days over multiple channels
# and store them in pandas dataframes for later use.

### FUNCTION TO COMPUTE CHANNEL TUNING ###

def compute_channel_tuning(data):
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

def create_channelwise_tuning_dataframe(preprocessingdir, outputdir):
    ### LOAD AND PREPROCESS DATA ###
    # Approximate total number of datasets
    total_datasets = 405

    # Create a progress bar
    #pbar = tqdm(total=total_datasets, desc="Processing datasets")

    # Path to the folder containing pkl files (FIND THIS IN HISHAMS STUDENT FOLDER -> BIG DATASET -> AUTOTRIMMING AND PREPROCESSING)
    data_folder = preprocessingdir

    # Get list of pkl files
    pkl_files = sorted(glob.glob(os.path.join(data_folder, '*.pkl')))

    # Dictionary to store results
    results = {}

    # Process each pkl file
    counter = 405
    for file in pkl_files:
        sys.stdout.write(f"\r Date Processing: {file}")
        sys.stdout.flush()
        # Extract date from filename (assuming format like 'YYYY-MM-DD_data.pkl')
        date = pd.to_datetime(os.path.basename(file).split('_')[0])

        # Load data
        with open(file, 'rb') as f:
            data = pickle.load(f)
            
        # Compute channel tuning
        try:
            channel_tuning = compute_channel_tuning(data)
            if channel_tuning is not None:
                results[date] = channel_tuning
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue

        results[date] = channel_tuning

        counter += 1
        if counter == 405:
            break
    
    # convert to desired dataframe
    data_rows = []

    # Iterate through the results dictionary
    for date, channels in results.items():
        for channel, metrics in channels.items():
            magnitude = metrics['magnitude']
            angle = metrics['angle']
            channel_value = int(channel.split('_')[-1])
            data_rows.append({'date': date, 'channel': channel_value, 'magnitude': magnitude, 'angle': angle})

    df = pd.DataFrame(data_rows)

    print(df)
    # Save the dataframe to a CSV file
    df.to_csv(os.path.join(outputdir, 'channelwise_stability_tuning.csv'), index=False)

def create_channelwise_tuning_plot(outputdir):
    # Load the channelwise tuning dataframe
    df = pd.read_csv(os.path.join(outputdir, 'channelwise_stability_tuning.csv'))

    # Create a pivot table for better visualization
    pivot_df = df.pivot(index='date', columns='channel', values='magnitude')

    # Set the date as index for plotting
    pivot_df.set_index('date', inplace=True)

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_df, cmap='viridis', annot=False, cbar_kws={'label': 'Magnitude'})
    plt.title('Channel-wise Stability Tuning')
    plt.xlabel('Channels')
    plt.ylabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(outputdir, 'channelwise_stability_tuning_plot.png'))