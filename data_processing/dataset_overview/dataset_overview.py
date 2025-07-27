import os
import pandas as pd
import numpy as np
from datetime import datetime
#import config
import matplotlib.pyplot as plt
import pickle
import pdb
import os
import glob

from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from data_tools import extract_dates_from_filenames, load_day
import mpl_config
# data_path = "Z:\Student Folders\\Nina_Gill\data\only_good_days_timeouts"
# output_dir = "D:\\University of Michigan Dropbox\Hisham Temmar\Science Communication\Papers\LINK_dataset\experimental setup"

def create_dataset_overview_figure(data_path, output_dir):
    # load in data, call create time plot - maybe dont need create time plot and dataset count over time?
    # load in one day with center out and random, make plots of target positions
    dates = extract_dates_from_filenames(data_path)
    create_time_plot(dates, data_path, output_dir)
    pass

def create_time_plot(ds, data_path, outputdir):
    n_sessions = []
    for date in ds:
        data_CO, data_RD = load_day(date,data_path)
        num_sessions = 0
        if data_CO:
            num_sessions += len(data_CO['trial_count'])
        if data_RD:
            num_sessions += len(data_RD['trial_count'])
        n_sessions.append(num_sessions)

    # ds = np.asarray([datetime.strptime(x, '%Y-%m-%d') for x in ds])
    data = pd.DataFrame({'date':ds, 'n_sessions':np.array(n_sessions)})
    per_week = data.groupby(pd.Grouper(key='date', freq="W-MON"))
    fig, ax = plt.subplots(figsize=(10, 2), layout='constrained')
    ax.bar(per_week.sum().index, per_week.sum()['n_sessions'], width=7, color='k', edgecolor='k', align='edge')
    ax.spines[['right','top','left']].set_visible(False)
    ax.set(title="Number of available trials, per week",ylabel="# of trials")
    ax.set_axisbelow(True)
    ax.grid(axis="y",zorder=10)
    fig.savefig(os.path.join(outputdir,f'dataset_timeline_v03.pdf'))

def target_positions(dates, data_path, isOneDay = False, dayIdx = None):
    targetpos_co = np.zeros((1,2))
    targetpos_rd = np.zeros((1,2))
    if False:
        a = 5
    else:
        cos = []
        for date in dates:
            file = os.path.join(data_path, f'{date.strftime("%Y-%m-%d")}_preprocess.pkl')
            # fig, ax = plt.subplots(1,2, figsize=(4, 2), layout='constrained', sharex=True, sharey=True)
            with open(file, 'rb') as f:
                data_CO, data_RD = pickle.load(f)
            pdb.set_trace()
            
            if data_CO:
                targets_CO = np.unique(data_CO['target_positions'], axis=0)
                cos.append(len(targets_CO))
                # ax[0].scatter(targets_CO[1:,0],targets_CO[1:,1],c='k')
            
            if data_RD:
                targets_RD = np.unique(data_RD['target_positions'], axis=0)
                # ax[1].scatter(targets_RD[1:,0],targets_RD[1:,1],c='k')
        plt.figure()
        plt.plot(cos)
        plt.show()

def create_behavior_metrics_figure():
    # load the dataset pkl files
    dataset_pkl_dir = "D:\\link_dataset_pkl"

    mean_bitrates = []
    std_bitrates = []
    dates = []
    for filename in tqdm(os.listdir(dataset_pkl_dir)):
        if filename.endswith(".pkl"):
            file_path = os.path.join(dataset_pkl_dir, filename)
            with open(file_path, "rb") as file:
                data_co, data_rd = pickle.load(file)
            
            # get date from filename
            date_str = filename.split('_')[0]

            #convert date string to datetime object
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')

        else:
            continue

        # check if data_co and data_rd are not None
        bitrates = []
        if data_co is not None:
            bitrates.append(calculate_bitrate(data_co))
        if data_rd is not None:
            bitrates.append(calculate_bitrate(data_rd))

        bitrates = np.concatenate(bitrates)
        mean_bitrates.append(np.mean(bitrates))
        std_bitrates.append(np.std(bitrates)/np.sqrt(len(bitrates)))
        dates.append(date_obj)

    # convert dates to relative day from first date
    first_date = min(dates)
    dates = [(date - first_date).days for date in dates]
    # plot the mean and std over time according to the dates
    fig, ax = plt.subplots(1,1, figsize=(10, 5), layout='constrained')
    ax.plot(dates, mean_bitrates, label='Mean Bitrate', color='black')
    ax.fill_between(dates, np.array(mean_bitrates) - np.array(std_bitrates), np.array(mean_bitrates) + np.array(std_bitrates), color='black', alpha=0.2)
    ax.set(xlabel='Days', ylabel='Bitrate (bps)', title='Mean Bitrate over Time', xlim=(0, max(dates)))
    plt.show()

def calculate_bitrate(data):
    """
    calculate per trial bitrate in bits per second using fitts law
    """
    # get behavior from data
    behavior = data['finger_kinematics']

    #get trial start indices
    trial_start_indices = data['trial_index']

    #get trial lengths
    trial_lengths = data['trial_count']

    # get target positions
    target_positions = data['target_positions']

    # organize data into trials using trial start indices and trial lengthsand calculate bitrate using target positions
    bitrates = []
    for i in range(len(trial_start_indices) - 1):
        trial_data = behavior[trial_start_indices[i]:trial_start_indices[i] + trial_lengths[i]]
        target_position = target_positions[i]
        
        # assuming target scaling is 100
        widths = np.array([.0375 * 2, .0375 * 2])  # Example widths for x and y dimensions


        if len(trial_data) > 0:
            # calculate distance berween trial starting position and target position
            trial_start_position = trial_data[0, :2]  # Assuming first two columns are x, y positions
            dists = np.abs(target_position - trial_start_position[:2])
            dists -= widths
            
            trial_id = np.log2(1 + dists / (2 * widths))
            trial_length = (trial_lengths[i] * 20 - 750)/1000 # assuming 20ms per sample and 750 hold time
            trial_throughput = np.sum(trial_id) / trial_length
            # Calculate bitrate for each trial
            bitrates.append(trial_throughput)
    return np.array(bitrates)

if __name__ == "__main__":
    # create_dataset_overview_figure()
    create_behavior_metrics_figure()