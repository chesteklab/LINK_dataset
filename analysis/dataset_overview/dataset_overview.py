import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime
#import config
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pdb
import os
import re
import sys
import glob
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats
from collections import defaultdict
from tutorial.utils.data_tools import extract_dates_from_filenames, load_day
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



if __name__ == "__main__":
    create_dataset_overview_figure()