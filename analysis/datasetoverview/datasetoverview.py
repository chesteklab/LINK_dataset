import os
import pandas as pd
import numpy as np
from datetime import datetime
#import config
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pdb
import re
import os 
import sys
import glob
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats
from collections import defaultdict
output_path = 'D:\\University of Michigan Dropbox\\Hisham Temmar\\Science Communication\\Papers\\LINK_dataset\\experimental setup'
data_path = "Z:\\Student Folders\\Nina_Gill\\data\\unnormalized"

def create_dataset_overview_figure():
    # load in data, call create time plot - maybe dont need create time plot and dataset count over time?
    # load in one day with center out and random, make plots of target positions

    dates = extract_dates_from_filenames()

    plot_data_distribution(dates, output_path)
    plot_target_positions(dates, data_path, output_path)

def extract_dates_from_filenames():
    # Find all matching .pkl files
    pkl_files = glob.glob(os.path.join(data_path, '*_preprocess.pkl'))

    dates = []
    for file_path in pkl_files:
        filename = os.path.basename(file_path)
        match = re.match(r'(\d{4}-\d{2}-\d{2})_preprocess\.pkl', filename)
        if match:
            dates.append(match.group(1))

    dates = np.asarray([datetime.strptime(date, '%Y-%m-%d') for date in dates])
    return dates #sorted(dates) 

def plot_data_distribution(ds, outputdir):
    total_days = len(ds)
    total_span = (ds[-1] - ds[0]).days
    fig, ax = plt.subplots(figsize=(10, 2), layout='constrained', dpi=300)
    ax.hist(ds, bins=100, color='k')
    ax.spines[['right','top','left']].set_visible(False)
    ax.set(yticks=(0,5,10),title=f"Distribution of {total_days} included days, spanning {total_span} days",ylabel="# of days")
    ax.set_axisbelow(True)
    ax.grid(axis="y",zorder=10)
    fig.savefig(os.path.join(outputdir,f'dataset_timeline_v02.pdf'))

def plot_target_positions(dates, data_path, outputdir, isOneDay = False, dayIdx = None):
    
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
            

    # for date in dates[0:]:
    #     if(isOneDay):
    #         selectedDateIdx = dayIdx
    #         date = dates[selectedDateIdx]
    #         file = os.path.join(data_path,f'{date}_preprocess.pkl')

    #         with open(file, 'rb') as f:
    #             data_CO, data_RD = pickle.load(f)

    #         if data_CO and data_RD:
    #             targetpos_co = np.concatenate((targetpos_co, data_CO['target_positions']),axis=0)
    #             targetpos_rd = np.concatenate((targetpos_rd, data_RD['target_positions']),axis=0)
    #         elif data_RD:
    #             targetpos_rd = np.concatenate((targetpos_rd, data_RD['target_positions']),axis=0)
    #         else:
    #             targetpos_co = np.concatenate((targetpos_co, data_CO['target_positions']),axis=0)
    #     else:
    #         file = os.path.join(data_path,f'{date}_preprocess.pkl')

    #         with open(file, 'rb') as f:
    #             data_CO, data_RD = pickle.load(f)
            
    #         if data_CO and data_RD:
    #             targetpos_co = np.concatenate((targetpos_co, data_CO['target_positions']),axis=0)
    #             targetpos_rd = np.concatenate((targetpos_rd, data_RD['target_positions']),axis=0)
    #         elif data_RD:
    #             targetpos_rd = np.concatenate((targetpos_rd, data_RD['target_positions']),axis=0)
    #         else:
    #             targetpos_co = np.concatenate((targetpos_co, data_CO['target_positions']),axis=0)

    # # switch to these for one day plotting
        
        
    # fig, ax = plt.subplots(1,2, figsize=(14.5, 6.5), layout='constrained', sharex=True, sharey=True)

    # title = ('Center-Out (CO)','Random (RD)')

    # ax[0].scatter(targetpos_co[1:,0],targetpos_co[1:,1],c='k')
    # ax[0].set_box_aspect(1)
    # ax[0].set(yticks=(0,0.5,1),
    #         xticks=(0,0.5,1),
    #         title=f'{title[0]} Targets',
    #         xlabel='IDX Flex %',
    #         ylabel='MRS Flex %')

    # ax[1].scatter(targetpos_rd[1:,0],targetpos_rd[1:,1],c='k')
    # ax[1].set_box_aspect(1)
    # ax[1].set(yticks=(0,0.5,1),
    #         xticks=(0,0.5,1),
    #         title=f'{title[1]} Targets',
    #         xlabel='IDX Flex %',
    #         ylabel='MRS Flex %')
        
    # fig.savefig(os.path.join(outputdir, "targpos.pdf"))


if __name__ == "__main__":
    create_dataset_overview_figure()