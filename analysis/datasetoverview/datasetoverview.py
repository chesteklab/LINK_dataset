import os
import pandas as pd
import numpy as np
from datetime import datetime
#import config
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pdb
import os 
import sys
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats
from collections import defaultdict

def create_dataset_overview_figure():
    # load in data, call create time plot - maybe dont need create time plot and dataset count over time?
    # load in one day with center out and random, make plots of target positions
    
    pass

def create_time_plot(ds, outputdir):
    ds = np.asarray([datetime.strptime(x, '%Y-%m-%d') for x in ds])
    fig, ax = plt.subplots(figsize=(10, 2), layout='constrained', dpi=1200)
    ax.hist(ds, bins=100, color='k')
    ax.spines[['right','top','left']].set_visible(False)
    ax.set(yticks=(0,5,10),title="Distribution of days included in v0.2",ylabel="# of days")
    ax.set_axisbelow(True)
    ax.grid(axis="y",zorder=10)
    fig.savefig(os.path.join(outputdir,f'dataset_timeline_v02.pdf'))

def dataset_count_over_time(dates, outputdir):
    create_time_plot(dates, outputdir)

def target_positions(dates, preprocessingdir, outputdir, isOneDay = False, dayIdx = None):
    targetpos_co = np.zeros((1,2))
    targetpos_rd = np.zeros((1,2))
    for date in dates[0:]:
        if(isOneDay):
            selectedDateIdx = dayIdx
            date = dates[selectedDateIdx]
            file = os.path.join(preprocessingdir,f'{date}_preprocess.pkl')

            with open(file, 'rb') as f:
                data_CO, data_RD = pickle.load(f)

            if data_CO and data_RD:
                targetpos_co = np.concatenate((targetpos_co, data_CO['target_positions']),axis=0)
                targetpos_rd = np.concatenate((targetpos_rd, data_RD['target_positions']),axis=0)
            elif data_RD:
                targetpos_rd = np.concatenate((targetpos_rd, data_RD['target_positions']),axis=0)
            else:
                targetpos_co = np.concatenate((targetpos_co, data_CO['target_positions']),axis=0)
        else:
            file = os.path.join(preprocessingdir,f'{date}_preprocess.pkl')

            with open(file, 'rb') as f:
                data_CO, data_RD = pickle.load(f)
            
            if data_CO and data_RD:
                targetpos_co = np.concatenate((targetpos_co, data_CO['target_positions']),axis=0)
                targetpos_rd = np.concatenate((targetpos_rd, data_RD['target_positions']),axis=0)
            elif data_RD:
                targetpos_rd = np.concatenate((targetpos_rd, data_RD['target_positions']),axis=0)
            else:
                targetpos_co = np.concatenate((targetpos_co, data_CO['target_positions']),axis=0)

    # switch to these for one day plotting
        
        
    fig, ax = plt.subplots(1,2, figsize=(14.5, 6.5), layout='constrained', sharex=True, sharey=True)

    title = ('Center-Out (CO)','Random (RD)')

    ax[0].scatter(targetpos_co[1:,0],targetpos_co[1:,1],c='k')
    ax[0].set_box_aspect(1)
    ax[0].set(yticks=(0,0.5,1),
            xticks=(0,0.5,1),
            title=f'{title[0]} Targets',
            xlabel='IDX Flex %',
            ylabel='MRS Flex %')

    ax[1].scatter(targetpos_rd[1:,0],targetpos_rd[1:,1],c='k')
    ax[1].set_box_aspect(1)
    ax[1].set(yticks=(0,0.5,1),
            xticks=(0,0.5,1),
            title=f'{title[1]} Targets',
            xlabel='IDX Flex %',
            ylabel='MRS Flex %')
        
    fig.savefig(os.path.join(outputdir, "targpos.pdf"))