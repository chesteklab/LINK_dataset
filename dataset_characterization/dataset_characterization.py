import os
import pandas as pd
import numpy as np
from datetime import datetime
import config
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pdb
import os 
import sys
import pdb

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats

from collections import defaultdict
import pdb


def get_good_days():
    # review_results = pd.read_csv(config.reviewpath)
    # # filter out bad days
    # good_days = review_results.loc[review_results['Status'] == 'good']

    # #create histogram for current version (fewer days I guess?)
    # dates = good_days['Date'].to_numpy()

    filenames = [f for f in os.listdir(config.preprocessingdir) if not f.startswith('.')] #in case there are any hidden files
    filenames.remove('bad_days.txt')
    filenames.sort()
    dates = [file[0:10] for file in filenames]
    return dates

def create_time_plot(dates):

    dates = np.asarray([datetime.strptime(x, '%Y-%m-%d') for x in dates])

    fig, ax = plt.subplots(figsize=(10, 1), layout='constrained', dpi=1200)
    ax.hist(dates, bins=100, color='k')
    ax.spines[['right','top','left']].set_visible(False)
    ax.set(yticks=(0,5,10),title="Distribution of days included in v0.2",ylabel="# of days")
    ax.set_axisbelow(True)
    ax.grid(axis="y",zorder=10)

    fig.savefig(os.path.join(config.outputdir,f'dataset_timeline_v02.pdf'))

def calc_avg_sbps(dates):
    sbp_avgs = pd.DataFrame(np.zeros((len(dates), 96), dtype=float), index=dates)
    sbp_avgs.index = pd.to_datetime(sbp_avgs.index)

    for date in dates:
        file = os.path.join(config.preprocessingdir,f'{date}_preprocess.pkl')

        with open(file, 'rb') as f:
            data_CO, data_RD = pickle.load(f)
        
        if data_CO and data_RD:
            sbp = np.concatenate((data_CO['sbp'], data_RD['sbp']),axis=0)
        elif data_RD:
            sbp = data_RD['sbp']
        else:
            sbp = data_CO['sbp']

        sbp_avgs.loc[date] = np.mean(sbp, axis=0)

    # once everything is in, take date out of index and make its own column
    pdb.set_trace()
    sbp_avgs.to_csv(os.path.join(config.characterizationdir, "sbp_avgs.csv"))

# tuning analyses
def signal_power_over_time(dates, ax, crunch=True):

    if crunch:
        calc_avg_sbps(dates)
    
    sbp_avgs = pd.read_csv(os.path.join(config.characterizationdir, "sbp_avgs.csv"), index_col=0)
    sbp_avgs.index = pd.to_datetime(sbp_avgs.index)

    sbp_avgs['days'] = (sbp_avgs.index - sbp_avgs.index[0]).to_series().dt.days.to_numpy()

    sbp_long = sbp_avgs.reset_index(names="date").melt(id_vars=["date",'days'], var_name='channel', value_name='sbp')
    sbp_long['sbp'] = sbp_long['sbp']*0.25
    
    # plot average
    slope, intercept, r, p_val, sterr = stats.linregress(x=sbp_long['days'], y=sbp_long['sbp'])

    sns.regplot(data=sbp_long, x="days", y="sbp", marker='.', color='k', 
                line_kws={'color':'r'}, scatter_kws={'alpha':0.5}, ax=ax, label="Per-channel, single day averages")
    
    ax.annotate(f'slope: {slope:.3e} PR/day\nint: {intercept:.3f}\nr^2:{r**2:.3f},\np:{p_val:.3f}', (0, 10), ha='left')

    ax.set(xlabel='Day since first recording',
           ylabel='Mean SBP (uV)',
           title='SBP across channels over time',
           ylim=(0,70*.25),
           yticks=(0,4,8,16))
    

def participation_ratio(dx_flat):
    # Dxflat is a 2D array of shape (n_features, n_samples)
    dx_flat = dx_flat - np.mean(dx_flat, axis=1)[:,np.newaxis] # subtract the mean so that we can actually get the covariance matrix
    DD = np.matmul(dx_flat, dx_flat.T) # 96 x 135 * 135 x 96
    U, S, V = np.linalg.svd(DD)
    pr = np.sum(S)**2/np.sum(S**2)
    return pr

def calc_participation_ratios(dates):
    pr_dict = {'date': [],
               'chan_mask': [],
               'pr':[],
               'pr_mask':[],
               'target_style': []
               }
    
    for date in dates:
        file = os.path.join(config.preprocessingdir,f'{date}_preprocess.pkl')

        with open(file, 'rb') as f:
            data_CO, data_RD = pickle.load(f)

        # use CO if its there
        feat = data_CO if data_CO else data_RD
        TS = 'CO' if data_CO else 'RD'

        tcfr = feat['tcfr'] * 1000 / config.data_params['binsize']
        sbp = feat['sbp'] * 1000 / config.data_params['binsize']

        if np.sum(np.isnan(sbp)) > 0:
            continue

        chanMask = np.where(np.mean(tcfr, axis=0) > 1)[0]
        if chanMask.shape[0] == 0:
            continue

        PR_mask = participation_ratio(sbp[:, chanMask].T)
        PR = participation_ratio(sbp.T)
        print(PR_mask, PR, chanMask.shape)

        pr_dict['date'].append(date)
        pr_dict['chan_mask'].append(chanMask)
        pr_dict['pr'].append(PR)
        pr_dict['pr_mask'].append(PR_mask)
        pr_dict['target_style'].append(TS)

    pr_df = pd.DataFrame.from_dict(pr_dict)
    pr_df.to_csv(os.path.join(config.characterizationdir,"participation_ratios.csv"))

def create_pr_plot(ax):
    pr_df = pd.read_csv(os.path.join(config.characterizationdir,"participation_ratios.csv"))
    pr_df.set_index(['date'], inplace=True)
    pr_df.index = pd.to_datetime(pr_df.index)
    pr_df['days'] = (pr_df.index - pr_df.index[0]).to_series().dt.days.to_numpy()

    p = sns.regplot(data=pr_df, x="days", y="pr_mask", marker='.', color='k', 
                line_kws={'color':'r'}, scatter_kws={'alpha':0.6}, ax=ax, label="Single-day PR", ci=None)
    
    slope, intercept, r, p_val, sterr = stats.linregress(x=pr_df['days'], y=pr_df['pr_mask'])

    ax.annotate(f'slope: {slope:.3e} PR/day\nint: {intercept:.3f}\nr^2:{r**2:.3f},\np:{p_val:.3f}', (0, 10), ha='left')
    xticks = (0, 500, 1000, 1500)
    ax.set(title='Participation ratio over time', ylabel='Participation Ratio (PR)*', 
           xlabel="Day since first recording", xticks=xticks, yticks=(0,4,8,12,16))
    ax.legend()

def create_time_plot(dates):
    dates_dt = np.asarray([datetime.strptime(x, '%Y-%m-%d') for x in dates])

    fig, ax = plt.subplots(figsize=(20, 2.25), layout='constrained')
    ax.vlines(dates, .5, '.')
    ax.axhline(0, c='black')

    fig, ax = plt.subplots(figsize=(10, 1), layout='constrained')
    ax.hist(dates_dt, bins=100, color='k')
    ax.spines[['right','top','left']].set_visible(False)
    fig.savefig(os.path.join(config.outputdir,"time_hist.pdf"))
    plt.show()

def plot_target_examples():
    path = "Z:\Student Folders\Hisham_Temmar\\big_dataset\sfn_round_2\\2021-10-15_preprocess.pkl"
    with open(path, 'rb') as f:
        data_co, data_rd = pickle.load(f)

    fig, ax = plt.subplots(1,2, figsize=(7.25, 3.25), layout='constrained', sharex=True, sharey=True)
    data = (data_co, data_rd)
    title = ('Center-Out (CO)','Random (RD)')
    for i in range(2):
        pos = data[i]['target_positions'][:,[0,1]]
        ax[i].scatter(pos[:,0],pos[:,1],c='k')
        ax[i].set_box_aspect(1)
        ax[i].set(yticks=(0,0.5,1),
                xticks=(0,0.5,1),
                title=f'{title[i]} Targets',
                xlabel='IDX Flex %',
                ylabel='MRS Flex %')
        
    outpath = "C:\\Users\\Hisham\\University of Michigan Dropbox\\Hisham Temmar\Science Communication\Conferences\SFN2024\\figures"
    fig.savefig(os.path.join(outpath, "targpos.pdf"))

def sfn_decoding():
    perfs_path = "Z:\Student Folders\Hisham_Temmar\\big_dataset\sfn_stuff\performances"

    files =[
        'RR_correlation_perfs_253_datasets.pkl',
        'LSTM_correlation_perfs_253_datasets_3_models',
        'RR_correlation_perfs_across_days_253_datasets',
        'LSTM_correlation_perfs_across_days_253_datasets',
        'hisham_sbp_dates.pkl'
    ]

    data = {
        'rr_performance':None,
        'lstm_performance':None,
        'rr_performance_across_days':None,
        'lstm_performance_across_days':None
    }

    for key, filename in zip(data.keys(), files):
        with open(os.path.join(perfs_path, filename), 'rb') as f:
            data[key] = pickle.load(f)
    
    data['dates'] = [datetime.combine(date, datetime.min.time()) for date in data['dates']]
    data['days'] = [(date - data['dates'][0]).days for date in data['dates']]

    # extract same day performances
    rr_same_day = np.diagonal(data['rr_performance'], axis1=0, axis2=1)

    # average over the 3 lstm models on each day
    lstm_perfs = data['lstm_performance']
    num_datasets = lstm_perfs.shape[1]
    models_per_dataset = lstm_perfs.shape[0] // num_datasets
    degrees_of_freedom  = lstm_perfs.shape[2]

    pdb.set_trace()
    lstm_data_overall = []
    for j in range(degrees_of_freedom):
        lstm_data = np.zeros([num_datasets])
        for i in range(num_datasets):
            lstm_data[i] = np.mean(lstm_perfs[i*models_per_dataset:i*models_per_dataset+models_per_dataset, i, j])
        lstm_data_overall.append(lstm_data)