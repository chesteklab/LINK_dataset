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
#
def calc_avg_sbps(dates,preprocessingdir,characterizationdir):
    sbp_avgs = pd.DataFrame(np.zeros((len(dates), 96), dtype=float), index=dates)
    sbp_avgs.index = pd.to_datetime(sbp_avgs.index)

    for date in dates:
        file = os.path.join(preprocessingdir,f'{date}_preprocess.pkl')

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
    #pdb.set_trace()
    sbp_avgs.to_csv(os.path.join(characterizationdir, "sbp_avgs.csv"))

def signal_power_over_time(dates, characterizationdir, crunch=False):
    fig, ax = plt.subplots(1, 1, figsize = (19.5, 6), sharex=True)
    if crunch:
        calc_avg_sbps(dates)
    
    sbp_avgs = pd.read_csv(os.path.join(characterizationdir, "sbp_avgs.csv"), index_col=0)
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
    
def sbp_distributions(ds,channels,preprocessingdir,characterizationdir):
    if isinstance(channels, int):
        ch_list = [channels]
        label = f"channel_{channels}"
    elif isinstance(channels, tuple) and len(channels) == 2:
        ch_start, ch_end = channels
        ch_list = list(range(ch_start, ch_end + 1))
        label = f"channel_{ch_start}-{ch_end}"
    elif isinstance(channels, list):
        ch_list = channels
        label = f"channel_{'_'.join(map(str, channels))}"

    distributions = []
    
    for date in ds:
        file = os.path.join(preprocessingdir, f'{date}_preprocess.pkl')

        with open(file, 'rb') as f:
            data_CO, data_RD = pickle.load(f)
        
        for ch in ch_list:
            if data_CO and data_RD:
                sbp = np.concatenate((data_CO['sbp'], data_RD['sbp']), axis=0)[:, ch]
            elif data_RD:
                sbp = data_RD['sbp'][:, ch]
                
            else:
                sbp = data_CO['sbp'][:, ch]
            
            date_df = pd.DataFrame({
                'date': pd.to_datetime(date),
                'value': sbp.reshape(-1)
            })
            distributions.append(date_df)
    
    distributions_df = pd.concat(distributions, ignore_index=True)
    
    distributions_df.to_csv(os.path.join(characterizationdir, f"sbp_distributions_{label}.csv"), index=False)
    
    return distributions_df

def sbp_distributions_per_ch(ds, ch, preprocessingdir, characterizationdir):
    distributions = []
    for date in ds:
        file = os.path.join(preprocessingdir,f'{date}_preprocess.pkl')

        with open(file, 'rb') as f:
            data_CO, data_RD = pickle.load(f)
        
        if data_CO and data_RD:
            sbp = np.concatenate((data_CO['sbp'], data_RD['sbp']),axis=0)[:,ch]
        elif data_RD:
            sbp = data_RD['sbp'][:,ch]
        else:
            sbp = data_CO['sbp'][:,ch]
        date_df = pd.DataFrame({
            'date': pd.to_datetime(date),
            'value': sbp
        })
        distributions.append(date_df)
    distributions_df = pd.concat(distributions, ignore_index=True)

    distributions_df.to_csv(os.path.join(characterizationdir, f"sbp_distributions_channel_{ch}.csv"), index=False)
    
def signal_distribution_per_ch(dates, ax, ch, preprocessingdir, characterizationdir, n_bins=10, display_mean = True, display_median = True, show_trend=True,crunch=False,robust=True, percentile_range=(0.3,99.7)):

    file_path = os.path.join(characterizationdir, f"sbp_distributions_channel_{ch}.csv")

    try:
        sbp_avgs = pd.read_csv(file_path, index_col=0)
    except FileNotFoundError:
        print(f"Error: Channel {ch} average SBP data not found.")
        sbp_avgs = None

    if crunch or not os.path.exists(file_path):
        distribution_df = sbp_distributions_per_ch(dates, ch, preprocessingdir, characterizationdir)
    else:
        try:
            #print('loaded')
            distribution_df = pd.read_csv(file_path)
            distribution_df['date'] = pd.to_datetime(distribution_df['date'])
        except FileNotFoundError:
            print(f"Error: Channel {ch} distribution data not found. Generating using sbp_distributions_per_ch.")
            distribution_df = sbp_distributions_per_ch(dates, ch, preprocessingdir, characterizationdir)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6))
    
    distribution_df['days'] = (distribution_df['date'] - distribution_df['date'].min()).dt.days
    
    if robust:
        # low_val = np.percentile(distribution_df['value'], percentile_range[0])
        # high_val = np.percentile(distribution_df['value'], percentile_range[1])
        low_val = 4.
        high_val = 30.
    else:
        low_val = distribution_df['value'].min() # outliers mess this up, use percentile instead
        high_val = distribution_df['value'].max()
    
    bins = np.linspace(low_val, high_val, n_bins + 1)
    bin_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(n_bins)]
    bin_labels = bin_labels[::-1]

    days = sorted(distribution_df['days'].unique())
    date_labels = [pd.to_datetime(dates)[list(days).index(d)].strftime('%Y-%m-%d') 
                   if d in days else None for d in days]
    
    heatmap_data = np.zeros((n_bins, len(days)))
    
    for i, day in enumerate(days):
        day_data = distribution_df[distribution_df['days'] == day]['value']
        
        hist, _ = np.histogram(day_data, bins=np.concatenate([[-np.inf], bins[1:-1], [np.inf]]), density=False)

        heatmap_data[:, i] = hist[::-1]
    
    heatmap_df = pd.DataFrame(heatmap_data, index=bin_labels, columns=date_labels)
    
    sns.heatmap(
        heatmap_df, 
        annot=False, 
        cbar_kws={'label': 'Freq'},
        cmap="Blues",
        ax=ax,
        robust=robust
    )
    
    ax.set_title(f'SBP Distribution Heatmap for Channel {ch} Over Time')
    ax.set_ylabel('SBP Value Bins (uV)')
    ax.set_xlabel('Date')

    day_means = [distribution_df[distribution_df['days'] == day]['value'].mean() for day in days]
    mean_positions = [n_bins - 1 - ((m - low_val) / (high_val - low_val) * (n_bins - 1)) + 0.5 for m in day_means]
    
    day_medians = [distribution_df[distribution_df['days'] == day]['value'].median() for day in days]
    median_positions = [n_bins - 1 - ((m - low_val) / (high_val - low_val) * (n_bins - 1)) + 0.5 for m in day_medians]
    
    handles, labels = [], []
    
    if display_mean:
        for i, mean_pos in enumerate(mean_positions):
            x_pos = i + 0.5
            ax.plot(x_pos, mean_pos, marker='o', color='red', label='Mean' if i == 0 else "")
        
        handles.append(plt.Line2D([0], [0], marker='o', color='red', linestyle='None'))
        labels.append('Mean')
        
        if show_trend and len(days) > 1:
            x_positions = np.arange(len(days)) + 0.5
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_positions, mean_positions)
            
            trend_x = np.array([0, len(days)])
            trend_y = intercept + slope * trend_x
            
            mean_trend_line = ax.plot(trend_x, trend_y, color='black',linestyle='--', linewidth=1.5)
            
            sbp_range = high_val - low_val
            bin_range = n_bins - 1
            mean_sbp_slope = -slope * (sbp_range / bin_range)
            mean_sbp_intercept = high_val - (intercept - 0.5) * (sbp_range / bin_range)

            handles.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5))
            labels.append(f'Mean Trend (r^2={r_value**2:.3f}; Mean Slope: {mean_sbp_slope:.4f} uV/day; Intercept: {mean_sbp_intercept:.4f}; p-value: {p_value:.4f})')

    if display_median:
        for i, median_pos in enumerate(median_positions):
            x_pos = i + 0.5
            ax.plot(x_pos, median_pos, marker='o', color='orange', label='Median' if i == 0 else "")
        
        handles.append(plt.Line2D([0], [0], marker='o', color='orange', linestyle='None'))
        labels.append('Median')
        
        if show_trend and len(days) > 1:
            x_positions = np.arange(len(days)) + 0.5
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_positions, median_positions)
            
            trend_x = np.array([0, len(days)])
            trend_y = intercept + slope * trend_x
            sbp_range = high_val - low_val
            bin_range = n_bins - 1
            median_sbp_slope = -slope * (sbp_range / bin_range)
            median_sbp_intercept = high_val - (intercept - 0.5) * (sbp_range / bin_range)

            
            median_trend_line = ax.plot(trend_x, trend_y, 'black', linestyle='dotted', linewidth=1.5)
            
            handles.append(plt.Line2D([0], [0], color='black', linestyle='dotted', linewidth=1.5))
            labels.append(f'Median Trend (r^2={r_value**2:.3f}; Median Slope: {median_sbp_slope:.4f} uV/day; Intercept: {median_sbp_intercept:.4f}; p-value: {p_value:.4f})')
    
    if handles:
        ax.legend(handles, labels, loc='upper right', framealpha=0.7)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return ax

def signal_distribution_all_ch(dates,preprocessingdir,characterizationdir):
    for i in range(96):
        sys.stdout.write(f"\r Channel Processing: {i}")
        sys.stdout.flush()
        fig, ax = plt.subplots(1, 1, figsize = (19.5, 6), sharex=True)
        signal_distribution_per_ch(dates, ax,i,preprocessingdir,characterizationdir,100,display_mean=False,display_median=False, crunch=False)
        plt.savefig(os.path.join(characterizationdir, f"sbp_distributions_{i}_same.png"))