import os
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
from tqdm import tqdm
#import config
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pdb
import os 
import sys
import glob
import re
from . import signal_utils
import matplotlib as mpl
# #some basic text parameters for figures
# mpl.rcParams['font.family'] = "Atkinson Hyperlegible" # if installed but not showing up, rebuild mpl cache
# mpl.rcParams['font.size'] = 10
# mpl.rcParams['savefig.format'] = 'pdf'
# mpl.rcParams['axes.unicode_minus'] = False
# # mpl.rcParams['axes.titlesize'] = 14
# # mpl.rcParams['axes.labelsize'] = 12
# mpl.rcParams['axes.titlelocation'] = 'center'
# mpl.rcParams['axes.titleweight'] = 'bold'
# mpl.rcParams['figure.constrained_layout.use'] = True
# # mpl.rcParams['figure.titlesize'] = 14
# mpl.rcParams['figure.titleweight'] = 'bold'
# mpl.rcParams['pdf.fonttype'] = 42
from utils import mpl_config
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from scipy import stats
from utils.data_tools import *
from collections import defaultdict

def create_signal_quality_figure(data_path, output_path, calc_avg_sbp = False, calculate_pr = False):
    dates = extract_dates_from_filenames(data_path)
    print(f"Found {len(dates)} dates")
    
    fig, ax = plt.subplots(3,1, sharex=True)
    # data_CO, data_RD = load_day(dates[0])
    # for i in range(96):
    #     plt.hist(data_CO['sbp'][:,i])
    # plt.show()

    #show example channel heatmap over time
    # calc_sbp_heatmaps = False
    # if calc_sbp_heatmaps:
    #     signal_utils.calc_sbp_heatmaps(dates, data_path, output_path)

    # create_channel_heatmaps(dates, channel = 0, output_path)

    #average sbp figure
    # calc_avg_sbp = False
    if calc_avg_sbp:
        signal_utils.calc_avg_sbps(dates, data_path, output_path)
    create_avg_sbp_plot(ax[0], output_path)
    
    # calculate participation ratio on each day
    # calculate_pr = False
    if calculate_pr:
        signal_utils.calc_pr_all_days(dates, data_path, output_path)

    # create pr figure
    create_pr_plot(ax[2], output_path)
    active_channels_plot(ax[1],os.path.join(output_path,"participation_ratios.csv"))
    plt.savefig(os.path.join(output_path, "signal_change"))
    plt.show()
    # per channel sbp distributions over time
    # channels = 4
    # channels = [1,3,5] # select multiple channels
    # channels = (1,5) # select range of channels
    # sbp_distributions(dates,channels,preprocessingdir,characterizationdir)
    # channel_chosen = 4
    # fig, ax = plt.subplots(1, 1, figsize = (19.5, 6), sharex=True)
    # signal_distribution_per_ch(dates, ax,channel_chosen, preprocessingdir, characterizationdir,100,display_mean=False,display_median=False, crunch=False)
    # plt.savefig(os.path.join(characterizationdir, f"sbp_distributions_{4}_same.png"))
    #maybe do this for all channels? find a way to combine
    
    # add mutual information calculations per day

    # calc_mutual_information(dates, characterizationdir="./output_dir")

def active_channels_plot(ax, path_to_pr_calcs):

    df = pd.read_csv(path_to_pr_calcs)
    
    # Convert string representation of lists into actual lists
    df['chan_mask'] = df['chan_mask'].apply(signal_utils.clean_chan_mask)

    # Count num of active channels
    df['num_active_channels'] = df['chan_mask'].apply(len)

    # convert and sort dates
    df['date_ordinal'] = pd.to_datetime(df['date']).apply(lambda date: date.toordinal())
    df = df.sort_values('date_ordinal')

    ax.plot(df['date_ordinal'], df['num_active_channels'], 'k.')
    ax.set(title='Number of Active Channels per Day', xlabel='Date', ylabel="# of Active Channels")
    ax.grid(True)


def create_pr_plot(ax, output_path):
    pr_df = pd.read_csv(os.path.join(output_path,"participation_ratios.csv"))
    pr_df.set_index(['date'], inplace=True)
    pr_df.index = pd.to_datetime(pr_df.index)
    pr_df['days'] = (pr_df.index - pr_df.index[0]).to_series().dt.days.to_numpy()
    pr_df['date_ordinal'] = pr_df.index.to_series().apply(lambda date: date.toordinal())

    p = sns.regplot(data=pr_df, x="date_ordinal", y="participation_ratio_active", marker='.', color='k', 
                line_kws={'color':'r'}, scatter_kws={'alpha':0.6}, ax=ax, label="Single-day PR", ci=None)
    
    slope, intercept, r, p_val, sterr = stats.linregress(x=pr_df['days'], y=pr_df['participation_ratio_active'])

    ax.annotate(f'slope: {slope:.3e} PR/day\nint: {intercept:.3f}\nr^2:{r**2:.3f},\np:{p_val:.3f}', (pr_df['date_ordinal'].min(), 10), ha='left')
    ax.set(title='Participation ratio over time (Active Channels)', ylabel='Participation Ratio (PR)', yticks=[0,5,10,15,20], xlabel=None)
    ax.legend()

def create_avg_sbp_plot(ax, output_path):
    sbp_avgs = pd.read_csv(os.path.join(output_path, "sbp_avgs.csv"), index_col=0)
    sbp_avgs.index = pd.to_datetime(sbp_avgs.index)

    sbp_avgs['days'] = (sbp_avgs.index - sbp_avgs.index[0]).to_series().dt.days.to_numpy()

    sbp_long = sbp_avgs.reset_index().rename(columns={'index': 'date'}).melt(id_vars=["date",'days'], var_name='channel', value_name='sbp')
    sbp_long['sbp'] = sbp_long['sbp']*0.25
    sbp_long['date_ordinal'] = sbp_long['date'].apply(lambda date: date.toordinal())
    
    # plot average
    slope, intercept, r, p_val, sterr = stats.linregress(x=sbp_long['days'], y=sbp_long['sbp'])

    per_day_mean = sbp_long.groupby('days')['sbp'].mean()
    # ax.plot(per_day_mean)
    sns.regplot(data=sbp_long, x="date_ordinal", y="sbp", marker='.', color='k', 
                line_kws={'color':'r'}, scatter_kws={'alpha':0.5}, ax=ax, label="Per-channel, single day averages")
    
    outliers = sbp_long.loc[sbp_long['sbp'] > 20]['date_ordinal']
    dummy_points = np.zeros(len(outliers))
    dummy_points = dummy_points + 19.9

    ax.plot(outliers, dummy_points, 'x', ms=10)
    ax.annotate(f'slope: {slope:.3e} PR/day\nint: {intercept:.3f}\nr^2:{r**2:.3f},\np:{p_val:.3f}', (sbp_long['date_ordinal'].min(), 10), ha='left')
    # new_labels = [date.fromordinal(int(item)) for item in ax.get_xticks()]
    ax.set(xlabel=None,
           ylabel='Mean SBP (uV)',
           title='SBP across channels over time',
           ylim=(0,20),
           yticks=(0,5,10,15,20))
    # Create tick positions
    tick_labels = ['2020-01-01','2020-07-01','2021-01-01','2021-07-01','2022-01-01','2022-07-01','2023-01-01','2023-07-01']
    tick_positions = [datetime.strptime(tick, '%Y-%m-%d').date().toordinal() for tick in tick_labels]
    ax.set(xticks=tick_positions, xticklabels=tick_labels)

def calc_mutual_information(dates, characterizationdir):
    mi_dict = {'date': [], 'channel': [], 'mutual_information': []}
    first_signal = None

    for date in tqdm(dates, desc="Calculating Mutual Information", file=sys.stdout):
        
        data_CO, data_RD = load_day(date)

        if data_CO and data_RD:
            sbp = np.concatenate((data_CO['sbp'], data_RD['sbp']), axis=0)
        elif data_RD:
            sbp = data_RD['sbp']
        else:
            sbp = data_CO['sbp']

        if date == dates[0]:
            first_signal = sbp.copy()

        for ch in range(sbp.shape[1]):
            signal = sbp[:, ch]

            # Use the first day for "target"
            original_signal = first_signal[:,ch]

            min_length = min(len(signal), len(original_signal))

            # Calculate mutual information between original and shifted signal
            mi = mutual_info_regression(original_signal[:min_length].reshape(-1,1), signal[:min_length],  random_state=0)
            mi_value = mi[0]
            # mi_value = mutual_info_score(
            #     pd.qcut(signal[:min_length], 10, duplicates='drop'),
            #     pd.qcut(original_signal[:min_length], 10, duplicates='drop')
            # )

            mi_dict['date'].append(date)
            mi_dict['channel'].append(ch)
            mi_dict['mutual_information'].append(mi_value)

    mi_df = pd.DataFrame.from_dict(mi_dict)
    mi_df.to_pickle(os.path.join(characterizationdir, "mutual_information_df.pkl"))
    mi_df.to_csv(os.path.join(characterizationdir, "mutual_information.csv"), index=False)


def plot_average_mutual_information(pkl_file_path):

    mi_df = pd.read_pickle(pkl_file_path)

    # Convert dates to datetime for sorting and grouping
    mi_df['date'] = pd.to_datetime(mi_df['date'])

    mi_df = mi_df.sort_values('date')

    # Get the first date to exclude
    first_date = mi_df['date'].min()

    # Filter out the first date
    mi_df = mi_df[mi_df['date'] != first_date]

    # Group by date and compute average MI
    avg_mi_per_day = mi_df.groupby('date')['mutual_information'].mean()

    plt.figure(figsize=(12, 6))
    avg_mi_per_day.plot(marker='o')
    plt.xlabel("Date")
    plt.ylabel("Average Mutual Information")
    plt.title("Average Mutual Information Across Channels Over Time (excluding first day)")
    plt.grid(True)
    plt.tight_layout()

    output_dir = os.path.dirname(pkl_file_path)
    output_path = os.path.join(output_dir, "avg_mutual_information_over_time.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_top_channel_mi_histograms(characterizationdir):

    mi_df = pd.read_pickle(os.path.join(characterizationdir, "mutual_information_df.pkl"))

    # Remove the first day's data
    first_date = mi_df['date'].min()
    mi_df = mi_df[mi_df['date'] != first_date]

    # Compute average MI per channel
    avg_mi_per_channel = mi_df.groupby('channel')['mutual_information'].mean()

    # Get top 3 channels by average MI
    top_channels = avg_mi_per_channel.sort_values(ascending=False).head(3).index.tolist()


    plt.figure(figsize=(15, 4))
    for i, ch in enumerate(top_channels, 1):
        plt.subplot(1, 3, i)
        sns.histplot(mi_df[mi_df['channel'] == ch]['mutual_information'], bins=20, kde=True)
        plt.title(f"Channel {ch} MI Histogram")
        plt.xlabel("Mutual Information")
        plt.ylabel("Frequency")

    plt.tight_layout()
    save_path = os.path.join(characterizationdir, "top3_channel_mi_histograms.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved histogram figure to: {save_path}")

def plot_mutual_information_heatmap(pkl_file_path):

    mi_df = pd.read_pickle(pkl_file_path)

    # Convert date column to datetime
    mi_df['date'] = pd.to_datetime(mi_df['date'])

    mi_df = mi_df.sort_values('date')

    # Exclude the first date
    first_date = mi_df['date'].min()
    mi_df = mi_df[mi_df['date'] != first_date]

    # Pivot table to create a 2D matrix of shape (date x channel)
    heatmap_data = mi_df.pivot_table(index='date', columns='channel', values='mutual_information')

    plt.figure(figsize=(16, 8))
    sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Mutual Information'}, xticklabels=8, yticklabels=False)
    plt.title("Channel-wise Mutual Information Over Time (Excluding First Day)")
    plt.xlabel("Channel")
    plt.ylabel("Date")
    plt.tight_layout()

    output_dir = os.path.dirname(pkl_file_path)
    output_path = os.path.join(output_dir, "mutual_information_heatmap.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

def create_channel_heatmaps(dates, channel, output_path):
    with open(os.path.join(output_path, 'sbp_heatmaps.pkl'), 'rb') as f:
        sbp_heatmaps  = pickle.load(f)
    
    channels = [0,1,2,3,4]
    fig, ax = plt.subplots(5, 1)
    for i in range(len(channels)):
        sbp_heatmap = sbp_heatmaps[:,:,channels[i]]
        ax[i].imshow(sbp_heatmap.T, origin='lower')

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



if __name__ == "__main__":
    create_signal_quality_figure()
