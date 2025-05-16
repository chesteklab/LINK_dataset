import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os 
import sys
import signal_utils
import matplotlib as mpl
from sklearn.feature_selection import mutual_info_regression
from scipy import stats

#some basic text parameters for figures
mpl.rcParams['font.family'] = "Atkinson Hyperlegible" # if installed but not showing up, rebuild mpl cache
mpl.rcParams['font.size'] = 10
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['axes.unicode_minus'] = False
# mpl.rcParams['axes.titlesize'] = 14
# mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlelocation'] = 'center'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['figure.constrained_layout.use'] = True
# mpl.rcParams['figure.titlesize'] = 14
mpl.rcParams['figure.titleweight'] = 'bold'
mpl.rcParams['pdf.fonttype'] = 42

def create_signal_quality_figure(calc_avg_sbp=True, calc_pr=True):
    dates = signal_utils.extract_dates_from_filenames()
    print(f"Found {len(dates)} dates")
    
    fig, ax = plt.subplots(3,1, sharex=True)

    #average sbp figure
    if calc_avg_sbp:
        signal_utils.calc_avg_sbps(dates)
    create_avg_sbp_plot(ax[0])
    
    # calculate participation ratio on each day
    if calc_pr:
        signal_utils.calc_pr_all_days(dates)

    # create pr figure
    create_pr_plot(ax[2])
    active_channels_plot(ax[1],os.path.join(signal_utils.output_path,"participation_ratios.csv"))
    plt.show()

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


def create_pr_plot(ax):
    pr_df = pd.read_csv(os.path.join(signal_utils.output_path,"participation_ratios.csv"))
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

def create_avg_sbp_plot(ax):
    sbp_avgs = pd.read_csv(os.path.join(signal_utils.output_path, "sbp_avgs.csv"), index_col=0)
    sbp_avgs.index = pd.to_datetime(sbp_avgs.index)

    sbp_avgs['days'] = (sbp_avgs.index - sbp_avgs.index[0]).to_series().dt.days.to_numpy()

    sbp_long = sbp_avgs.reset_index(names="date").melt(id_vars=["date",'days'], var_name='channel', value_name='sbp')
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

if __name__ == "__main__":
    create_signal_quality_figure(False, False)
