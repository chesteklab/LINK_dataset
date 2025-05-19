### IMPORTS ###
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib as mpl
import glob
import sys
import pdb
#from dataset_characterization import dataset_characterization
import matplotlib.gridspec as gridspec
#import config

from . import tuning_utils
from . import tuning_plotter
import seaborn as sns

TICK_OVERRIDE = True
def create_single_channel_tuning_figure(data_path, output_path):
    output_dir = os.path.join(output_path,'channel_stability_tuning.csv')
    calc_tunings = False
    if calc_tunings:
        tuning_utils.compute_tuning_data(data_path, output_dir)
    
    tuning_df = tuning_utils.load_tuning_data(output_dir)
    selected_channels = [-1,7,32]

    fig = plt.figure(figsize=(8, 10), layout='constrained')  # Adjusted figure size for the additional row
    subfigs = fig.subfigures(3,1, height_ratios=[1,2,1])

    top_sfs = subfigs[0].subfigures(1,2, width_ratios=[1,2])
    dummy_ax = top_sfs[0].subplots(1,1, subplot_kw={'projection':'polar'})
    example_channels_axs = top_sfs[1].subplots(1,2, subplot_kw={'projection':'polar'}, sharex=True, sharey=True)

    mid_axs = subfigs[1].subplots(2,1, sharex=True)
    tuning_angle_heatmap_ax = mid_axs[0]
    tuning_strength_heatmap_ax = mid_axs[1]

    avg_tuning_ax = subfigs[2].subplots(1,3, subplot_kw={'projection':'polar'})

    # create dummy polar plot with circle color bar around it
    top_sfs[0].suptitle("A. Preferred Tuning ")
    tuning_plotter.plot_dummy_ax(dummy_ax)

    params = {'ylim':(0, 0.06), 'cmap':'mako', 's':5, 'alpha':0.8, 'tick_override':TICK_OVERRIDE}
    for i, channel in enumerate(selected_channels[1:]):
        im = tuning_plotter.plot_polar_tuning(example_channels_axs[i], tuning_df, channel, params=params)
    cb = top_sfs[1].colorbar(im, ax=example_channels_axs, label='days')
    cb.outline.set_visible(False)
    top_sfs[1].suptitle("B. Example tunings over time")

    #plot tuning angles
    # subfigs[1].suptitle("C. tuning heatmaps")
    fig2, supp_tcr_ax = plt.subplots(1,1,figsize=(8,2.5), layout='constrained')
    cbar_kw = {'ticks':[-180, -90, 0, 90, 180]}
    tuning_plotter.plot_tuning_heatmap(tuning_angle_heatmap_ax, tuning_df, metric='angle', cmap='hsv')
    tuning_plotter.plot_tuning_heatmap(tuning_strength_heatmap_ax, tuning_df, metric='magnitude', cmap='plasma')
    tuning_plotter.plot_tuning_heatmap(supp_tcr_ax, tuning_df, metric='avg_tcr', cmap='plasma')
    tuning_angle_heatmap_ax.set(xlabel=None)
    tuning_strength_heatmap_ax.set(xlabel=None)
    #plot tuning spreads
    ta = tuning_utils.calc_tuning_avgs(tuning_df)
    qt = tuning_utils.calc_medians_iqrs(tuning_df)
    for i in np.arange(3):
        a = i*32
        b = i*32 + 32
        def angular_difference_rad(a, b):
            diff = (a - b + np.pi) % (2 * np.pi) - np.pi
            return np.abs(diff)
        xerr = [angular_difference_rad(np.radians(qt['ang_lower_quartile'][a:b]), np.radians(qt['ang_median'])[a:b]), 
                angular_difference_rad(np.radians(qt['ang_upper_quartile'][a:b]), np.radians(qt['ang_median'])[a:b])]
        # print(f'bank_{i} avg iqr: {np.mean(np.sum(xerr))}')
        avg_tuning_ax[i].errorbar(np.radians(qt['ang_median'])[a:b], qt['mag_median'][a:b], 
                                  xerr=xerr, 
                                  yerr=[qt['mag_lower_quartile'][a:b], qt['mag_upper_quartile'][a:b]], 
                                  fmt='none', linestyle='none', elinewidth=0.8, marker='o', ms=10, ecolor = 'k')
        if(TICK_OVERRIDE):
            avg_tuning_ax[i].set_xticklabels(['0°','45°', '90°','135°', '180°','-135°','-90°','-45°'])
    subfigs[2].suptitle("E. tuning spreads")

    plt.show()

if __name__=="__main__":
    create_single_channel_tuning_figure()
