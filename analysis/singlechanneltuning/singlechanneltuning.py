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
import tuning_utils
import tuning_plotter

def create_single_channel_tuning_figure():
    output_dir = os.path.join(tuning_utils.output_path,'channel_stability_tuning.csv')
    calc_tunings = False
    if calc_tunings:
        tuning_utils.compute_tuning_data(output_dir)
    
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

    params = {'ylim':(0, 0.05), 'cmap':'crest', 's':7, 'alpha':0.6}
    for i, channel in enumerate(selected_channels[1:]):
        im = tuning_plotter.plot_polar_tuning(example_channels_axs[i], tuning_df, channel, params=params)
    cb = top_sfs[1].colorbar(im, ax=example_channels_axs, label='days')
    cb.outline.set_visible(False)
    top_sfs[1].suptitle("B. Example tunings over time")

    #plot tuning angles
    # subfigs[1].suptitle("C. tuning heatmaps")
    cbar_kw = {'ticks':[-180, -90, 0, 90, 180]}
    tuning_plotter.plot_tuning_heatmap(tuning_angle_heatmap_ax, tuning_df, metric='angle', cmap='hsv')
    tuning_plotter.plot_tuning_heatmap(tuning_strength_heatmap_ax, tuning_df, metric='magnitude', cmap='plasma')
    tuning_angle_heatmap_ax.set(xlabel=None)
    tuning_strength_heatmap_ax.set(xlabel=None)
    #plot tuning spreads
    ta = tuning_utils.calc_tuning_avgs(tuning_df)
    for i in np.arange(3):
        a = i*32
        b = i*32 + 32
        avg_tuning_ax[0].errorbar(np.radians(ta['ang_mean'])[a:b], ta['mag_mean'][a:b], xerr=np.radians(ta['ang_std'][a:b]), yerr=ta['mag_std'][a:b], fmt='.')
    subfigs[2].suptitle("D. tuning spreads")

    plt.show()

if __name__=="__main__":
    create_single_channel_tuning_figure()
