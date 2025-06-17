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
from data_processing.signal_changes import signal_utils
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
from data_processing.signal_changes import *

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
