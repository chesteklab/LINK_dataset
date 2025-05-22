import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter
import glob
import pickle
from datetime import datetime
from tqdm import tqdm
from .population_level_analyses import *
from utils import mpl_config

def create_dimensionality_across_days_figures(data_path, output_path):
    mpath = data_path
    # mpath = os.path.join("..", "..","..", "AdaptiveAlignment", "data", "hisham_good_days")
    results = load_all_datasets(mpath, 303)

    ### LOAD AND PREPROCESS DATA ###
    df_tuning = prepare_tuning_data(results)
    plot_avg_trajectories(df_tuning, type_of_data='sbps', output_path=output_path, group_by = 'year', trim_method = trim_neural_data_at_movement_onset_std_and_smooth, trim_pt = max_jerk, sigma = .5, years_to_skip=[2021, 2022], directions='ext_flex', remove_RT=False)
    plot_centroid_of_pca_data_across_time(df_tuning,output_path, group_by='quarter', remove_RT=False, normalization_method = 'all', years_to_skip = [], plot_centr_across_time=True)
