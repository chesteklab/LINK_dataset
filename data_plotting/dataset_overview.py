import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime
#import config
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pdb
import os
import re
import sys
import glob
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats
from collections import defaultdict
from utils.data_tools import extract_dates_from_filenames, load_day
from data_processing.dataset_overview import *
# data_path = "Z:\Student Folders\\Nina_Gill\data\only_good_days_timeouts"
# output_dir = "D:\\University of Michigan Dropbox\Hisham Temmar\Science Communication\Papers\LINK_dataset\experimental setup"

def create_dataset_overview_figure(data_path, output_dir):
    # load in data, call create time plot - maybe dont need create time plot and dataset count over time?
    # load in one day with center out and random, make plots of target positions
    dates = extract_dates_from_filenames(data_path)
    create_time_plot(dates, data_path, output_dir)
    pass
