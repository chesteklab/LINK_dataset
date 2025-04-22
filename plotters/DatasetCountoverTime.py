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