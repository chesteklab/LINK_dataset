import os 
import sys
import pdb
from copy import deepcopy
sys.path.append('/Users/yixuan/Documents/GitHub/pybmi')
from pybmi.utils.ZTools import ZStructTranslator,zarray
from pybmi.utils import ZTools, TrainingUtils
from pybmi.decoders import NNDecoders
from pybmi.offline import TrainingOffline
from pybmi.utils.AnalysisTools import adjustfeats

import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
from scipy import stats
import numpy as np
import pandas as pd
import pickle 

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from collections import defaultdict
import datetime
import pdb
import re

data_folder = "/Volumes/share/Student Folders/Bianca_Wang/updated_retrieved"

linear_reg_dict = {'Date': [],
                    'Day': [],
                    'TS': [],
                    'Coef':[],
                    'Year': [],
                    'Month' : [],
                    'Quarter': []}

signal_power_dict = {'Date': [],
                    'Day': [],
                    'TS': [],
                    'Power':[],
                    'Year': [],
                    'Month': [],
                    'Quarter': []}

def year_month_quarter(date, dictionary):
    year, month, _ = date.split('-')
    dictionary['Year'].append(year)
    dictionary['Month'].append(month)
    if month in ['01', '02', '03']:
        dictionary['Quarter'].append('1')
    elif month in ['04', '05', '06']:
        dictionary['Quarter'].append('2')
    elif month in ['07', '08', '09']:
        dictionary['Quarter'].append('3')
    else:
        dictionary['Quarter'].append('4')

def split_fit_single_chan(feats):
    
    coef_matrix = np.zeros((2, feats['NeuralFeature'].shape[1]))
    TrialIndex = feats['TrialIndex']
    
    for channel in range(0, feats['NeuralFeature'].shape[1]):

        if len(TrialIndex) > 300:
            test_len = np.min((len(TrialIndex)-1, 399))
       
            neural_training = (feats['NeuralFeature'][:TrialIndex[300], channel]).reshape(-1, 1)
            neural_testing = (feats['NeuralFeature'][TrialIndex[300]:TrialIndex[test_len], channel]).reshape(-1, 1)

            finger_training = feats['FingerAnglesTIMRL'][:TrialIndex[300], 2:]
            finger_testing = feats['FingerAnglesTIMRL'][TrialIndex[300]:TrialIndex[test_len], 2:]

        else:
            raise Exception('not enough trials')

        reg = linear_model.LinearRegression()
        reg.fit(neural_training, finger_training)
        coef_matrix[:, channel] = reg.coef_.flatten()
    
    return coef_matrix

# Loading in data for each day
for file_name in os.listdir(data_folder):

    file = os.path.join(data_folder, file_name)
    with open(file, 'rb') as f:
        feats = pickle.load(f)
        print("Data loaded")
    
    #if feats["Date"] in dates_to_exclude:
        #continue
    date = feats['Date']
    # only taking TS34 for now
    if feats['Target Style'] == 34.0:
        coef_matrix = split_fit_single_chan(feats)
        linear_reg_dict['Coef'].append(coef_matrix)
        linear_reg_dict['Date'].append(date)
        linear_reg_dict['TS'].append(feats['Target Style'])
        year_month_quarter(date, linear_reg_dict)
    
    avg_signal_power = []
    for channel in range(0, feats['NeuralFeature'].shape[1]):
        avg_signal_power.append(np.mean(feats['NeuralFeature'][:, channel]))    
    
    signal_power_dict['Power'].append(avg_signal_power)
    signal_power_dict['Date'].append(date)
    signal_power_dict['TS'].append(feats['Target Style'])
    
    year_month_quarter(date, signal_power_dict)

day0_reg = datetime.datetime.strptime(linear_reg_dict['Date'][0], '%Y-%m-%d')
linear_reg_dict['Day'] = [(datetime.datetime.strptime(day, '%Y-%m-%d') - day0_reg).days for day in linear_reg_dict['Date']]
day0_power = datetime.datetime.strptime(linear_reg_dict['Date'][0], '%Y-%m-%d')
signal_power_dict['Day'] = [(datetime.datetime.strptime(day, '%Y-%m-%d') - day0_power).days for day in signal_power_dict['Date']]

### SIGNAL POWER ANALYSIS
# Grouping by month or quarters, expanding the channels out, putting in dataframe

signal_power_df = pd.DataFrame(signal_power_dict)
channel_power_expanded = pd.DataFrame(signal_power_df['Power'].tolist(), index=signal_power_df.index)
signal_power_df = pd.concat([signal_power_df.drop(columns=['Power']), channel_power_expanded], axis=1)

quarterly_mean = signal_power_df.groupby(['Year', 'Quarter']).mean().drop(['Day', 'TS'], axis =1)
quarterly_median = signal_power_df.groupby(['Year', 'Quarter']).median().drop(['Day', 'TS'], axis =1)
quarter_count = signal_power_df.groupby(['Year', 'Quarter']).size().reset_index(name='Count')

monthly_mean = signal_power_df.groupby(['Year', 'Month']).mean().drop(['Day', 'TS'], axis =1)
monthly_median = signal_power_df.groupby(['Year', 'Month']).median().drop(['Day', 'TS'], axis =1)
month_count = signal_power_df.groupby(['Year', 'Month']).size().reset_index(name='Count')

def reformat_grouped(df, Month_or_Quarter):
    df.columns = df.columns.astype(str)
    df.reset_index(inplace=True)
    df[f'Year-{Month_or_Quarter}'] = df['Year'].astype(str) + '-' + df[Month_or_Quarter]

reformat_grouped(quarterly_mean, 'Quarter')
reformat_grouped(quarterly_median, 'Quarter')
reformat_grouped(monthly_mean, 'Month')
reformat_grouped(monthly_median, 'Month')

quarterly_mean = quarterly_mean.reset_index().merge(quarter_count, on = ['Year', 'Quarter'])
quarterly_median = quarterly_median.reset_index().merge(quarter_count, on = ['Year', 'Quarter'])
monthly_mean = monthly_mean.reset_index().merge(month_count, on = ['Year', 'Month'])
monthly_median = monthly_median.reset_index().merge(month_count, on = ['Year', 'Month'])

# some arbitrary filtering
quarterly_mean_filtered = quarterly_mean[quarterly_mean['Count'] >= 5]
quarterly_median_filtered = quarterly_median[quarterly_mean['Count'] >= 5]
monthly_mean_filtered = monthly_mean[monthly_mean['Count'] >= 3]
monthly_median_filtered = monthly_median[monthly_mean['Count'] >= 3]

# Reformatting the dataframes to be more compatible with seaborn
# Adding new column of standard error

channel_order = [f'{ch:02}' for ch in range(1, 97)]  # Generate '01' to '96'

def long_df(df, Month_or_Quarter):
    channel_columns = df.columns.difference([f'Year-{Month_or_Quarter}', 'Count', 'index', Month_or_Quarter, 'Year'])
    df['Power'] = df[channel_columns].values.tolist() # Adding a column with lists of channel power
    df['SEM'] = df['Power'].apply(lambda x: np.std(x, ddof=1) / np.sqrt(len(x))).values
    df['Mean'] = df['Power'].apply(np.mean).values
    df['Median'] = df['Power'].apply(np.mean).values
    df['SD'] = df['Power'].apply(np.std).values
    
    long_df = pd.melt(df, id_vars = [f'Year-{Month_or_Quarter}'], value_vars=channel_columns, 
                      var_name='Channel', value_name = 'Power')
    long_df['Channel'] = (long_df['Channel'].astype(int) + 1).apply(lambda x: str(x).zfill(2))
    long_df['Channel'] = pd.Categorical(long_df['Channel'], categories=channel_order)
    return long_df

long_quarterly_mean = long_df(quarterly_mean_filtered, 'Quarter')
long_quarterly_median = long_df(quarterly_median_filtered, 'Quarter')
long_monthly_mean = long_df(monthly_mean_filtered, 'Month')
long_monthly_median = long_df(monthly_median_filtered, 'Month')

# plotting signal power over time
def strip_plot(long_df, short_df, Mean_or_Median, Month_or_Quarter):

    sns.set(style = 'ticks', context = 'talk')
    plt.figure(figsize=(24, 12))
    
    palette = sns.color_palette("hsv", len(channel_order))
    channel_color_map = {channel: color for channel, color in zip(channel_order, palette)}
    long_df['Channel'] = pd.Categorical(long_df['Channel'], categories=channel_order)

    strip = sns.stripplot(x = f'Year-{Month_or_Quarter}', y = 'Power', 
                          hue = 'Channel', data = long_df, linewidth = 0.5)
    
    # Extract x coordinates of the stripplot dots
    strip_x_coords = []
    strip_y_coords = []
    strip_colors = []
    for strip_collection in strip.collections[:short_df.shape[0]]:
        offsets = strip_collection.get_offsets()
        face_colors = strip_collection.get_facecolors()
        x_coords = offsets[:, 0]
        y_coords = offsets[:, 1]
        strip_x_coords.append(x_coords)
        strip_y_coords.append(y_coords)
        strip_colors.append(face_colors)

    for channel in range(96):
        jittered_x = [x_coords[channel] for x_coords in strip_x_coords]
        jittered_y = [y_coords[channel] for y_coords in strip_y_coords]
        plt.plot(jittered_x, jittered_y, 
                 color = strip_colors[0][channel], alpha=0.2)
        
    mean_line, = plt.plot(short_df[f'Year-{Month_or_Quarter}'], short_df['Mean'], color = 'black', linewidth = 3,
                          label = 'Mean')
    fill_between= plt.fill_between(short_df[f'Year-{Month_or_Quarter}'], 
                                   short_df['Mean'] - short_df['SD'], 
                                   short_df['Mean'] + short_df['SD'], 
                                   alpha = 0.4, color = 'grey', edgecolor='none', label = 'Mean ± 1SD')

    plt.xticks(rotation=45)        
    plt.xlabel(f'Year-{Month_or_Quarter}', labelpad = 8)
    plt.ylabel(f'{Mean_or_Median} Power')
    mid = (fig.subplotpars.right + fig.subplotpars.left)/2
    suptitle = plt.suptitle(f'{Mean_or_Median} {Month_or_Quarter}ly Power Over Time', 
                            fontsize = 22, fontweight = 'semibold', x = mid)
    if Month_or_Quarter == "Month":
        plt.title("(Only months with at least 3 sessions are shown)", pad = 8, ha = 'center')
    else:
        plt.title("(Only quarters with at least 5 sessions are shown)", pad = 8, ha = 'center')
    plt.subplots_adjust(top=0.92)
    
    # Create a custom legend for the strip plot
    handles, labels = strip.get_legend_handles_labels()
    strip_legend = plt.legend(handles=handles, labels=labels[:96], title='Channels', bbox_to_anchor=(0.48, -0.14), loc='upper center', ncol=16, fontsize='small', title_fontsize='medium')
    strip_legend.get_title().set_position((mid+100, 0))
    
    # Extract legend handles and labels for mean and fill_between
    handles_mean_fill = [mean_line, fill_between]
    labels_mean_fill = ['Mean', 'Mean ± 1 SD']
    mean_fill_legend = plt.legend(handles=handles_mean_fill, labels=labels_mean_fill, loc='upper right')
    
    # Add the strip legend to the plot
    plt.gca().add_artist(strip_legend)
    
    plt.savefig(f'{Month_or_Quarter}ly {Mean_or_Median}.png', 
                bbox_extra_artists = (strip_legend, mean_fill_legend, suptitle), dpi=300, bbox_inches='tight')
    plt.show()

strip_plot(long_quarterly_mean, quarterly_mean_filtered, 'Mean', 'Quarter')
strip_plot(long_monthly_mean, monthly_mean_filtered, 'Mean', 'Month')
strip_plot(long_quarterly_median, quarterly_median_filtered, 'Median', 'Quarter')
strip_plot(long_monthly_median, monthly_median_filtered, 'Median', 'Month')

# THE CHANNEL POWER HEATMAP THINGY
example_month_64 = monthly_mean_filtered['Power'][1][:64]
example_month_32 = monthly_mean_filtered['Power'][1][64:]

matrix_64 = np.zeros((8, 8))
matrix_32 = np.zeros((8, 4))

def power_matrix(data, matrix, size):
    for channel in range(size):
        row = -(channel % 8 + 1)  
        col = -(channel // 8 + 1)
        matrix[row, col] = data[channel]

power_matrix(example_month_64, matrix_64, 64)
power_matrix(example_month_32, matrix_32, 32)

plt.figure(figsize=(6, 8))
ax = sns.heatmap(matrix_32, annot=True, fmt=".2f", cmap="viridis", cbar=True, vmin = 0, vmax = 60)

ax.set_xticks(np.arange(4) + 0.5)
ax.set_yticks(np.arange(8) + 0.5)
ax.set_xticklabels(np.arange(12, 8, -1))
ax.set_yticklabels(np.arange(8, 0, -1))
ax.set_xlabel('Column Index')
ax.set_ylabel('Row Index')

plt.title("2020-01 mean power by channel")
plt.show()

### Coefficients stuff
linear_reg_df = pd.DataFrame(linear_reg_dict)

# expand the dataframe
fingers = ['index', 'MRP']
index_cols = [f'index_{i+1}' for i in range(96)]
MRP_cols = [f'MRP_{i+1}' for i in range(96)]
channel_cols = index_cols + MRP_cols

def expand_coef(coef_matrix, fingers):
    # Flatten the coef_matrix and create a new series with the correct column names
    coef_matrix = np.array(coef_matrix)
    flattened = np.concatenate(coef_matrix, axis=0)
    return pd.Series(flattened, index = channel_cols)

expanded_coef = linear_reg_df['Coef'].apply(lambda x: expand_coef(x, fingers))
expanded_coef = pd.concat([linear_reg_df, expanded_coef], axis=1)
expanded_coef[f'Year-Month'] = expanded_coef['Year'].astype(str) + '-' + expanded_coef['Month']

# converting into long format for easier grouping
long_expanded_coef = pd.melt(expanded_coef, id_vars = 'Year-Month', value_vars=channel_cols, 
                  var_name='Channel', value_name = 'coef')
long_expanded_coef['Type'] = long_expanded_coef['Channel'].apply(lambda x: 'index' if x.startswith('index') else 'MRP')

# grouping by month for each channel and doing summary statistics
# monthly absolute deltas
long_monthly_delta = long_expanded_coef.groupby(['Year-Month', 'Channel']).agg(max_coef=('coef', 'max'),
                                                    min_coef=('coef', 'min')).reset_index()

long_monthly_delta['max_delta'] = long_monthly_delta['max_coef'] - long_monthly_delta['min_coef']

long_monthly_delta['prev_max_delta'] = long_monthly_delta.groupby(['Channel'])['max_delta'].shift(1)
long_monthly_delta['delta'] = long_monthly_delta['max_delta'] - long_monthly_delta['prev_max_delta']
long_monthly_delta['abs_monthly_delta'] = long_monthly_delta['delta'].abs()

delta_summary_stats = long_monthly_delta[['abs_monthly_delta', 'Channel']].groupby('Channel').describe().reset_index()
delta_summary_stats['Channel_Num'] = delta_summary_stats['Channel'].apply(lambda x: int(re.findall(r'\d+', x)[0])) # just sorting so the table's easier to look at
delta_summary_stats = delta_summary_stats.sort_values(by='Channel_Num').drop(columns=['Channel_Num'])
print(delta_summary_stats)


# box plot, plotting coef over time for a single channel
def plot_coef_over_time(long_expanded_coef, channel):
    df = long_expanded_coef[(long_expanded_coef['Channel'] == f'index_{channel}') | (long_expanded_coef['Channel'] == f'MRP_{channel}')]
    
    plt.figure(figsize=(30, 8))
    sns.boxplot(x='Year-Month', y='coef', hue='Type', data = df, width=0.5)
    plt.xticks(rotation=45)
    plt.title(f'Box Plot of {channel} Coefficients Over Time')
    plt.xlabel('Year-Month')
    plt.ylabel('Coefficient')
    plt.legend(title='Finger Type')
    plt.savefig(f'Coef Over Time for Channel {channel}')
    plt.show()
plot_coef_over_time(long_expanded_coef, 1)

# finding max deltas from the expanded coef df
max_delta_all_time = {'index': [],
                     'MRP': []}
for finger in fingers:
    for channel in range(1, 97):
        max_delta_all_time[f'{finger}'].append(
            max(expanded_coef[f'{finger}_{channel}']) - min(expanded_coef[f'{finger}_{channel}']))

# converting into a long format df for easier plotting
long_max_delta_all_time = pd.melt(pd.DataFrame(max_delta_all_time), var_name='finger', value_name='max_delta')

# strip plot for the all time max deltas
plt.figure(figsize=(10, 6))
sns.stripplot(x = 'finger', y = 'max_delta', hue = 'finger', data=long_max_delta_all_time)

# Customize the plot
plt.title('Max Delta of Linear Reg Coefficients over the entire time period')
plt.xlabel('Finger')
plt.ylabel('Max Delta')
plt.show()