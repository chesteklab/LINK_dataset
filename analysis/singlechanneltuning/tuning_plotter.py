import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.signal import savgol_filter
from scipy import stats
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pdb
import scipy
import tuning_utils

def plot_dummy_ax(ax):
    norm = mpl.colors.Normalize(-1*np.pi, np.pi)
    n=200
    t = np.linspace(-1*np.pi,np.pi, n)
    r = np.linspace(0.9, 1, 2)
    rg, tg = np.meshgrid(r, t)
    c = tg
    im = ax.pcolormesh(t, r, c.T, norm=norm, cmap='hsv')
    # dummy_ax.sety_ticklabels([])
    ax.tick_params()      #cosmetic changes to tick labels
    # ax.spines['polar'].set_visible(False)    #turn off the axis spine.
    ax.grid(True)
    ax.set_yticklabels([])
    ax.set_thetagrids([0,90,180,270])
    labels = []

    # for label, angle in zip(ax.get_xticklabels(), [0,90,180,270]):
    #     x,y = label.get_position()
    #     lab = ax.text(x,y, label.get_text(), transform=label.get_transform(),
    #                 ha=label.get_ha(), va=label.get_va())
    #     lab.set_rotation(angle+90)
    #     labels.append(lab)
    
    ax.set_xticklabels([])

def plot_polar_tuning(ax, dataframe, channel_number, params = {'ylim':None, 'cmap':'crest', 's':4, 'alpha':0.5, 'tick_override':False}):
    channel_data = dataframe.loc[dataframe['channel'] == channel_number].copy()

    days = (channel_data['date'] - channel_data['date'].min()).dt.days
    qt = tuning_utils.calc_medians_iqrs(dataframe)
    cmap = sns.color_palette(params['cmap'], as_cmap=True)
    
    scatter = ax.scatter(np.radians(channel_data['angle']), 
                         channel_data['magnitude'],
                         s=params['s'],
                         c=days, 
                         cmap=params['cmap'], 
                         alpha=params['alpha'])
    def angular_difference_rad(a, b):
        diff = (a - b + np.pi) % (2 * np.pi) - np.pi
        return np.abs(diff)

    xerr = np.asarray([angular_difference_rad(np.radians(qt['ang_lower_quartile'][channel_number]), np.radians(qt['ang_median'][channel_number])), 
            angular_difference_rad(np.radians(qt['ang_upper_quartile'][channel_number]), np.radians(qt['ang_median'][channel_number]))]).reshape(2,1)
    yerr = np.asarray([qt['mag_lower_quartile'][channel_number], qt['mag_upper_quartile'][channel_number]]).reshape(2,1)
    ax.errorbar(np.radians(qt['ang_median'][channel_number]), 
                qt['mag_median'][channel_number], 
                xerr=xerr, 
                yerr=yerr,
                fmt='k.',
                elinewidth=3)
    if params['tick_override']:
        ax.set_xticklabels(['0°','45°', '90°','135°', '180°','-135°','-90°','-45°','-180°'])
    if params['ylim']:
        ax.set_ylim(params['ylim'][0], params['ylim'][1])

    avg_avg_tcr = np.round(np.mean(channel_data['avg_tcr']), 2)
    std_avg_tcr = np.round(np.std(channel_data['avg_tcr']), 2)
    print(f'Ch. {channel_number} perday FR: {avg_avg_tcr} +/- {std_avg_tcr}')
    
    # pdb.set_trace()
    # ax.set_title(f'Channel {channel_number}', va='bottom')
    #ax.set_yticks((.05, .1))
    return scatter

def plot_tuning_heatmap(ax, dataframe, metric = 'magnitude', cmap = 'coolwarm'):
        
    all_dates = pd.date_range(dataframe['date'].min(), dataframe['date'].max())
    matrix = (dataframe.pivot(index='channel', columns='date', values=metric).reindex(columns=all_dates))

    if metric == 'magnitude':
        lower_limit = 0
        upper_limit = np.nanquantile(matrix.values, 0.99)
        matrix = matrix.clip(upper=upper_limit)
        cbar_kws = {'label':'Tuning Strength'}
        title="Preferred Tuning Strengths"
    elif metric == 'angle':
        lower_limit = -180
        upper_limit = 180
        cbar_kws = {'ticks':[-180,-90,0,90,180], 'label':'Tuning Angle'}
        title="Preferred Tuning Angles"
    elif metric == 'avg_tcr':
        lower_limit = 0
        upper_limit = 10
        cbar_kws = {'label':'Tuning Strength'}
        title='Average TCR Per Day'
    else:
        Exception("Unsupported Metric")

    if isinstance(cmap, str):
        cmapn = sns.color_palette(cmap, as_cmap=True).copy()
        cmapn.set_bad('black')
    else:
        cmapn = cmap
    
    # if derive == 0:
    #     data = matrix
    # elif derive == 1:
    #     data = matrix.diff(axis=1)
    # elif derive == 2:
    #     data = matrix.diff(axis=1).diff(axis=1)
    
    sns.heatmap(matrix, ax=ax, cmap=cmapn, vmin=lower_limit, vmax=upper_limit, cbar_kws=cbar_kws)
    ax.set_aspect('auto', adjustable='box')
    ax.set(title=title, xlabel='Date', ylabel='Channel')
    xticks = [i for i, date in enumerate(all_dates) if date.month in [3, 9] and date.day == 1]
    ax.set_xticks(xticks)
    ax.set_xticklabels(pd.to_datetime(all_dates[xticks]).strftime('%Y-%m-%d'), rotation = 0)

    ax.set_yticks([0 , 32, 64, 95])
    ax.set_yticklabels([0 , 31, 63, 95])

def plot_tuning_histogram(ax, tuning_avgs, metric = 'std', type='ang', bins = 10, colour = 'blue', banks = [(0, 32), (32, 64), (64, 96)]):
    '''
    metric can be 'variance', 'mean', 'std', 'circstd', or 'circmean'
    '''
    key = f'{type}_{metric}'
    if not banks:
        values = tuning_avgs[key]
        ax.hist(values, bins=bins, color=colour, density=True)
        mn = values.mean()
    else:
        sns.pointplot(data=tuning_avgs, x='bank', y=key, hue='bank', ax=ax, errorbar='sd')
        mn = tuning_avgs.groupby('bank')[key].mean()
    
    ax.set_title(key)
    ax.set_xlabel(key)
    ax.set_ylabel('# channels')
    ax.legend()
    return mn

# DO NOT REMOVE, structure kept for writing future functions
# def plot_heat_map_uniform(ax, dataframe, choice = 'original', type = 'magnitude', cmap = 'coolwarm'):
#     all_dates = pd.date_range(start=dataframe['date'].min(), end=dataframe['date'].max())
#     matrix = dataframe.pivot(index='channel', columns='date', values=type).reindex(columns=all_dates)
#     lower_bound = np.nanquantile(matrix.values.flatten(), 0.01)
#     upper_bound = np.nanquantile(matrix.values.flatten(), 0.99)
#     clipped_matrix = matrix.clip(lower=lower_bound, upper=upper_bound)

#     no_data_value = lower_bound - (upper_bound - lower_bound) * 0.1
#     clipped_matrix = clipped_matrix.fillna(no_data_value)

#     # Define a custom colormap with a specific color for no_data_value
#     custom_cmap = sns.color_palette(cmap, as_cmap=True)
#     custom_cmap.set_bad('gray')  # Replace 'gray' with your desired color for blanks

#     if choice == 'original':
#         sns.heatmap(clipped_matrix, ax=ax, cmap=custom_cmap, cbar_kws={'label': type, 'orientation': 'vertical'}, vmin=no_data_value)
#         ax.set_title('Heatmap of ' + choice.capitalize() + ' ' + type.capitalize())
#     elif choice == 'first_derivative':
#         first_derivative = matrix.diff(axis=1)
#         sns.heatmap(first_derivative, ax=ax, cmap=custom_cmap, cbar_kws={'label': 'First Derivative', 'orientation': 'vertical'})
#         ax.set_title('Heatmap of ' + choice.capitalize() + ' ' + type.capitalize())
#     elif choice == 'second_derivative':
#         second_derivative = matrix.diff(axis=1).diff(axis=1)
#         sns.heatmap(second_derivative, ax=ax, cmap=custom_cmap, cbar_kws={'label': 'Second Derivative', 'orientation': 'vertical'})
#         ax.set_title('Heatmap of ' + choice.capitalize() + ' ' + type.capitalize())

#     ax.set_xlabel('Date')
#     ax.set_ylabel('Channel')
#     ax.set_xticks(np.linspace(0, len(all_dates) - 1, 10))
#     ax.set_xticklabels(pd.to_datetime(all_dates[np.linspace(0, len(all_dates) - 1, 10, dtype=int)]).strftime('%Y-%m-%d'), rotation=45)

def plot_time_graph(ax, dataframe, derivative = 'original', type='magnitude', channels=None, apply_smoothing=False, smoothing_params=None):
    #todo: choice change to derivative
    '''
    type can be 'magnitude' or 'angle'
    channels can be a single channel (int) or a list of channels
    apply_smoothing applies Savitzky-Golay filter if True
    '''
    if channels is None:
        raise ValueError("Please specify a channel or a list of channels.")

    if isinstance(channels, int):
        selected_data = dataframe[dataframe['channel'] == channels]
        title_suffix = f"Channel {channels}"
    elif isinstance(channels, list):
        selected_data = dataframe[dataframe['channel'].isin(channels)]
        selected_data = selected_data.groupby('date').mean().reset_index()
        title_suffix = f"Mean of Channels"
    else:
        raise ValueError("Channels must be an int or a list of ints.")

    if apply_smoothing:
        if 'window_length' in smoothing_params and 'polyorder' in smoothing_params:
            selected_data[type] = savgol_filter(
                selected_data[type],
                window_length=smoothing_params['window_length'],
                polyorder=smoothing_params['polyorder']
            )
        else:
            raise ValueError("Smoothing parameters must include 'window_length' and 'polyorder'.")
    matrix = selected_data.pivot(index='channel', columns='date', values=type)
    lower_bound = matrix.quantile(0.01).min()
    upper_bound = matrix.quantile(0.99).max()
    clipped_matrix = matrix.clip(lower=lower_bound, upper=upper_bound)
    if derivative == 'original':
        if isinstance(channels, int):
            ax.plot(selected_data['date'], selected_data[type], label=f'Channel {channels}')
        elif isinstance(channels, list):
            start_channel = channels[0]
            end_channel = channels[-1]
            ax.plot(selected_data['date'], selected_data[type], label=f'Channels {start_channel}-{end_channel}')
        ax.set_title(f'Time Graph of {type.capitalize()} ({title_suffix})')
    elif derivative == 'first_derivative':
        selected_data['first_derivative'] = selected_data[type].diff()
        if isinstance(channels, int):
            ax.plot(selected_data['date'], selected_data['first_derivative'], label=f'First Derivative (Channel {channels})')
        elif isinstance(channels, list):
            start_channel = channels[0]
            end_channel = channels[-1]
            ax.plot(selected_data['date'], selected_data['first_derivative'], label=f'First Derivative (Channels {start_channel}-{end_channel})')
        ax.set_title(f'Time Graph of First Derivative of {type.capitalize()} ({title_suffix})')
    elif derivative == 'second_derivative':
        selected_data['second_derivative'] = selected_data[type].diff().diff()
        if isinstance(channels, int):
            ax.plot(selected_data['date'], selected_data['second_derivative'], label=f'Second Derivative (Channel {channels})')
        elif isinstance(channels, list):
            start_channel = channels[0]
            end_channel = channels[-1]
            ax.plot(selected_data['date'], selected_data['second_derivative'], label=f'Second Derivative (Channels {start_channel}-{end_channel})')
        ax.set_title(f'Time Graph of Second Derivative of {type.capitalize()} ({title_suffix})')

    ax.set_xlabel('Date')
    ax.set_ylabel(type.capitalize())
    ax.legend()
        
def plot_daily_spatial_tuning(ax,dataframe,date_choice,mapping_dir,correspondence = False):
    mapping = pd.read_csv(mapping_dir)
    day_choice = date_choice
    daily_tuning = dataframe[dataframe['date'] == day_choice]
    spatial_magnitude = np.zeros((8,8,2))
    spatial_angles = np.zeros((8,8,2))
    array_correspondence = np.zeros((8,8,2))
    cmapv = sns.color_palette('viridis', as_cmap=True).copy()
    cmapv.set_bad('gainsboro')
    cmaph = sns.color_palette('hsv', as_cmap=True).copy()
    cmaph.set_bad('gainsboro')
    for ch in daily_tuning['channel']:
        row = mapping[mapping['Channel']== ch+1]['Array Row'].values[0]
        col = mapping[mapping['Channel']== ch+1]['Array Column'].values[0]
        array_name = mapping[mapping['Channel']== ch+1]['Array Name'].values[0]
        spatial_magnitude[row-1,col-1,array_name-1] = daily_tuning[daily_tuning['channel'] == ch]['magnitude'].values[0]
        spatial_angles[row-1,col-1,array_name-1] = daily_tuning[daily_tuning['channel'] == ch]['angle'].values[0]
        array_correspondence[row-1,col-1,array_name-1] = ch+1
    spatial_magnitude[:,4:,1] = np.nan
    spatial_angles[:,4:,1] = np.nan
    if(not correspondence):
        sns.heatmap(spatial_magnitude[:, :, 0], ax=ax[0, 0], cmap=cmapv, cbar=True, vmin=0.000828924594908148, vmax=0.05302147033920138)
        ax[0, 0].set_title('Spatial Magnitude (Array 1)')

        sns.heatmap(spatial_magnitude[:, :, 1], ax=ax[0, 1], cmap=cmapv, cbar=True, vmin=0.000828924594908148, vmax=0.05302147033920138)
        ax[0, 1].set_title('Spatial Magnitude (Array 2)')

        sns.heatmap(spatial_angles[:, :, 0], ax=ax[1, 0], cmap=cmaph, cbar=True, vmin=-180, vmax=180)
        ax[1, 0].set_title('Spatial Angles (Array 1)')

        sns.heatmap(spatial_angles[:, :, 1], ax=ax[1, 1], cmap=cmaph, cbar=True, vmin=-180, vmax=180)
        ax[1, 1].set_title('Spatial Angles (Array 2)')
    else:
        sns.heatmap(array_correspondence[:, :, 0], annot=True, fmt=".0f", cmap=None, cbar=False, ax=ax[0])
        ax[0].set_title("Array Correspondence (Array 1)")

        # Plot Array Correspondence for Array 2
        sns.heatmap(array_correspondence[:, :, 1], annot=True, fmt=".0f", cmap=None, cbar=False, ax=ax[1])
        ax[1].set_title("Array Correspondence (Array 2)")