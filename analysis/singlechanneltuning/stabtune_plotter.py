import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.signal import savgol_filter
from scipy import stats
import pdb

def plot_polar_tuning(ax, dataframe, channel_number, ylim = None, cmap = 'crest'):
    
    channel_data = dataframe.loc[dataframe['channel'] == channel_number].copy()

    channel_data['date'] = pd.to_datetime(channel_data['date'])
    time_normalized = (channel_data['date'] - channel_data['date'].min()) / (channel_data['date'].max() - channel_data['date'].min())

    if ax.name != 'polar':
        fig = ax.figure
        pos = ax.get_position()
        ax.remove()
        ax = fig.add_axes(pos, projection='polar')

    cmap = sns.color_palette(cmap, as_cmap=True).reversed()

    scatter = ax.scatter(
        np.radians(channel_data['angle']),
        channel_data['magnitude'],
        c=time_normalized,
        cmap=cmap,
        alpha=0.6
    )
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    #cbar = fig.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.1)
    #cbar.set_label('Time')
    #cbar.set_ticks([0, 1])
    #cbar.set_ticklabels([channel_data['date'].min().strftime('%Y-%m-%d'), channel_data['date'].max().strftime('%Y-%m-%d')])
    #ax.set_title(f'Tuning Direction and Magnitude over Time for Channel {channel_number}', va='bottom', pad=50)
    ax.set_title(f'Channel {channel_number}', va='bottom', pad=20)

def plot_heat_map(ax, dataframe, choice = 'original', type = 'magnitude', cmap = 'coolwarm'):
    matrix = dataframe.pivot(index='channel', columns='date', values=type)
    lower_bound = np.quantile(matrix.values.flatten(), 0.01)
    upper_bound = np.quantile(matrix.values.flatten(), 0.99)
    clipped_matrix = matrix.clip(lower=lower_bound, upper=upper_bound)

    if choice == 'original':
        sns.heatmap(clipped_matrix, ax=ax, cmap=cmap, cbar_kws={'label': type, 'orientation': 'vertical'})
        ax.set_title('Heatmap of ' + choice.capitalize() + ' ' + type.capitalize())
    elif choice == 'first_derivative':
        first_derivative = matrix.diff(axis=1)
        sns.heatmap(first_derivative, ax=ax, cmap=cmap, cbar_kws={'label': 'First Derivative', 'orientation': 'vertical'})
        ax.set_title('Heatmap of ' + choice.capitalize() + ' ' + type.capitalize())
    elif choice == 'second_derivative':
        second_derivative = matrix.diff(axis=1).diff(axis=1)
        sns.heatmap(second_derivative, ax=ax, cmap=cmap, cbar_kws={'label': 'Second Derivative', 'orientation': 'vertical'})
        ax.set_title('Heatmap of ' + choice.capitalize() + ' ' + type.capitalize())

    ax.set_xlabel('Date')
    ax.set_ylabel('Channel')
    ax.set_xticklabels([])  # Disable x-axis tick labels

def plot_heat_map_uniform_Experimental(ax, dataframe, choice = 'original', type = 'magnitude', cmap = 'coolwarm'):
    all_dates = pd.date_range(dataframe['date'].min(), dataframe['date'].max())
    matrix    = (dataframe
                 .pivot(index='channel', columns='date', values=type)
                 .reindex(columns=all_dates))

    lower, upper = np.nanquantile(matrix.values, [0.01, 0.99])
    clipped      = matrix.clip(lower=lower, upper=upper)

    cmapn = sns.color_palette(cmap, as_cmap=True).copy()
    cmapn.set_bad('black')

    if choice == 'original':
        data      = clipped
        cbar_label = type
    elif choice == 'first_derivative':
        data      = matrix.diff(axis=1)
        cbar_label = 'First Derivative'
    elif choice == 'second_derivative':
        data      = matrix.diff(axis=1).diff(axis=1)
        cbar_label = 'Second Derivative'

    sns.heatmap(data,
                ax=ax,
                cmap=cmapn,
                vmin=lower, vmax=upper,
                cbar_kws={'label': cbar_label, 'orientation': 'vertical'})

    ax.set(
        title=f'Heatmap of {choice.capitalize()} {type.capitalize()}',
        xlabel='Date', ylabel='Channel')
    xticks = np.linspace(0, len(all_dates) - 1, 10, dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels(pd.to_datetime(all_dates[xticks]).strftime('%Y-%m-%d'), rotation=45)

def plot_heat_map_uniform_Experimental(ax, dataframe, choice = 'original', type = 'magnitude', cmap = 'coolwarm', plot_xlabel = True):
    all_dates = pd.date_range(dataframe['date'].min(), dataframe['date'].max())
    matrix    = (dataframe.pivot(index='channel', columns='date', values=type).reindex(columns=all_dates))

    lower, upper = np.nanquantile(matrix.values, [0.01, 0.99])
    clipped      = matrix.clip(lower=lower, upper=upper)

    if isinstance(cmap, str):
        cmapn = sns.color_palette(cmap, as_cmap=True).copy()
        cmapn.set_bad('grey')
    else:
        cmapn = cmap
    if choice == 'original':
        data      = clipped
        cbar_label = type
    elif choice == 'first_derivative':
        data      = matrix.diff(axis=1)
        cbar_label = 'First Derivative'
    elif choice == 'second_derivative':
        data      = matrix.diff(axis=1).diff(axis=1)
        cbar_label = 'Second Derivative'

    sns.heatmap(data,ax=ax,cmap=cmapn,vmin=lower, vmax=upper,cbar_kws={'label': cbar_label, 'orientation': 'vertical'})

    ax.set(title=f'Heatmap of {choice.capitalize()} {type.capitalize()}', xlabel='Date', ylabel='Channel')
    xticks = [i for i, date in enumerate(all_dates) if date.is_month_end]
    ax.set_xticks(xticks)
    ax.set_yticks([0 , 32, 64, 95])
    ax.set_yticklabels([f'Ch {i}' for i in ax.get_yticks()])
    if(plot_xlabel):
        ax.set_xticklabels(pd.to_datetime(all_dates[xticks]).strftime('%Y-%m-%d'), rotation=45)
    else:
        ax.set_xticklabels([])
    
def plot_heat_map_HSV(ax, df, q_clip=(0.01, 0.99), sat=1.0):

    all_dates = pd.date_range(df['date'].min(), df['date'].max())
    mag_mat = (df.pivot(index='channel', columns='date', values='magnitude').reindex(columns=all_dates))
    ang_mat = (df.pivot(index='channel', columns='date', values='angle').reindex(columns=all_dates))

    mask = mag_mat.isna() | ang_mat.isna()

    lo, hi = np.nanquantile(mag_mat.values, q_clip)
    V = (mag_mat.clip(lower=lo, upper=hi) - lo) / (hi - lo)
    V = np.power(V, 0.5)
    V[mask] = 1

    H = ((ang_mat + 180) % 360) / 360
    H[mask] = 0

    
    S = np.full_like(H, sat)
    S[mask] = 0
    hsv = np.dstack([H.values, S, V.values])
    rgb = mcolors.hsv_to_rgb(hsv)

    ax.imshow(rgb, aspect='auto', origin='lower',interpolation='none')
    ax.set(title='Angle (H), Magnitude (V)', xlabel='Date', ylabel='Channel')

    ticks = np.linspace(0, len(all_dates) - 1, 10, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels(pd.to_datetime(all_dates[ticks]).strftime('%Y-%m-%d'), rotation=45)

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

def plot_time_graph(ax, dataframe, choice = 'original', type='magnitude', channels=None, apply_smoothing=False, smoothing_params=None):
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
    if choice == 'original':
        if isinstance(channels, int):
            ax.plot(selected_data['date'], selected_data[type], label=f'Channel {channels}')
        elif isinstance(channels, list):
            start_channel = channels[0]
            end_channel = channels[-1]
            ax.plot(selected_data['date'], selected_data[type], label=f'Channels {start_channel}-{end_channel}')
        ax.set_title(f'Time Graph of {type.capitalize()} ({title_suffix})')
    elif choice == 'first_derivative':
        selected_data['first_derivative'] = selected_data[type].diff()
        if isinstance(channels, int):
            ax.plot(selected_data['date'], selected_data['first_derivative'], label=f'First Derivative (Channel {channels})')
        elif isinstance(channels, list):
            start_channel = channels[0]
            end_channel = channels[-1]
            ax.plot(selected_data['date'], selected_data['first_derivative'], label=f'First Derivative (Channels {start_channel}-{end_channel})')
        ax.set_title(f'Time Graph of First Derivative of {type.capitalize()} ({title_suffix})')
    elif choice == 'second_derivative':
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

# the following functions are from Nicholas's Stability Tuning Code adopted to work with the new dataframe format
def stability_tuning_load(dataframe):
    ### FUNCTION TO COMPUTE CHANNEL TUNING ###
    df = dataframe.copy()

    df['date'] = pd.to_datetime(df['date'])
    
    # Convert results to organized DataFrame
    df_tuning = df.pivot(index='date', columns='channel')

    df_tuning = df_tuning[:407]

    ### VISUALIZATION ###
    def plot_tuning_analysis(df_tuning, window_length=49, poly_order=3):
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Magnitude Plot
        ax1 = plt.subplot(311)
        for channel in df_tuning['magnitude'].columns[:80]:  # First 5 channels
            y = df_tuning['magnitude'][channel].values
            y_smooth = savgol_filter(y, window_length, poly_order)
            ax1.plot(df_tuning.index, y_smooth, label=channel)
        ax1.set_title('Tuning Magnitude Over Time')
        ax1.set_ylabel('Magnitude')
        # ax1.legend()
        
        # 2. Angle Plot
        ax2 = plt.subplot(312)
        for channel in df_tuning['angle'].columns[:80]:
            y = df_tuning['angle'][channel].values
            y_smooth = savgol_filter(y, window_length, poly_order)
            ax2.plot(df_tuning.index, y_smooth, label=channel)
        ax2.set_title('Tuning Angle Over Time')
        ax2.set_ylabel('Angle (degrees)')
        # ax2.legend()
        
        # 3. Polar Plot for Selected Time Points
        ax3 = plt.subplot(313, projection='polar')
        # Select a few time points to show evolution
        time_points = df_tuning.index[::len(df_tuning.index)//10][:5]  # 5 evenly spaced time points
        
        for idx, time_point in enumerate(time_points):
            angles = df_tuning['angle'].loc[time_point][:50].values  # First 5 channels
            magnitudes = df_tuning['magnitude'].loc[time_point][:50].values
            ax3.scatter(np.radians(angles), magnitudes, 
                    label=time_point.strftime('%Y-%m-%d'),
                    alpha=0.6, s=100)
        ax3.set_title('Tuning Direction and Magnitude\nat Different Time Points')
        # ax3.legend()
        
        plt.tight_layout()
        plt.show()

    # Call the visualization functions
    plot_tuning_analysis(df_tuning)

    def plot_magnitude_distribution_outline(df_tuning):
        # Collect magnitudes and years
        magnitudes = []
        years = []
        
        for channel in df_tuning['magnitude'].columns:
            channel_magnitudes = df_tuning['magnitude'][channel]
            
            for date, mag in channel_magnitudes.items():
                if not np.isnan(mag):
                    magnitudes.append(mag)
                    years.append(date.year)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create outline for each year
        unique_years = sorted(set(years))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_years)))
        
        # Create bins for histogram
        range_min = 0
        range_max = max(magnitudes) if magnitudes else 0.1
        bin_width = (range_max - range_min) / 100  # Adjust for desired resolution
        bins = np.arange(range_min, range_max + bin_width, bin_width)

        for year, color in zip(unique_years, colors):
            year_magnitudes = [m for m, y in zip(magnitudes, years) if y == year]
            
            # Get histogram values for outline
            hist_values, _ = np.histogram(year_magnitudes, bins=bins, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Plot outline
            plt.plot(bin_centers, hist_values, color=color, linewidth=2, label=str(year))
        
        plt.xlabel('Tuning Magnitude')
        plt.ylabel('Density')
        plt.title('Distribution of Neural Tuning Magnitudes by Year')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set x-axis limits based on your magnitude ranges
        plt.xlim(0, 0.5)  # Adjust based on your data
        
        plt.tight_layout()
        plt.show()
        
    plot_magnitude_distribution_outline(df_tuning)

    from scipy.spatial.distance import jensenshannon
    def analyze_distributions_with_means(df_tuning):
        # Collect magnitudes and years
        magnitudes = []
        years = []
        for channel in df_tuning['magnitude'].columns:
            channel_magnitudes = df_tuning['magnitude'][channel]
            for date, mag in channel_magnitudes.items():
                if not np.isnan(mag):
                    magnitudes.append(mag)
                    years.append(date.year)
        
        # Prepare distributions
        unique_years = sorted(set(years))
        bins = np.linspace(0, 0.5, 100)
        distributions = {}
        means = {}
        
        for year in unique_years:
            year_magnitudes = [m for m, y in zip(magnitudes, years) if y == year]
            hist, _ = np.histogram(year_magnitudes, bins=bins, density=True)
            distributions[year] = hist / hist.sum()
            means[year] = np.mean(year_magnitudes)
        
        # Calculate all statistics
        comparisons = {}
        for i, year1 in enumerate(unique_years):
            for year2 in unique_years[i+1:]:
                year_mags1 = [m for m, y in zip(magnitudes, years) if y == year1]
                year_mags2 = [m for m, y in zip(magnitudes, years) if y == year2]
                
                # Original tests
                js_div = jensenshannon(distributions[year1], distributions[year2])
                ks_stat, ks_pvalue = stats.ks_2samp(year_mags1, year_mags2)
                
                # Add t-test
                t_stat, t_pvalue = stats.ttest_ind(year_mags1, year_mags2, equal_var=False)
                
                comparisons[f"{year1}-{year2}"] = {
                    "JS_divergence": js_div,
                    "KS_statistic": ks_stat,
                    "KS_p_value": ks_pvalue,
                    "t_statistic": t_stat,
                    "t_p_value": t_pvalue,
                    "mean_diff": means[year2] - means[year1]
                }
        
        return comparisons, means

    def print_extended_comparisons(comparisons, means):
        print("\n## Yearly Means ##")
        for year, mean in means.items():
            print(f"Year {year}: {mean:.4f}")
            
        print("\n## Distribution Comparisons Between Years ##")
        print(f"{'Years':12} {'JS Div':10} {'KS Stat':10} {'KS p-val':10} {'t-stat':10} {'t p-val':10} {'Mean Diff':10}")
        print("-" * 72)
        
        for years, stats in comparisons.items():
            print(f"{years:12} "
                f"{stats['JS_divergence']:.4f}    "
                f"{stats['KS_statistic']:.4f}    "
                f"{stats['KS_p_value']:.2e}  "
                f"{stats['t_statistic']:.4f}    "
                f"{stats['t_p_value']:.2e}  "
                f"{stats['mean_diff']:.4f}")

    # Usage
    comparisons, means = analyze_distributions_with_means(df_tuning)
    print_extended_comparisons(comparisons, means)

    def plot_magnitude_distribution_outline_rolling(df_tuning):
        # Convert the nested dictionary structure to a DataFrame
        magnitudes_by_channel = {}
        
        for channel in df_tuning['magnitude'].columns:
            channel_magnitudes = df_tuning['magnitude'][channel]
            magnitudes_by_channel[channel] = channel_magnitudes
        
        df_magnitudes = pd.DataFrame(magnitudes_by_channel)
        
        # Sort by date
        df_magnitudes.sort_index(inplace=True)
        
        # Compute time differences in days
        time_diff = df_magnitudes.index.to_series().diff().dt.total_seconds() / (24 * 3600)
        
        # Compute time-weighted derivatives
        derivatives = df_magnitudes.diff() / time_diff.values[:, np.newaxis]
        
        # Compute rolling average of derivatives (we are just doing 1-day here so no effect)
        window_size = '1D'
        rolling_avg = derivatives.rolling(window=window_size, center=True, min_periods=1).mean()
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create outline for each year
        unique_years = sorted(set(df_magnitudes.index.year))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_years)))
        
        # Create bins for histogram
        all_rates = rolling_avg.values.flatten()
        all_rates = all_rates[~np.isnan(all_rates)]
        
        range_min = np.percentile(all_rates, 1)  # 1st percentile to remove outliers
        range_max = np.percentile(all_rates, 99)  # 99th percentile to remove outliers
        bin_width = (range_max - range_min) / 100  # Adjust for desired resolution
        bins = np.arange(range_min, range_max + bin_width, bin_width)

        for year, color in zip(unique_years, colors):
            # Get rates for this year
            year_mask = rolling_avg.index.year == year
            year_rates = rolling_avg[year_mask].values.flatten()
            year_rates = year_rates[~np.isnan(year_rates)]
            
            if len(year_rates) > 0:
                # Get histogram values for outline
                hist_values, _ = np.histogram(year_rates, bins=bins, density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                # Plot outline
                plt.plot(bin_centers, hist_values, color=color, linewidth=2, label=str(year))
        
        plt.xlabel('Time-weighted Rate of Change in Tuning Magnitude (per day)')
        plt.ylabel('Density')
        plt.title('Distribution of Magnitude Changes by Year')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set reasonable x-axis limits based on the data distribution
        plt.xlim(range_min, range_max)
        
        plt.tight_layout()
        plt.show()


    plot_magnitude_distribution_outline_rolling(df_tuning)

    def analyze_time_weighted_distributions(df_tuning):
        # Convert to DataFrame
        magnitudes_by_channel = {}
        for channel in df_tuning['magnitude'].columns:
            channel_magnitudes = df_tuning['magnitude'][channel]
            magnitudes_by_channel[channel] = channel_magnitudes
        
        df_magnitudes = pd.DataFrame(magnitudes_by_channel)
        df_magnitudes.sort_index(inplace=True)
        
        # Calculate time-weighted derivatives
        time_diff = df_magnitudes.index.to_series().diff().dt.total_seconds() / (24 * 3600)
        derivatives = df_magnitudes.diff() / time_diff.values[:, np.newaxis]
        
        # Rolling average
        window_size = '1D'
        rolling_avg = derivatives.rolling(window=window_size, center=True, min_periods=1).mean()
        
        # Prepare distributions and means
        unique_years = sorted(set(rolling_avg.index.year))
        bins = np.linspace(-0.1, 0.1, 100)
        distributions = {}
        means = {}
        
        for year in unique_years:
            year_rates = rolling_avg[rolling_avg.index.year == year].values.flatten()
            year_rates = year_rates[~np.isnan(year_rates)]
            hist, _ = np.histogram(year_rates, bins=bins, density=True)
            distributions[year] = hist / hist.sum()
            means[year] = np.mean(year_rates)
        
        # Calculate statistics
        comparisons = {}
        for i, year1 in enumerate(unique_years):
            for year2 in unique_years[i+1:]:
                year_rates1 = rolling_avg[rolling_avg.index.year == year1].values.flatten()
                year_rates2 = rolling_avg[rolling_avg.index.year == year2].values.flatten()
                
                year_rates1 = year_rates1[~np.isnan(year_rates1)]
                year_rates2 = year_rates2[~np.isnan(year_rates2)]
                
                js_div = jensenshannon(distributions[year1], distributions[year2])
                t_stat, t_pvalue = stats.ttest_ind(year_rates1, year_rates2, equal_var=False)
                
                comparisons[f"{year1}-{year2}"] = {
                    "JS_divergence": js_div,
                    "t_statistic": t_stat,
                    "t_p_value": t_pvalue,
                    "mean_diff": means[year2] - means[year1]
                }
        
        return comparisons, means

    def print_derivative_comparisons(comparisons, means):
        print("\n## Yearly Mean Rates of Change ##")
        for year, mean in means.items():
            print(f"Year {year}: {mean:.6f} per day")
        
        print("\n## Time-Weighted Rate Comparisons ##")
        print(f"{'Years':12} {'JS Div':10} {'t-stat':10} {'t p-val':10} {'Mean Diff':10}")
        print("-" * 60)
        
        for years, stats in comparisons.items():
            print(f"{years:12} "
                f"{stats['JS_divergence']:.4f}    "
                f"{stats['t_statistic']:.4f}    "
                f"{stats['t_p_value']:.2e}  "
                f"{stats['mean_diff']:.6f}")

    # Usage
    comparisons, means = analyze_time_weighted_distributions(df_tuning)
    print_derivative_comparisons(comparisons, means)

    def plot_angle_distribution_outline(df_tuning):
        # Collect magnitudes and years
        magnitudes = []
        years = []
        
        for channel in df_tuning['angle'].columns:
            channel_magnitudes = df_tuning['angle'][channel]
            
            for date, mag in channel_magnitudes.items():
                if not np.isnan(mag):
                    magnitudes.append(mag)
                    years.append(date.year)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create outline for each year
        unique_years = sorted(set(years))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_years)))
        
        # Create bins for histogram
        range_min = -180
        range_max = 180 #max(magnitudes) if magnitudes else 0.1
        bin_width = (range_max - range_min) / 100  # Adjust for desired resolution
        bins = np.arange(range_min, range_max + bin_width, bin_width)

        for year, color in zip(unique_years, colors):
            year_magnitudes = [m for m, y in zip(magnitudes, years) if y == year]
            
            # Get histogram values for outline
            hist_values, _ = np.histogram(year_magnitudes, bins=bins, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Plot outline
            plt.plot(bin_centers, hist_values, color=color, linewidth=2, label=str(year))
        
        plt.xlabel('Tuning Angles')
        plt.ylabel('Density')
        plt.title('Distribution of Neural Tuning Angles by Year')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set x-axis limits based on your magnitude ranges
        plt.xlim(-180, 180)  # Adjust based on your data
        
        plt.tight_layout()
        plt.show()
        
    plot_angle_distribution_outline(df_tuning)

    def analyze_angle_distributions(df_tuning):
        # Collect angles and years
        angles = []
        years = []
        
        for channel in df_tuning['angle'].columns:
            channel_angles = df_tuning['angle'][channel]
            for date, ang in channel_angles.items():
                if not np.isnan(ang):
                    angles.append(ang)
                    years.append(date.year)
        
        # Prepare distributions
        unique_years = sorted(set(years))
        bins = np.linspace(-180, 180, 100)
        distributions = {}
        means = {}
        
        for year in unique_years:
            year_angles = [a for a, y in zip(angles, years) if y == year]
            hist, _ = np.histogram(year_angles, bins=bins, density=True)
            distributions[year] = hist / hist.sum()
            means[year] = np.mean(year_angles)
        
        # Calculate statistics
        comparisons = {}
        for i, year1 in enumerate(unique_years):
            for year2 in unique_years[i+1:]:
                year_angles1 = [a for a, y in zip(angles, years) if y == year1]
                year_angles2 = [a for a, y in zip(angles, years) if y == year2]
                
                # Calculate JS divergence
                js_div = jensenshannon(distributions[year1], distributions[year2])
                
                # Circular statistics might be more appropriate for angles,
                # but using t-test for consistency with previous analysis
                t_stat, t_pvalue = stats.ttest_ind(year_angles1, year_angles2, equal_var=False)
                
                comparisons[f"{year1}-{year2}"] = {
                    "JS_divergence": js_div,
                    "t_statistic": t_stat,
                    "t_p_value": t_pvalue,
                    "mean_diff": means[year2] - means[year1]
                }
        
        return comparisons, means

    def print_angle_comparisons(comparisons, means):
        print("\n## Yearly Mean Angles ##")
        for year, mean in means.items():
            print(f"Year {year}: {mean:.2f} degrees")
        
        print("\n## Angular Distribution Comparisons ##")
        print(f"{'Years':12} {'JS Div':10} {'t-stat':10} {'t p-val':10} {'Mean Diff':10}")
        print("-" * 60)
        
        for years, stats in comparisons.items():
            print(f"{years:12} "
                f"{stats['JS_divergence']:.4f}    "
                f"{stats['t_statistic']:.4f}    "
                f"{stats['t_p_value']:.2e}  "
                f"{stats['mean_diff']:.2f}")

    # Usage
    comparisons, means = analyze_angle_distributions(df_tuning)
    print_angle_comparisons(comparisons, means)

    def plot_angle_distribution_outline_rolling(df_tuning):
        # Convert the nested dictionary structure to a DataFrame
        magnitudes_by_channel = {}
        
        for channel in df_tuning['angle'].columns:
            channel_magnitudes = df_tuning['angle'][channel]
            magnitudes_by_channel[channel] = channel_magnitudes

        
        
        df_magnitudes = pd.DataFrame(magnitudes_by_channel)
        # df_magnitudes = np.degrees(np.unwrap(np.radians(df_magnitudes)))


        
        # Sort by date
        df_magnitudes.sort_index(inplace=True)


        # Compute time differences in days
        time_diff = df_magnitudes.index.to_series().diff().dt.total_seconds() / (24 * 3600)

        # # Now do the unwrapping
        # df_magnitudes = pd.DataFrame(
        #     np.degrees(np.unwrap(np.radians(df_magnitudes.values))),
        #     index=df_magnitudes.index,
        #     columns=df_magnitudes.columns
        # )
        
        # Compute time-weighted derivatives
        derivatives = df_magnitudes.diff() / time_diff.values[:, np.newaxis]
        
        # Compute rolling average of derivatives (we are just doing 1-day here so no effect)
        window_size = '1D'
        rolling_avg = derivatives.rolling(window=window_size, center=True, min_periods=1).mean()
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create outline for each year
        unique_years = sorted(set(df_magnitudes.index.year))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_years)))
        
        # Create bins for histogram
        all_rates = rolling_avg.values.flatten()
        all_rates = all_rates[~np.isnan(all_rates)]
        
        range_min = np.percentile(all_rates, 1)  # 1st percentile to remove outliers
        range_max = np.percentile(all_rates, 99)  # 99th percentile to remove outliers
        bin_width = (range_max - range_min) / 100  # Adjust for desired resolution
        bins = np.arange(range_min, range_max + bin_width, bin_width)

        for year, color in zip(unique_years, colors):
            # Get rates for this year
            year_mask = rolling_avg.index.year == year
            year_rates = rolling_avg[year_mask].values.flatten()
            year_rates = year_rates[~np.isnan(year_rates)]
            
            if len(year_rates) > 0:
                # Get histogram values for outline
                hist_values, _ = np.histogram(year_rates, bins=bins, density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                # Plot outline
                plt.plot(bin_centers, hist_values, color=color, linewidth=2, label=str(year))
        
        plt.xlabel('Time-weighted Rate of Change in Tuning Angle, per day (degrees)')
        plt.ylabel('Density')
        plt.title('Distribution of Angle Changes by Year')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set reasonable x-axis limits based on the data distribution
        plt.xlim(-180, 180)
        
        plt.tight_layout()
        plt.show()


    plot_angle_distribution_outline_rolling(df_tuning)

    def analyze_time_weighted_angle_distributions(df_tuning):
        # Convert to DataFrame
        magnitudes_by_channel = {}
        for channel in df_tuning['angle'].columns:
            channel_magnitudes = df_tuning['angle'][channel]
            magnitudes_by_channel[channel] = channel_magnitudes
        
        df_magnitudes = pd.DataFrame(magnitudes_by_channel)
        df_magnitudes.sort_index(inplace=True)
        
        # Calculate time-weighted derivatives
        time_diff = df_magnitudes.index.to_series().diff().dt.total_seconds() / (24 * 3600)
        derivatives = df_magnitudes.diff() / time_diff.values[:, np.newaxis]
        
        # Rolling average
        window_size = '1D'
        rolling_avg = derivatives.rolling(window=window_size, center=True, min_periods=1).mean()
        
        # Prepare distributions and means
        unique_years = sorted(set(rolling_avg.index.year))
        bins = np.linspace(-180, 180, 100)
        distributions = {}
        means = {}
        
        for year in unique_years:
            year_rates = rolling_avg[rolling_avg.index.year == year].values.flatten()
            year_rates = year_rates[~np.isnan(year_rates)]
            hist, _ = np.histogram(year_rates, bins=bins, density=True)
            distributions[year] = hist / hist.sum()
            means[year] = np.mean(year_rates)
        
        # Calculate statistics
        comparisons = {}
        for i, year1 in enumerate(unique_years):
            for year2 in unique_years[i+1:]:
                year_rates1 = rolling_avg[rolling_avg.index.year == year1].values.flatten()
                year_rates2 = rolling_avg[rolling_avg.index.year == year2].values.flatten()
                
                year_rates1 = year_rates1[~np.isnan(year_rates1)]
                year_rates2 = year_rates2[~np.isnan(year_rates2)]
                
                js_div = jensenshannon(distributions[year1], distributions[year2])
                t_stat, t_pvalue = stats.ttest_ind(year_rates1, year_rates2, equal_var=False)
                
                comparisons[f"{year1}-{year2}"] = {
                    "JS_divergence": js_div,
                    "t_statistic": t_stat,
                    "t_p_value": t_pvalue,
                    "mean_diff": means[year2] - means[year1]
                }
        
        return comparisons, means

    def print_angle_derivative_comparisons(comparisons, means):
        print("\n## Yearly Mean Rates of Change ##")
        for year, mean in means.items():
            print(f"Year {year}: {mean:.6f} per day")
        
        print("\n## Time-Weighted Rate Comparisons ##")
        print(f"{'Years':12} {'JS Div':10} {'t-stat':10} {'t p-val':10} {'Mean Diff':10}")
        print("-" * 60)
        
        for years, stats in comparisons.items():
            print(f"{years:12} "
                f"{stats['JS_divergence']:.4f}    "
                f"{stats['t_statistic']:.4f}    "
                f"{stats['t_p_value']:.2e}  "
                f"{stats['mean_diff']:.6f}")

    # Usage
    comparisons, means = analyze_time_weighted_angle_distributions(df_tuning)
    print_angle_derivative_comparisons(comparisons, means)

    def calculate_angle_changes(df_tuning):
        angles_by_channel = {}
        for channel in df_tuning['angle'].columns:
            channel_angles = df_tuning['angle'][channel]
            angles_by_channel[channel] = channel_angles
        
        df_angles = pd.DataFrame(angles_by_channel)[:405]
        df_angles.sort_index(inplace=True)
        
        # Calculate changes for each channel
        daily_changes = df_angles.resample('D').mean().diff().abs()
        mean_daily_changes = daily_changes.median()
        
        monthly_changes = df_angles.resample('ME').mean().diff().abs()
        mean_monthly_changes = monthly_changes.median()
        
        
        return {
            'daily_mean': mean_daily_changes,
            'monthly_mean': mean_monthly_changes
        }


    output_dict = calculate_angle_changes(df_tuning)

    print("Mean of Mean Day-to-Day Changes: ", output_dict["daily_mean"].mean(), "Mean of Mean Month-to-Month Changes: ", output_dict["monthly_mean"].mean())

    def plot_change_distributions(df_tuning):
        # Calculate changes
        output_dict = calculate_angle_changes(df_tuning)
        daily_means = output_dict['daily_mean']
        monthly_means = output_dict['monthly_mean']
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Daily changes histogram
        ax1.hist(daily_means, bins=20, edgecolor='black')
        ax1.set_title('Distribution of Mean Daily Changes Across Channels')
        ax1.set_xlabel('Mean Daily Changes')
        ax1.set_ylabel('Number of Channels')
        ax1.set_xlim(0, 120)  # Set x-axis range
        ax1.grid(True, alpha=0.3)
        
        # Monthly changes histogram
        ax2.hist(monthly_means, bins=20, edgecolor='black')
        ax2.set_title('Distribution of Mean Monthly Changes Across Channels')
        ax2.set_xlabel('Mean Monthly Changes')
        ax2.set_ylabel('Number of Channels')
        ax2.set_xlim(0, 120)  # Set x-axis range
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\nDaily Changes Summary:")
        print(f"Mean: {daily_means.mean():.4f}")
        print(f"Std: {daily_means.std():.4f}")
        print(f"Min: {daily_means.min():.4f}")
        print(f"Max: {daily_means.max():.4f}")
        
        print("\nMonthly Changes Summary:")
        print(f"Mean: {monthly_means.mean():.4f}")
        print(f"Std: {monthly_means.std():.4f}")
        print(f"Min: {monthly_means.min():.4f}")
        print(f"Max: {monthly_means.max():.4f}")

    # Call the function
    plot_change_distributions(df_tuning)

    def analyze_tuning_characteristics(df_tuning):
        # Define angle boundaries for categories
        angle_categories = {
            'index_positive': (-22.5, 22.5),
            'mixed_pos': (22.5, 67.5),
            'mrp_positive': (67.5, 112.5),
            'index_neg_mrp_pos': (112.5, 180),
            'index_pos_mrp_neg': (-90, -22.5),
            'both_negative': (-180, -90)
        }

        df_tuning = df_tuning.resample('D').interpolate(method='linear')
        
        # Use vectorized operations instead of loops
        def categorize_angles(angles, magnitudes):
            counts = {k: 0 for k in angle_categories}
            mask = magnitudes > 0
            valid_angles = angles[mask]
            
            for category, (lower, upper) in angle_categories.items():
                counts[category] = np.sum((valid_angles >= lower) & (valid_angles <= upper))
            
            return counts
        
        # Process all channels at once for each timepoint
        tuning_counts = {category: [] for category in angle_categories}
        
        # Vectorized operation over timepoints
        for time in df_tuning.index[::5]:  # Sample every 5th timepoint
            angles = df_tuning['angle'].loc[time]
            magnitudes = df_tuning['magnitude'].loc[time]
            counts = categorize_angles(angles, magnitudes)
            
            for category in angle_categories:
                tuning_counts[category].append(counts[category])
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        timepoints = df_tuning.index[::5]  # Use same sampling as above
        
        # Time series plot
        for category, values in tuning_counts.items():
            ax1.plot(timepoints, values, label=category)
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Number of Channels')
        ax1.set_title('Number of Channels by Tuning Type Over Time')
        ax1.legend()
        ax1.grid(True)
        
        # Stacked area plot
        data = np.array(list(tuning_counts.values()))
        ax2.stackplot(timepoints, data, 
                    labels=tuning_counts.keys(),
                    alpha=0.7)
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Number of Channels')
        ax2.set_title('Proportion of Channel Tuning Types Over Time')
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Summary statistics
        print("\nAverage number of channels in each category:")
        for category in tuning_counts:
            mean_count = np.mean(tuning_counts[category])
            print(f"{category}: {mean_count:.1f}")


        mask_negative = (df_tuning['angle'] >= -180) & (df_tuning['angle'] < -90)
        mask_positive = (df_tuning['angle'] >= 0) & (df_tuning['angle'] < 90)
        avg_magnitude_negative = df_tuning['magnitude'][mask_negative].mean().mean()
        
        mask_elsewhere = ~mask_negative
        avg_magnitude_elsewhere = df_tuning['magnitude'][mask_positive].mean().mean()
        
        print("\nAverage Magnitudes:")
        print(f"Negative angles (-180° to -90°): {avg_magnitude_negative:.2f}")
        print(f"All other angles: {avg_magnitude_elsewhere:.2f}")
        

    analyze_tuning_characteristics(df_tuning)
