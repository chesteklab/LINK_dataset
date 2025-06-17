import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import glob
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import copy
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

### Data loading  ###
def load_channel_population_data(data):
    
    # Check if data is a tuple and extract the dictionary if necessary
    if isinstance(data, tuple):
        if data[0] is None and isinstance(data[1], dict):
            data = data[1]
        elif isinstance(data[0], dict):
            data = data[0]
        else:
            print(f"Unexpected data structure: {type(data)}")
            return None

    return data

def load_all_datasets(data_folder = '../../Hisham_Temmar/big_dataset/2_autotrimming_and_preprocessing/preprocessing_092024_no7822nofalcon', total_datasets = None):
    ### LOAD AND PREPROCESS DATA ###

    print(os.getcwd())
    # Approximate total number of datasets
    if total_datasets is None:
        total_datasets = 384

    # Create a progress bar
    pbar = tqdm(total=total_datasets, desc="Processing datasets")

    # Path to the folder containing pkl files (FIND THIS IN HISHAMS STUDENT FOLDER -> BIG DATASET -> AUTOTRIMMING AND PREPROCESSING)

    # Get list of pkl files
    pkl_files = sorted(glob.glob(os.path.join(data_folder, '*.pkl')))

    print(len(pkl_files))

    # Dictionary to store results
    results = {}

    # Process each pkl file
    counter = 0
    for i in range(0, total_datasets, 1): # range(0, 384, 25):
        file = pkl_files[i]

        # Extract date from filename (assuming format like 'YYYY-MM-DD_data.pkl')
        date = pd.to_datetime(os.path.basename(file).split('_')[0])

        # Load data
        with open(file, 'rb') as f:
            data = pickle.load(f)
            
        # Compute channel tuning
        try:
            population_data = load_channel_population_data(data)
            if population_data is not None:
                results[date] = population_data
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue

        # Update the progress bar
        pbar.update(1)
        
        # Store results
        results[date] = population_data

        counter += 1
        if counter == 384:
            break

    # Close the progress bar
    pbar.close()
    return results

def prepare_tuning_data(results, normalize = True, normalize_all = True):
    ### PREPARE DATA FOR VISUALIZATION ###
    
    # Create lists to store data
    dates = []
    kinematics = []
    threshold_crossings = []
    sbps = []
    tcfrs = []
    target_styles = []
    target_positions = []
    trial_indices = []
    trial_counts = []
    
    for date, pop_data in results.items():
        # print(date)
        dates.append(date)
        kinematics.append(pop_data['finger_kinematics'])
        threshold_crossings.append(pop_data['tcfr'])
        sbps.append(pop_data['sbp'])
        tcfrs.append(pop_data['tcfr'])
        target_styles.append(pop_data['target_style'])
        target_positions.append(pop_data['target_positions'])
        trial_indices.append(pop_data['trial_index'])
        trial_counts.append(pop_data['trial_count'])
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'sbps': sbps,
        'tcfr': tcfrs,
        'threshold_crossings': threshold_crossings,
        'finger_kinematics': kinematics,
        'target_styles': target_styles,
        'target_positions': target_positions,
        'trial_counts': trial_counts,
        'trial_indices': trial_indices
    })
    
    df.set_index('date', inplace=True)

    return df

def normalize_data(df, data_type, normalization_method, pca_by_day = False):
    df_out = copy.deepcopy(df)

    if normalization_method == 'all':
    
        all_data = np.concatenate(df[data_type].values)
        scaler = StandardScaler().fit(all_data)
    
        for i, row in df.iterrows():
            scaled_data = scaler.transform(np.array(row[data_type]))
            
            df_out.at[i, data_type] = scaled_data
            
        print(all_data.shape)
        
    elif normalization_method == 'day':
        for i, row in df.iterrows():
            data = np.array(row[data_type])
            scaled_data = StandardScaler().fit_transform(data)
            if pca_by_day:
                pca = PCA(n_components=3)
                scaled_data = pca.fit_transform(scaled_data)
            
            df_out.at[i, data_type] = scaled_data
        
    elif normalization_method is None:
        return df_out
    else:
        raise ValueError
         
    for (_, df_row), (_, df_out_row)  in zip(df_out.iterrows(), df.iterrows()):
        df_data = df_row[data_type]
        df_out_data = df_out_row[data_type]
        if pca_by_day:
            assert (df_data.shape[0] == df_out_data.shape[0])

        else:
            assert (df_data.shape == df_out_data.shape)
        
    return df_out        
    
    
## Analysis ##
def day_by_day_PCA(df_tuning, threshold = 80):
    ## Analyze number of components to reach threshold variance for each day 
    dec_threshold = threshold / 100
    # Extract sbps data
    sbps_data = np.array(df_tuning['sbps'].tolist(), dtype=object)
    # Initialize lists to store results
    dates = df_tuning.index
    explained_variance_ratios = []
    num_components_80_variance = []

    # Perform PCA day-to-day
    for day_data in sbps_data:
        pca = PCA()
        pca.fit(day_data)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.argmax(cumulative_variance >= dec_threshold) + 1
        
        explained_variance_ratios.append(pca.explained_variance_ratio_)
        num_components_80_variance.append(num_components)

    # Store results in DataFrame
    pca_results = pd.DataFrame({
        'date': dates,
        'num_components_80_variance': num_components_80_variance,
        'explained_variance_ratios': explained_variance_ratios
    })
    
    
    # Plot 1: Number of components needed for 80% variance
    plt.figure(figsize=(10, 6))
    plt.plot(pca_results['date'], pca_results['num_components_80_variance'], 
            marker='o', label='Components for 80% Variance')
    plt.xlabel('Date')
    plt.ylabel('Number of Components')
    plt.title(f'Number of PCA Components for {threshold}% Variance Over Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.ylim(0,50)
    plt.show()

    return pca_results

def explained_var_over_time(df_tuning):
    ## TODO vairable grouping
    # Extract sbps data
    sbps_data = np.array(df_tuning['tcfr'].tolist(), dtype=object)
    dates = df_tuning.index

    # Create quarter labels
    dates = pd.to_datetime(dates)
    quarters = pd.PeriodIndex(dates, freq='Q').quarter
    years = pd.PeriodIndex(dates, freq='Q').year
    quarter_labels = [f'Q{q} {y}' for q, y in zip(quarters, years)]

    # Group data by quarters
    quarter_groups = pd.DataFrame({
        'data': sbps_data.tolist(),
        'quarter_label': quarter_labels
    }).groupby('quarter_label')


    # display(sbps_data)
    
    # display(quarter_groups)
    # Set up the plot
    plt.figure(figsize=(15, 10))

    # Define base colors for each year
    # year_colors = {
    #     2020: '#FFD700',  # Gold/Yellow
    #     2021: '#4169E1',  # Royal Blue
    #     2022: '#32CD32',  # Lime Green
    #     2023: '#FF4500'   # Orange Red
    # }
    year_colors = {2020: 'red', 2021: 'green', 2022: 'blue' , 2023: "orange"}


    # Process each quarter
    for quarter_label, group in quarter_groups:
        year = int(quarter_label.split()[-1])
        quarter = int(quarter_label[1])
        
        # Get the color for this quarter
        if year == 2024:
            continue
        
        quarter_color = get_quarter_color(quarter, year_colors[year])
        
        # Get the first day's data to determine number of features
        sample_data = group['data'].iloc[0]
        n_features = sample_data.shape[1]
        
        # Calculate average variance explained for the quarter
        quarter_variance = np.zeros(n_features)
        count = 0
        
        for day_data in group['data']:
            pca = PCA()
            pca.fit(day_data)
            quarter_variance += np.cumsum(pca.explained_variance_ratio_)
            count += 1
        
        # Calculate average
        avg_quarter_variance = quarter_variance / count
        
        
        print(quarter_color)
        # Plot the line for this quarter
        plt.plot(range(1, n_features + 1), avg_quarter_variance, 
                label=quarter_label, color=quarter_color)

    # Add the 80% threshold line
    plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80% Threshold')

    # Customize the plot
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.title('PCA Variance Explained by Quarter (2020-2023)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.show()

## Utils ##
# Create color variations for quarters within each year
def get_quarter_color(quarter, base_color, n_periods = 4):
    base_rgb = mcolors.to_rgb(base_color)
    # Darken the color progressively for each quarter
    scale = 1/(n_periods)
    factor = 1 - (quarter - 1) * scale
    return tuple(min(1, c * factor) for c in base_rgb)

def direction_map(directions='not_all'): 
    ''' 
    First step is to define the reach directions and mapping from coords to directions that we are looking for
    '''

    dir_list = None
    position_map = None

    if directions == 'not_all':
        label_list = {'N': 'MRP Flexion', 
                      'W': 'Index Extension',
                      'E': 'Index Flexion', 
                      'S': 'MRP Extension',
                      'NW':'MRP Flexion, \nIndex Extension',
                      'NE':'MRP Flexion, \nIndex Flexion', 
                      'SW':'MRP Extension, \nIndex Extension',
                      'SE':'MRP Flexion, \nIndex Flexion'
                    }
        dir_list = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        position_map = {
            (0.5, 0.7): 'N', (0.5, 0.9): 'N',
            (0.5, 0.1): 'S', (0.5, 0.3): 'S',
            (0.7, 0.5): 'E', (0.9, 0.5): 'E',
            (0.1, 0.5): 'W', (0.3, 0.5): 'W',
            (0.7, 0.7): 'NE', (0.7, 0.9): 'NE',
            (0.9, 0.7): 'NE', (0.9, 0.9): 'NE',
            (0.1, 0.7): 'NW', (0.1, 0.9): 'NW',
            (0.3, 0.7): 'NW', (0.3, 0.9): 'NW',
            (0.7, 0.1): 'SE', (0.7, 0.3): 'SE',
            (0.9, 0.1): 'SE', (0.9, 0.3): 'SE',
            (0.1, 0.1): 'SW', (0.1, 0.3): 'SW',
            (0.3, 0.1): 'SW', (0.3, 0.3): 'SW'
        }
    elif directions == 'all':
        dir_list = ['N', 'W', 'E', 'S', 'NW1', 'NW2', 'NW3', 'NW4', 'NE1', 'NE2', 'NE3', 'NE4', 'SW1','SW2','SW3','SW4', 'SE1','SE2','SE3','SE4']
        position_map = {
            (0.5, 0.7): 'N', (0.5, 0.9): 'N',
            (0.5, 0.1): 'S', (0.5, 0.3): 'S',
            (0.7, 0.5): 'E', (0.9, 0.5): 'E',
            (0.1, 0.5): 'W', (0.3, 0.5): 'W',
            (0.7, 0.7): 'NE1', (0.7, 0.9): 'NE2',
            (0.9, 0.7): 'NE3', (0.9, 0.9): 'NE4',
            (0.1, 0.7): 'NW1', (0.1, 0.9): 'NW2',
            (0.3, 0.7): 'NW3', (0.3, 0.9): 'NW4',
            (0.7, 0.1): 'SE1', (0.7, 0.3): 'SE2',
            (0.9, 0.1): 'SE3', (0.9, 0.3): 'SE4',
            (0.1, 0.1): 'SW1', (0.1, 0.3): 'SW2',
            (0.3, 0.1): 'SW3', (0.3, 0.3): 'SW4'
        }
        label_list = {'N': 'MRP Flexion', 
                'W': 'Index Extension',
                'E': 'Index Flexion', 
                'S': 'MRP Extension',
                'NW1':'MRP Flexion, \nIndex Extension',
                'NW2':'MRP Flexion, \nIndex Extension',
                'NW3':'MRP Flexion, \nIndex Extension',
                'NW4':'MRP Flexion, \nIndex Extension',
                'NE1':'MRP Flexion, \nIndex Flexion',
                'NE2':'MRP Flexion, \nIndex Flexion',
                'NE3':'MRP Flexion, \nIndex Flexion',
                'NE4':'MRP Flexion, \nIndex Flexion', 
                'SW1':'MRP Extension, \nIndex Extension',
                'SW2':'MRP Extension, \nIndex Extension',
                'SW3':'MRP Extension, \nIndex Extension',
                'SW4':'MRP Extension, \nIndex Extension',
                'SE1':'MRP Flexion, \nIndex Flexion',
                'SE2':'MRP Flexion, \nIndex Flexion',
                'SE3':'MRP Flexion, \nIndex Flexion',
                'SE4':'MRP Flexion, \nIndex Flexion',
            }
        
    elif directions == 'extreme':
        dir_list = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

        position_map = {
            (0.5, 0.7): 'X', (0.5, 0.9): 'N',
            (0.5, 0.1): 'S', (0.5, 0.3): 'X',
            (0.7, 0.5): 'X', (0.9, 0.5): 'E',
            (0.1, 0.5): 'W', (0.3, 0.5): 'X',
            
            (0.7, 0.7): 'X', (0.7, 0.9): 'X',
            (0.9, 0.7): 'X', (0.9, 0.9): 'NE',
            
            (0.1, 0.7): 'X', (0.1, 0.9): 'NW',
            (0.3, 0.7): 'X', (0.3, 0.9): 'X',
            
            (0.7, 0.1): 'X', (0.7, 0.3): 'X',
            (0.9, 0.1): 'SE', (0.9, 0.3): 'X',
            
            (0.1, 0.1): 'SW', (0.1, 0.3): 'X',
            (0.3, 0.1): 'X', (0.3, 0.3): 'X'
        }
        label_list = {'N': 'MRP Flexion', 
                'W': 'Index Extension',
                'E': 'Index Flexion', 
                'S': 'MRP Extension',
                'NW':'MRP Flexion, \nIndex Extension',
                'NE':'MRP Flexion, \nIndex Flexion', 
                'SW':'MRP Extension, \nIndex Extension',
                'SE':'MRP Flexion, \nIndex Flexion'
            }
    elif directions == 'small':
        dir_list = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

        position_map = {
            (0.5, 0.7): 'N', (0.5, 0.9): 'X',
            (0.5, 0.1): 'X', (0.5, 0.3): 'S',
            (0.7, 0.5): 'E', (0.9, 0.5): 'X',
            (0.1, 0.5): 'X', (0.3, 0.5): 'W',
            
            (0.7, 0.7): 'NE', (0.7, 0.9): 'X',
            (0.9, 0.7): 'X', (0.9, 0.9): 'X',
            
            (0.1, 0.7): 'X', (0.1, 0.9): 'X',
            (0.3, 0.7): 'NW', (0.3, 0.9): 'X',
            
            (0.7, 0.1): 'X', (0.7, 0.3): 'SE',
            (0.9, 0.1): 'X', (0.9, 0.3): 'X',
            
            (0.1, 0.1): 'X', (0.1, 0.3): 'X',
            (0.3, 0.1): 'X', (0.3, 0.3): 'SW',
            (0.5, 0.5): 'X'
        }
        label_list = {'N': 'MRP Flexion', 
                'W': 'Index Extension',
                'E': 'Index Flexion', 
                'S': 'MRP Extension',
                'NW':'MRP Flexion, \nIndex Extension',
                'NE':'MRP Flexion, \nIndex Flexion', 
                'SW':'MRP Extension, \nIndex Extension',
                'SE':'MRP Flexion, \nIndex Flexion'
            }
    elif directions == 'ext_flex':
        dir_list = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

        position_map = {
            (0, 0): 'X',
            (1, 0): 'N', 
            (-1, 0): 'S',
            (0, 1): 'E', 
            (0, -1): 'W', 
            (1, 1): 'NE', 
            (1, -1): 'NW', 
            (-1, 1): 'SE', 
            (-1, -1): 'SW'
        }
        
        label_list = {'N': 'MRP Flexion', 
                'NE':'MRP Flexion, \nIndex Flexion', 
                'E': 'Index Flexion', 
                'SE':'MRP Extension, \nIndex Flexion',
                'S': 'MRP Extension',  
                'SW':'MRP Extension, \nIndex Extension',              
                'W': 'Index Extension',
                'NW':'MRP Flexion, \nIndex Extension'
            }

    else:
        raise ValueError
    
    return dir_list, position_map, label_list
     
## Data Processing ##
def get_all_trial_classes(df, plot_targs = False, class_span = 45):
    df_out = copy.deepcopy(df)
    del df_out['target_positions']
    df_out['target_positions'] = [[] for _ in range(len(df))]
  
    if plot_targs:
        dir_list, position_map, direction_key = direction_map(directions='ext_flex')

        c_list = ['red','blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        colors = {}
        
        for i, dir in enumerate(sorted(dir_list)):
            colors[dir] = c_list[i]    
            
    diffs = []
    targs = []
    
    for day_idx, (_, day) in enumerate(df.iterrows()):
        classes = []  # Store movement classes
        last = None
        for target_pos in day['target_positions'].tolist():
            if last is None:
                last = target_pos
                classes.append(np.array([0, 0]))
            else: 
                
                targ = get_targ(target_pos, last, class_span = class_span)
                if (targ[0]==0) and (targ[1]==0):
                    a = 1
                else:
                    targs.append(targ)
                    diffs.append(np.array(target_pos) - np.array(last))

                classes.append(targ)
                last = target_pos  # Update for next iteration

        # Add the classes to the DataFrame
        index = day.name  # Get the actual index label from `df.iterrows()`
        df_out.loc[index, 'target_positions'] = [np.array(classes)]

    if plot_targs:
        plt.figure()
        color_list = [colors[position_map[tuple(targ)]] for targ in targs if position_map[tuple(targ)] != 'X']
        coords = np.vstack(diffs)  # Shape becomes (N, 2)

        plt.scatter(coords[:, 0], coords[:, 1], c = color_list)
        legend_elements = []
        used_indices = set(position_map[tuple(targ)] for targ in targs if position_map[tuple(targ)] != 'X')
        for idx in sorted(used_indices):
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', label=direction_key[idx],
                    markerfacecolor=colors[idx], markersize=8)
            )

        plt.legend(
            handles=legend_elements,
            title="Directions",
            loc='center left',
            bbox_to_anchor=(1, 0.5)
        )

        plt.xlabel("MRP Movement")
        plt.ylabel("Index Movement")
        plt.title("Movement (current target position - last target position)")

        plt.axis('equal')  # Ensures square aspect ratio

        plt.tight_layout()
        angles_deg = np.arange(0, 360, 22.5)
        radius = np.max(np.linalg.norm(coords, axis=1)) * 1.1  # Slightly beyond the furthest point

        for angle in angles_deg:
            if angle % 45 != 0:

                theta = np.deg2rad(angle)
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                plt.plot([0, x], [0, y], color='black', linewidth=0.8, linestyle='--')
                
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.show()
        

        
    # print(x_counter)
    return df_out

## Trim Methods ##
def max_jerk(trial_data, kinematics, std_multiplier):
    
    mvmt_idx, _ = movement_onset(trial_data, kinematics, std_multiplier)
    if mvmt_idx < 3:
        ind = -1
    else:
        acceleration = np.hstack(([0], np.diff(kinematics)))
        jerk = np.hstack(([0], np.diff(acceleration)))
        ind = np.argmax(abs(jerk[:mvmt_idx]))
    return ind, 0

def movement_onset(trial_data, kinematics, std_multiplier):
    # Use first 10% of trial kinematics as baseline
    baseline_period = kinematics[:int(kinematics.shape[0] * 0.1)]
    baseline_mean = np.mean(baseline_period)
    baseline_std = np.std(baseline_period)

    # Calculate threshold
    threshold = baseline_mean + (baseline_std * std_multiplier)
    min_threshold = baseline_mean - (baseline_std * std_multiplier)

    # Find first index where SBP exceeds threshold
    movement_onset_idx = next(
        (i for i, value in enumerate(kinematics) if ((value > threshold) or (value < min_threshold))),
        0
    )
    buffer = 150
    return movement_onset_idx, buffer
      
def trim_neural_data_at_movement_onset_std_and_smooth(data_dict, std_multiplier=2, sigma=5, display_alignment=False, trim_pt = movement_onset, direction_key = None, position_data = None):
    """
    Trims neural data using standard deviations above baseline mean, then smooths using Gaussian kernel
    
    Parameters:
    - data_dict: Dictionary containing neural data organized by year, and target position
    - std_multiplier: Number of standard deviations above baseline mean (default: 2)
    - sigma: Standard deviation for Gaussian kernel (in units of samples)
    - display_alignment: flag that specifies whether to visualize kinematic alignment
    
    Returns:
    - Dictionary with trimmed and smoothed neural data maintaining the same structure
    """
    trimmed_data = {}
    kin_data = {}

    for year in data_dict:
        trimmed_data[year] = {}
        kin_data[year] = {}
        
        for target_pos in data_dict[year]:
            # print(target_pos)
            if target_pos == 'X':
                continue
            if display_alignment:
                fig, ax = plt.subplots(figsize=(8, 4))

            for i, (trial_data, kinematics) in enumerate(data_dict[year][target_pos]):
                # print(f"Movement Onset: {movement_onset(trial_data, kinematics, std_multiplier)}, Max Jerk: {max_jerk(trial_data, kinematics, None)} ")
                ind, buffer = trim_pt(trial_data, kinematics, std_multiplier) 
                  # Trim the data to start from movement onset and smooth it 
                start = ind - (buffer//20) # want 150 ms pre movement and 20 ms bins 
                end = ind + ((750-buffer) // 20) # want 600 ms post movement and 20 ms bins 
                # pre_movement_start = movement_onset_idx # want 150 ms pre movement and 20 ms bins 
                # post_movement_end = len(kinematics)-1 # want 600 ms post movement and 20 ms bins 

                if start < 0: # (drop trials where movement onset seems too close to start or too close to end)
                    continue
                if end > trial_data.shape[0]-1:
                    continue

                trimmed_neural = trial_data[start:end, :]
                if sigma != 0:
                    # print(f'smoothing, sigma = {sigma}')
                    trimmed_neural = gaussian_filter1d(trial_data[start:end, :], sigma= sigma)
                trimmed_kin = kinematics[start:end]
                
                if target_pos not in trimmed_data[year]:
                    trimmed_data[year][target_pos] = [trimmed_neural]
                    kin_data[year][target_pos] =  [trimmed_kin]
                else:
                    trimmed_data[year][target_pos].append(trimmed_neural)
                    kin_data[year][target_pos].append(trimmed_kin)
                  
                if display_alignment:
                    pos = position_data[year][target_pos][i][1]
                    trimmed_pos = pos[start:end]
                    time = np.arange(len(trimmed_pos))
                    ax.plot(time, trimmed_pos, alpha = .8)   


            if display_alignment:
                ax.set_xlabel("Bins")
                ax.set_ylabel("Value")
                ax.set_title(f"Kinematics Over Time for Target {target_pos}, {direction_key[target_pos]} (Year {year})")
                ax.grid(True)
                # plt.plot(np.array(traces).mean(axis = 0), color = 'black')
                plt.show()

    
    return trimmed_data,  kin_data

def get_avg_trajs(df_tuning, group_by, period_to_plot_trajs, years_to_skip = [], trim_method = trim_neural_data_at_movement_onset_std_and_smooth, trim_pt = max_jerk, data_type = 'sbps', directions = 'ext_flex', norm_method = 'day', pca_all = True):
    
    df_tuning = copy.deepcopy(df_tuning)
    dir_list, position_map, direction_key = direction_map(directions=directions)

    df_tuning = normalize_data(df_tuning, data_type, norm_method)
    
    df_time, time_periods, _ = get_grouped_data(df_tuning, group_by, years_to_skip=years_to_skip)
    
    neural_data_for_direction, _ = split_and_pca_all_trials(df_time=df_time, time_periods=time_periods, type_of_data=data_type, kinematic_type='vel', position_map = position_map, pca_all = pca_all)
    # pos_data, _ = split_and_pca_all_trials(df_time=df_time, time_periods=time_periods, type_of_data=data_type, pca = pca, kinematic_type='pos')

    processed_neural_data_for_direction, kinematics_data = trim_method(neural_data_for_direction, std_multiplier=2, sigma=0, display_alignment=False, trim_pt = trim_pt, direction_key = direction_key)

    # plot_kin(kinematics_data)
    print("Trimmed Data")


    # average the PCA data across trials
    averaged_pca_results, averaged_kin_results = average_trial_PCA_data(dir_list=dir_list, kinematics=kinematics_data, processed_neural_data_for_direction=processed_neural_data_for_direction)

    keys = [str(x) for x in period_to_plot_trajs]
    traj_dict = {k: averaged_pca_results[k] for k in keys}
    
    return traj_dict


## Visualization
def plot_trajectories(averaged_pca_results, time_periods, label,  direction_key, output_path, directions = 'ext_flex', dim_inds = [0, 1, 2], elev = 20, azim= -45):
    dir_list, position_map, direction_key = direction_map(directions=directions)

    if label == 'Year':
        main_cols = len(list(set([str(x)[:4] for x in time_periods])))
        main_rows = 1
        figsize = (18, 8)
    elif label == 'Quarter':
        main_cols = len(list(set([str(x)[:4] for x in time_periods])))
        main_rows = 4
    elif label == 'Month':
        main_cols = len(list(set([str(x)[:4] for x in time_periods])))*2
        main_rows = 6
    elif label == "Week":
        return

    colors = {'N': (0.3, 0.0, 0.9), 'NE': (0.15, 0.4,  0.8), # N is Blue
              'E': (0.0, 0.8, 0.7), 'SE': (0.45, 0.8, 0.35), # W is greenish
              'S': (0.9, 0.7, 0.0), 'SW': (0.9, 0.35, 0.1),  # S is Yellowish
              'W': (0.9, 0.0, 0.2), 'NW': (0.65, 0.0,  0.55)}  #E is Red

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    fig.suptitle(f"Neural Trajectories by Reach Direction and {label}s (PCA)", fontsize=16)
    fig.patch.set_facecolor('white') # Sets the figure (outer) background to white

    
    subfigs = fig.subfigures(1, 2, width_ratios=[4.5, 1.5])

    # === Left Subfigure: PCA Trajectories ===
    # main_cols = len(list(set([str(x)[:4] for x in time_periods])))
    axs = subfigs[0].subplots(main_rows, main_cols, subplot_kw={'projection': '3d'}, squeeze=False)

    # gs = gridspec.GridSpec(main_rows, main_cols + 2, figure=fig, width_ratios=[1]*main_cols + [0.01, 0.7])

    # Add all PCA subplots
    for i, year in enumerate(time_periods):

        row = i // main_cols
        col = i % main_cols
        
        ax = axs[row][col]
        ax.set_box_aspect([1,1,1])
        
        # ax = fig.add_subplot(gs[row, col], projection='3d')
        # ax.set_box_aspect([1,1,1])
        # ax.grid(True)

        for direction in sorted(averaged_pca_results[str(year)].keys()):
            if averaged_pca_results[str(year)][direction] is not None:
                trajectory = averaged_pca_results[str(year)][direction]
                ax.plot(trajectory[:, dim_inds[0]], trajectory[:, dim_inds[1]], trajectory[:, dim_inds[2]],
                        color=colors[direction], label=f'{direction_key[direction]}')

                ax.scatter(trajectory[0, dim_inds[0]], trajectory[0, dim_inds[1]], trajectory[0, dim_inds[2]],
                        color=colors[direction], s=30, marker='o', edgecolor='black')

        print(year)
        ax.set_xlabel("  PC1")
        ax.set_ylabel("  PC2")
        ax.set_zlabel("  PC3  ")
        ax.set_title(f'{label} {str(year)}')
        if year == 2020:
            if directions == 'not_all':
                ax.view_init(elev = 0, azim = -70)
            elif directions == 'ext_flex':
                ax.view_init(elev = 0, azim = -65)
            elif directions == 'small':
                ax.view_init(elev = 0, azim = -70)
            elif directions == 'extreme':
                ax.view_init(elev = 0, azim = -70)

        elif year == 2021:
            ax.view_init(elev = 65, azim = -15)
        elif year == 2022:
            if directions == 'not_all':
                ax.view_init(elev = -155, azim = -65)
            elif directions == 'ext_flex':
                ax.view_init(elev = 65, azim = -75)
            elif directions == 'small':
                ax.view_init(elev = -160, azim = -70)
            elif directions == 'extreme':
                ax.view_init(elev = -130, azim = -83)
            
        elif year == 2023:
            ax.view_init(elev = -95, azim = -60)
        else:
            ax.view_init(elev=elev, azim=azim)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))  # Set max 5 x-ticks
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4)) 
        ax.zaxis.set_major_locator(MaxNLocator(nbins=4)) 
        ax.grid(False)
        ax.set_facecolor('white')
        ax.xaxis.set_pane_color((1, 1, 1, 1))
        ax.yaxis.set_pane_color((1, 1, 1, 1))
        ax.zaxis.set_pane_color((1, 1, 1, 1))




    # === Add Direction Circle as Final Subplot ===
    ax_circle = subfigs[1].add_subplot()
    ax_circle.set_aspect('equal')
    ax_circle.axis('off')  # Optional

    # Direction circle settings
    num_circles = 8
    radius = 1
    circle_radius = 0.25

    for i, dir in enumerate(dir_list):
        angle = (np.pi / 2) - (2 * np.pi * i / num_circles)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        circle = plt.Circle((x, y), circle_radius, color=colors[dir], ec='black')
        ax_circle.add_patch(circle)

    # Labels
    ax_circle.text(0, radius+.4, f"{direction_key['N']}", ha='center', va='bottom', fontsize=12)
    ax_circle.text(0, -radius-.4, f"{direction_key['S']}", ha='center', va='top', fontsize=12)
    ax_circle.text(-radius-.4, 0, f"{direction_key['W'][:5]}\n{direction_key['W'][6:]}", ha='right', va='center', fontsize=12)
    ax_circle.text(radius+.4, 0, f"{direction_key['E'][:5]}\n{direction_key['E'][6:]}", ha='left', va='center', fontsize=12)

    # ax_circle.text(radius+.2, radius+.2, f"{direction_key['NE']}", ha='left', va='top', fontsize=10)
    # ax_circle.text(radius+.2, -radius-.4, f"{direction_key['SE']}", ha='left', va='bottom', fontsize=10)
    # ax_circle.text(-radius-.2, radius+.2, f"{direction_key['NW']}", ha='right', va='top', fontsize=10)
    # ax_circle.text(-radius-.2, -radius-.4, f"{direction_key['SW']}", ha='right', va='bottom', fontsize=10)
   
    # Set bounds of circle subplot
    lim = radius + 1
    ax_circle.set_xlim(-lim, lim)
    ax_circle.set_ylim(-lim, lim)
    # fig.subplots_adjust(top=.9)
    
    plt.savefig(os.path.join(output_path, f"average_trajectories_{directions}"))

def centroids_across_time(averaged_pca_results, time_periods, color_dict, label, output_path, day_centroids = None, dim_inds = [0, 1, 2], PART_TO_PLOT = 'center', cmap = None, norm = None, radius = 'std'):
           
    legend = False
    # Color scheme by year (2020-2023)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_box_aspect([.2, .2, .2])
    ax.grid(True)
    ax.set_title(f"Center of Data in PCA Space Across {label}s", pad=20)

    # Store legend handles manually
    legend_handles = []
    centroids = []

    colors = []

    xlim = [-6, 14]
    ylim = [-3, 3]
    zlim = [-3, 5]
    for year_idx, year in enumerate(time_periods):

        color = color_dict[str(year)]
        colors.append(color)
        
        # Add to legend once per year
        legend_handles.append(plt.Line2D([0], [0], 
                            linestyle='-', 
                            color=color,
                            label=f'{label} {year}'))
        
        # Plot all directions for this year
        if PART_TO_PLOT == 'beginning':
            pts = []
            for direction in sorted(averaged_pca_results[str(year)].keys()):
                trajectory = averaged_pca_results[str(year)][direction]
                if trajectory is not None:
                    pts.append(trajectory[0, :])
                    
            pts = np.array(pts)
        elif PART_TO_PLOT == 'center':
            pts = []
            for dir in averaged_pca_results[str(year)].keys():

                data = [x[0] for x in averaged_pca_results[str(year)][dir]]

                pts.append(np.vstack(data))
            pts = np.vstack(pts)
        if len(pts) > 0:
            centroid = np.mean(pts, axis = 0)
            centroids.append(centroid)


            if radius == 'std':
                std_dev = np.std(pts, axis=0)  # Std dev for each dimension (x, y, z)

                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x_sphere = centroid[0] + std_dev[0] * np.outer(np.cos(u), np.sin(v))
                y_sphere = centroid[1] + std_dev[1] * np.outer(np.sin(u), np.sin(v))
                z_sphere = centroid[2] + std_dev[2] * np.outer(np.ones_like(u), np.cos(v))
                
                ax.plot_surface(x_sphere, y_sphere, z_sphere,
                                # rstride=10, cstride=10,  # Skip grid points
                                edgecolor='none',
                                color=color, alpha=0.07, linewidth=0, antialiased=True, zorder = 1)
            else:          
                pass
            
            if day_centroids is not None: 
                # print(f"N days in {year}: {len(day_centroids[str(year)])}")
                d_centroids = np.vstack(day_centroids[str(year)])
                mask = (
                    (d_centroids[:, 0] >= xlim[0]) & (d_centroids[:, 0] <= xlim[1]) &
                    (d_centroids[:, 1] >= ylim[0]) & (d_centroids[:, 1] <= ylim[1]) &
                    (d_centroids[:, 2] >= zlim[0]) & (d_centroids[:, 2] <= zlim[1])
                )
                
                d_centroids_in_bounds = d_centroids[mask]

                ax.scatter(d_centroids_in_bounds[:, 0], 
                            d_centroids_in_bounds[:, 1], 
                            d_centroids_in_bounds[:, 2],
                            color=color,
                            s=20,
                            marker='o',
                            edgecolor=None, zorder = 2)
                
                clamped = d_centroids.copy()
                was_clipped = np.zeros(clamped.shape[0], dtype=bool)
                for i, lims in enumerate([xlim, ylim, zlim]):
                    lower_mask = clamped[:, i] < lims[0]
                    upper_mask = clamped[:, i] > lims[1]

                    was_clipped |= lower_mask | upper_mask  # Track if any coordinate was clipped
                    clamped[lower_mask, i] = lims[0]
                    clamped[upper_mask, i] = lims[1]

                # Only plot points that were clipped
                out_of_bounds_points = clamped[was_clipped]

                ax.scatter(out_of_bounds_points[:, 0],
                        out_of_bounds_points[:, 1],
                        out_of_bounds_points[:, 2],
                        color=color,
                        s=20,
                        marker='x',
                        edgecolor=None,
                        zorder=2)


                
            ax.scatter(centroid[0], centroid[1], centroid[2],
                    color='black',
                    s=100,
                    marker='o',
                    depthshade=False)

            # Foreground (actual point)
            ax.scatter(centroid[0], centroid[1], centroid[2],
                    color=color,
                    s=50,
                    marker='o',
                    depthshade=False)                

    centroids = np.array(centroids)

    # ind = np.argmax(centroids[])
    # print(colors)
    for i in range(len(centroids)-1):
        x = [centroids[i, dim_inds[0]], centroids[i+1, dim_inds[0]]]
        y = [centroids[i, dim_inds[1]], centroids[i+1, dim_inds[1]]]
        z = [centroids[i, dim_inds[2]], centroids[i+1, dim_inds[2]]]

        color = (np.array(colors[i]) + np.array(colors[i + 1])) / 2
        ax.plot(x, y, z, color=color, linewidth=2, zorder = 2)
    # Axis labels and legend
    ax.set_xlabel(f"PC{dim_inds[0]+1}", labelpad=10)
    ax.set_ylabel(f"PC{dim_inds[1]+1}", labelpad=10)
    ax.set_zlabel(f"PC{dim_inds[2]+1}", labelpad=10)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
            
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Set max 5 x-ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5)) 
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5)) 

    if legend:
        ax.legend(handles=legend_handles,
            loc='upper left',
            bbox_to_anchor=(1.25, 1),
            borderaxespad=0.)
    else:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Required for colorbar to work
        cbar = plt.colorbar(sm, ax=ax, shrink=0.2, pad=0.1)
        cbar.set_label(f'Time', rotation=270, labelpad=15)
        
        years = list(set([str(x)[:4] for x in time_periods]))
        years.sort()
        # c_ticks = cbar.get_ticks()
        # ticks = np.linspace(0, int(norm.vmax), len(time_periods))
        ticks = [0, 4, 8, 12]

        labels = [f"Jan {year}" for year in years]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)
    fig.subplots_adjust(left=0.1, right=1.1, top=0.9, bottom=0.1)

    # Better viewing angle
    # if label == "Week":
    #     ax.view_init(elev=40, azim=-45)
    # else: 
    ax.view_init(elev=20, azim=80)
    
    # plt.tight_layout()
    plt.savefig(os.path.join(output_path, "pca_centroids_across_time"))

    # plt.show()

## PCA and data processing 
def average_trial_PCA_data(dir_list, kinematics ,processed_neural_data_for_direction):
    '''
    Average the PCA data across trials for eventual plotting
    '''

    averaged_pca_results = {year: 
                {direction: None for direction in dir_list} 
                for year in processed_neural_data_for_direction.keys()
    }

    averaged_kinematic_results = {year: 
                {direction: None for direction in dir_list} 
                for year in processed_neural_data_for_direction.keys()
    }

    for year in processed_neural_data_for_direction.keys():
        for direction in dir_list:
                      
            if direction in processed_neural_data_for_direction[year]:
    
                #num_trials = len(processed_neural_data_for_direction[year][direction])
                #time_points = processed_neural_data_for_direction[year][direction][0].shape[0]
                #direction_pca_results_reshaped = stacked_trials.reshape(num_trials, time_points, n_components)  # Shape: (num_trials, time_points, n_components)

                averaged_pca = np.mean(processed_neural_data_for_direction[year][direction], axis=0)
                averaged_pca_results[year][direction] = averaged_pca

                if kinematics is not None:
                    averaged_kin = np.mean(kinematics[year][direction], axis=0)
                    averaged_kinematic_results[year][direction] = averaged_kin
                else:
                    averaged_kinematic_results = None

    return averaged_pca_results, averaged_kinematic_results
     
def split(df_time, time_periods, position_map, type_of_data, jpca=False, kinematic_type = 'vel', only_CO = True):
 
    neural_data_for_direction = {str(key): {} for key in time_periods}

    if kinematic_type =="vel":
        index_ind = 2
        MRP_ind = 3
    elif kinematic_type == "pos":
        index_ind = 0
        MRP_ind = 1
    else: 
        raise ValueError("kinematic_type must be either 'vel' or 'pos")


    # extract data
    for j, year in enumerate(time_periods):
        # for channel_num in top_channel_indices:
        neural_data_for_direction[str(year)] = {}

        for  _, yearly_data in df_time.get_group(year).iterrows(): #[year].iterrows(): 
            

            # pca = PCA(n_components=n_components) # if you want to apply PCA transforms to each day
            # day_pca_results = pca.fit_transform(gaussian_filter1d(np.sqrt(yearly_data[type_of_data]), sigma=2))

            if only_CO:
                if yearly_data["target_styles"] != "CO": # only include it if its CO data
                    continue
            
            targ_coords = np.asarray(yearly_data['target_positions'])
            if len(targ_coords.shape) > 2 and targ_coords.shape[0] ==1:
                ## Handles difference between old and new pandas versions, which extract series differently 
                targ_coords = targ_coords[0, :, :]
           
            for i in range(0, targ_coords.shape[0]):
                trial_index_start = yearly_data["trial_indices"][i]
                trial_length = yearly_data['trial_counts'][i]

                target_pos_coords = targ_coords[i, :]
                target_pos_coords = (round(float(target_pos_coords[0]), 1), round(float(target_pos_coords[1]), 1))
                if target_pos_coords in position_map: # skips (0.5, 0.5)    

                    target_pos = position_map[target_pos_coords]

                    if (target_pos == "N") or (target_pos == "S"):
                        trial_kinematics = yearly_data['finger_kinematics'][:, MRP_ind][trial_index_start:trial_index_start+trial_length] # finger kinematics is [index_position, MRP_position, index_velocity, MRP_velocity]. We are indexing the MRP velocity here (N,S)
                    else:
                        trial_kinematics = yearly_data['finger_kinematics'][:, index_ind][trial_index_start:trial_index_start+trial_length] # finger kinematics is [index_position, MRP_position, index_velocity, MRP_velocity]. We are indexing the Index velocity here (E,W, or default to index in all other directions)
                        
                    neural_data = yearly_data[type_of_data][trial_index_start:trial_index_start+trial_length, :]
                    if target_pos not in neural_data_for_direction[str(year)]:
                        neural_data_for_direction[str(year)][target_pos] = [(neural_data, trial_kinematics)]
                    else:
                        neural_data_for_direction[str(year)][target_pos].append((neural_data, trial_kinematics))


    return neural_data_for_direction, None

def get_grouped_data(df_tuning, group_by = 'year', years_to_skip = ['2023', '2024']):
    
    for year in years_to_skip:
        df_tuning = df_tuning[~df_tuning.index.astype(str).str.startswith(str(year))]
        
    if group_by == "year":
        df_time = df_tuning.groupby(df_tuning.index.year) 
        time_periods = [g for g in list(df_time.groups.keys())]
        label = 'Year'
        
    elif group_by == "quarter":
        
        dates = df_tuning.index

        # Create quarter labels
        dates = pd.to_datetime(dates)
        quarters = pd.PeriodIndex(dates, freq='Q').quarter
        years = pd.PeriodIndex(dates, freq='Q').year
        quarter_label = [f'{y} Q{q} ' for q, y in zip(quarters, years)]

        df_tuning = df_tuning.copy()  # (optional, good habit)
        df_tuning['quarter_label'] = quarter_label

        # Now group by quarter_label
        df_time = df_tuning.groupby('quarter_label') 
        time_periods = [g for g in list(df_time.groups.keys())]
        label = "Quarter"
        
    elif group_by == "month":
        df_time = df_tuning.groupby(df_tuning.index.to_period("M"))
        time_periods = [g for g in list(df_time.groups.keys())]
        label = "Month"
    elif group_by == 'week':
        df_time = df_tuning.groupby(df_tuning.index.to_period("W"))
        time_periods = [g for g in list(df_time.groups.keys())]
        label = "Week"
        
    elif group_by == 'day':
        df_time = df_tuning.groupby(df_tuning.index.to_period("D"))
        time_periods = [g for g in list(df_time.groups.keys())]
        label = "Day"

    else:
        raise ValueError("group_by must be either 'year', 'quarter', 'month', 'week', or 'day")
    
    return df_time, time_periods, label
     
def split_and_pca_all_trials(df_time, time_periods, type_of_data, position_map, kinematic_type = 'vel', n_components = 3, pca_all = True):
 
    # print('new')
    neural_data_for_direction = {str(key): {} for key in time_periods}
    
    if pca_all:
        all_data = []
        for j, year in enumerate(time_periods):
            # for channel_num in top_channel_indices:
            for  _, yearly_data in df_time.get_group(year).iterrows(): #[year].iterrows(): 
                data = yearly_data[type_of_data]
                all_data.append(data)

        data = np.vstack(all_data)
        pca = PCA(n_components=n_components)    
        pca.fit(data)


    if kinematic_type =="vel":
        index_ind = 2
        MRP_ind = 3
    elif kinematic_type == "pos":
        index_ind = 0
        MRP_ind = 1
    else: 
        raise ValueError("kinematic_type must be either 'vel' or 'pos")

    centroids = {}

    x_counter = 0
    # extract data
    for j, year in enumerate(time_periods):
        # for channel_num in top_channel_indices:
        neural_data_for_direction[str(year)] = {}
        centroids[str(year)] = []

        for  _, yearly_data in df_time.get_group(year).iterrows(): #[year].iterrows():
        
            data = yearly_data[type_of_data]
            
        
            if pca_all:
                day_pca_results = pca.transform(data)
            else: 
                pca = PCA(n_components=n_components)
                day_pca_results = pca.fit_transform(data)
                
            centroids[str(year)].append(day_pca_results.mean(axis = 0))

            targ_coords = np.asarray(yearly_data['target_positions'])
            if len(targ_coords.shape) > 2 and targ_coords.shape[0] ==1:
                ## Handles difference between old and new pandas versions, which extract series differently 
                targ_coords = targ_coords[0, :, :]

            for i in range(0, targ_coords.shape[0]):
                trial_index_start = yearly_data["trial_indices"][i]
                trial_length = yearly_data['trial_counts'][i]

                target_pos_coords = targ_coords[i, :]
                target_pos_coords = (round(float(target_pos_coords[0]), 1), round(float(target_pos_coords[1]), 1))
                channel_pca_sbps = day_pca_results[trial_index_start:trial_index_start+trial_length] 

                if target_pos_coords in position_map:
                    target_pos = position_map[target_pos_coords]
                else:
                    continue
            
                if target_pos == 'X':
                    x_counter +=1
                    continue

                if (target_pos == "N") or (target_pos == "S"):
                    trial_kinematics = yearly_data['finger_kinematics'][:, MRP_ind][trial_index_start:trial_index_start+trial_length] # finger kinematics is [index_position, MRP_position, index_velocity, MRP_velocity]. We are indexing the MRP velocity here (N,S)
                else:
                    trial_kinematics = yearly_data['finger_kinematics'][:, index_ind][trial_index_start:trial_index_start+trial_length] # finger kinematics is [index_position, MRP_position, index_velocity, MRP_velocity]. We are indexing the Index velocity here (E,W, or default to index in all other directions)
                    
                if target_pos not in neural_data_for_direction[str(year)]:
                    neural_data_for_direction[str(year)][target_pos] = [(channel_pca_sbps, trial_kinematics)]
                else:
                    neural_data_for_direction[str(year)][target_pos].append((channel_pca_sbps, trial_kinematics))
    
    return neural_data_for_direction, centroids
  
def get_targ(targ_pos, last, class_span=45):
    # Fixed 8 directions (45° apart)
    direction_angles = [0, 45, 90, 135, 180, 225, 270, 315]  # Centers of the bins
    direction_vectors = {
        0: [1, 0],     # 0°
        1: [1, 1],     # 45° (unnormalized)
        2: [0, 1],     # 90°
        3: [-1, 1],    # 135° (unnormalized)
        4: [-1, 0],    # 180°
        5: [-1, -1],   # 225° (unnormalized)
        6: [0, -1],    # 270°
        7: [1, -1],    # 315° (unnormalized)
    }

    diff = np.array(targ_pos) - np.array(last)
    if (diff == 0).all():
        return np.array([0, 0])

    angle = np.arctan2(diff[1], diff[0])  # [-π, π]
    angle_deg = (np.degrees(angle) + 360) % 360  # [0°, 360°)

    # Find the closest fixed direction
    closest_dir = min(direction_angles, key=lambda x: min((angle_deg - x) % 360, (x - angle_deg) % 360))
    dir_index = direction_angles.index(closest_dir)

    # Check if the angle is within ±class_span/2 of the closest direction
    angle_diff = min((angle_deg - closest_dir) % 360, (closest_dir - angle_deg) % 360)
    if angle_diff <= class_span / 2:
        return np.array(direction_vectors[dir_index])
    else:
        return np.array([0, 0])
## Main fxns ##
def plot_centroid_of_pca_data_across_time(df_tuning, output_path, n_components = 3, dpca = False, group_by = "year", years_to_skip = [], data_type = 'sbps', remove_RT = False, directions = 'ext_flex', trim_pt = max_jerk, trim_method = trim_neural_data_at_movement_onset_std_and_smooth, PART_TO_PLOT = 'center', normalization_method = 'all', pca_all = True, plot_centr_across_time= True, sigma = .5):
    
    # print(directions)
    dir_list, position_map, direction_key = direction_map(directions=directions)

    if remove_RT:
        df_tuning = df_tuning[df_tuning['target_styles'] != 'RD']
    else:
        if directions != 'ext_flex':
            raise ValueError
      
    if directions == 'ext_flex':
        # display(df_tuning)
        df_tuning = get_all_trial_classes(df_tuning)
        # display(df_tuning)
       
    print("Made trial classes")

    # if normalization_method is not None:
    df_tuning = normalize_data(df_tuning, data_type=data_type, normalization_method=normalization_method)
  
    df_time, time_periods, label = get_grouped_data(df_tuning, group_by, years_to_skip)

    cmap = get_cmap('plasma_r')
    norm = Normalize(vmin=0, vmax=len(time_periods)+1)
    colors = [cmap(norm(j+1)) for j in range(len(time_periods))]
    color_dict = {}
    for i, time_period in enumerate(time_periods): 

        color_dict[str(time_period)] = colors[i]
        
    print('Grouped Data')
    neural_data_for_direction, day_centroids = split_and_pca_all_trials(df_time=df_time, time_periods=time_periods, type_of_data=data_type, kinematic_type='vel', position_map = position_map, pca_all = pca_all)

    if plot_centr_across_time:
        centroids_across_time(averaged_pca_results=neural_data_for_direction, time_periods=time_periods, color_dict=color_dict, label = label,output_path = output_path, PART_TO_PLOT=PART_TO_PLOT, cmap = cmap, norm = norm, radius = 'std', day_centroids = day_centroids)
  
def plot_avg_trajectories(df_tuning, type_of_data, output_path, group_by = "year", display_alignment = False, directions = 'not_all', trim_method = trim_neural_data_at_movement_onset_std_and_smooth, trim_pt = movement_onset, years_to_skip = [], sigma = 0, remove_RT = False):
    #  print(directions)
    dir_list, position_map, direction_key = direction_map(directions=directions)

    for year in years_to_skip:
        df_tuning = df_tuning[~df_tuning.index.astype(str).str.startswith(str(year))]

    if remove_RT:
        df_tuning = df_tuning[df_tuning['target_styles'] != 'RD']
        
    if directions == 'ext_flex':
        # display(df_tuning)
        df_tuning = get_all_trial_classes(df_tuning)

    df_tuning = normalize_data(df_tuning, data_type=type_of_data, normalization_method='day', pca_by_day=True)
    
    # display(df_tuning)
    df_time, time_periods, label = get_grouped_data(df_tuning, group_by, years_to_skip = [])
    # print(time_periods)
    cmap = get_cmap('plasma_r')
    norm = Normalize(vmin=0, vmax=len(time_periods)+1)
    colors = [cmap(norm(j+1)) for j in range(len(time_periods))]
    color_dict = {}
    for i, time_period in enumerate(time_periods): 

        color_dict[str(time_period)] = colors[i]
        
    neural_data_for_direction, _ = split(df_time=df_time, time_periods=time_periods, type_of_data=type_of_data, kinematic_type='vel', position_map = position_map, only_CO=remove_RT)

    processed_neural_data_for_direction, kinematics_data = trim_method(neural_data_for_direction, std_multiplier=2, sigma=sigma, display_alignment=display_alignment, trim_pt = trim_pt, direction_key = direction_key)

    averaged_pca_results, averaged_kin_results = average_trial_PCA_data(dir_list=dir_list, kinematics=kinematics_data, processed_neural_data_for_direction=processed_neural_data_for_direction)

    plot_trajectories(averaged_pca_results=averaged_pca_results, time_periods=time_periods, direction_key = direction_key,output_path = output_path, directions = directions, label = label, elev = 15, azim = -45)

