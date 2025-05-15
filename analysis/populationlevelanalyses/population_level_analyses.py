import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy import stats
from scipy.signal import savgol_filter
import glob
import pickle
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from IPython.display import display
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
from scipy.spatial import procrustes
from itertools import islice
# import jPCA
# from jPCA.util import plot_projections
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import copy
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy.linalg import subspace_angles




# dimensionality_across_days_analysis.ipynb needs to be ported into here essentially


def create_population_level_figure():
    # load the data
    # 
    pass


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
        dir_list = ['N', 'W', 'E', 'S', 'NW', 'NE', 'SW', 'SE']
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
        dir_list = ['N', 'W', 'E', 'S', 'NW', 'NE', 'SW', 'SE']

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
        dir_list = ['N', 'W', 'E', 'S', 'NW', 'NE', 'SW', 'SE']

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

def neural_dist(a, b):
    return abs(np.linalg.norm(a - b))

def get_closest_points(arrays, distance, center_pt = False):
    best_total_distance = np.inf
    best_path = None
    best_closest_points = None

    start_array = arrays[0]
    other_arrays = arrays[1:]

    for start_idx, start_point in enumerate(start_array):
        path = [start_idx]  # Start with the index in array 0
        total_distance = 0
        current_point = start_point
        closest_points = [start_point]  # Start with the initial point's coordinates

        for arr in other_arrays:
            if len(arr) < 3:
                path.append(-1)
            else:
                distances = np.array([distance(current_point, p) for p in arr])
                closest_idx = np.argmin(distances)
                total_distance += distances[closest_idx]
                path.append(closest_idx)
                current_point = arr[closest_idx]
                closest_points.append(current_point)

        if total_distance < best_total_distance:
            best_total_distance = total_distance
            best_path = path
            best_closest_points = closest_points
            
    # Compute the center of all closest points
    if center_pt:
        return np.mean(best_closest_points, axis=0)
    else:
        return best_path
          
def compute_velocity(pca_trajectories):
    """
    Compute velocity as the discrete time derivative of PCA trajectories.
    """
    if not isinstance(pca_trajectories, np.ndarray):
        return 0

    if pca_trajectories.size == 0 or pca_trajectories.shape[0] < 2:
        return 0

    return np.gradient(pca_trajectories, axis=0)  # Axis=0 corresponds to time

def compute_entanglement(pca_velocity, movement_velocity):
    """
    Compute entanglement metric by regressing PCA velocity onto movement velocity.
    Returns the entanglement index (higher = more entangled, lower = less entangled).
    """
    reg = LinearRegression()
    reg.fit(movement_velocity.reshape(1,-1), pca_velocity)  # Predict PCA velocity from movement velocity
    predicted_velocity = reg.predict(movement_velocity.reshape(1,-1))
    
    unexplained_variance = np.var(pca_velocity - predicted_velocity, axis=0).sum()
    total_variance = np.var(pca_velocity, axis=0).sum()

    entanglement_index = unexplained_variance / total_variance  # Ratio of unexplained variance
    return entanglement_index

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

                    
                # M = np.sign(diff[0])
                # I = np.sign(diff[1])
                
                # targ = np.array([M, I])
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

def trim_neural_data_at_movement_onset_std_align_at_year_and_smooth(data_dict, std_multiplier=2, sigma=5, display_alignment=False, trim_pt = movement_onset, direction_key = None, position_data = None):
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

    # trim_pt = movement_onset
    for year in data_dict:
        trimmed_data[year] = {}
        kin_data[year] = {}
        
        avgs = []
        # m_onsets = [[] for k in data_dict[year].keys()]
        for i, target_pos in enumerate(data_dict[year]):
            neurals = []

            for j, (trial_data, kinematics) in enumerate(data_dict[year][target_pos]):
                
                ind, _ = trim_pt(trial_data, kinematics, std_multiplier) 
                  # Trim the data to start from movement onset and smooth it 
                start = ind - 3 # want 150 ms pre movement and 20 ms bins 
                end = ind +((750) // 20)  # want 600 ms post movement and 20 ms bins 
                # start = movement_onset_idx # want 150 ms pre movement and 20 ms bins 
                # post_movement_end = len(kinematics)-1 # want 600 ms post movement and 20 ms bins 

                if (start < 0) or (end < 2) or (end > trial_data.shape[0]-1): # (drop trials where movement onset seems too close to start or too close to end)
                    continue
                else:
                    if sigma != 0:
                        traj = gaussian_filter1d(trial_data[start:end, :], sigma=sigma)
                        # print('smoothing')
                    else:
                        traj = trial_data[start:end, :]
                        
                    neurals.append(traj)
            
            # print(len(neurals))
            avgs.append(np.array(neurals).mean(axis = 0))
            # print(np.array(neurals).mean(axis = 0).shape)

        q_avgs = [x[:5, :] for x in avgs]
        # [print(x.shape) for x in q_avgs]
        path = get_closest_points(q_avgs, neural_dist)
        # print(path)
        for i, target_pos in enumerate(data_dict[year]):
            if display_alignment:
                fig, ax = plt.subplots(figsize=(8, 4))
                
            trimmed_data[year][target_pos] = [avgs[i][path[i]:, :]]
            kin_data[year][target_pos] =  [0]
                    # else:
                    #     trimmed_data[year][target_pos].append(trimmed_neural)
                    #     kin_data[year][target_pos].append(trimmed_kin)
                    
                    # if display_alignment:
                    #     time = np.arange(len(trimmed_kin))
                    #     ax.plot(time, trimmed_kin, alpha = .8)        

            if display_alignment:
                ax.set_xlabel("Bins")
                ax.set_ylabel("Value")
                ax.set_title(f"Kinematics Over Time for Target {target_pos} (Year {year})")
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

def visualize_trajectories(averaged_pca_results, time_periods, color_dict, label, dim_inds = [0, 1, 2], direction_key = None, cmap = None, norm = None, ALL_DIRS_ACROSS_YEARS=True, ALL_DIRS_ALL_YEARS_ONE_PLOT=True, ALL_DIRS_GROUPED_BY_YEAR=True, COMBINED_PLOT = True):

    # dim_inds = [0, 1, 2]
    if ALL_DIRS_ACROSS_YEARS:
        plot_all_dirs_across_years(averaged_pca_results, time_periods, label, dim_inds, color_dict, direction_key)
       
        
    if ALL_DIRS_ALL_YEARS_ONE_PLOT:
        plot_all_dirs_all_years_one_plot(averaged_pca_results, time_periods, label, dim_inds, color_dict, cmap, norm)
     

    if ALL_DIRS_GROUPED_BY_YEAR:
        plot_all_dirs_grouped_by_year(averaged_pca_results, time_periods, label, direction_key)

    if COMBINED_PLOT:
        plot_galaxy_and_trajs(averaged_pca_results, time_periods, label, dim_inds, color_dict, cmap, norm, direction_key)
       
def plot_all_dirs_across_years(averaged_pca_results, time_periods, label, dim_inds, color_dict, direction_key):
    fig = plt.figure(figsize=(10,25))
    fig.suptitle(f"Neural Trajectories by Reach Direction and {label} (PCA)", fontsize=16)

    for row, direction in enumerate(sorted(averaged_pca_results[str(time_periods[0])].keys())): # iterate through directions (2020 just used to get dirs)

        ax = fig.add_subplot(8,1, row+1, projection='3d') # have to expand this if want to break it down by magnitude
        #ax = fig.add_subplot(1,1, row+1, projection='3d')
        ax.set_box_aspect([1,1,1])  # Make the plot cubic
        ax.grid(True)
        
        for year in time_periods:
            if averaged_pca_results[str(year)][direction] is not None:
                trajectory = averaged_pca_results[str(year)][direction]          

                ax.plot(trajectory[:, dim_inds[0]], trajectory[:, dim_inds[1]], trajectory[:,dim_inds[2]], color=color_dict[str(year)], label=f'Year {year}')

                # Plot starting point as a dot
                ax.scatter(trajectory[0, dim_inds[0]], trajectory[0, dim_inds[1]], trajectory[0, dim_inds[2]], color=color_dict[str(year)], s=30, marker='o', edgecolor='black')

                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_zlabel("PC3")
                ax.set_title(f'{direction_key[direction]}')
                #ax.legend()

    # global legend
    handles = [plt.Line2D([0], [0], color=color_dict[year], lw=3, label=f'{year}') for year in color_dict]
    fig.legend(handles=handles, loc='upper right', fontsize=14, bbox_to_anchor=(1.05, 0.98))

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.tight_layout(pad=3.5)

    plt.show()

def plot_all_dirs_all_years_one_plot(averaged_pca_results, time_periods, label, dim_inds, color_dict, cmap, norm):
            # Color scheme by year (2020-2023)
    fig = plt.figure(figsize=(10, 25))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])
    ax.grid(True)
    ax.set_title(f"Combined Neural Trajectories by {label} (PCA)", pad=20)

    # Store legend handles manually
    legend_handles = []

    for year_idx, year in enumerate(time_periods):
        # Get color for this year
        color = color_dict[str(year)]
        
        # Add to legend once per year
        legend_handles.append(plt.Line2D([0], [0], 
                            linestyle='-', 
                            color=color,
                            label=f'{label} {year}'))
        
        # Plot all directions for this year
        trajs = []
        for direction in sorted(averaged_pca_results[str(year)].keys()):
            trajectory = averaged_pca_results[str(year)][direction]
            if trajectory is not None:
                # Plot trajectory line
                line = ax.plot(trajectory[:,dim_inds[0]], 
                            trajectory[:,dim_inds[1]], 
                            trajectory[:,dim_inds[2]],
                            color=color,
                            alpha=0.6,
                            linewidth=1.5)[0]
                
                ax.scatter(trajectory[0,dim_inds[0]], 
                        trajectory[0,dim_inds[1]], 
                        trajectory[0,dim_inds[2]],
                        color=color,
                        s=40,
                        marker='o',
                        edgecolor='black')
                
                trajs.append(trajectory)
        
    # Axis labels and legend
    ax.set_xlabel(f"PC{dim_inds[0]+1}", labelpad=10)
    ax.set_ylabel(f"PC{dim_inds[1]+1}", labelpad=10)
    ax.set_zlabel(f"PC{dim_inds[0]+1}", labelpad=10)
    # ax.legend(handles=legend_handles, 
    #         loc='upper left',
    #         bbox_to_anchor=(0.05, 0.95),
    #         frameon=True)
    
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for colorbar to work
    cbar = plt.colorbar(sm, ax=ax, shrink=0.2, pad=0.1)
    cbar.set_label(f'Time', rotation=270, labelpad=15)
    
    years = list(set([str(x)[:4] for x in time_periods]))
    years.sort()
    # c_ticks = cbar.get_ticks()
    ticks = np.linspace(0, int(norm.vmax), len(years))
    labels = years
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    

    # Better viewing angle
    ax.view_init(elev=20, azim=-45)

    plt.tight_layout()
    plt.show()


def plot_all_dirs_grouped_by_year(averaged_pca_results, time_periods, label, direction_key, dim_inds = [0, 1, 2], elev = 20, azim= -45):
    
    if label == 'Year':
        cols = len(list(set([str(x)[:4] for x in time_periods])))
        rows = 1
        figsize = (18, 8)
    elif label == 'Quarter':
        cols = len(list(set([str(x)[:4] for x in time_periods])))
        rows = 4
    elif label == 'Month':
        cols = len(list(set([str(x)[:4] for x in time_periods])))*2
        rows = 6
    elif label == "Week":
        return
   
    colors = {'N': (0.0, 0.0, 1.0), 'NE': (0.5,  0.0,  0.5),
              'E': (1.0, 0.0, 0.0), 'SE': (0.95, 0.35, 0.0), 
              'S': (0.9, 0.7, 0.0), 'SW': (0.6, 0.8, 0.0), 
              'W': (0.0, 0.8, 0.4), 'NW': (0.0,  0.45, 0.7)}
    
    
    fig = plt.figure(figsize=(26, 8), constrained_layout = True)
    fig.suptitle(f"Neural Trajectories by Reach Direction and {label}s (PCA)", fontsize=16)

    for i, year in enumerate(time_periods):  # One plot per year
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        ax.set_box_aspect([1,1,1])  # Make the plot cubic
        ax.grid(True)

        handles = []
        for j, direction in enumerate(sorted(averaged_pca_results[str(year)].keys())):
            if averaged_pca_results[str(year)][direction] is not None:
                trajectory = averaged_pca_results[str(year)][direction]
                
                handles.append(plt.Line2D([0], [0], 
                            linestyle='-', 
                            color=colors[direction],
                            label=f'{direction_key[direction]}'))
                ax.plot(trajectory[:,dim_inds[0]], trajectory[:,dim_inds[1]], trajectory[:,dim_inds[2]], color=colors[direction], label=f'{direction_key[direction]}')

                # Plot starting point as a dot
                ax.scatter(trajectory[0, dim_inds[0]], trajectory[0, dim_inds[1]], trajectory[0, dim_inds[2]], 
                        color=colors[direction], s=30, marker='o', edgecolor='black')

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f'{label} {str(time_periods[i])}')
        ax.view_init(elev=elev, azim=azim)

    # fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.95, 0.95))

    plt.tight_layout(rect=[0, 0.1, 1.01, .95])
    make_dir_key(colors = colors)


def plot_trajectories(averaged_pca_results, time_periods, label, direction_key, dim_inds = [0, 1, 2], elev = 20, azim= -45):
    dir_list, position_map, direction_key = direction_map(directions='ext_flex')

    if label == 'Year':
        cols = len(list(set([str(x)[:4] for x in time_periods])))
        rows = 1
        figsize = (18, 8)
    elif label == 'Quarter':
        cols = len(list(set([str(x)[:4] for x in time_periods])))
        rows = 4
    elif label == 'Month':
        cols = len(list(set([str(x)[:4] for x in time_periods])))*2
        rows = 6
    elif label == "Week":
        return

    colors = {'N': (0.3, 0.0, 0.9), 'NE': (0.15, 0.4,  0.8), # N is Blue
              'E': (0.0, 0.8, 0.7), 'SE': (0.45, 0.8, 0.35), # W is greenish
              'S': (0.9, 0.7, 0.0), 'SW': (0.9, 0.35, 0.1),  # S is Yellowish
              'W': (0.9, 0.0, 0.2), 'NW': (0.65, 0.0,  0.55)}  #E is Red

    fig = plt.figure(figsize=(18, 7), constrained_layout=False)
    fig.suptitle(f"Neural Trajectories by Reach Direction and {label}s (PCA)", fontsize=16)

    # Define number of rows and columns for plots + 1 extra column for the legend
    main_cols = len(list(set([str(x)[:4] for x in time_periods])))
    main_rows = rows
    gs = gridspec.GridSpec(main_rows, main_cols + 2, figure=fig, width_ratios=[1]*main_cols + [0.01, 0.7])

    # Add all PCA subplots
    for i, year in enumerate(time_periods):
        row = i // main_cols
        col = i % main_cols
        ax = fig.add_subplot(gs[row, col], projection='3d')
        ax.set_box_aspect([1,1,1])
        ax.grid(True)

        for direction in sorted(averaged_pca_results[str(year)].keys()):
            if averaged_pca_results[str(year)][direction] is not None:
                trajectory = averaged_pca_results[str(year)][direction]
                ax.plot(trajectory[:, dim_inds[0]], trajectory[:, dim_inds[1]], trajectory[:, dim_inds[2]],
                        color=colors[direction], label=f'{direction_key[direction]}')

                ax.scatter(trajectory[0, dim_inds[0]], trajectory[0, dim_inds[1]], trajectory[0, dim_inds[2]],
                        color=colors[direction], s=30, marker='o', edgecolor='black')

        print(year)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f'{label} {str(year)}')
        if year == 2020:
            ax.view_init(elev = 0, azim = -65)
        elif year == 2021:
            ax.view_init(elev = 65, azim = -15)
        elif year == 2022:
            ax.view_init(elev = 65, azim = -75)
        elif year == 2023:
            ax.view_init(elev = 75, azim = -75)
        else:
            ax.view_init(elev=elev, azim=azim)

    # === Add Direction Circle as Final Subplot ===
    ax_circle = fig.add_subplot(gs[:, -1])  # Last column across all rows
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
    ax_circle.text(0, radius+.4, f"{direction_key['N']}", ha='center', va='bottom', fontsize=10)
    ax_circle.text(0, -radius-.4, f"{direction_key['S']}", ha='center', va='top', fontsize=10)
    ax_circle.text(-radius-.4, 0, f"{direction_key['W'][:5]}\n{direction_key['W'][6:]}", ha='right', va='center', fontsize=10)
    ax_circle.text(radius+.4, 0, f"{direction_key['E'][:5]}\n{direction_key['E'][6:]}", ha='left', va='center', fontsize=10)

    # ax_circle.text(radius+.2, radius+.2, f"{direction_key['NE']}", ha='left', va='top', fontsize=10)
    # ax_circle.text(radius+.2, -radius-.4, f"{direction_key['SE']}", ha='left', va='bottom', fontsize=10)
    # ax_circle.text(-radius-.2, radius+.2, f"{direction_key['NW']}", ha='right', va='top', fontsize=10)
    # ax_circle.text(-radius-.2, -radius-.4, f"{direction_key['SW']}", ha='right', va='bottom', fontsize=10)
   
    # Set bounds of circle subplot
    lim = radius + 1
    ax_circle.set_xlim(-lim, lim)
    ax_circle.set_ylim(-lim, lim)
    fig.subplots_adjust(left=0.05, right=0.95, top=1, bottom=0.2, wspace=0.2)
    
    plt.savefig("average_trajectories.svg")

    
def plot_galaxy_and_trajs(averaged_pca_results, time_periods, label, dim_inds, color_dict, cmap, norm, direction_key):
    
    if label == "Year":
        n_rows = len(time_periods)
        n_cols = 2
        width_ratios=[3, 1]
        splts = [[i, 1] for i in range(len(time_periods))]
    elif label == "Quarter":
        n_rows = len(list(set([str(x)[:4] for x in time_periods])))

        n_cols = 5
        width_ratios=[3, 1, 1, 1, 1]
        splts = [[int(np.floor((i)/(n_rows+1))), (i % (n_cols-1))+1] for i in range(len(time_periods))]
    else:
        # plot_all_dirs_all_years_one_plot(averaged_pca_results, time_periods, label, dim_inds, color_dict, cmap, norm)
        # plot_all_dirs_grouped_by_year(averaged_pca_results, time_periods, label, direction_key)
        return
        
    # if label != 'Year':
    #     return
    colors = {
        0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 
        4: 'orange', 5: 'brown', 6: 'pink', 7: 'gray'
    }  # Different colors for each direction

    
    # Create figure
    fig = plt.figure(figsize=(26, 14), constrained_layout=True)
    # gs = gridspec.GridSpec(3, 4, width_ratios=[3, 0, 0.01, 1], wspace=.01, hspace=0.5)
    gs = gridspec.GridSpec(n_rows, n_cols, width_ratios=width_ratios, hspace=.35, wspace = 0)  # 2 columns: [main, smalls]

    # Large main plot on the right (spans all 3 rows)
    ax_main = fig.add_subplot(gs[:, 0], projection = '3d')
    ax_main.set_box_aspect([1,1,1])
    ax_main.grid(True)
    ax_main.set_title(f"Combined Neural Trajectories by {label} (PCA)", pad=20, fontweight = 'bold', fontsize = 14)
    legend_handles = []

    for year_idx, year in enumerate(time_periods):
        # Get color for this year
        color = color_dict[str(year)]
        
        # Add to legend once per year
        legend_handles.append(plt.Line2D([0], [0], 
                            linestyle='-', 
                            color=color,
                            label=f'{label} {year}'))
        
        # Plot all directions for this year
        trajs = []
        for direction in sorted(averaged_pca_results[str(year)].keys()):
            trajectory = averaged_pca_results[str(year)][direction]
            if trajectory is not None:
                # Plot trajectory line
                line = ax_main.plot(trajectory[:,dim_inds[0]], 
                            trajectory[:,dim_inds[1]], 
                            trajectory[:,dim_inds[2]],
                            color=color,
                            alpha=0.6,
                            linewidth=1.5)[0]
                
                ax_main.scatter(trajectory[0,dim_inds[0]], 
                        trajectory[0,dim_inds[1]], 
                        trajectory[0,dim_inds[2]],
                        color=color,
                        s=40,
                        marker='o',
                        edgecolor='black')
                
                trajs.append(trajectory)
                
    # Axis labels and legend
    ax_main.set_xlabel(f"PC{dim_inds[0]+1}", labelpad=10)
    ax_main.set_ylabel(f"PC{dim_inds[1]+1}", labelpad=10)
    ax_main.set_zlabel(f"PC{dim_inds[2]+1}", labelpad=10)
    ax_main.legend(handles=legend_handles, 
            loc='upper left',
            bbox_to_anchor=(0.05, 0.95),
            frameon=True)

    # Better viewing angle
    ax_main.view_init(elev=20, azim=-35)
    # ax_main.view_init(elev=90, azim=-90)


    # Smaller plots
    for i, year in enumerate(time_periods):  # One plot per year
        # print(year)
        # col = i % (n_cols)
        # row = np.floor((i)/(n_rows+1)) 
        # print(splts[i])
        # print(splts[i])

        
        ax = fig.add_subplot(gs[splts[i][0], splts[i][1]], projection='3d')
        ax.set_box_aspect([1,1,1])  # Make the plot cubic
        ax.grid(True)

        lines = []
        for j, direction in enumerate(sorted(averaged_pca_results[str(year)].keys())):
            if averaged_pca_results[str(year)][direction] is not None:
                trajectory = averaged_pca_results[str(year)][direction]
                line, = ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], color=colors[j], label=f'{direction_key[direction]}')
                lines.append(line)
                # Plot starting point as a dot
                ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                        color=colors[j], s=30, marker='o', edgecolor='black')

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f'{year}')
        
    fig.legend(
        handles=lines,
        loc='center left',
        bbox_to_anchor=(.92, 0.5),  # X=0.35 places it between columns
        borderaxespad=0., 
        fontsize = 14
        )

    x = (width_ratios[0] + (sum(width_ratios[1:])/2)-.3)/(sum(width_ratios))
    # print(width_ratios[0]/sum(width_ratios) )
    # print(sum(width_ratios) - width_ratios[0])/(sum(width_ratios)*2)
    print(x)
    fig.text(x, 0.92, "Neural Trajectories by Reach\n Direction and Year (PCA)", ha='center', va='bottom', fontsize=14, fontweight='bold')

    # plt.subplots(layout="constrained")
    # plt.subplots_adjust(right=0.5)
    # fig.tight_layout()
    # plt.show()

def plot_across_time_and_trajs(df_tuning, left_group_by = 'weeks',  right_group_by = 'quarters', trim_method = trim_neural_data_at_movement_onset_std_and_smooth, trim_pt = max_jerk, remove_RT = False, part_to_plot = 'center', normalization_method = 'all', pca_all = True, directions = 'ext_flex'):
    left_time_periods,  left_averaged_results,  left_label, _, color_dict, cmap, norm = plot_pca_all(df_tuning, group_by=left_group_by,  trim_method = trim_method, trim_pt = trim_pt, remove_RT=remove_RT, PART_TO_PLOT=part_to_plot, normalization_method = 'all', pca_all = True, plot_centr_across_time = False, directions = directions)
    right_time_periods, right_averaged_results, right_label, direction_key, _, _, _ = plot_pca_all(df_tuning, group_by=right_group_by, trim_method = trim_method, trim_pt = trim_pt, remove_RT=remove_RT, PART_TO_PLOT=None, normalization_method = normalization_method, pca_all = pca_all, plot_centr_across_time = False, directions = directions)

    if right_group_by == "year":
        n_rows = len(right_time_periods)
        n_cols = 2
        width_ratios=[3, 1]
        splts = [[i, 1] for i in range(len(right_time_periods))]
    elif right_group_by == "quarter":
        n_rows = len(list(set([str(x)[:4] for x in right_time_periods])))

        n_cols = 5
        width_ratios=[3, 1, 1, 1, 1]
        splts = [[int(np.floor((i)/(n_rows))), (i % (n_cols-1))+1] for i in range(len(right_time_periods))]
    else:
        raise ValueError
        
    print(n_rows)
    # Create figure
    fig = plt.figure(figsize=(26, 14), constrained_layout=True)
    # gs = gridspec.GridSpec(3, 4, width_ratios=[3, 0, 0.01, 1], wspace=.01, hspace=0.5)
    gs = gridspec.GridSpec(n_rows, n_cols, width_ratios=width_ratios, hspace=.35, wspace = 0)  # 2 columns: [main, smalls]

    # Large main plot on the right (spans all 3 rows)
    ax_main = fig.add_subplot(gs[:, 0], projection = '3d')
    ax_main.set_box_aspect([1,1,1])
    ax_main.grid(True)
    ax_main.set_title(f"Center of Data in PCA Space Across {left_label}s", pad=20, fontweight = 'bold', fontsize = 14)
         
    centroids = []
    colors = []
    for year_idx, year in enumerate(left_time_periods):

        color = color_dict[str(year)]
        colors.append(color)
        
        # Plot all directions for this year
        if part_to_plot == 'beginning':
            pts = []
            for direction in sorted(left_averaged_results[str(year)].keys()):
                trajectory = left_averaged_results[str(year)][direction]
                if trajectory is not None:
                    pts.append(trajectory[0, :])
                    
            pts = np.array(pts)
        elif part_to_plot == 'center':
            pts = []
            for dir in left_averaged_results[str(year)].keys():
                data = [x[0] for x in left_averaged_results[str(year)][dir]]
                pts.append(np.vstack(data))
            
            pts = np.vstack(pts)
        if len(pts) > 0:
            centroid = np.mean(pts, axis = 0)
            centroids.append(centroid)
            ax_main.scatter(centroid[0], 
                            centroid[1], 
                            centroid[2],
                            color=color,
                            s=40,
                            marker='o',
                            edgecolor=color)

    centroids = np.array(centroids)
    
    for i in range(len(centroids)-1):
        x = [centroids[i, 0], centroids[i+1, 0]]
        y = [centroids[i, 1], centroids[i+1, 1]]
        z = [centroids[i, 2], centroids[i+1, 2]]

        color = (np.array(colors[i]) + np.array(colors[i + 1])) / 2
        ax_main.plot(x, y, z, color=color, linewidth=2)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for colorbar to work
    cbar = plt.colorbar(sm, ax=ax_main, shrink = .9, orientation='horizontal')
    cbar.set_label(f'Time', labelpad=15)
    
    years = list(set([str(x)[:4] for x in left_time_periods]))
    years.sort()
    ticks = np.linspace(0, int(norm.vmax), len(years))
    labels = years
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    # fig.subplots_adjust(left=0.1, right=1.1, top=0.9, bottom=0.1)

    # Axis labels and legend
    ax_main.set_xlabel(f"PC 1", labelpad=10)
    ax_main.set_ylabel(f"PC 2", labelpad=10)
    ax_main.set_zlabel(f"PC 3", labelpad=10)

    # Better viewing angle
    ax_main.view_init(elev=20, azim=-35)

    dir_colors = {
        'E': 'red', 'N': 'blue', 'NE': 'green', 'NW': 'purple', 
        'S': 'orange', 'SE': 'brown', 'SW': 'pink', 'W': 'gray'
    }  # Different colors for each direction

    # print(direction_key)
    # Smaller plots

    for i, year in enumerate(right_time_periods):  # One plot per year
        # print(i)
        # print(splts[i])
        ax = fig.add_subplot(gs[splts[i][0], splts[i][1]], projection='3d')
        ax.set_box_aspect([1,1,1])  # Make the plot cubic
        ax.grid(True)
        
        lines = []
        for j, direction in enumerate(sorted(right_averaged_results[str(year)].keys())):
            # print(direction)
            if right_averaged_results[str(year)][direction] is not None:
                # print({direction_key[direction]})
                trajectory = right_averaged_results[str(year)][direction]
                line, = ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], color=dir_colors[direction], label=f'{direction_key[direction]}')
                lines.append(line)
                # Plot starting point as a dot
                ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                        color=dir_colors[direction], s=30, marker='o', edgecolor='black')

        if i == 0:
            leg_lines = lines
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f'{year}')
        
    fig.legend(
        handles=leg_lines,
        loc='center left',
        bbox_to_anchor=(.92, 0.5),  # X=0.35 places it between columns
        borderaxespad=0., 
        fontsize = 14
        )

    x = (width_ratios[0] + (sum(width_ratios[1:])/2)-.3)/(sum(width_ratios))
    fig.text(x, 0.92, f"Neural Trajectories by Reach\n Direction and {right_label} (PCA)", ha='center', va='bottom', fontsize=14, fontweight='bold')

def visualize_monthly_trajectories(averaged_pca_results, time_periods):

    base_cmaps = {
        2020: cm.Reds,
        2021: cm.Greens,
        2022: cm.Blues,
        2023: cm.Oranges
    }

    fig = plt.figure(figsize=(10, 25))
    fig.suptitle("Neural Trajectories by Reach Direction and Month (PCA)", fontsize=16)

    example_period = next(iter(averaged_pca_results))
    for row, direction in enumerate(sorted(averaged_pca_results[example_period].keys())):
        if direction != "N":
            continue  # Skip until we find "N"

        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_box_aspect([1,1,1])
        ax.grid(True)

        for period in time_periods:
            year = period.year
            month = period.month
            if year not in base_cmaps or averaged_pca_results.get(period) is None:
                continue
            if averaged_pca_results[period][direction] is None:
                continue

            # Get shade for this month (normalize 1–12 to 0–1)
            norm_month = (month - 1) / 11
            color = base_cmaps[year](norm_month)

            trajectory = averaged_pca_results[period][direction]
            ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2],
                    color=color, label=f'{period.strftime("%b %Y")}')

            ax.scatter(trajectory[0,0], trajectory[0,1], trajectory[0,2],
                    color=color, s=30, marker='o', edgecolor='black')

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            ax.set_title(f'Direction {direction}')

        break  # Done after "N"

def cross_corr_viz(pca_data):

    X = pca_data[2020]['N']
    Y = pca_data[2021]['N']
    Z = pca_data[2022]['N']

    X_aligned, Y_aligned, _ = procrustes(X, Y)
    X2_aligned, Z_aligned, _ = procrustes(X, Z)


    cca = CCA(n_components=3)
    X_c1, Y_c1 = cca.fit_transform(X_aligned, Y_aligned)
    X_c2, Z_c1 = cca.fit_transform(X2_aligned, Z_aligned)


    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X_aligned[:,0], X_aligned[:,1], X_aligned[:,2], label='2020')
    ax.plot(Y_aligned[:,0], Y_aligned[:,1], Y_aligned[:,2], label='2021')
    ax.legend()
    plt.title("Procrustes-aligned neural trajectories (2020-2021)")
    plt.show()

    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    ax.plot(X2_aligned[:,0], X2_aligned[:,1], X2_aligned[:,2], label='2020')
    ax.plot(Z_aligned[:,0], Z_aligned[:,1], Z_aligned[:,2], label='2022')
    ax.legend()
    plt.title("Procrustes-aligned neural trajectories (2020-2022)")
    plt.show()


    r_vals_before_XY = [pearsonr(X[:, i], Y[:, i])[0] for i in range(X.shape[1])]
    r_vals_after_XY = [pearsonr(X_c1[:, i], Y_c1[:, i])[0] for i in range(X_c1.shape[1])]

    r_vals_before_XZ = [pearsonr(X[:, i], Z[:, i])[0] for i in range(X.shape[1])]
    r_vals_after_XZ = [pearsonr(X_c2[:, i], Z_c1[:, i])[0] for i in range(X_c2.shape[1])]


    print("Mean r (2020-2021) before:", np.mean(r_vals_before_XY))
    print("Mean r (2020-2021) after:", np.mean(r_vals_after_XY))
    print("Improvement:", np.mean(r_vals_after_XY) - np.mean(r_vals_before_XY))

    print("Mean r (2020-2022) before:", np.mean(r_vals_before_XZ))
    print("Mean r (2020-2022) after:", np.mean(r_vals_after_XZ))
    print("Improvement:", np.mean(r_vals_after_XZ) - np.mean(r_vals_before_XZ))

def plot_one_day(df_tuning, date, data_type = 'sbps'):
    
    dir_list, position_map, direction_key = direction_map(all_directions=False)

    df_time = df_tuning.groupby(df_tuning.index.to_period("D"))
    period = pd.Period(date.replace('_', '-'), freq='D')

    neural_data_for_direction, pca_results = calculate_pca_and_split(df_time, [period], position_map, data_type, jpca=False, kinematic_type = 'vel', pca_all = True)
    
    
    
    
    # for 
    
def histogram_target_styles(df, time_periods, dir_list):
    pass
    for period in time_periods:
        counts = np.zeros(len(dir_list))
        for target_pos in df[period]:
            # print(target_pos)
            if target_pos == 'X':
                continue
            dir = np.argwhere(dir_list == target_pos)
            counts[dir] +=1
        
def centroids_across_time(averaged_pca_results, time_periods, color_dict, label, day_centroids = None, dim_inds = [0, 1, 2], PART_TO_PLOT = 'center', cmap = None, norm = None, radius = 'std'):
           
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
            # elif radius == 'cov':
            #     cov = np.cov(pts, rowvar=False)
            #     eigvals, eigvecs = eig(cov)

            #     # Scale ellipsoid to desired confidence level (chi-square)
            #     chi2_val = chi2.ppf(.5, df=3)
            #     radii = np.sqrt(eigvals * chi2_val)

            #     # Build sphere
            #     u = np.linspace(0, 2 * np.pi, 30)
            #     v = np.linspace(0, np.pi, 30)
            #     x = np.outer(np.cos(u), np.sin(v))
            #     y = np.outer(np.sin(u), np.sin(v))
            #     z = np.outer(np.ones_like(u), np.cos(v))
            #     sphere = np.stack((x, y, z), axis=-1)

            #     # Transform sphere into ellipsoid
            #     for i in range(len(u)):
            #         for j in range(len(v)):
            #             point = sphere[i, j, :]
            #             sphere[i, j, :] = eigvecs @ (point * radii) + centroid

            #     ax.plot_surface(
            #         sphere[:, :, 0], sphere[:, :, 1], sphere[:, :, 2],
            #         color=color, alpha=.07, edgecolor='none', zorder = 1
            #     )
            # else:
                pass
            
            if day_centroids is not None: 
                # print(f"N days in {year}: {len(day_centroids[str(year)])}")
                d_centroids = np.vstack(day_centroids[str(year)])
                
                ax.scatter(d_centroids[:, 0], 
                            d_centroids[:, 1], 
                            d_centroids[:, 2],
                            color=color,
                            s=20,
                            marker='o',
                            edgecolor=None, zorder = 2)
                
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
    
    ax.set_xlim([-5, 10])
    ax.set_zlim([-3, 7])

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
        ticks = [1.5, 5.5, 9.5, 12.5]

        labels = years
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)
    fig.subplots_adjust(left=0.1, right=1.1, top=0.9, bottom=0.1)

    # Better viewing angle
    # if label == "Week":
    #     ax.view_init(elev=40, azim=-45)
    # else: 
    ax.view_init(elev=30, azim=-45)
    
    # plt.tight_layout()
    plt.savefig("pca_centroids_across_time.svg")

    # plt.show()

def point_cloud(averaged_pca_results, time_periods, color_dict, label, dim_inds = [0, 1, 2], PART_TO_PLOT = 'center', cmap = None, norm = None):
           
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
    for year_idx, year in enumerate(time_periods):

        color = color_dict[str(year)]
        colors.append(color)
        
        # Add to legend once per year
        legend_handles.append(plt.Line2D([0], [0], 
                            linestyle='-', 
                            color=color,
                            label=f'{label} {year}'))
        
        # Plot all directions for this year

        pts = []
        for dir in averaged_pca_results[str(year)].keys():
            # if dir == 'X':
            #     continue
            # else:
            data = [x[0] for x in averaged_pca_results[str(year)][dir]]

            # print(a.shape)
            pts.append(np.vstack(data))
        
        # pts = []
        # for direction in sorted(averaged_pca_results[str(year)].keys()):
        #     data = [x[0][0, :] for x in averaged_pca_results[str(year)][direction]]
        #     pts.append(np.vstack(data))
        # pts = np.vstack(pts)                
        # # pts = np.array(pts)
        # print(pts.shape)
        
        # print(color)
        # print(pts.shape)
        if len(pts) > 0:
            pts = np.vstack(pts)
            ax.scatter(xs = pts[:, 0], ys = pts[:, 1], zs = pts[:, 2], alpha = .01, color = color )
    
    ax.set_xlabel(f"PC{dim_inds[0]+1}", labelpad=10)
    ax.set_ylabel(f"PC{dim_inds[1]+1}", labelpad=10)
    ax.set_zlabel(f"PC{dim_inds[2]+1}", labelpad=10)
    ax.set_xlim([-15, 15])
    ax.set_ylim([-5, 10])
    ax.set_zlim([-5, 10])


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
        ticks = np.linspace(0, int(norm.vmax), len(years))
        labels = years
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)
    fig.subplots_adjust(left=0.1, right=1.1, top=0.9, bottom=0.1)

    # Better viewing angle
    # if label == "Week":
    #     ax.view_init(elev=40, azim=-45)
    # else: 
    ax.view_init(elev=20, azim=-45)
    
    plt.tight_layout()
    # plt.savefig("pca_centroids_across_time.svg")

    # plt.show()

def within_time_group_principal_angles(averaged_pca_results, time_periods):
    angles_across_time = []
    for i, year in enumerate(time_periods):
        angles = []
        base_dir = list(sorted(averaged_pca_results[str(year)].keys()))[0]
        base_traj = averaged_pca_results[str(year)][base_dir]
        for direction in sorted(averaged_pca_results[str(year)].keys()):
            if direction == base_dir:
                continue
            if averaged_pca_results[str(year)][direction] is not None:
                trajectory = averaged_pca_results[str(year)][direction]
                angles.append(principal_angle(base_traj, trajectory))
                
        # print(np.array(angles).shape)
        angles_across_time.append(np.array(angles).mean())
            
    plt.figure()
    plt.plot(angles_across_time)

def across_time_group_principal_angles(averaged_pca_results, time_periods):
    angles_across_time = {}
    time_0_trajs = averaged_pca_results[str(time_periods[0])]
    for dir in time_0_trajs.keys():
        angles_across_time[dir] = []
        
    for i, year in enumerate(time_periods[1:]):
        for direction in sorted(averaged_pca_results[str(year)].keys()):
            if averaged_pca_results[str(year)][direction] is not None:
                trajectory = averaged_pca_results[str(year)][direction]
                angles_across_time[direction].append(principal_angle(time_0_trajs[direction], trajectory))
            else:
                print(f"WARNING: no traj for {direction}, {year}")
                angles_across_time[direction].append(None)
                
    colors = {'N': (0.3, 0.0, 0.9), 'NE': (0.15, 0.4,  0.8), # N is Blue
            'E': (0.0, 0.8, 0.7), 'SE': (0.45, 0.8, 0.35), # W is greenish
            'S': (0.9, 0.7, 0.0), 'SW': (0.9, 0.35, 0.1),  # S is Yellowish
            'W': (0.9, 0.0, 0.2), 'NW': (0.65, 0.0,  0.55)}  #E is Red
    
    plt.figure()
    for dir in angles_across_time.keys():
        plt.plot(angles_across_time[dir], color = colors[dir])  
        
    data = np.array(list(angles_across_time.values()))  

    # Average across the keys (i.e., along axis 0)
    average_list = data.mean(axis=0) 
    plt.figure()   
    plt.plot(average_list) 

def principal_angle(traj_1, traj_2):

    traj_1 = traj_1 - traj_1.mean(axis=0)
    traj_2 = traj_2 - traj_2.mean(axis=0)

    # Orthonormal bases using SVD (or use QR for speed)
    U_i, _, _ = np.linalg.svd(traj_1, full_matrices=False)
    U_j, _, _ = np.linalg.svd(traj_2, full_matrices=False)

    # Compute principal angles in radians
    angles_rad = subspace_angles(U_i, U_j)

    # Convert to degrees
    angles_deg = np.array(np.degrees(angles_rad))
    # print(angles_deg.shape)

    return angles_deg.max()




## PCA and data processing 
def calculate_pca_and_split(df_time, time_periods, position_map, type_of_data, jpca=False, kinematic_type = 'vel', pca_all = True, normalize = True):
    '''
    Calculates PCA on all data and splits the trials up using that data

        Params: df_time: dataframe that contains all the relevant data
                time_periods: list containing years we want to look at
                position_map: hash_map that maps target direction coordinates to the compass direction, so that its possible to group by target direction
                type_of_data: str that specifies SBPs vs TCFR
                jpca: flag indicating whether to do JPCA or PCA (not done yet)
        
        Returns: neural_data_for_direction: hashmap containing grouped neural PCA data; indexed by year, and target_pos
                 pca_results: hash_map containing pca results, indexed by each year
    '''

    # print(time_periods)
    # Now we will begin to Calculate neural trajectories with PCA 
    neural_data_for_direction = {str(key): {} for key in time_periods}

    #neural_data_for_direction = {period: {} for period in time_periods}

    # initialize PCA storage for ALL data
    n_components = 3
    pca_results = {year: [] for year in neural_data_for_direction.keys()}
    #pca_results = {period: [] for period in time_periods}


    #Collect all sbps data across years and trials (that are CO)

    print(f"pca all: {pca_all}")
    if pca_all:
        all_sbps = []
        for year in time_periods:
            year_group = df_time.get_group(year)
            for _, yearly_data in year_group.iterrows():
                if yearly_data["target_styles"] == "CO":
                    all_sbps.append(yearly_data[type_of_data])
 
        global_data = np.vstack(all_sbps)  # Shape: (total_trials*timepoints, n_channels)
        if normalize:
            scaler = StandardScaler()
            global_data = scaler.fit_transform(global_data) 
        # print(global_data.mean(axis = 0))

        n_components = 3
        global_pca = PCA(n_components=n_components).fit(global_data)
        pcas = [global_pca for x in time_periods]

    else:
        raise ValueError
        pcas = []

        for year in time_periods:
            all_sbps = []
            year_group = df_time.get_group(year)
            for _, yearly_data in year_group.iterrows():
                if yearly_data["target_styles"] == "CO":
                    all_sbps.append(yearly_data[type_of_data])
 
            global_data = np.vstack(all_sbps)  # Shape: (total_trials*timepoints, n_channels)
            if normalize: 
                
                scaler = StandardScaler()
                # print(global_data.mean(axis = 0))
                global_data = scaler.fit_transform(global_data) 
            # print(global_data.mean(axis = 0))

            n_components = 3         
            year_pca = PCA(n_components=n_components).fit(global_data)
            pcas.append(year_pca)

    # Fit global PCA
    if not jpca:
        n_components = 3
        global_pca = PCA(n_components=n_components).fit(global_data)
        # pcas = [global_pca for x in time_periods]

    else:
        # TODO: JPCA stuff
        # jpca_trials = [trial_array for trial_array in processed_neural_data_for_direction[year][direction]]


        # jpca = jPCA.JPCA(num_jpcs=2)
        # times = np.linspace(-150, 600, 37)

        # (direction_pca_results, 
        # full_data_var,
        # pca_var_capt,
        # jpca_var_capt) = jpca.fit(jpca_trials, times=list(times), tstart=-150, tend=600)
        # plot_projections(direction_pca_results)
        pass
    
    
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

            data = yearly_data[type_of_data]
            if normalize:
                # scaler = StandardScaler()
                data = scaler.transform(data) 
            
            day_pca_results = pcas[j].transform(data)
            pca_results[str(year)].append(day_pca_results.copy())

            if yearly_data["target_styles"] != "CO": # only include it if its CO data
                continue

            for i in range(0, len(yearly_data['target_positions'])):
                trial_index_start = yearly_data["trial_indices"][i]
                trial_length = yearly_data['trial_counts'][i]
                channel_pca_sbps = day_pca_results[trial_index_start:trial_index_start+trial_length] 


                target_pos_coords = tuple(yearly_data['target_positions'][i])
                target_pos_coords = (round(float(target_pos_coords[0]), 1), round(float(target_pos_coords[1]), 1))
                if target_pos_coords in position_map: # skips (0.5, 0.5)

                    target_pos = position_map[target_pos_coords]

                    # if (target_pos != "N"):
                    #     continue
                    if (target_pos == "N") or (target_pos == "S"):
                        trial_kinematics = yearly_data['finger_kinematics'][:, MRP_ind][trial_index_start:trial_index_start+trial_length] # finger kinematics is [index_position, MRP_position, index_velocity, MRP_velocity]. We are indexing the MRP velocity here (N,S)
                    else:
                        trial_kinematics = yearly_data['finger_kinematics'][:, index_ind][trial_index_start:trial_index_start+trial_length] # finger kinematics is [index_position, MRP_position, index_velocity, MRP_velocity]. We are indexing the Index velocity here (E,W, or default to index in all other directions)
                        
                    if target_pos not in neural_data_for_direction[str(year)]:
                        neural_data_for_direction[str(year)][target_pos] = [(channel_pca_sbps, trial_kinematics)]
                    else:
                        neural_data_for_direction[str(year)][target_pos].append((channel_pca_sbps, trial_kinematics))


    
    return neural_data_for_direction, pca_results

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

            for i in range(0, len(yearly_data['target_positions'])):
                trial_index_start = yearly_data["trial_indices"][i]
                trial_length = yearly_data['trial_counts'][i]

                target_pos_coords = tuple(yearly_data['target_positions'][i])
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

def pca_trial_data(data_dict, time_periods, pca_all, average_before_pca = True, sigma = 0):
    data_to_pca = {}
    for year in data_dict: 
        data_to_pca[year] = {}       
        for target_pos in data_dict[year]:
            if target_pos == 'X':
                continue
            if average_before_pca:
                data_to_pca[year][target_pos] = [np.array(data_dict[year][target_pos]).mean(axis = 0)] 
            else:
                data_to_pca[year][target_pos] = data_dict[year][target_pos]               
            
            
    # print(len(data_to_pca[year][target_pos]))
    # print(data_to_pca[year][target_pos][0].shape)

    if pca_all:
        all_sbps = []
        for year in data_to_pca.keys():
            for target_pos in data_to_pca[year].keys():
                [all_sbps.append(x) for x in data_to_pca[year][target_pos]]
 
        global_data = np.vstack(all_sbps)  # Shape: (total_trials*timepoints, n_channels)
        scaler = StandardScaler()
        global_data = scaler.fit_transform(global_data) 

        n_components = 3
        global_pca = PCA(n_components=n_components).fit(global_data)
        pcas = [global_pca for x in time_periods] 
    else:
        pcas = []
        for year in data_to_pca.keys():
            all_sbps = []
            for target_pos in data_to_pca[year].keys():
                [all_sbps.append(x) for x in data_to_pca[year][target_pos]]
 
            global_data = np.vstack(all_sbps)  # Shape: (total_trials*timepoints, n_channels)
            scaler = StandardScaler()
            global_data = scaler.fit_transform(global_data) 
            n_components = 3         
            year_pca = PCA(n_components=n_components).fit(global_data)
            pcas.append(year_pca)
        
    assert(len(pcas) == len(time_periods))
    # print(pcas[0] == pcas[1] == pcas[2])
    pca_data = {}
    for i, year in enumerate(data_to_pca.keys()): 
        pca_data[year] = {}       
        for target_pos in data_to_pca[year].keys():
            if average_before_pca:
                scaler = StandardScaler()
                data = scaler.fit_transform(data_to_pca[year][target_pos][0]) 
                traj = pcas[i].transform(data)
            else:
                pca_trials = []
                for x in data_to_pca[year][target_pos]:
                    scaler = StandardScaler()
                    data = scaler.fit_transform(x) 
                    pca_trials.append(pcas[i].transform(data))

                traj = np.array(pca_trials).mean(axis = 0)
            if sigma != 0:
                traj = gaussian_filter1d(traj, sigma)
                # print('smoothing')
            pca_data[year][target_pos] = traj
    # print(data_to_pca[year][target_pos][0].shape)

    return pca_data
      
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

            for i in range(0, len(yearly_data['target_positions'])):
                trial_index_start = yearly_data["trial_indices"][i]
                trial_length = yearly_data['trial_counts'][i]
                channel_pca_sbps = day_pca_results[trial_index_start:trial_index_start+trial_length] 

                target_pos_coords = tuple(yearly_data['target_positions'][i])
                target_pos_coords = (round(float(target_pos_coords[0]), 1), round(float(target_pos_coords[1]), 1))
                
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
    
# def get_targ(targ_pos, last, boundary = 45):
#     direction_vectors = {
#         0: [1, 0],
#         1: [1, 1],
#         2: [0, 1],
#         3: [-1, 1],
#         4: [-1, 0],
#         5: [-1, -1],
#         6: [0, -1],
#         7: [1, -1]
#     }
    
#     angles = {
#         0: 0,
#         1: 45,
#         2: 90,
#         3: 135,
#         4: 180,
#         5: 225,
#         6: 270,
#         7: 315,
#     }

#     diff = np.array(targ_pos) - np.array(last)
#     if np.allclose(diff, 0):
#         return np.array([0, 0])

#     angle = np.degrees(np.arctan2(diff[1], diff[0])) % 360

#     for direction, ref_angle in angles.items():
#         lower = (ref_angle - boundary/2) % 360
#         upper = (ref_angle + boundary/2) % 360

#         if lower < upper:
#             if lower <= angle <= upper:
#                 return np.array(direction_vectors[direction])
#         else:
#             # Wraparound case (e.g., 350° to 10°)
#             if angle >= lower or angle <= upper:
#                 return np.array(direction_vectors[direction])

#     return np.array([0, 0])
#     diff = np.array(targ_pos) - np.array(last)
#     if (diff == 0).all():
#         return np.array([0, 0])

#     # Compute angle in radians from x-axis
#     angle = np.arctan2(diff[1], diff[0])  # Returns [-π, π]

#     # Convert to degrees and shift to [0, 360)
#     angle_deg = (np.degrees(angle) + 360) % 360

#     # Bin into 8 directions (0 to 7), each 45°
#     # 0 = 0° to 45°, 1 = 45° to 90°, ..., 7 = 315° to 360°
#     direction = int(np.floor((angle_deg + 22.5) / 45)) % 8
    
#     # print(f"diff: {diff}, targ: {targ}")

#     return np.array(direction_vectors[direction])

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
def pca_of_reach_directions(df_tuning, type_of_data, group_by = "year", smoothing=False, display_alignment = False, directions = 'not_all', trim_method = trim_neural_data_at_movement_onset_std_and_smooth, trim_pt = movement_onset, years_to_skip = [2023, 2024], dim_inds = [0, 1, 2], pca_all = True, ALL_DIRS_ACROSS_GROUPS=False, ALL_DIRS_ALL_GROUPS_ONE_PLOT=True, ALL_DIRS_GROUPED_BY_GROUP=False, COMBINED_PLOT = False, CENTROIDS_ACROSS_TIME = False, normalize = True):

    # define directions
    dir_list, position_map, direction_key = direction_map(directions=directions)
    # year_color_dict = {'2020': 'red', '2021': 'green', '2022': 'blue' , '2023': "orange" }
    cmap = get_cmap('plasma_r')

    ## Remove 2024 and 2023 from data
    for year in years_to_skip:
        df_tuning = df_tuning[~df_tuning.index.astype(str).str.startswith(str(year))]
        
    df_time, time_periods, label = get_grouped_data(df_tuning, group_by)

    norm = Normalize(vmin=0, vmax=len(time_periods)+1)
    colors = [cmap(norm(j+1)) for j in range(len(time_periods))]
    color_dict = {}

    for i, time_period in enumerate(time_periods): 

        color_dict[str(time_period)] = colors[i]
    
    # calculates PCA on all data and split up the trials using that data
    neural_data_for_direction, pca_results = calculate_pca_and_split(df_time=df_time, time_periods=time_periods, position_map=position_map, type_of_data=type_of_data, jpca=False, kinematic_type='vel', pca_all=pca_all, normalize=normalize)
    pos_data, _ = calculate_pca_and_split(df_time=df_time, time_periods=time_periods, position_map=position_map, type_of_data=type_of_data, jpca=False, kinematic_type='pos', pca_all=pca_all)

    print("Calculated PCA")

    # Trim and Smooth data (note that data is binned in 20 ms increments, so for example 40 ms smoothing is sigma=2)
    chosen_sigma = 0.1 if not smoothing else 1
    # histogram_target_styles(neural_data_for_direction, time_periods, dir_list)
    processed_neural_data_for_direction, kinematics_data = trim_method(neural_data_for_direction, std_multiplier=2, sigma=chosen_sigma, display_alignment=display_alignment, trim_pt = trim_pt, direction_key = direction_key, position_data = pos_data)

    # plot_kin(kinematics_data)
    print("Trimmed Data")


    # average the PCA data across trials
    averaged_pca_results, averaged_kin_results = average_trial_PCA_data(dir_list=dir_list, kinematics=kinematics_data, processed_neural_data_for_direction=processed_neural_data_for_direction)

    print("Averaged PCA")
    
    # visualize PCA trajectories
    visualize_trajectories(averaged_pca_results=averaged_pca_results, time_periods=time_periods, direction_key = direction_key, color_dict=color_dict, label = label, dim_inds = dim_inds, ALL_DIRS_ACROSS_YEARS=ALL_DIRS_ACROSS_GROUPS, ALL_DIRS_ALL_YEARS_ONE_PLOT=ALL_DIRS_ALL_GROUPS_ONE_PLOT, ALL_DIRS_GROUPED_BY_YEAR=ALL_DIRS_GROUPED_BY_GROUP, COMBINED_PLOT=COMBINED_PLOT)
    # visualize_monthly_trajectories(averaged_pca_results=averaged_pca_results, time_periods=time_periods)
    if CENTROIDS_ACROSS_TIME:
        centroids_across_time(averaged_pca_results=averaged_pca_results, time_periods=time_periods, color_dict=color_dict, label = label, dim_inds = dim_inds)

def averaged_pca_of_reach_directions(df_tuning, type_of_data, group_by = "year", smoothing=False, display_alignment = False, directions = 'not_all', trim_method = trim_neural_data_at_movement_onset_std_and_smooth, trim_pt = movement_onset, years_to_skip = [2023, 2024], dim_inds = [0, 1, 2], pca_all = True, average_before_pca = True, ALL_DIRS_ACROSS_GROUPS=False, ALL_DIRS_ALL_GROUPS_ONE_PLOT=False, ALL_DIRS_GROUPED_BY_GROUP=False, COMBINED_PLOT = False, CENTROIDS_ACROSS_TIME = False):

    # define directions
    dir_list, position_map, direction_key = direction_map(directions=directions)

    for year in years_to_skip:
        df_tuning = df_tuning[~df_tuning.index.astype(str).str.startswith(str(year))]
        
    df_time, time_periods, label = get_grouped_data(df_tuning, group_by)

    cmap = get_cmap('plasma_r')
    norm = Normalize(vmin=0, vmax=len(time_periods)+1)
    colors = [cmap(norm(j+1)) for j in range(len(time_periods))]
    color_dict = {}

    for i, time_period in enumerate(time_periods): 

        color_dict[str(time_period)] = colors[i]

    # calculates PCA on all data and split up the trials using that data
    neural_data_for_direction, _ = split(df_time=df_time, time_periods=time_periods, position_map=position_map, type_of_data=type_of_data, jpca=False, kinematic_type='vel')
    pos_data, _ = split(df_time=df_time, time_periods=time_periods, position_map=position_map, type_of_data=type_of_data, jpca=False, kinematic_type='pos')

    print("Calculated PCA")

    # Trim and Smooth data (note that data is binned in 20 ms increments, so for example 40 ms smoothing is sigma=2)
    chosen_sigma = 0 if not smoothing else .6
    # histogram_target_styles(neural_data_for_direction, time_periods, dir_list)
    processed_neural_data_for_direction, kinematics_data = trim_method(neural_data_for_direction, std_multiplier=2, sigma=0, display_alignment=display_alignment, trim_pt = trim_pt, direction_key = direction_key, position_data = pos_data)

    print("Trimmed Data")

    averaged_pca_results = pca_trial_data(processed_neural_data_for_direction, time_periods, pca_all = pca_all, average_before_pca = average_before_pca, sigma = chosen_sigma)
    # plot_kin(kinematics_data)

    print("Averaged PCA")

    # visualize PCA trajectories
    visualize_trajectories(averaged_pca_results=averaged_pca_results, time_periods=time_periods, direction_key = direction_key, color_dict=color_dict, label = label, dim_inds = dim_inds, ALL_DIRS_ACROSS_YEARS=ALL_DIRS_ACROSS_GROUPS, ALL_DIRS_ALL_YEARS_ONE_PLOT=ALL_DIRS_ALL_GROUPS_ONE_PLOT, ALL_DIRS_GROUPED_BY_YEAR=ALL_DIRS_GROUPED_BY_GROUP, COMBINED_PLOT=COMBINED_PLOT)
    # visualize_monthly_trajectories(averaged_pca_results=averaged_pca_results, time_periods=time_periods)
    if CENTROIDS_ACROSS_TIME:
        centroids_across_time(averaged_pca_results=averaged_pca_results, time_periods=time_periods, color_dict=color_dict, label = label, dim_inds = dim_inds)

def plot_pca_all(df_tuning, n_components = 3, dpca = False, group_by = "year", years_to_skip = [], data_type = 'sbps', remove_RT = False, directions = 'ext_flex', trim_pt = max_jerk, trim_method = trim_neural_data_at_movement_onset_std_and_smooth, PART_TO_PLOT = 'center', normalization_method = 'all', pca_all = True, plot_centr_across_time= True, sigma = .5):
    
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
        centroids_across_time(averaged_pca_results=neural_data_for_direction, time_periods=time_periods, color_dict=color_dict, label = label, PART_TO_PLOT=PART_TO_PLOT, cmap = cmap, norm = norm, radius = 'std', day_centroids = day_centroids)

def plot_alignment_across_time(df_tuning, n_components = 3, dpca = False, group_by = "year", years_to_skip = [2023, 2024], data_type = 'sbps', remove_RT = False, directions = 'ext_flex', trim_pt = max_jerk, trim_method = trim_neural_data_at_movement_onset_std_and_smooth, PART_TO_PLOT = 'center', normalize = True):
     
    dir_list, position_map, direction_key = direction_map(directions=directions)

    for year in years_to_skip:
        df_tuning = df_tuning[~df_tuning.index.astype(str).str.startswith(str(year))]
        
    if remove_RT:
        df_tuning = df_tuning[df_tuning['target_styles'] != 'RD']
    else:
        if directions != 'ext_flex':
            raise ValueError
        
    if directions == 'ext_flex':
        # display(df_tuning)
        df_tuning = get_all_trial_classes(df_tuning)
        # display(df_tuning)
        
    sbps = np.vstack(df_tuning[data_type].to_list())
    if normalize: 
        scaler = StandardScaler()
        sbps = scaler.fit_transform(sbps) 
    else:
        scaler = None
    pca = PCA(n_components=n_components)
    pca.fit(sbps)
    
    df_time, time_periods, label = get_grouped_data(df_tuning, group_by)
    
    cmap = get_cmap('plasma_r')
    norm = Normalize(vmin=0, vmax=len(time_periods)+1)
    colors = [cmap(norm(j+1)) for j in range(len(time_periods))]
    color_dict = {}
    for i, time_period in enumerate(time_periods): 

        color_dict[str(time_period)] = colors[i]
        
    
    neural_data_for_direction, _ = split_and_pca_all_trials(df_time=df_time, time_periods=time_periods, type_of_data=data_type, pca = pca, scaler = scaler, kinematic_type='vel', position_map = position_map, normalize = normalize)
    # pos_data, _ = split_and_pca_all_trials(df_time=df_time, time_periods=time_periods, type_of_data=data_type, pca = pca, kinematic_type='pos')

    processed_neural_data_for_direction, kinematics_data = trim_method(neural_data_for_direction, std_multiplier=2, sigma=0, display_alignment=False, trim_pt = trim_pt, direction_key = direction_key)

    # plot_kin(kinematics_data)
    print("Trimmed Data")


    # average the PCA data across trials
    averaged_pca_results, averaged_kin_results = average_trial_PCA_data(dir_list=dir_list, kinematics=kinematics_data, processed_neural_data_for_direction=processed_neural_data_for_direction)

    
    return
    
def plot_avg_trajectories(df_tuning, type_of_data, group_by = "year", display_alignment = False, directions = 'not_all', trim_method = trim_neural_data_at_movement_onset_std_and_smooth, trim_pt = movement_onset, years_to_skip = [], sigma = 0, remove_RT = False):
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

    plot_trajectories(averaged_pca_results=averaged_pca_results, time_periods=time_periods, direction_key = direction_key, label = label, elev = 15, azim = -45)


def plot_principal_angles(df_tuning, type_of_data, group_by = "year", display_alignment = False, directions = 'ext_flex', trim_method = trim_neural_data_at_movement_onset_std_and_smooth, trim_pt = movement_onset, years_to_skip = [], sigma = 0, remove_RT = False, normalization_method = 'day', pca_all = True):
    #  print(directions)
    dir_list, position_map, direction_key = direction_map(directions=directions)

    for year in years_to_skip:
        df_tuning = df_tuning[~df_tuning.index.astype(str).str.startswith(str(year))]

    if remove_RT:
        df_tuning = df_tuning[df_tuning['target_styles'] != 'RD']
        
    if directions == 'ext_flex':
        # display(df_tuning)
        df_tuning = get_all_trial_classes(df_tuning)

    df_tuning = normalize_data(df_tuning, data_type=type_of_data, normalization_method=normalization_method, pca_by_day= not pca_all)
    
    # display(df_tuning)
    df_time, time_periods, label = get_grouped_data(df_tuning, group_by, years_to_skip = [])
    # print(time_periods)
    cmap = get_cmap('plasma_r')
    norm = Normalize(vmin=0, vmax=len(time_periods)+1)
    colors = [cmap(norm(j+1)) for j in range(len(time_periods))]
    color_dict = {}
    for i, time_period in enumerate(time_periods): 

        color_dict[str(time_period)] = colors[i]
        
    if pca_all:
        neural_data_for_direction, _ = split_and_pca_all_trials(df_time=df_time, time_periods=time_periods, type_of_data=type_of_data, kinematic_type='vel', position_map = position_map, pca_all = pca_all)
    else:
        neural_data_for_direction, _ = split(df_time=df_time, time_periods=time_periods, type_of_data=type_of_data, kinematic_type='vel', position_map = position_map, only_CO=remove_RT)

    processed_neural_data_for_direction, kinematics_data = trim_method(neural_data_for_direction, std_multiplier=2, sigma=sigma, display_alignment=display_alignment, trim_pt = trim_pt, direction_key = direction_key)

    averaged_pca_results, averaged_kin_results = average_trial_PCA_data(dir_list=dir_list, kinematics=kinematics_data, processed_neural_data_for_direction=processed_neural_data_for_direction)

    within_time_group_principal_angles(averaged_pca_results, time_periods)
    
    across_time_group_principal_angles(averaged_pca_results, time_periods)

    

# def trim_neural_data_at_movement_onset_std_align_and_smooth(data_dict, std_multiplier=2, sigma=5, display_alignment=False, trim_pt = movement_onset, direction_key = None, position_data = None):
#     """
#     Trims neural data using standard deviations above baseline mean, then smooths using Gaussian kernel
    
#     Parameters:
#     - data_dict: Dictionary containing neural data organized by year, and target position
#     - std_multiplier: Number of standard deviations above baseline mean (default: 2)
#     - sigma: Standard deviation for Gaussian kernel (in units of samples)
#     - display_alignment: flag that specifies whether to visualize kinematic alignment
    
#     Returns:
#     - Dictionary with trimmed and smoothed neural data maintaining the same structure
#     """
#     trimmed_data = {}
#     kin_data = {}

#     trim_pt = movement_onset
#     for year in data_dict:
#         trimmed_data[year] = {}
#         kin_data[year] = {}
        
#         neurals = []
#         m_onsets = [[] for k in data_dict[year].keys()]
#         for i, target_pos in enumerate(data_dict[year]):

#             for j, (trial_data, kinematics) in enumerate(data_dict[year][target_pos]):
                
#                 ind, _ = trim_pt(trial_data, kinematics, std_multiplier) 
#                   # Trim the data to start from movement onset and smooth it 
#                 start = ind - 3 # want 150 ms pre movement and 20 ms bins 
#                 end = ind+3  # want 600 ms post movement and 20 ms bins 
#                 # start = movement_onset_idx # want 150 ms pre movement and 20 ms bins 
#                 # post_movement_end = len(kinematics)-1 # want 600 ms post movement and 20 ms bins 

#                 if (start < 0) or (end < 2) or (end > trial_data.shape[0]-1): # (drop trials where movement onset seems too close to start or too close to end)
#                     m_onsets[i].append(-1)
#                     continue
#                 else:
#                     m_onsets[i].append(ind)
#                     neurals.append(gaussian_filter1d(trial_data[start:end, :], sigma=sigma))
                
#         center = get_closest_points(neurals, neural_dist, True)
#         for i, target_pos in enumerate(data_dict[year]):
#             if display_alignment:
#                 fig, ax = plt.subplots(figsize=(8, 4))

#             for j, (trial_data, kinematics) in enumerate(data_dict[year][target_pos]):
#                 if m_onsets[i][j] == -1:
#                     continue
#                 else: 
#                     ind = m_onsets[i][j]
#                     smoothed_neural = gaussian_filter1d(trial_data, sigma=sigma)
#                     distances = np.array([neural_dist(center, p) for p in smoothed_neural[ind-3:ind, :]])
#                     start = np.argmin(distances) + ind-3
#                     # start = start +5
#                     # print(f"{ind}, {start}")
#                     end = start + (400 // 20)
            
#                     if start < 0: # (drop trials where movement onset seems too close to start or too close to end)
#                         continue
#                     if end > trial_data.shape[0]-1:
#                         continue

#                     trimmed_neural = smoothed_neural[start:end, :]
#                     trimmed_kin = kinematics[start:end]
                    
#                     if target_pos not in trimmed_data[year]:
#                         trimmed_data[year][target_pos] = [trimmed_neural]
#                         kin_data[year][target_pos] =  [trimmed_kin]
#                     else:
#                         trimmed_data[year][target_pos].append(trimmed_neural)
#                         kin_data[year][target_pos].append(trimmed_kin)
                    
#                     if display_alignment:
#                         time = np.arange(len(trimmed_kin))
#                         ax.plot(time, trimmed_kin, alpha = .8)        

#             if display_alignment:
#                 ax.set_xlabel("Bins")
#                 ax.set_ylabel("Value")
#                 ax.set_title(f"Kinematics Over Time for Target {target_pos} (Year {year})")
#                 ax.grid(True)
#                 # plt.plot(np.array(traces).mean(axis = 0), color = 'black')
#                 plt.show()

             
        
    
#     return trimmed_data,  kin_data
