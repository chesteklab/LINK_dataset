import pandas as pd
import os
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
import pickle
import glob
import tqdm
import pdb
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl
#some basic text parameters for figures
mpl.rcParams['font.family'] = "Atkinson Hyperlegible" # if installed but not showing up, rebuild mpl cache
mpl.rcParams['font.size'] = 10
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlelocation'] = 'center'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['figure.constrained_layout.use'] = True
mpl.rcParams['figure.titlesize'] = 14
mpl.rcParams['figure.titleweight'] = 'bold'
mpl.rcParams['pdf.fonttype'] = 42

data_path = "Z:\Student Folders\\Nina_Gill\data\only_good_days"
output_path = 'Z:\Student Folders\Hisham_Temmar\\big_dataset\output\single_channel_tuning'
binsize = 20
# (for debugging stage) please dont remove these, comment them out would be fine. otherwise ill have to write the directories again which is annoying because windows hates single \
# data_path = "C:\\Files\\UM\\ND\\SFN\\only_good_days"
# output_path = 'C:\\Files\\UM\\ND\\github\\big_nhp_dataset_code\\outputs'

def compute_channel_tuning(neural, full_behavior, velocity_tuning = False):

    if velocity_tuning:
        behavior = full_behavior[:,[2,3]]
    else:
        behavior = full_behavior[:,[0,1]]

    channel_tunings = np.zeros((neural.shape[1], behavior.shape[1]))
    for channel in range(neural.shape[1]):
        single_channel = neural[:, channel]

        eps = 1e-8
        std = np.std(single_channel)

        if std < eps:
            single_channel = single_channel - np.mean(single_channel)
        else:
            single_channel = (single_channel - np.mean(single_channel)) / std

        # calculate tuning to each dof at a variety of lags, pick the one which gives the largest l2 norm across the 2 dof
        lag_range = 11 # look up to 10 bins of lag
        tuning_vec = np.zeros((lag_range, behavior.shape[1]))
        for lag in range(lag_range):
            if lag != 0:
                lagged_channel = single_channel[:-lag]
                lagged_behavior = behavior[lag:,:]
            else:
                lagged_channel = single_channel
                lagged_behavior = behavior
            # calculate tuning for each dof
            lm= LinearRegression(fit_intercept=True)
            lm.fit(lagged_channel.reshape(-1,1), lagged_behavior)
            tuning_vec[lag, :] = lm.coef_[:,0]
        
        best_lag = np.argmax(np.linalg.norm(tuning_vec, 2, axis=1))
        channel_tunings[channel, :] = tuning_vec[best_lag, :]

    # calculate magnitudes and angles
    magnitudes = np.linalg.norm(channel_tunings, 2, 1)
    angles = np.degrees(np.arctan2(channel_tunings[:,1], channel_tunings[:,0]))
    
    channels = np.arange(neural.shape[1])
    tuning_df = pd.DataFrame(np.concatenate((channels.reshape(-1,1), channel_tunings, magnitudes.reshape(-1,1), angles.reshape(-1,1)), axis=1), columns=('channel', 'idx', 'mrs', 'magnitude', 'angle'))

    return tuning_df

def compute_tuning_data(save_dir = None):
    # Path to the folder containing pkl files
    data_folder = data_path

    # check if the directories exists
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Directory {data_folder} does not exist.")
    # if is_save and save_dir is None:
    #     raise ValueError("save_dir must be provided if is_save is True.")
    # elif is_save and not os.path.exists(save_dir):
    #     print(f"Directory {save_dir} does not exist. Creating.")
    #     os.makedirs(save_dir)

    # Get list of pkl files
    if not os.path.join(data_folder, '*.pkl'):
        raise ValueError(f"No pkl files found in {data_folder}.")

    pkl_files = sorted(glob.glob(os.path.join(data_folder, '*.pkl')))

    # Dictionary to store results
    df_list = []

    # Process each pkl file
    for file in tqdm.tqdm(pkl_files, desc=f"Processing files in {data_folder}"):
        # Extract date from filename (assuming format like 'YYYY-MM-DD_data.pkl')
        date = pd.to_datetime(os.path.basename(file).split('_')[0])

        # Load data
        with open(file, 'rb') as f:
            data_CO, data_RD = pickle.load(f)
        
        if data_CO and data_RD:
            sbp = np.concatenate((data_CO['sbp'], data_RD['sbp']), axis=0)
            beh = np.concatenate((data_CO['finger_kinematics'], data_RD['finger_kinematics']), axis=0)
            tcr = np.concatenate((data_CO['tcfr'], data_RD['tcfr']), axis=0)
        elif data_RD:
            sbp = data_RD['sbp']
            beh = data_RD['finger_kinematics']
            tcr = data_RD['tcfr']
        else:
            sbp = data_CO['sbp']
            beh = data_CO['finger_kinematics']
            tcr = data_CO['tcfr']
        
        sbp = sbp * 0.25 # convert to uV
        tcr = tcr * 1000 / binsize # convert from # spikes / bin to # spikes / second
        # Compute channel tuning
        channel_tuning = compute_channel_tuning(sbp, beh, velocity_tuning=False)
        # also save the average TCR
        channel_tuning['avg_tcr'] = np.mean(tcr, axis=0)
        channel_tuning['date'] = date
        df_list.append(channel_tuning)
    
    df = pd.concat(df_list)
    df['channel'] = df['channel'].astype(int)
    df.loc[df['channel'] < 32, 'bank'] = 'A'
    df.loc[(df['channel'] >= 32) &  (df['channel'] < 64), 'bank'] = 'B'
    df.loc[df['channel'] >= 64, 'bank'] = 'C'
    # Save the DataFrame to a CSV file
    df.to_csv(save_dir, index=False)
    print(f"Data saved to {save_dir}")
    return df

def load_tuning_data(dir, overwrite=False):
    if not os.path.exists(dir) or overwrite:
        print(f"Directory {dir} does not exist or overwrite, computing tuning data")
        df = compute_tuning_data(save_dir=dir)
    else:
        df = pd.read_csv(dir)
    df['date'] = pd.to_datetime(df['date'])
    return df

def wrap(x):
    return (x + np.pi) % (2.0*np.pi) - np.pi

def circular_median_rad(a, tol=1e-12):
    a = np.asarray(a, dtype=float)
    diff = wrap(a[:, None] - a)
    loss = np.sum(np.abs(diff), axis=1)
    med = a[np.argmin(loss)]
    ties = a[np.abs(loss - loss.min()) < tol]
    return med, ties

def circular_quantile_rad_signed(angles, probs):
    a = angles
    p = probs

    m, _ = circular_median_rad(a)

    tx = wrap(a - m)

    lin_q = np.quantile(tx, p, method="linear")

    return wrap(lin_q + m)

def calc_medians_iqrs(tuning_df):
    tuning_df_copy = tuning_df.copy()
    clower_quartiles = []
    cupper_quartiles = []
    cmedians = []

    lower_quartiles = []
    upper_quartiles = []
    medians = []
    for channel, group in tuning_df.groupby('channel'):
        angles = np.radians(group['angle'].values)
        mags = group['magnitude'].values

        lower_quartile, upper_quartile = circular_quantile_rad_signed(angles, [0.25, 0.75])
        clower_quartiles.append(np.degrees(lower_quartile))
        cupper_quartiles.append(np.degrees(upper_quartile))
        median, _ = circular_median_rad(angles)
        cmedians.append(np.degrees(median))

        lower_quartiles.append(group['magnitude'].quantile(0.25))
        upper_quartiles.append(group['magnitude'].quantile(0.75))
        medians.append(group['magnitude'].median())
        
    quartiles_df = pd.DataFrame({
        'channel': tuning_df['channel'].unique(),
        'ang_lower_quartile': clower_quartiles,
        'ang_upper_quartile': cupper_quartiles,
        'ang_median': cmedians,
        'mag_lower_quartile':lower_quartiles,
        'mag_upper_quartile':upper_quartiles,
        'mag_median':medians
    })
    return quartiles_df

def calc_tuning_iqrs(tuning_df):
    tuning_df_copy = tuning_df.copy()
    lower_q = []
    upper_q = []
def calc_tuning_avgs(tuning_df):
    tuning_df_copy = tuning_df.copy()
    mag_avg = tuning_df_copy.groupby('channel')['magnitude'].agg(('mean','std'))
    mag_avg = mag_avg.rename(columns={'mean':'mag_mean','std':'mag_std'}) 

    ang_avg = tuning_df_copy.groupby('channel')['angle'].agg((lambda x: np.degrees(stats.circmean(np.radians(x), high=np.pi, low=-1*np.pi)), 
                                                         lambda x: np.degrees(stats.circstd(np.radians(x), high=np.pi, low=-1*np.pi))))
    ang_avg = ang_avg.rename(columns={'<lambda_0>':'ang_mean', '<lambda_1>':'ang_std'})

    tuning_avgs = pd.concat((mag_avg, ang_avg), axis=1)
    tuning_avgs['bank'] = 'A'
    tuning_avgs.loc[32:64,'bank'] = 'B'
    tuning_avgs.loc[64:,'bank'] = 'C'
    return tuning_avgs

def desaturate_hsv(colormap = 'hsv', s = 0.75):
    # 1 ) grab the built‑in cyclic map (256 samples give a smooth gradient)
    base = plt.cm.get_cmap(colormap, 256)

    # 2 ) split RGB and α
    rgba = base(np.linspace(0, 1, 256))
    rgb  = rgba[:, :3]                 # (N,3)
    alpha = rgba[:, 3]                 # keep original transparency

    # 3 ) RGB → HSV, desaturate, HSV → RGB
    hsv  = colors.rgb_to_hsv(rgb)
    hsv[:, 1] *= s                 # ← 0 = grey‑scale … 1 = original saturation
    desat_rgb = colors.hsv_to_rgb(hsv)

    # 4 ) re‑attach α and build the new map
    desat_rgba = np.column_stack([desat_rgb, alpha])
    cmap_hsv_desat = colors.ListedColormap(desat_rgba, name='hsv_desat')
    cmap_hsv_desat.set_bad('black')    # NaNs will plot in black
    return cmap_hsv_desat



def compute_channel_mutual_information(data):
    # Check if data is a tuple and extract the dictionary if necessary
    if isinstance(data, tuple):
        if data[0] is None and isinstance(data[1], dict):
            data = data[1]
        elif isinstance(data[0], dict):
            data = data[0]
        else:
            print(f"Unexpected data structure: {type(data)}")
            return None

    # Extract relevant data
    finger_kinematics = data['finger_kinematics']  # shape (n_samples, num_dofs)
    sbp = data['sbp']                              # shape (n_samples, 96)

    num_channels = sbp.shape[1]
    num_dofs = finger_kinematics.shape[1]
    
    channel_mi = np.zeros((num_channels, num_dofs))

    # For each channel
    for ch in range(num_channels):
        channel_data = sbp[:, ch]

        # Check if channel_data is constant
        if np.all(channel_data == channel_data[0]):
            continue  # MI would be zero anyway

        # For each DoF
        for dof in range(num_dofs):
            dof_data = finger_kinematics[:, dof]

            if np.all(dof_data == dof_data[0]):
                continue  # skip constant dof

            # Compute mutual information
            mi = mutual_info_regression(channel_data.reshape(-1,1), dof_data, discrete_features=False)
            channel_mi[ch, dof] = mi[0]

            # mi_value = mutual_info_score(
            #     pd.qcut(channel_data, 10, duplicates='drop'),
            #     pd.qcut(dof_data, 10, duplicates='drop')
            # )
            # channel_mi[ch, dof] = mi_value

    return channel_mi

def create_channelwise_mutual_information_matrix(preprocessingdir, outputdir):
    # Path to the folder containing pkl files
    data_folder = preprocessingdir

    # Get list of pkl files
    pkl_files = sorted(glob.glob(os.path.join(data_folder, '*.pkl')))

    # Find number of channels and dofs by looking at one file
    with open(pkl_files[0], 'rb') as f:
        sample_data = pickle.load(f)
    if isinstance(sample_data, tuple):
        if sample_data[0] is None:
            sample_data = sample_data[1]
        else:
            sample_data = sample_data[0]

    num_channels = sample_data['sbp'].shape[1]
    num_dofs = sample_data['finger_kinematics'].shape[1]

    # Initialize storage
    all_mi = []
    dates = []

    for file in pkl_files:
        sys.stdout.write(f"\rProcessing for MI: {file}")
        sys.stdout.flush()

        date = pd.to_datetime(os.path.basename(file).split('_')[0])

        # Load data
        with open(file, 'rb') as f:
            data = pickle.load(f)

        try:
            mi_matrix = compute_channel_mutual_information(data)
            all_mi.append(mi_matrix)
            dates.append(date)
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue

    all_mi = np.array(all_mi)  # Shape: (num_days, 96, num_dofs)
    dates = np.array(dates)

    # Save matrix and dates
    np.save(os.path.join(outputdir, 'channelwise_mutual_information.npy'), all_mi)
    np.save(os.path.join(outputdir, 'channelwise_mi_dates.npy'), dates)

    print(f"\nSaved MI matrix of shape {all_mi.shape}")



def create_channelwise_mutual_information_plot(outputdir):
    # Load
    all_mi = np.load(os.path.join(outputdir, 'channelwise_mutual_information.npy'))  # (num_days, 96, num_dofs)
    dates = np.load(os.path.join(outputdir, 'channelwise_mi_dates.npy'), allow_pickle=True)

    # Average across channels
    mean_mi = np.mean(all_mi, axis=1)  # (num_days, num_dofs)

    # Plot
    legend_dict = {0: "index_position", 1: "MRP_position", 2: "index_velocity", 3: "MRP_velocity"}
    plt.figure(figsize=(10,6))
    for dof_idx in range(mean_mi.shape[1]):
        plt.plot(dates, mean_mi[:, dof_idx], label=legend_dict[dof_idx])

    plt.title('Mutual Information over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Mutual Information')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(outputdir, 'channelwise_mutual_information_plot.png'))
    plt.show()

if __name__ == "__main__":
    create_channelwise_mutual_information_matrix("../test_data", "../test_data")
    create_channelwise_mutual_information_plot("../test_data")