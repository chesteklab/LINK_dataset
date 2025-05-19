import numpy as np
import pandas as pd
import os
import pickle
import glob
import re
from tqdm import tqdm
from datetime import datetime
import ast

# # server_dir = "/run/user/1000/gvfs/smb-share:server=cnpl-drmanhattan.engin.umich.edu,share=share/"
# # data_path = os.path.join(server_dir, 'Student Folders','Nina_Gill','data','only_good_days')
# data_path = "Z:\Student Folders\\Nina_Gill\data\only_good_days"
# output_path = "Z:\Student Folders\Hisham_Temmar\\big_dataset\output\signal_quality"


# data_path = "C:\\Files\\UM\\ND\\SFN\\only_good_days"
# output_path = 'C:\\Files\\UM\\ND\\github\\big_nhp_dataset_code\\outputs'
binsize = 20
def extract_dates_from_filenames(data_path):
    # Find all matching .pkl files
    pkl_files = glob.glob(os.path.join(data_path, '*_preprocess.pkl'))

    dates = []
    for file_path in pkl_files:
        filename = os.path.basename(file_path)
        match = re.match(r'(\d{4}-\d{2}-\d{2})_preprocess\.pkl', filename)
        if match:
            dates.append(match.group(1))

    dates = np.asarray([datetime.strptime(date, '%Y-%m-%d') for date in dates])
    return dates #sorted(dates) 

def load_day(date, data_path):
        file = os.path.join(data_path, f'{date.strftime("%Y-%m-%d")}_preprocess.pkl')

        with open(file, 'rb') as f:
            data_CO, data_RD = pickle.load(f)
        
        return data_CO, data_RD

def calc_avg_sbps(dates, data_path, output_path):
    sbp_avgs = pd.DataFrame(np.zeros((len(dates), 96), dtype=float), index=dates)
    sbp_avgs.index = pd.to_datetime(sbp_avgs.index)

    sbp_stds = pd.DataFrame(np.zeros((len(dates), 96), dtype=float), index=dates)
    sbp_stds.index = pd.to_datetime(sbp_stds.index)

    for date in tqdm(dates):
        data_CO, data_RD = load_day(date, data_path)
        
        if data_CO and data_RD:
            sbp = np.concatenate((data_CO['sbp'], data_RD['sbp']),axis=0)
        elif data_RD:
            sbp = data_RD['sbp']
        else:
            sbp = data_CO['sbp']

        sbp_avgs.loc[date] = np.mean(sbp, axis=0)
        sbp_stds.loc[date] = np.std(sbp, axis=0)

    #pdb.set_trace()
    sbp_avgs.to_csv(os.path.join(output_path, "sbp_avgs.csv"))
    sbp_stds.to_csv(os.path.join(output_path, "sbp_stds.csv"))


def calc_pr_all_days(dates, data_path, output_path):
    pr_dict = {'date': [],
               'chan_mask': [],
               'participation_ratio':[],
               'participation_ratio_active':[],
               'target_style': []
               }
    
    for date in tqdm(dates):
        data_CO, data_RD = load_day(date, data_path)
        # use CO if its there
        feat = data_CO if data_CO else data_RD
        target_style = 'CO' if data_CO else 'RD'

        tcfr = feat['tcfr'] * 1000 / binsize
        sbp = feat['sbp'] * 1000 / binsize

        # find all channels >1hz mean fr (CHECK)
        chanMask = np.where(np.mean(tcfr, axis=0) > 1)[0]
        if chanMask.shape[0] == 0:
            continue
        
        daily_pr = participation_ratio(sbp.T)
        daily_pr_active = participation_ratio(sbp[:, chanMask].T)

        pr_dict['date'].append(date)
        pr_dict['chan_mask'].append(chanMask)
        pr_dict['participation_ratio'].append(daily_pr)
        pr_dict['participation_ratio_active'].append(daily_pr_active)
        pr_dict['target_style'].append(target_style)

    pr_df = pd.DataFrame.from_dict(pr_dict)
    pr_df.to_csv(os.path.join(output_path,"participation_ratios.csv"))

def participation_ratio(dx_flat):
    # Dxflat is a 2D array of shape (n_features, n_samples)
    dx_flat = dx_flat - np.mean(dx_flat, axis=1)[:,np.newaxis] # subtract the mean so that we can actually get the covariance matrix
    DD = np.matmul(dx_flat, dx_flat.T) # 96 x 135 * 135 x 96
    U, S, V = np.linalg.svd(DD)
    pr = np.sum(S)**2/np.sum(S**2)
    return pr

def calc_sbp_heatmaps(dates, data_path, output_path):
    n_bins = 30
    bin_range = (0,30)
    bin_edges = np.linspace(bin_range[0], bin_range[1], n_bins + 1)
    bin_edges = np.concatenate([[-np.inf], bin_edges, [np.inf]])

    sbp_heatmaps = np.zeros((len(dates), n_bins+2, 96))

    for i, date in enumerate(tqdm(dates)):
        data_CO, data_RD = load_day(date, data_path)
        if data_CO and data_RD:
            sbp = np.concatenate((data_CO['sbp'], data_RD['sbp']),axis=0)
        elif data_RD:
            sbp = data_RD['sbp']
        else:
            sbp = data_CO['sbp']
        if np.any(np.isnan(sbp)):
            print('found nans')
        for j in range(sbp.shape[1]):
            sbp_heatmaps[i, :, j], _ = np.histogram(sbp[:,j], bins=bin_edges, range=bin_range, density=True)
    
    with open(os.path.join(output_path, 'sbp_heatmaps.pkl'), 'wb') as f:
        pickle.dump(sbp_heatmaps, f)

def clean_chan_mask(s):
    try:
        s = s.strip("[]")              
        s = s.lstrip(',')         
        new_s = [int(x) for x in s.split(' ') if x.strip().isdigit()]
        return new_s
    except Exception as e:
        print(f"Failed to parse {s}: {e}")
        return []