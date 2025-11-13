data_dir = "../../link_dataset_pkl"
neural_decoding_dir = "/home/chesteklab/Code/Chang/neuraldecoding"
target_day = '20210731'

cfg_path = "config_LSTM_cpu"
import os
import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--trial_name', type=str, default=None, help='Trial name for saving results')
args = parser.parse_args()

if args.trial_name is None:
    trial_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
else:
    trial_name = args.trial_name
import sys
import os

sys.path.append(neural_decoding_dir)

import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


from tqdm import tqdm

from hydra import initialize, compose

from neuraldecoding.decoder import NeuralNetworkDecoder
from neuraldecoding.trainer.NeuralNetworkTrainer import NNTrainer
from neuraldecoding.utils import load_one_nwb, prep_data_and_split, add_hist
from neuraldecoding.utils import parse_verify_config
from neuraldecoding.preprocessing import Preprocessing
np.set_printoptions(suppress=True)


files = os.listdir(data_dir)
days = []
pattern = re.compile(r"sub-Monkey-N_ses-(\d{8})_ecephys\.nwb")
for fname in files:
    match = pattern.match(fname)
    if match:
        days.append(match.group(1))

total_days = len(days)

from datetime import timedelta
from datetime import datetime
import numpy as np
from sklearn.metrics import r2_score
import pickle
from sklearn.preprocessing import StandardScaler

days.sort()
print(days)


target_index = days.index(target_day)


for i, sel_day in tqdm(enumerate(days[(target_index + 1):])):
    print(f"==Processing day: {sel_day}, day {target_index + i + 1} out of {total_days} days==")
    log_r_multi_day = []
    log_r2_multi_day = []
    log_mse_multi_day = []
    log_rr_pred_multi_day = []
    log_finger_multi_day = []

    sel_day_dt = datetime.strptime(sel_day, '%Y%m%d')
    six_months_later = sel_day_dt + timedelta(days=180)
    future_days = []
    for day in days:
        day_dt = datetime.strptime(day, '%Y%m%d')
        if day_dt <= six_months_later:
            future_days.append(day)

    file_name = f"sub-Monkey-N_ses-{sel_day}_ecephys.nwb"
    data_dict = load_one_nwb(os.path.join(data_dir, file_name))
    with initialize(version_base=None, config_path=cfg_path):
            cfg = compose("config")
    preprocessing_config = parse_verify_config(cfg, 'preprocessing')
    preprocessing_trainer_config = preprocessing_config['preprocessing_trainer']
    
    preprocessor_trainer = Preprocessing(preprocessing_trainer_config)
    _, _, finger_train, finger_test = preprocessor_trainer.preprocess_pipeline(data_dict, params={'is_train': True})
    scaler = StandardScaler()
    scaler.fit(finger_train)

    for num_train_day in range(1, 6):
        log_r_single_day = []
        log_r2_single_day = []
        log_mse_single_day = []
        log_rr_pred_single_day = []
        log_finger_single_day = []
        sel_idx = days.index(sel_day)
        train_days = days[max(0, sel_idx - 5):sel_idx + 1]
        train_days = train_days[::-1]
        train_days = train_days[:num_train_day]

        config = cfg.copy()
        cfg_trainer = parse_verify_config(config, 'trainer')
        cfg_decoder = parse_verify_config(config, 'decoder')
        preprocessing_config = parse_verify_config(config, 'preprocessing')
        preprocessing_trainer_config = preprocessing_config['preprocessing_trainer']
        preprocessing_decoder_config = preprocessing_config['preprocessing_decoder']
        device = cfg_decoder["model"]['params']['device']

        cfg_decoder["fpath"] = cfg_decoder["fpath"].format(date=sel_day, day=num_train_day)

        preprocessor_decoder = Preprocessing(preprocessing_decoder_config)
        decoder = NeuralNetworkDecoder(cfg_decoder)
        decoder.load_model()
        decoder.model.to(device)

        for test_day in future_days:
            file_name = f"sub-Monkey-N_ses-{test_day}_ecephys.nwb"
            data_dict = load_one_nwb(os.path.join(data_dir, file_name))
            _, neural, _, finger = preprocessor_decoder.preprocess_pipeline(data_dict, params={'is_train': False})
            rr_prediction = decoder.predict(neural)

            rr_prediction = scaler.inverse_transform(rr_prediction.cpu().numpy())

            finger_detach = finger.detach().cpu().numpy()
            r = np.array([pearsonr(finger_detach[:, i], rr_prediction[:, i])[0] for i in range(finger.shape[1])])
            r2 = r2_score(finger_detach, rr_prediction)
            mse = np.mean((finger_detach - rr_prediction)**2, axis=0)
            log_r_single_day.append(r.mean())
            log_r2_single_day.append(r2)
            log_mse_single_day.append(mse.mean())
            log_rr_pred_single_day.append(rr_prediction)
            log_finger_single_day.append(finger_detach)

            print(f"Train Day: {sel_day}, Dur: {num_train_day}, Test Day: {test_day}, r: {r.mean():.4f}, r2: {r2:.4f}, mse: {mse.mean():.4f}")
            df_row = pd.DataFrame({
                'train_day': [sel_day],
                'train_duration': [num_train_day],
                'test_day': [test_day],
                'r_mean': [r.mean()],
                'r2': [r2],
                'mse_mean': [mse.mean()]
            })

            if 'results_df' not in locals():
                results_df = pd.DataFrame(columns=['train_day', 'train_duration', 'test_day', 'r_mean', 'r2', 'mse_mean'])

            results_df = pd.concat([results_df, df_row], ignore_index=True)
    if i % 10 == 0:
        results_df.to_csv(f'results_{trial_name}_day_{sel_day}_{i}.csv', index=False)
        print(f"Saved results for day {sel_day} at iteration {i}")
    