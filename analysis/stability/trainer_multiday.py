data_dir = "/home/chesteklab/Code/Chang/LINK_dataset/001201/sub-Monkey-N"
neural_decoding_dir = "/home/chesteklab/Code/Chang/neuraldecoding"

cfg_path = "config_LSTM"
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

days.sort()
print(days)


for i, sel_day in tqdm(enumerate(days[5:])):
    print(f"==Processing day: {sel_day}, day {i + 1} out of {total_days} days==")
    log_r_multi_day = []
    log_r2_multi_day = []
    log_mse_multi_day = []
    log_rr_pred_multi_day = []
    log_finger_multi_day = []
    for num_train_day in range(1, 6):
    #for num_train_day in range(1, 2):
        log_r_single_day = []
        log_r2_single_day = []
        log_mse_single_day = []
        log_rr_pred_single_day = []
        log_finger_single_day = []
        sel_idx = days.index(sel_day)
        train_days = days[max(0, sel_idx - 5):sel_idx + 1]
        train_days = train_days[::-1]
        train_days = train_days[:num_train_day]

        with initialize(version_base=None, config_path=cfg_path):
            config = compose("config")
        cfg_trainer = parse_verify_config(config, 'trainer')
        cfg_decoder = parse_verify_config(config, 'decoder')
        preprocessing_config = parse_verify_config(config, 'preprocessing')
        preprocessing_trainer_config = preprocessing_config['preprocessing_trainer']
        preprocessing_decoder_config = preprocessing_config['preprocessing_decoder']
        preprocessor_trainer = Preprocessing(preprocessing_trainer_config)
        device = cfg_decoder["model"]['params']['device']

        trainer = NNTrainer(preprocessor_trainer, cfg_trainer)

        all_neural_train = []
        all_neural_test = []
        all_finger_train = []
        all_finger_test = []
        print(train_days)
        for d in train_days:
            file_name = f"sub-Monkey-N_ses-{d}_ecephys.nwb"
            data_dict = load_one_nwb(os.path.join(data_dir, file_name))
            neural_train_ind, neural_test_ind, finger_train_ind, finger_test_ind = preprocessor_trainer.preprocess_pipeline(data_dict, params={'is_train': True})
            all_neural_train.append(neural_train_ind)
            all_neural_test.append(neural_test_ind)
            all_finger_train.append(finger_train_ind)
            all_finger_test.append(finger_test_ind)

        neural_train = torch.cat(all_neural_train, dim=0).to(device)
        neural_test = torch.cat(all_neural_test, dim=0).to(device)
        finger_train = torch.cat(all_finger_train, dim=0).to(device)
        finger_test = torch.cat(all_finger_test, dim=0).to(device)
        
        trainer.train_X, trainer.train_Y, trainer.valid_X, trainer.valid_Y = neural_train, finger_train, neural_test, finger_test
        trainer.train_loader, trainer.valid_loader = trainer.create_dataloaders()
        print(f"----Training on {num_train_day} days for {sel_day}, day {i + 1} out of {total_days} days----")
        model, results = trainer.train_model()
        cfg_decoder["fpath"] = cfg_decoder["fpath"].format(date=sel_day, day=num_train_day)
        model_path = cfg_decoder["fpath"]
        model.save_model(cfg_decoder["fpath"])

        preprocessor_decoder = Preprocessing(preprocessing_decoder_config)
        decoder = NeuralNetworkDecoder(cfg_decoder)
        decoder.load_model()
        decoder.model.to(device)