import numpy as np
import pandas as pd
import glob
import os
import re
from datetime import datetime
import pickle

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