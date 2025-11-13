"""
Helper script to make a csv list of all the days, their day idx, and whether they have CO or RD data.
"""

import os
import pickle
import datetime
import pandas as pd

# Set your data folder path here
data_folder = '../../link_dataset_pkl'

# Get all files ending with _preprocess.pkl
files = [f for f in os.listdir(data_folder) if f.endswith('_preprocess.pkl')]

# Extract dates and sort them
dates = []
for f in files:
    try:
        date_str = f.split('_preprocess.pkl')[0]
        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        dates.append((date_obj, date_str, f))
    except Exception as e:
        print(f"Skipping file {f}: {e}")

dates.sort()
if not dates:
    print("No valid date files found.")
    exit()

first_date_obj = dates[0][0]

rows = []
for date_obj, date_str, fname in dates:
    print(date_str)
    row = {'date': date_str, 'days_since_first': (date_obj - first_date_obj).days, 'CO': '', 'RD': ''}
    try:
        with open(os.path.join(data_folder, fname), 'rb') as f:
            data_CO, data_RD = pickle.load(f)
        if data_CO is not None:
            row['CO'] = 'x'
        if data_RD is not None:
            row['RD'] = 'x'
    except Exception as e:
        print(f"Error loading {fname}: {e}")
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv('data_overview.csv', index=False)
print("Saved to data_overview.csv")