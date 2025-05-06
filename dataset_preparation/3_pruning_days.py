import os
import pandas as pd
import config
import pdb
from datetime import datetime
import numpy as np
import shutil

def move_good_days():
    review_path = "Z:\Student Folders\\Nina_Gill\data\datareview\\review_results_ht.csv"
    review_results = pd.read_csv(review_path)
    review_results = review_results.loc[review_results['Status'] == 'good']
    dates = review_results['Date'].to_numpy()
    dates = np.asarray([datetime.strptime(x, '%m/%d/%Y') for x in dates])
    outpath = 'Z:\Student Folders\\Nina_Gill\data\only_good_days'
    for date in dates:
        datestr = date.strftime('%Y-%m-%d')
        filepath = os.path.join(config.preprocessingdir, f'{datestr}_preprocess.pkl')
        shutil.copy2(filepath, outpath)
        print(f'copied {filepath}')
    
    pdb.set_trace()

if __name__=="__main__":
    move_good_days()