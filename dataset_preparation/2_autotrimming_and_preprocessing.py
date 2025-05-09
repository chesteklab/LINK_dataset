import pandas as pd
import os 
import numpy as np
import pickle
import config
import pdb
import sys
from tqdm import tqdm
sys.path.append(config.pybmipath)
print(config.pybmipath)
from pybmi.utils import ZTools # type: ignore


def prep_data(resume=True):
    datesruns = load_sheet()
    bad_days = []
    days = np.arange(len(datesruns))
    if resume:
        resumeidx = np.argwhere(datesruns['Date'].to_numpy() == '2020-10-21')[0,0]
    else:
        resumeidx = 0
    idxs = np.arange(resumeidx, len(datesruns))
    extra_bad_days = ['2022-06-09','2023-05-05','2024-01-29'] # first and second, notes file is wrong, third pickling went wrong, can't import.

    for i in tqdm(idxs):
        date = datesruns['Date'].iloc[i]
        runs = datesruns['Runs'].iloc[i]
        print(f'{date} with runs {runs}')
        if date in extra_bad_days:
            bad_days.append(f'{date}')
            continue
        data_CO, data_RD = load_day(date)

        if data_CO == None and data_RD == None:
            #save to bad days txt file
            bad_days.append(f'{date}')
        else:
            filename = f'{date}_preprocess.pkl'
            with open(os.path.join(config.preprocessingdir,filename),'wb') as f:
                pickle.dump((data_CO, data_RD), f)

        with open(os.path.join(config.preprocessingdir,'bad_days.txt'), 'a') as f:
            for day in bad_days:
                f.write(f"{day}\n")

def load_sheet():
    # load the spreadsheet
    if not os.path.isfile(config.dfpath):
        datesruns = pd.read_csv(config.sheetpath)
        datesruns['Date'] = pd.to_datetime(datesruns['Date'], format='%m/%d/%Y')
        datesruns['Date'] = datesruns['Date'].dt.strftime('%Y-%m-%d')
        datesruns['Runs'] = datesruns['Runs'].apply(lambda x: [int(i.strip()) for i in x.split(',')])
        with open(config.dfpath, 'wb') as f:
            pickle.dump(datesruns, f)
    else:
        with open(config.dfpath, 'rb') as f:
            datesruns = pickle.load(f)
    
    return datesruns

def load_day(date):
    #load runs within the day, do the logic on picking which ones to use
    print(f"LOADING {date}")
    datesruns = load_sheet()
    runs = datesruns.loc[datesruns['Date'] == date]['Runs'].to_numpy()[0]
    enough_trials = False
    # load all the runs for this day:
    data_CO = []
    runs_CO = []
    data_RD = []
    runs_RD = []
    for run in runs:
        data, target_style = load_run(date, run)
        if target_style == 'NO' or len(data) == 0:
            print(f'Run {run} on {date} does not have valid trials, ignoring.')
        elif target_style == 'CO':
            data_CO.append(data)
            runs_CO.append(run)
        elif target_style == 'RD':
            data_RD.append(data)
            runs_RD.append(run)

    #then do the logic to decide what to keep and preprocess
    (data_CO, run_CO) = choose_data(data_CO, runs_CO) if data_CO else (None, None)
    (data_RD, run_RD) = choose_data(data_RD, runs_RD) if data_RD else (None, None)

    #once we know we have enough trials, do preprocessing
    if data_CO is not None:
        print("PREPROCESSING CO")
        data_CO = preprocessing(data_CO, run_CO, "CO")
    if data_RD is not None:
        print("PREPROCESSING RD")
        data_RD = preprocessing(data_RD, run_RD, "RD")
    
    return data_CO, data_RD

def load_run(date, run):
    '''
    Loads a specified run from the given date trims the first few trials,
    removes unsuccessful trials, and target styles
    '''

    # LOAD REQUESTED Z STRUCT
    runstr = 'Run-' + str(run).zfill(3)
    fpath = os.path.join(config.datapath, config.data_params['monkey'], date, runstr)
    z = ZTools.ZStructTranslator(fpath, use_py=False).asdataframe()


    # filters based on the most common style AND if it's 29 or 34, so we don't have one 29 in a mix of 34 
    # remove closed loop and unsuccessful trials
    z = z[5:] #trim first 5
    style_mode = z['TargetPosStyle'].mode()[0]
    z = z[(z['TargetPosStyle'] == style_mode) 
          & (z['TargetPosStyle'].isin([29.0, 34.0])) 
          & (z['ClosedLoop'] == 0) 
          & (z['TrialSuccess'] == 1)
          & (z['TargetHoldTime'] == 750)]
          

    if style_mode == 34.0:
        target_style = 'CO'
    elif style_mode == 29.0:
        target_style = 'RD'
    else:
        target_style = 'NO'

    return z, target_style

def choose_data(data, runs):
    # check trials are in each run
    max_trials = np.max(np.asarray([len(z) for z in data]))
    argmax_trials = np.argmax(np.asarray([len(z) for z in data]))
    
    # from the largest - see if there are 375 trials
    if max_trials > 375:
        # take it
        trim_amount = max_trials - 375
        new_data = data[argmax_trials].iloc[0:-1*trim_amount]
        run = runs[argmax_trials]
    # else if
    elif max_trials == 375:
        new_data = data[argmax_trials]  
        run = runs[argmax_trials]
    else:
        new_data = None
        run = None

    return new_data, run

def preprocessing(data, run, target_style):
    
    processed_run = {}

    # first 5 trials have already been trimmed
    feats = ZTools.getZFeats(data, 
                             binsize=config.data_params['binsize'],
                             removeFirstTrial = False,
                             featList=('FingerAnglesTIMRL',
                                       'NeuralFeature',
                                       'TrialNumber',
                                       'TargetPos',
                                       'ExperimentTime',
                                       'Channel'))
    
    FingerAngles = feats["FingerAnglesTIMRL"][:, (1, 3, 6, 8)]  #selecting only position and velocity
    TrialNumber, TrialIndex, TrialCount = np.unique(feats["TrialNumber"], return_index = True, return_counts = True)

    sbp = np.abs(feats['NeuralFeature'])
    # sbp = (sbp - np.mean(sbp, axis=0)) / np.std(sbp, axis=0)

    # putting things together. I'm doing the adjust index thing here
    processed_run['target_style'] = target_style
    processed_run['trial_number'] = TrialNumber
    processed_run['trial_index'] = TrialIndex
    processed_run['trial_count'] = TrialCount
    processed_run['run_id'] = run
    processed_run['target_positions'] = feats["TargetPos"][TrialIndex][:, [1, 3]]
    processed_run['time'] = feats['ExperimentTime']
    processed_run['finger_kinematics'] = FingerAngles
    processed_run['sbp'] = sbp
    processed_run['tcfr'] = feats['Channel']

    return processed_run

if __name__=="__main__":
    prep_data(resume=False)