import pandas as pd
import os 
import numpy as np
import pickle
import config
import pdb
import sys
sys.path.append(config.pybmipath)
from pybmi.utils import ZTools

def prep_data(resume=True):
    datesruns = load_sheet()
    bad_days = []
    days = np.arange(len(datesruns))
    if resume:
        resumeidx = np.argwhere(datesruns['Date'].to_numpy() == '2022-10-28')[0,0]
    else:
        resumeidx = 0
    idxs = np.arange(resumeidx, len(datesruns))
    # data, data = load_day('2020-03-12')
    extra_bad_days = ['2022-06-09','2023-05-05','2024-01-29'] # first and second, notes file is wrong, third pickling went wrong, can't import.

    for i in idxs:
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
            filename = f'{date}_plotpreprocess.pkl'
            with open(os.path.join(config.cwd,'plot_preprocessing_new',filename),'wb') as f:
                pickle.dump((data_CO, data_RD), f)

        with open(os.path.join(config.cwd, 'plot_preprocessing','bad_days.txt'), 'a') as f:
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

    data_CO = choose_data(data_CO) if data_CO else None
    data_RD = choose_data(data_RD) if data_RD else None
    #once we know we have enough trials, do preprocessing
    if data_CO != None:
        print("PREPROCESSING CO")
        data_CO = plot_preprocessing(data_CO, runs_CO)
    if data_RD != None:
        print("PREPROCESSING RD")
        data_RD = plot_preprocessing(data_RD, runs_RD)
    
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
          & (z['TrialSuccess'] == 1)]
    # TODO: only 750ms hold time

    if style_mode == 34.0:
        target_style = 'CO'
    elif style_mode == 29.0:
        target_style = 'RD'
    else:
        target_style = 'NO'

    return z, target_style

num_runs_by_trial = []

def choose_data(data):
    # check how many runs are in a day
    num_runs = len(data)
    trials_per_run = np.asarray([len(z) for z in data])
    num_runs_by_trial = trials_per_run
    trials_per_run_sorted = np.flip(np.sort(trials_per_run))
    tpr_indices = np.flip(np.argsort(trials_per_run))
    
    # logic (THIS COULD BE DONE BETTER)
    enough_trials = False
    i = 0
    num_trials = 0

    new_data = [data[tpr_indices[i]]]
    while not enough_trials:
        # check num trials in next run
        num_trials += trials_per_run_sorted[i]
        # pdb.set_trace()
        if num_trials >= 400:
            enough_trials = True
            # trim the last run so that there are only 400 trials
            if num_trials > 400:
                trim_amount = num_trials - 400
                new_data[i] = new_data[i].iloc[0:-1*trim_amount]
        else:
            if num_runs > i+1:
                # add run and continue
                i += 1
                new_data.append(data[tpr_indices[i]])
            elif num_trials >= 375:
                # if we've reached 375 and there are no other runs, call it good.
                enough_trials == True
            else:
                # if there's no other runs and we don't have at least 375, reject entirely
                new_data = None
                enough_trials = True
    return new_data

def concat_runs(processed_runs):
    concatenated_data = {}
    for key in processed_runs[0].keys():
        if key == 'target_style':
            concatenated_data[key] = processed_runs[0][key]
        else:
            concatenated_data[key] = []
            for run in processed_runs:
                concatenated_data[key].extend(run[key])
            concatenated_data[key] = np.array(concatenated_data[key])
    return concatenated_data

def preprocessing(data, runs):
    processed = []
    
    # I'm doing the trial number trial index things here, not in the concat_runs function
    # keeping track of the last index and count here, will need to add to the index of later runs 
    trial_index_prev = []
    trial_count_prev = []
    count = 0

    for z, run in zip(data, runs):
        processed_run = {}
        style_mode = z['TargetPosStyle'].mode()[0]
        target_style = 'CO' if style_mode == 34.0 else 'RD'

        # first 5 trials have already been trimmed
        feats = ZTools.getZFeats(z, 
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
        
        trial_index_prev.append(TrialIndex[-1])
        trial_count_prev.append(TrialCount[-1])

        sbp = np.abs(feats['NeuralFeature'])
        sbp = (sbp - np.mean(sbp, axis=0)) / np.std(sbp, axis=0)

        # putting things together. I'm doing the adjust index thing here
        processed_run['target_style'] = target_style
        processed_run['trial_number'] = [x + 1000*(count) for x in TrialNumber]
        if run == runs[0]:
            processed_run['trial_index'] = TrialIndex
        else:
            processed_run['trial_index'] = [x + sum(trial_index_prev[:count]) + trial_count_prev[count-1] for x in TrialIndex]
        processed_run['trial_count'] = TrialCount
        processed_run['run_id'] = np.zeros_like(TrialCount, dtype=int) + run
        processed_run['target_positions'] = feats["TargetPos"][TrialIndex][:, [1, 3]]
        processed_run['time'] = feats['ExperimentTime']
        processed_run['finger_kinematics'] = FingerAngles
        processed_run['sbp'] = sbp
        processed_run['tcfr'] = feats['Channel']
        processed.append(processed_run)

        count += 1 

    data = concat_runs(processed)
    return data

if __name__=="__main__":
    prep_data()