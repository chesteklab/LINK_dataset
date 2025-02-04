# MONKEY TIME: INFO & FORMAT

## DATASET V1.1

#### **WHERE IS IT:**
* {cnpl-drmanhattan}\Student Folders\Bianca_Wang\plot_preprocessing

#### **WHAT DOES IT CONTAIN:**
* 439 session files labeled 'YYYY-MM-DD_plotpreprocess.pkl'
* Each contains preprocessed neural and behavioral data from one day, along with metadata and trial data
* Each contains 400 trials of data for one target style, center-out (CO) or random (RD), or 400 trials for *each* of the two target styles if both target styles are present on the same day, under two seperate dictionaries
* The trials may contain trials from more than a single run, but only if the best run of the day does not contain enough trials and there are other runs with the same target style

#### **HOW TO OPEN:**
* data_CO, data_RD = pickle.load(filepath)

## SESSION FILE CONTENTS
Each 'YYYY-MM-DD_plotpreprocess.pkl' file contains a dictionary with the following keys:

**METADATA:**
'target_style'

**TRIAL DATA:**
'trial_number', 'trial_index', 'trial_count', 'target_positions', 'run_id'

**TIMESERIES DATA:**
'sbp', 'finger_kinematics', 'time'

## METADATA
* **'target_style' (str)**: indicates how targets were presented, either CO (center-out) or RD (random targets)

## TRIAL DATA
* **'trial_number' (int64 np.ndarray)**: Mx1 array, M = # of trials included for a particular target style on this day, usually 400. Contains trial id’s of included trials (not necessarily continuous – some trials maybe be removed). If multiple runs are concatinated together from the same day, the first processed run has 1-3 digit trial numbers, the second processed run has trial numbers 1xxx where xxxx indicate the trial id's within that specific run, the third processed run has trial numbers 2xxx, etc.
* **'trial_index' (int64 np.ndarray)**: Mx1 array. Contains start indices of each trial in timeseries data
* **'trial_count' (int64 np.ndarray)**: Mx1 array. Contains length of each trial in the timeseries data
* **'target_positions' (float32 np.ndarray)**: Mx2 array. Each row contains the target position for the index finger and MRP (middle-ring-pinky) fingers: [index, MRP]
* **'run_id' (int64 np.ndarray)**: Mx1 array. Contains the run id of each trial

## TIMESERIES DATA
* **'sbp' (float64 np.ndarray)**: Nx96 array, N = # of 32ms bins across all trials included for a particular target style on this day. Spiking band power averaged into 32ms bins for all 96 channels
* **'finger_kinematics' (float64 np.ndarray)**: Nx4 array. Finger kinematics averaged into 32ms bins, each row contains: [index_position, MRP_position, index_velocity, MRP_velocity]
* **'time' (float64 np.ndarray)**: Nx1 array. Experiment time averaged into 32ms bins
