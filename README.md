# Code for LINK: Long-Term Intracortical Neural Activity and Kinematics

In this repository, you will find all the code used to create the figures and dataset presented in `Long-term Intracortical Neural activity and Kinematics (LINK): An intracortical neural dataset for chronic brain-machine interfaces, neuroscience, and machine learning`, Temmar et al. (In Submission, NeurIPS D&B 2025)

The repository is split into two sections, for dataset preparation and analysis.

## Getting the Data
1. Clone this repository
2. Create a conda environment using the requirement.txt file, and also install pytorch
3. Install the dandi-cli tool using: `pip install dandi`
4. Download the dataset with `dandi download DANDI:001201`
5. Then, adjusting filepaths, run the script `convert_dandi_nwb_to_pkl.py`

## Reproducing figures
Once the data is prepared, you can recreate the figures in the paper by running the following scripts:

To recreate the target position and data distribution plots in figure 1, please run the script `datasetoverview.py`
To recreate the subfigures a,b and c in the neural signals over time figure, run the script `signal_changes.py`, making sure to set filepath in `signal_utils.py` and if running for the first time, set the calc_avg_sbp and calc_pr parameters to true.
To recreate the subfigures d and e in the neural signals over time figure, run `dimensionality_across_days_analysis.ipynb`
To recreate the tuning figure, run the script `single_channel_tuning.py`, making sure to update the filepaths in `tuning_utils.py` and setting the calc_tunings flag to true.
To recreate the decoding figure, refer to the readme in the bci_decoding folder.

## Info for NWB files
For a NWBHDF5IO object named nwb:
All timeseries data for the nwb files is contained in the analysis module (nwb.analysis)
Trial information is contained inthe trial table (nwb.trials)
Additional information about the electrodes, like positional mappings and impedances, are stored in the electrodes

## Info & Format for .pkl preprocessed files
This format does not contain descriptions and as much detail about the subject, device, and electrodes, but contains all the essential data for performing analyses in the paper.

### What does each file contain?
* Each contains preprocessed neural and behavioral data from one day, along with metadata and trial data
* Each contains 375 trials of data for one target style, center-out (CO) or random (RD), or 400 trials for *each* of the two target styles if both target styles are present on the same day, under two seperate dictionaries

### How to open:
`data_CO, data_RD = pickle.load(filepath)`

## SESSION FILE CONTENTS
Each 'YYYY-MM-DD_plotpreprocess.pkl' file contains a dictionary with the following keys:

**METADATA:**
'target_style'

**TRIAL DATA:**
'trial_number', 'trial_index', 'trial_count', 'target_positions', 'run_id', 'trial_timeout'

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
* **'trial_timeouts' (int64 np.ndarray)**: Mx1 array. Contains the trial timeout in ms before trials were considered failures. Longer timeouts are more forgiving

## TIMESERIES DATA
* **'sbp' (float64 np.ndarray)**: Nx96 array, N = # of 32ms bins across all trials included for a particular target style on this day. Spiking band power averaged into 32ms bins for all 96 channels
* **'finger_kinematics' (float64 np.ndarray)**: Nx4 array. Finger kinematics averaged into 32ms bins, each row contains: [index_position, MRP_position, index_velocity, MRP_velocity]
* **'time' (float64 np.ndarray)**: Nx1 array. Experiment time averaged into 32ms bins
