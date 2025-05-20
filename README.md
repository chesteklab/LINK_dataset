# Code for LINK: Long-Term Intracortical Neural Activity and Kinematics

In this repository, you will find all the code used to create the figures and dataset presented in `Long-term Intracortical Neural activity and Kinematics (LINK): An intracortical neural dataset for chronic brain-machine interfaces, neuroscience, and machine learning`, Temmar et al. (In Submission, NeurIPS D&B 2025)

The repository is split into two sections, for dataset preparation and analysis.

## Getting the Data
1. Clone this repository
2. Create a conda environment using the requirement.txt file, python 3.9, and also install pytorch
3. Install the dandi-cli tool using: `pip install dandi`, if not already installed.
4. Download the dataset with `dandi download DANDI:001201`

## Accessing the data
Data can be accessed aas an nwb file, or converted to dictionaries (with sligtly less information) by running: `python dataset_preparation/convert_dandi_nwb_to_pkl.py`. Directories in line 74 and 77 should be changed to match user environment. N.B. In line 77 processed_data_path needs to be created before running the code.

## Visualizing the data
For a quick way to look at the timeseries data, use the script above to convert the data to dictionaries, then specify the location of the new .pkl files in `dataset_preparation/config.py` (default `./outputs`), then run the cell in `dataset_preparation/data_review_tool.ipynb`. You may also want to set an output directory in the config file or the tool may break when terminating plotting. Use the `+500` and `-500` buttons to scrub through the data. If two target styles are present on a day, you can use the `Change Target Style` button to view one or the other. You can ignore the rest of the UI, this was for manual data review.

## Reproducing figures
To recreate the figures in the paper, first convert the data to dictionaries using the script mentioned above. Once the data is prepared, you can recreate the figures in the paper by running the following scripts in the analysis folder. You will also need to download the font Atkinson Hyperlegible (available [here](https://www.brailleinstitute.org/freefont/)) and rebuild your mpl cache, or change the font in the rcparams of the scripts:

* To recreate the target position and data distribution plots in Figure 1, please run the `python analysis/dataset_overview/dataset_overview.py`. Line 21 and 22 can be changed to match user directories.
* To recreate Figure 2A-C, change the filepaths at line 11 and 12 (if needed) in `analysis/signal_changes/signal_utils.py`. Then run `python analysis/signal_changes/signal_changes.py`. After running for the first time, if you'd like to avoid crunching the numbers again, set the `calc_avg_sbp` and `calc_pr` flags back to `False` in `analysis/signal_changes/signal_changes.py`.
* To recreate Figure 2D-E, run `python analysis/pop_level_analyses/dimensionality_across_days_analysis.ipynb` in the pop_level_analyses folder. mpath in cell 4 should be changed to user data directory.
* To recreate Figure 3, change the filepaths at line 29 and 30 (if needed) in `analysis/tuning_utils.py`, then run `python analysis/single_channel_tuning/single_channel_tuning.py`. After running for the first time, if you'd like to avoid crunching the numbers again, set the `calc_tunings` flag `False` in `analysis/single_channel_tuning/single_channel_tuning.py`.
* To recreate Figure 4, refer to `analysis/bci_decoding/readme.md`.

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

### Preprocessed file contents
Each 'YYYY-MM-DD_plotpreprocess.pkl' file contains a dictionary with the following keys:

**METADATA:**
'target_style'

**TRIAL DATA:**
'trial_number', 'trial_index', 'trial_count', 'target_positions', 'run_id', 'trial_timeout'

**TIMESERIES DATA:**
'sbp', 'finger_kinematics', 'time'

### metadata
* **'target_style' (str)**: indicates how targets were presented, either CO (center-out) or RD (random targets)

## trial data
* **'trial_number' (int64 np.ndarray)**: Mx1 array, M = # of trials included for a particular target style on this day, usually 400. Contains trial id’s of included trials (not necessarily continuous – some trials maybe be removed). If multiple runs are concatinated together from the same day, the first processed run has 1-3 digit trial numbers, the second processed run has trial numbers 1xxx where xxxx indicate the trial id's within that specific run, the third processed run has trial numbers 2xxx, etc.
* **'trial_index' (int64 np.ndarray)**: Mx1 array. Contains start indices of each trial in timeseries data
* **'trial_count' (int64 np.ndarray)**: Mx1 array. Contains length of each trial in the timeseries data
* **'target_positions' (float32 np.ndarray)**: Mx2 array. Each row contains the target position for the index finger and MRP (middle-ring-pinky) fingers: [index, MRP]
* **'run_id' (int64 np.ndarray)**: Mx1 array. Contains the run id of each trial
* **'trial_timeouts' (int64 np.ndarray)**: Mx1 array. Contains the trial timeout in ms before trials were considered failures. Longer timeouts are more forgiving

## timeseries data
* **'sbp' (float64 np.ndarray)**: Nx96 array, N = # of 32ms bins across all trials included for a particular target style on this day. Spiking band power averaged into 32ms bins for all 96 channels
* **'finger_kinematics' (float64 np.ndarray)**: Nx4 array. Finger kinematics averaged into 32ms bins, each row contains: [index_position, MRP_position, index_velocity, MRP_velocity]
* **'time' (float64 np.ndarray)**: Nx1 array. Experiment time averaged into 32ms bins
