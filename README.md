# Code for LINK: Long-Term Intracortical Neural Activity and Kinematics

In this repository, you will find all the code used to create the figures and dataset presented in `Long-term Intracortical Neural activity and Kinematics (LINK): An intracortical neural dataset for chronic brain-machine interfaces, neuroscience, and machine learning`, Temmar et al. (NeurIPS D&B 2025)

## Getting the Data
1. Clone this repository!
2. In terminal, navigate to this repository and run `conda create -n LINK_dataset python=3.9 --file requirements.txt -y`
3. Run `conda activate LINK_dataset`
4. Install the appropriate pytorch for your computer [here](https://pytorch.org/get-started/locally/)
5. Install the pyNWB API with `pip install -U pynwb`
6. Install the dandi-cli tool using: `pip install dandi`, if not already installed.
7. Download the dataset with `dandi download DANDI:001201`

Alternatively, you can run the prereq.sh bash file from this directory, and install pytorch on the side as well.

## Accessing the data
In the `tutorials` folder, you can find initial scripts to demonstrate the process of loading the NWB data and converting them to dictionary (pkl) files. They also include some small samples of how to wrangle the data and use it for decoding. An important note is that may need to modify the directories in these scripts to denote where you have stored your nwb and pkl files. For an in-depth description of the NWB file contents, please refer to the Supplementary Materials section of the paper.

## Visualizing the data
As a quick way to visualize the SBP, TCFR, and timeseries data, open the jupyter notebook `data_review/data_review_tool.ipynb`. Be sure to replace the file paths in the top cell with the absolute paths of the data and outputs, respectively. Use the `+500` and `-500` buttons to scrub through the data. If two target styles are present on a day, you can use the `Change Target Style` button to view one or the other. You can ignore the rest of the UI, this was for manual data review.

## Reproducing figures
To recreate the figures in the paper, first convert the data to dictionaries using the script mentioned above. Once the data is prepared, you can recreate the figures in the paper by running the following scripts in the analysis folder. You will also need to download the font Atkinson Hyperlegible (available [here](https://www.brailleinstitute.org/freefont/)) and rebuild your mpl cache, or change the font in the rcparams of the scripts:

* To recreate Figures 1-4, you can run the `generate_figures.py` script or use the `generate_figures.ipynb` notebook. Again, be careful with the directories of data and output storage here, you may need to change them to fit your set-up.

* To recreate Figure 4 without pre-calculated data, refer to `data_processing/bci_decoding/readme.md`. 

* Figure 5 is a bit more complicated to recreate due to the nature of the training process. Figure 5A can be recreated by following the steps in `data_processing\stability\readme.md` file, using the configs and notebooks in the `data_processing\stability` folder. Figure 5B can be recreated by using the files in the `data_processing\continual_learning` folder (start with the `data_processing\continual_learning\continual_learning_lstm.py` file for calculations).

## Info & Format for .pkl preprocessed files
This format does not contain descriptions and as much detail about the subject, device, and electrodes, but contains all the essential data for performing analyses in the paper.

### What does each file contain?
Each contains preprocessed neural and behavioral data from one day, along with metadata and trial data. Each file contains a tuple of length 2, with either two dictionaries and one dictionary and one None. Each available dictionary contains 375 trials of data for one of the two target styles. The first is center-out (CO), the second is random (RD). If a day has both dictionaries present, both target styles are available. You can open the file with:
`data_CO, data_RD = pickle.load(filepath)`

### Preprocessed file contents
Each 'YYYY-MM-DD_plotpreprocess.pkl' file contains a dictionary with the following keys:

**METADATA:**
'target_style', 'run_id'

**TRIAL DATA:**
'trial_number', 'trial_index', 'trial_count', 'target_positions', 'trial_timeout'

**TIMESERIES DATA:**
'sbp', 'finger_kinematics', 'time'

### metadata
* **'target_style' (str)**: indicates how targets were presented, either CO (center-out) or RD (random targets)
* **'run_id' (int)**: indicates the chronological order of that session out of all sessions recorded that day (not super relevant info)

## trial data
* **'trial_number' (int64 np.ndarray)**: Mx1 array, M = # of trials included for a particular target style on this day (should be 375). Contains trial id’s of included trials (not necessarily continuous – some trials maybe be removed).
* **'trial_index' (int64 np.ndarray)**: Mx1 array. Contains start indices of each trial in timeseries data
* **'trial_count' (int64 np.ndarray)**: Mx1 array. Contains length of each trial in the timeseries data
* **'target_positions' (float32 np.ndarray)**: Mx2 array. Each row contains the target position for the index finger and MRP (middle-ring-pinky) fingers: [index, MRP]
* **'run_id' (int64 np.ndarray)**: Mx1 array. Contains the run id of each trial
* **'trial_timeouts' (int64 np.ndarray)**: Mx1 array. Contains the trial timeout in ms before trials were considered failures. Longer timeouts are more forgiving

## timeseries data
* **'sbp' (float64 np.ndarray)**: Nx96 array, N = # of 32ms bins across all trials included for a particular target style on this day. Spiking band power averaged into 32ms bins for all 96 channels
* **'finger_kinematics' (float64 np.ndarray)**: Nx4 array. Finger kinematics averaged into 32ms bins, each row contains: [index_position, MRP_position, index_velocity, MRP_velocity]
* **'time' (float64 np.ndarray)**: Nx1 array. Experiment time averaged into 32ms bins
