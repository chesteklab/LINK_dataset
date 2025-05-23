import pickle
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from datetime import datetime, timezone
from pynwb import NWBFile, TimeSeries, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries
from pynwb.file import Subject
from typing import Tuple, Optional
import dataset_preparation.archive.config as config
from dataset_preparation.archive.find_impedances import get_impedances


def convert_pkl_to_nwb(data_dir, electrode_table_csv_path, end_dir=None):
    if end_dir is None:
        end_dir = data_dir
    CHANNEL_MAP = pd.read_csv(electrode_table_csv_path)
    CHANNEL_MAP.sort_values("Channel", inplace=True)
    trial_fields = [
        "trial_number",
        "trial_count",
        "target_positions",
    ]
    ts_fields = ["time", "finger_kinematics", "sbp", "tcfr"]
    for filename in tqdm(os.listdir(data_dir)):
        if not filename.endswith(".pkl"):
            continue
        file_path = os.path.join(data_dir, filename)
        no_ext_filename = filename.split(".")[0]
        date = no_ext_filename.split("_")[0]
        # Load the dictionary from a pickle file
        with open(file_path, "rb") as f:
            data_tuple = pickle.load(f)

        for data_dict in data_tuple:
            if data_dict is None:
                continue
            # First, save the trial info
            trial_dfs = []
            # Trial
            for field in trial_fields:
                value = data_dict[field]
                df = pd.DataFrame(
                    value,
                    columns=(
                        [field]
                        if len(value.shape) == 1
                        else [f"{field}_{i}" for i in range(value.shape[1])]
                    ),
                )
                trial_dfs.append(df)
            trials_df = pd.concat(trial_dfs, axis=1)
            # Add target style
            trials_df["target_style"] = data_dict["target_style"]
            # Add trial timeout
            trials_df["trial_timeout"] = data_dict["trial_timeout"]
            trials_df["run_id"] = data_dict["run_id"]
            # Rename target positions to index and MRS
            renaming_trials = {
                "target_positions_0": "index_target_position",
                "target_positions_1": "mrs_target_position",
            }
            # Apply renaming to trials DataFrame
            trials_df = trials_df.rename(columns=renaming_trials)

            # Timeseries
            ts_dfs = []
            for field in ts_fields:
                value = data_dict[field]
                df = pd.DataFrame(
                    value,
                    columns=(
                        [field]
                        if len(value.shape) == 1 or value.shape[1] == 1
                        else [f"{field}_{i}" for i in range(value.shape[1])]
                    ),
                )
                ts_dfs.append(df)
            # Combine
            ts_df = pd.concat(ts_dfs, axis=1)
            # Set the trial numbers
            time_steps = np.arange(len(data_dict["time"]))
            trial_indices = (
                np.searchsorted(data_dict["trial_index"], time_steps, side="right") - 1
            )
            ts_df.insert(1, "trial_number", data_dict["trial_number"][trial_indices])
            # Rename columns to channels
            renaming_ts = {
                "finger_kinematics_0": "index_position",
                "finger_kinematics_1": "mrs_position",
                "finger_kinematics_2": "index_velocity",
                "finger_kinematics_3": "mrs_velocity",
            }
            renaming_ts.update({f"sbp_{i}": f"sbp_channel_{i}" for i in range(96)})
            renaming_ts.update({f"tcfr_{i}": f"tcfr_channel_{i}" for i in range(96)})
            ts_df = ts_df.rename(columns=renaming_ts)

            # Discard trials from runs other than the first run
            # Determine the first run_id
            first_run_id = trials_df["run_id"].iloc[0]
            print(
                f"Processing date {date}, target style {trials_df['target_style'][0]}, first run_id: {first_run_id}"
            )

            # Keep only trials with run_id == first_run_id
            trials_df = trials_df[trials_df["run_id"] == first_run_id]

            # Get the list of valid trial numbers
            valid_trial_numbers = trials_df["trial_number"].unique()

            # Filter timeseries_df to keep only data from valid trials
            ts_df = ts_df[ts_df["trial_number"].isin(valid_trial_numbers)]

            # Add target style information to the dataframe
            ts_df["target_style"] = trials_df["target_style"]

            # Adjust the timestamps to milliseconds
            times = ts_df["time"].values / 1000  # Convert to seconds

            # Create an NWB file with timezone-aware datetime objects
            nwbfile = NWBFile(
                session_description=f"Neural and behavioral data for target style {trials_df['target_style'][0]}",
                identifier=f"{date}_{trials_df['target_style']}_nwb",
                session_start_time=datetime.strptime(date, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                ),
                experimenter=["Chestek Lab"],
                file_create_date=datetime.now(timezone.utc),
                lab="Chestek Lab",
                institution="University of Michigan",
                experiment_description="Behavioral and neural data from Utah Array recordings from a macaque doing finger movements",
                keywords=["neural decoding", "finger movements", "motor cortex"],
            )
            # Create and add the subject
            subject = Subject(
                subject_id="Monkey N",
                description="Monkey N",
                species="Macaca mulatta",
                sex="M",
                date_of_birth=datetime.strptime("2012-05-26", "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                ),
            )
            nwbfile.subject = subject

            # Iterate over arrays in electrode table
            groups = {}
            for array in CHANNEL_MAP["Array location"].unique():
                dev = nwbfile.create_device(
                    name=array, description=f"{array} | Utah Array"
                )
                grp = nwbfile.create_electrode_group(
                    name=array,
                    description=f"{array} | Utah Array",
                    location="Hand area of M1",
                    device=dev,
                )
                groups[array] = grp

            for name, descr in {
                "array_name": "Name of the array this contact belongs to",
                "bank": "Headstage bank (A-C)",
                "pin": "Headstage bank pin number (1-32)",
                "row": "Row index in the 8x8 array grid (0-7)",
                "col": "Column index in the 8x8 array grid (0-7)",
            }.items():
                nwbfile.add_electrode_column(name, descr)

            # # Add electrodes to the electrode table
            # for idx in range(96):
            #     nwbfile.add_electrode(
            #         id=idx,
            #         location="Hand area of primary motor cortex",
            #         filtering="",
            #         group=electrode_group,
            #     )
            impedances = get_impedances(date)
            if impedances is None:
                print(f"No impedances found for date {date}")
            for _, ch in CHANNEL_MAP.iterrows():
                ch_id = int(ch["Channel"]) - 1  # 0-based to match sbp_channel_0 …
                row = int(ch["Array Row"]) - 1  # make rows/cols 0-based
                col = int(ch["Array Column"]) - 1
                # Convert impedances from kOhm to Ohm
                imp = (
                    impedances[ch["Channel"]] * 1e3 if impedances is not None else None
                )

                nwbfile.add_electrode(
                    id=ch_id,
                    location="Hand area of M1",
                    filtering="",
                    group=groups[ch["Array location"]],
                    array_name=ch["Array location"],
                    bank=ch["Bank"],
                    pin=int(ch["Pin"]),
                    row=row,
                    col=col,
                    imp=imp,
                )

            # Extract kinematic data
            index_position = ts_df["index_position"].values
            mrs_position = ts_df["mrs_position"].values

            # Positions are in flexion units from 0 to 1
            # Velocities are the difference of positions per time bin (20 ms)
            # Velocities are provided in the data
            index_velocity = ts_df["index_velocity"].values
            mrs_velocity = ts_df["mrs_velocity"].values

            # Create a processing module for behavior
            # behavior_module = nwbfile.create_processing_module(
            #     name="behavior", description="Processed behavioral data"
            # )

            # Create TimeSeries objects for positions
            index_position_ts = TimeSeries(
                name="index_position",
                data=index_position[:, None],
                unit="flexion units",  # From 0 to 1
                timestamps=times,
                description="Index flexion position, from fully extended (0) to fully flexed (1)",
                comments="Processed position data (averaged over 20 ms bins)",
            )
            mrs_position_ts = TimeSeries(
                name="mrs_position",
                data=mrs_position[:, None],
                unit="flexion units",  # From 0 to 1
                timestamps=times,
                description="Middle-ring-pinky flexion position, from fully extended (0) to fully flexed (1)",
                comments="Processed position data (averaged over 20 ms bins)",
            )

            # Add positions to analysis
            nwbfile.add_analysis(index_position_ts)
            nwbfile.add_analysis(mrs_position_ts)
            # Add positions to the behavior module
            # behavior_module.add_data_interface(index_position_ts)
            # behavior_module.add_data_interface(mrs_position_ts)

            # Create TimeSeries objects for velocities
            index_velocity_ts = TimeSeries(
                name="index_velocity",
                data=index_velocity[:, None],
                unit="flexion units per time bin",  # Units per 20 ms
                timestamps=times,
                description="Velocity of index flexion",
            )
            mrs_velocity_ts = TimeSeries(
                name="mrs_velocity",
                data=mrs_velocity[:, None],
                unit="flexion units per time bin",  # Units per 20 ms
                timestamps=times,
                description="Velocity of middle-ring-pinky flexion",
            )
            # Add velocities to analysis
            nwbfile.add_analysis(index_velocity_ts)
            nwbfile.add_analysis(mrs_velocity_ts)

            # Add velocities to the behavior module
            # behavior_module.add_data_interface(index_velocity_ts)
            # behavior_module.add_data_interface(mrs_velocity_ts)

            # Process neural data
            # SBP channels
            sbp_channels = [
                col for col in ts_df.columns if col.startswith("sbp_channel")
            ]
            sbp_data = ts_df[
                sbp_channels
            ].values  # Shape: (num_timepoints, num_channels)

            # TCFR channels
            tcfr_channels = [
                col for col in ts_df.columns if col.startswith("tcfr_channel")
            ]
            tcfr_data = ts_df[
                tcfr_channels
            ].values  # Shape: (num_timepoints, num_channels)

            # Create the ecephys processing module
            # ecephys_module = nwbfile.create_processing_module(
            #     name="ecephys", description="Binned (20ms) and processed neural data"
            # )
            elec_region = nwbfile.create_electrode_table_region(  # :contentReference[oaicite:1]{index=1}
                region=list(range(sbp_data.shape[1])),  # [0 … 95] in sheet order
                description="All recorded contacts",
            )

            # Create TimeSeries for SpikingBandPower
            sbp_timeseries = ElectricalSeries(
                name="SpikingBandPower",
                data=sbp_data * 0.25,  # Scaling to convert to microvolts
                timestamps=times,
                conversion=1e-6,  # measured in microvolts
                description="Spiking Band Power across time, in 20ms bins",
                electrodes=elec_region,
            )

            # Create TimeSeries for ThresholdCrossings
            tcfr_timeseries = ElectricalSeries(
                name="ThresholdCrossings",
                data=tcfr_data,
                electrodes=elec_region,
                timestamps=times,
                description="Threshold crossings (threshold = -4.5RMS) across time, in 20ms bins",
            )
            # Add sbp and tcfr to analysis
            nwbfile.add_analysis(sbp_timeseries)
            nwbfile.add_analysis(tcfr_timeseries)

            # Add TimeSeries to the ecephys processing module
            # ecephys_module.add_data_interface(sbp_timeseries)
            # ecephys_module.add_data_interface(tcfr_timeseries)

            # Add trials with start and stop times
            # Add custom columns to the trials table
            nwbfile.add_trial_column(name="trial_number", description="Trial number")
            nwbfile.add_trial_column(name="trial_count", description="Trial count")
            nwbfile.add_trial_column(name="run_id", description="Run ID")
            nwbfile.add_trial_column(
                name="index_target_position", description="Index target position"
            )
            nwbfile.add_trial_column(
                name="mrs_target_position", description="MRS target position"
            )
            nwbfile.add_trial_column(
                name="target_style", description="Target style (CO or RD)"
            )
            nwbfile.add_trial_column(
                name="trial_timeout", description="Trial timeout (milliseconds)"
            )

            unique_trial_numbers = trials_df["trial_number"].unique()

            for trial_number in unique_trial_numbers:
                trial_indices = ts_df["trial_number"] == trial_number
                trial_times = times[trial_indices]
                if len(trial_times) == 0:
                    print(
                        f"No timeseries data found for trial {trial_number}. Skipping."
                    )
                    continue
                start_time = trial_times[0]
                stop_time = trial_times[-1]
                # Get trial metadata
                trial_meta = trials_df[trials_df["trial_number"] == trial_number].iloc[
                    0
                ]
                # Add trial with metadata
                nwbfile.add_trial(
                    start_time=start_time,
                    stop_time=stop_time,
                    trial_number=trial_meta["trial_number"],
                    trial_count=trial_meta["trial_count"],
                    run_id=trial_meta["run_id"],
                    index_target_position=trial_meta["index_target_position"],
                    mrs_target_position=trial_meta["mrs_target_position"],
                    target_style=trial_meta["target_style"],
                    trial_timeout=trial_meta["trial_timeout"],
                    timeseries=[
                        index_position_ts,
                        mrs_position_ts,
                        index_velocity_ts,
                        mrs_velocity_ts,
                        sbp_timeseries,
                        tcfr_timeseries,
                    ],
                )

            # Write the NWB file
            output_filename = os.path.join(
                end_dir, f"{date}_{trials_df['target_style'][0]}.nwb"
            )
            with NWBHDF5IO(output_filename, "w") as io:
                io.write(nwbfile)
            print(
                f"NWB file for date {date} and target style '{trials_df['target_style'][0]}' saved as {output_filename}"
            )





if __name__ == "__main__":
    convert_pkl_to_nwb(
        config.good_daysdir,
        os.path.join(os.path.dirname(__file__), "../channel_map.csv"),
        end_dir=config.nwbdir,
    )
    # dicts = dicts_from_pickle(
    #     f"{config.good_daysdir}/2020-01-27_CO.nwb",
    #     config.nwbdir,
    # )
    # print(dicts)
