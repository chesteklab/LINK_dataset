import os
import re
from datetime import datetime
import numpy as np
from pynwb import NWBFile, TimeSeries, NWBHDF5IO
from collections import defaultdict
import pickle
from tqdm import tqdm

def load_one_nwb(fp: str) -> dict:
    with NWBHDF5IO(fp, "r", load_namespaces=True) as io:
        nwb = io.read()
        ana = nwb.analysis  # TimeSeries are stored only here

        # helper → fetch from /analysis or raise clear error
        def ts(name: str):
            try:
                return ana[name]
            except KeyError as e:
                raise KeyError(
                    f"TimeSeries '{name}' not in /analysis of '{fp}'"
                ) from e

        # --- time and kinematics ---
        t_sec = ts("index_position").timestamps[:]
        time_ms = (np.round(t_sec * 1000)).astype(np.int32)

        finger_kinematics = np.column_stack(
            [
                ts(n).data[:].ravel()
                for n in (
                    "index_position",
                    "mrs_position",
                    "index_velocity",
                    "mrs_velocity",
                )
            ]
        ).astype(np.float64)

        # --- neural features ---
        sbp = (ts("SpikingBandPower").data[:] / 0.25).astype(np.float64)
        tcfr = ts("ThresholdCrossings").data[:].astype(np.int16)

        # --- trial info ---
        tr = nwb.trials.to_dataframe()
        trial_number = tr["trial_number"].to_numpy()
        trial_count = tr["trial_count"].to_numpy()
        target_positions = tr[
            ["index_target_position", "mrs_target_position"]
        ].to_numpy()
        run_id = tr["run_id"].iloc[0]
        target_style = tr["target_style"]
        trial_timeout = tr["trial_timeout"]

        trial_index = np.searchsorted(t_sec, tr["start_time"].to_numpy()).astype(
            np.int32
        )

        return dict(
            trial_number=trial_number,
            trial_count=trial_count,
            target_positions=target_positions,
            trial_timeout=trial_timeout,
            time=time_ms,
            finger_kinematics=finger_kinematics,
            sbp=sbp,
            tcfr=tcfr,
            trial_index=trial_index,
            target_style=target_style[0],
            run_id=np.full_like(trial_number, run_id),
        )

if __name__=="__main__":
    dandiset_path = '/home/chang/Documents/ND/github/LINK_dataset/001201'
    nwb_path = os.path.join(dandiset_path, 'sub-Monkey-N')

    processed_data_path = '/home/chang/Documents/ND/github/LINK_dataset/data_test'

    # get filenames
    files = os.listdir(nwb_path)
    files = [f for f in files if os.path.isfile(os.path.join(nwb_path, f))]

    date_to_entries = defaultdict(list)
    for file in tqdm(files):
        data = load_one_nwb(os.path.join(nwb_path,file))
        match = re.search(r'\d{8}', file)
        if not match:
            raise ValueError(f"No valid date found in filename: {file}")
        
        raw_date = match.group()
    
        # Convert to datetime and reformat
        date_obj = datetime.strptime(raw_date, '%Y%m%d')
        date = date_obj.strftime('%Y-%m-%d')

        date_to_entries[date].append(data)

    for date, entries in tqdm(date_to_entries.items()):
        if len(entries) > 1:
            if entries[0]['target_style'] == 'CO':
                output = (entries[0], entries[1])
            elif entries[0]['target_style'] == 'RD':
                output = (entries[1], entries[0])
            else:
                raise ValueError("Incorrect target style found")
        else:
            data = entries[0]
            if data['target_style'] == 'CO':
                output = (data, None)
            elif data['target_style'] == 'RD':
                output = (None, data)
            else:
                raise ValueError("Incorrect target style found")
    
        with open(os.path.join(processed_data_path, f'{date}_preprocess.pkl'), 'wb') as f:
            pickle.dump(output, f)