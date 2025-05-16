## Functions to find nearest impedance file for each day in the dataset - Aren Hite, 2025
from shutil import copy
from datetime import datetime, timedelta
import os
import re
import numpy as np
from pathlib import Path

_LINE_RE = re.compile(r"chan(\d+)\s+([0-9]*\.?[0-9]+)\s+(kOhm|uV)", re.IGNORECASE)


def copy_imp_files(
    start_date,
    end_date,
    data_path=r"Z:\Data\Monkeys\Joker",
    output_path=r"Z:\Student Folders\Nina_Gill\data\impedances",
):
    """
    Finds and copies all impedance files in the range [start_date, end_date] from Joker's data
    into the output path.

    Inputs:
    - start_date (str): earliest date to look for impedance files, formatted "YYYY-MM-DD"
    - end_date (str): last date to look for impedance files, formatted "YYYY-MM-DD"
    - data_path (str): path to Joker's data on the server
    - output_path (str, optional): folder to copy impedance files into, default is Nina's
        impedance folder in the server
    """
    # convert dates to datetime objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # iterate through folders in joker's data
    for folder_name in os.listdir(data_path):
        try:
            # convert folder title to a datetime object so it can be compared
            folder_date = datetime.strptime(folder_name, "%Y-%m-%d")
            if start_date <= folder_date <= end_date:
                # folder is int date range - search for impedance file
                folder_path = os.path.join(data_path, folder_name)
                for filename in os.listdir(folder_path):
                    if (
                        "impedance" in filename or "Impedance" in filename
                    ) and os.path.isfile(os.path.join(folder_path, filename)):
                        # found impedance file
                        imp_path = os.path.join(folder_path, filename)
                        # search the file for its real date
                        name_date = None
                        with open(imp_path, "r") as file:
                            for line in file:
                                match = re.search(
                                    r"\* (\d{2} \w+ \d{4} \d{2}:\d{2}:\d{2})", line
                                )
                                # found the date line: save the date and break
                                if match:
                                    extracted_date = match.group(1)
                                    name_date = datetime.strptime(
                                        extracted_date, "%d %B %Y %H:%M:%S"
                                    )
                                    break
                        if name_date is None:
                            # couldn't find date in the file, using folder date instead
                            name_date = folder_date
                        # rename file
                        is_sensory = "Sensory" in filename or "sensory" in filename
                        new_filename = f"{name_date.strftime('%Y-%m-%d')}_{'Sensory' if is_sensory else 'Motor'}.txt"
                        file_loc = os.path.join(output_path, new_filename)
                        copy(imp_path, file_loc)
                        # print(
                        #     f"Found impedance in folder {folder_name}: {new_filename}"
                        # )
        except ValueError:
            # skip any folder with improper date formatting
            print(f"Skipping folder {folder_name} due to improper date formatting")
            continue


def find_nearest_imp(
    date, imp_path=r"Z:\Student Folders\\Nina_Gill\data\\impedances", imp_type="Motor"
):
    """
    Finds the nearest impedance recording to the date and returns its filename,
    as found in imp_path

    Inputs:
        - date (str): Date to find the nearest impedance from, formatted as "YYYY-MM-DD"
        - imp_path (str, optional): folder with impedance data to search for the nearest date,
            default is the impedance folder in Nina's student folder
        - imp_type (str, optional):  which brain location to return impedance data from. options are
        "Both", "Motor", or "Sensory". If "motor" or "sensory" is specified, the nearest motor or sensory
        impedance is returned. Otherwise, both nearest files will be returned by default

    Outputs: filename of the chronologically nearest sensory and/or motor impedance data
    """
    # verify input argument
    if imp_type != "Both" and imp_type != "Sensory" and imp_type != "Motor":
        raise Exception(
            f"Invalid input argument for imp_type: {imp_type}. Valid options are Both, Motor, or Sensory"
        )
    # extract list of impedance filenames
    all_files = os.listdir(imp_path)
    # separate motor and sensory lists
    sensory_files = []
    motor_files = []
    for filename in all_files:
        if "Sensory" in filename:
            sensory_files.append(filename)
        if "Motor" in filename:
            motor_files.append(filename)
    # find nearest motor and/or sensory impedance
    if imp_type == "Motor":
        return search_date(date, motor_files)
    elif imp_type == "Sensory":
        return search_date(date, sensory_files)
    else:
        nearest_motor = search_date(date, motor_files)
        nearest_sensory = search_date(date, sensory_files)
        return nearest_motor, nearest_sensory


def _read_impedances(file_path) -> np.ndarray:
    """
    Parse *AutoImpedance* text file and return a vector (index 0 → chan1)
    with impedances in kΩ.  Channels whose value is not reported in kΩ are
    set to `np.nan`.

    Parameters
    ----------
    file_path : str | Path
        Path to the *.txt* file produced by Blackrock/NeuroPort Auto-Impedance.
    max_chans : int, optional
        Maximum number of channels to read.

    Returns
    -------
    np.ndarray
        1-D float array where index i corresponds to chan(i+1).
    """
    file_path = Path(file_path)
    impedances: dict[int, float] = {}

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = _LINE_RE.search(line)
            if not m:
                continue

            chan = int(m.group(1))
            value = float(m.group(2))
            unit = m.group(3).lower()

            # Only treat kOhm entries as impedance; everything else → NaN
            impedances[chan] = value if unit == "kohm" else np.nan

    if not impedances:
        raise ValueError("No channel lines found in the file.")

    return impedances


def search_date(date, files, max_allowed_diff=timedelta(weeks=3)):
    """
    Helper function for find_nearest_imp(). Returns the filename of the file closest to
    date in files, or None if no file is within max_allowed_diff.

    Parameters
    ----------
    date : str
        Date to find the nearest file for, formatted "YYYY-MM-DD".
    files : list of str
        List of files to search for the nearest date within. Each file
        should begin with the date in the format "YYYY-MM-DD".
    max_allowed_diff : timedelta, optional
        Maximum allowed time difference between the target date and the file date.
        Defaults to 3 weeks.
    """
    if not files:
        return None

    # sort file list
    files.sort()
    # convert date to datetime object
    date = datetime.strptime(date, "%Y-%m-%d")

    # Find the first file with date >= target date
    index = 0
    while index < len(files):
        file_date = datetime.strptime(files[index][0:10], "%Y-%m-%d")
        if file_date >= date:
            break
        index += 1

    # Find the closest file
    candidates = []
    # Check the file just before the target date (if it exists)
    if index > 0:
        candidates.append(files[index - 1])
    # Check the file at or just after the target date (if it exists)
    if index < len(files):
        candidates.append(files[index])

    # Find the closest among the candidates
    min_diff = timedelta.max
    nearest_file = None
    for imp_filename in candidates:
        file_date = datetime.strptime(imp_filename[0:10], "%Y-%m-%d")
        diff = abs(file_date - date)
        if diff < min_diff:
            min_diff = diff
            nearest_file = imp_filename

    # Return None if the closest file is more than 3 weeks away
    if min_diff > max_allowed_diff:
        return None

    return nearest_file


def get_impedances(
    date: str,
    imp_type: str = "Motor",
    impedance_path: str = "Z:/Student Folders/Nina_Gill/data/impedances",
) -> np.ndarray:
    """
    Get the impedances from a given date and impedance type.
    """
    imp_file = find_nearest_imp(date, imp_type=imp_type)
    if imp_file is None:
        print(f"No impedance file found for {date} and {imp_type}")
        return None
    return _read_impedances(os.path.join(impedance_path, imp_file))


if __name__ == "__main__":
    # run some test code demonstrating the functions

    # start = "2020-01-01"
    # end = "2023-12-31"
    # copy_imp_files(start, end)
    test_date = "2023-04-19"
    motor_impedances = get_impedances(test_date, imp_type="Motor")
    print(motor_impedances)
