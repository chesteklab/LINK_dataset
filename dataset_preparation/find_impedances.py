## Functions to find nearest impedance file for each day in the dataset - Aren Hite, 2025
from shutil import copy
from datetime import datetime, timedelta
import os
import re

def copy_imp_files(start_date, end_date, data_path=r"Z:\Data\Monkeys\Joker", output_path=r"Z:\Student Folders\Nina_Gill\data\impedances"):
    '''
    Finds and copies all impedance files in the range [start_date, end_date] from Joker's data
    into the output path. 

    Input:
    - start_date (str): earliest date to look for impedance files, formatted "YYYY-MM-DD"
    - end_date (str): last date to look for impedacne files, formatted "YYYY-MM-DD"
    - data_path (str): path to Joker's data on the server 
    - output_path (str, optional): folder to copy impedance files into, default is Nina's 
        impedance folder in the server
    '''
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
                    if "impedance" in filename or "Impedance" in filename:
                        # found impedance file
                        imp_path = os.path.join(folder_path, filename)
                        # search the file for its real date
                        name_date = None
                        with open(imp_path, 'r') as file:
                            for line in file:
                                match = re.search(r'\* (\d{2} \w+ \d{4} \d{2}:\d{2}:\d{2})', line)
                                # found the date line: save the date and break
                                if match:
                                    extracted_date = match.group(1)
                                    name_date = datetime.strptime(extracted_date, "%d %B %Y %H:%M:%S")
                                    break
                        if name_date == None:
                            # couldn't find date in the file, using folder date instead
                            name_date = folder_date
                        # rename file
                        new_filename = filename.replace(filename[0:9], f"{name_date.strftime('%Y-%m-%d')}_")
                        # add .txt if it's not on there
                        if new_filename[-4:] != ".txt":
                            new_filename = f"{new_filename}.txt"
                        file_loc = os.path.join(output_path, new_filename)
                        copy(imp_path, file_loc)
                        print(f"Found impedance in folder {folder_name}: {new_filename}")
        except ValueError:
            # skip any folder with improper date formatting
            continue

def find_nearest_imp(date, imp_path=r"Z:\Student Folders\Nina_Gill\data\impedances", imp_type="Both"):
    '''
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
    '''
    # verify input argument
    if imp_type != "Both" and imp_type != "Sensory" and imp_type != "Motor":
        raise Exception(f"Invalid input argument for imp_type: {imp_type}. Valid options are Both, Motor, or Sensory")
    # extract list of impedance filenames
    all_files = os.listdir(imp_path)
    # separate motor and sensory lists
    sensory_files = []
    motor_files = []
    for filename in all_files:
        if "Sensory" in filename: sensory_files.append(filename)
        if "Motor" in filename: motor_files.append(filename)
    # find nearest motor and/or sensory impedance
    if imp_type == "Motor":
        return search_date(date, motor_files)
    elif imp_type == "Sensory":
        return search_date(date, sensory_files)
    else: 
        nearest_motor = search_date(date, motor_files)
        nearest_sensory = search_date(date, sensory_files)
        return nearest_motor, nearest_sensory

def search_date(date, files):
    '''
    Helper function for find_nearest_imp(). Returns the filename of the file closest to 
    date in files. 

    Inputs: 
        - date (str): date to find the nearest file for, formatted "YYYY-MM-DD"
        - files (str): list of files to search for the nearest date within. Each file 
            should begin with the date in the format "YYYYMMDD"
    '''
    # sort file list
    files.sort()
    # convert date to datetime object
    date = datetime.strptime(date, "%Y-%m-%d")
    min_diff = timedelta.max
    nearest_file = None
    for imp_filename in files:
        file_date = datetime.strptime(imp_filename[0:10], "%Y-%m-%d")
        diff = abs(file_date - date)
        if diff < min_diff: 
            min_diff = diff
            nearest_file = imp_filename
        if diff > min_diff:
            break
    return nearest_file

if __name__ == "__main__":
    # run some test code demonstrating the functions
    import dataset_preparation.find_impedances as imp
    start = "2023-01-10"
    end = "2023-06-06"
    data_path = "/Volumes/share/Data/Monkeys/Joker"
    out_path = "/Volumes/share/Student Folders/Nina_Gill/data/impedances"
    imp.copy_imp_files(start, end, data_path=data_path, output_path=out_path)
    test_date = "2023-04-19"
    motor_imp, sensory_imp = imp.find_nearest_imp(test_date, out_path, imp_type="Both")
    print(f"{test_date} motor impedance file: {motor_imp}\n{test_date} sensory impedance file: {sensory_imp}")