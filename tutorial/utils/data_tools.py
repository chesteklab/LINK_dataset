import numpy as np
import pandas as pd

def load_data(data_path, file_name):
    """
    Load data from a CSV file.
    
    Parameters:
    - data_path: str, path to the directory containing the data file
    - file_name: str, name of the CSV file to load
    
    Returns:
    - DataFrame: loaded data
    """
    return pd.read_csv(f"{data_path}/{file_name}")