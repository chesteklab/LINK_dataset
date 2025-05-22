#!/bin/bash

read -p "Enter the name of the conda environment: " env_name
echo "Creating a new conda environment '$env_name' with Python 3.9."
conda create -n "$env_name" python=3.9 -y

echo "Activating the conda environment."
conda activate "$env_name"  

echo "Installing dependencies from requirements.txt."
conda install --file requirements.txt -y

echo "Installing pyNWB."
pip install -U pynwb

echo "Installing ipympl."
conda install anaconda::ipympl

echo "Installing dandi-cli."
pip install dandi

echo "Downloading dataset with dandi."
dandi download DANDI:001201

