import os

# get cwd for making the folders - so that all the data should generate/get saved into the right places
cwd = os.getcwd()

filedir = "Z:\Student Folders\Hisham_Temmar\\big_dataset"
datasetdir = os.path.join(filedir,"4_preprocesses_32ms")
reviewpath = os.path.join(filedir,"3_data_review_results","bianca_review_results.csv")

outputdir = os.path.join(filedir, "dataset_characterization")

dataset_version = "0.2"