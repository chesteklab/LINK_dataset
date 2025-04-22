import os

pybmipath = 'C:\Repos\pybmi'
datapath = 'Z:\Data\Monkeys'
outputdir = "Z:\Student Folders\Hisham_Temmar\\big_dataset"

firstpassdir = os.path.join(outputdir, "1_notes_data_pruning")
notesdir = os.path.join(firstpassdir,'joker_notes_firstpass')
sheetpath = os.path.join(firstpassdir, 'firstpass_datesruns.csv')
# sheetpath = os.path.join(firstpassdir, 'sfn_datasets.csv')
dfpath = os.path.join(firstpassdir, 'firstpass_datesruns.pkl')
# dfpath = os.path.join(firstpassdir, 'sfn_datasets.pkl')

preprocessingdir = os.path.join("Z:\Student Folders\Hisham_Temmar\\big_dataset\\2_autotrimming_and_preprocessing","preprocessing_092024_no7822nofalcon")
# preprocessingpath = os.path.join("Z:\Student Folders\Hisham_Temmar\\big_dataset\\sfn_round_2")

datareviewdir = "Z:\Student Folders\Hisham_Temmar\\big_dataset\\3_data_review_results"
reviewpath = os.path.join(datareviewdir, 'review_results.csv')
savestatepath = os.path.join(datareviewdir, 'savestate.pkl')

characterizationdir = os.path.join(outputdir, "dataset_characterization")


#data review config and params
dataset_version = "0.3"
data_params = {
    'monkey':'Joker',
    'binsize':20,
    'cutoff':400,
    'trim':5
}

good_chans = list(range(1, 97))
good_chans_indexed = [chan - 1 for chan in good_chans]

