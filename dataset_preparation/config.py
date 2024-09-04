import os

# get cwd for making the folders - so that all the data should generate/get saved into the right places
cwd = os.getcwd()
firstpassdir = "Z:\Student Folders\Hisham_Temmar\\big_dataset\\1_notes_data_pruning"
notesdir = os.path.join(firstpassdir,'joker_notes_firstpass')
sheetpath = os.path.join(firstpassdir, 'firstpass_datesruns.csv')
dfpath = os.path.join(firstpassdir, 'firstpass_datesruns.pkl')

datapath = 'Z:\Data\Monkeys'

preprocessingpath = os.path.join("Z:\Student Folders\Hisham_Temmar\\big_dataset\\2_autotrimming_and_preprocessing","preprocessing")

datareviewpath = "Z:\Student Folders\Hisham_Temmar\\big_dataset\\3_data_review_results"
resultspath = os.path.join(datareviewpath, 'review_results.csv')
savestatepath = os.path.join(datareviewpath, 'savestate.pkl')

pybmipath = 'C:\Repos\pybmi'



#data review config and params
data_params = {
    'monkey':'Joker',
    'binsize':10,
    'cutoff':400,
    'trim':5
}

good_chans = list(range(1, 97))
good_chans_indexed = [chan - 1 for chan in good_chans]

