import os

# get cwd for making the folders - so that all the data should generate/get saved into the right places
cwd = os.getcwd()
firstpassdir = "Z:\Student Folders\Hisham_Temmar\\big_dataset\\1_notes_data_pruning"
notesdir = os.path.join(firstpassdir,'joker_notes_firstpass')
sheetpath = os.path.join(firstpassdir, 'firstpass_datesruns.csv')
dfpath = os.path.join(firstpassdir, 'firstpass_datesruns.pkl')

datapath = 'Z:\Data\Monkeys'

datareviewpath = "Z:\Student Folders\Hisham_Temmar\\big_dataset\\3_Data_Review_Results"
resultspath = os.path.join(datareviewpath, 'preprocessing_results.csv')
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

