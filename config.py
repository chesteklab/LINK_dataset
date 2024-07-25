import os

# get cwd for making the folders - so that all the data should generate/get saved into the right places
cwd = os.getcwd()
notesdir = os.path.join(cwd,'joker_notes_firstpass')
sheetpath = os.path.join(cwd, 'firstpass_datesruns.csv')
dfpath = os.path.join(cwd, 'firstpass_datesruns.pkl')
datapath = 'Z:\Data\Monkeys'
dataoutpath = os.path.join(cwd,'review_output')
resultspath = os.path.join(cwd, 'preprocessing_results.csv')
savestatepath = os.path.join(cwd, 'savestate.pkl')
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

