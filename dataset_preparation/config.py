import os

# get cwd for making the folders - so that all the data should generate/get saved into the right places
cwd = os.getcwd()

server_dir = "/run/user/1000/gvfs/smb-share:server=cnpl-drmanhattan.engin.umich.edu,share=share/"
#server_dir = "Z:\"

#pybmipath = 'C:\Repos\pybmi'
pybmipath = '/home/chesteklab/Repos/pybmi'
datapath = os.path.join(server_dir, 'Data', 'Monkeys')
outputdir = os.path.join(server_dir, 'Student Folders', 'Hisham_Temmar', 'big_dataset')

firstpassdir = os.path.join(outputdir, "1_notes_data_pruning")
notesdir = os.path.join(firstpassdir,'joker_notes_firstpass')
sheetpath = os.path.join(firstpassdir, 'firstpass_datesruns.csv')
# sheetpath = os.path.join(firstpassdir, 'sfn_datasets.csv')
dfpath = os.path.join(firstpassdir, 'firstpass_datesruns.pkl')
# dfpath = os.path.join(firstpassdir, 'sfn_datasets.pkl')

preprocessingdir = os.path.join(server_dir, 'Student Folders', 'Nina_Gill', 'data', 'adjusted_times')

datareviewdir = "Z:\Student Folders\Hisham_Temmar\\big_dataset\\3_data_review_results"
reviewpath = os.path.join(datareviewdir, 'review_results_test.csv')
savestatepath = os.path.join(datareviewdir, 'savestate.pkl')

#data review config and params
data_params = {
    'monkey':'Joker',
    'binsize':20,
    'cutoff':400,
    'trim':5
}

good_chans = list(range(1, 97))
good_chans_indexed = [chan - 1 for chan in good_chans]

