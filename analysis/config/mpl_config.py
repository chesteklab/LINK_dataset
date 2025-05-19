# import os

# pybmipath = 'C:\Repos\pybmi'
# datapath = 'Z:\Data\Monkeys'
# outputdir = "C:\\Files\\UM\\ND\\SFN"

# firstpassdir = os.path.join(outputdir, "1_notes_data_pruning")
# notesdir = os.path.join(firstpassdir,'joker_notes_firstpass')
# sheetpath = os.path.join(firstpassdir, 'firstpass_datesruns.csv')
# # sheetpath = os.path.join(firstpassdir, 'sfn_datasets.csv')
# dfpath = os.path.join(firstpassdir, 'firstpass_datesruns.pkl')
# # dfpath = os.path.join(firstpassdir, 'sfn_datasets.pkl')

# preprocessingdir = os.path.join("C:\\Files\\UM\\ND\\SFN","preprocessing_092024_no7822nofalcon")
# # preprocessingpath = os.path.join("Z:\Student Folders\Hisham_Temmar\\big_dataset\\sfn_round_2")

# datareviewdir = "Z:\Student Folders\Hisham_Temmar\\big_dataset\\3_data_review_results"
# reviewpath = os.path.join(datareviewdir, 'review_results.csv')
# savestatepath = os.path.join(datareviewdir, 'savestate.pkl')

# characterizationdir = os.path.join(outputdir, "dataset_characterization")


# #data review config and params
# dataset_version = "0.3"
# data_params = {
#     'monkey':'Joker',
#     'binsize':20,
#     'cutoff':400,
#     'trim':5
# }

# good_chans = list(range(1, 97))
# good_chans_indexed = [chan - 1 for chan in good_chans]

import matplotlib as mpl
#some basic text parameters for figures
#mpl.rcParams['font.family'] = "Atkinson Hyperlegible" # if installed but not showing up, rebuild mpl cache
mpl.rcParams['font.size'] = 10
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlelocation'] = 'center'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['figure.constrained_layout.use'] = True
mpl.rcParams['figure.titlesize'] = 14
mpl.rcParams['figure.titleweight'] = 'bold'
mpl.rcParams['pdf.fonttype'] = 42

