
import h5py
import numpy as np
import os
import sys
import data_utils
import scipy.io as sio
# sys.path.append('/Users/yixuan/Documents/GitHub/pybmi')
# from pybmi.utils.ZTools import ZStructTranslator, zarray
# from pybmi.utils import ZTools, spikePlot_JC, AnalysisTools
# from pybmi.utils.ZTools import ZStructTranslator,zarray
# from pybmi.utils.AnalysisTools import calcBitRate

import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.max_columns = None
from ipywidgets import Text, IntSlider, VBox, HBox, Button, RadioButtons, Textarea, Output
from IPython.display import display, clear_output
plt.ioff()
import interactive_plot
import pickle
import math
import time

data_utils.prep_data_for_plotting()