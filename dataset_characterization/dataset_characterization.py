import os
import pandas as pd
import numpy as np
from datetime import datetime
import config
import matplotlib.pyplot as plt

# filter out bad days
review_results = pd.read_csv(config.reviewpath)

good_days = review_results.loc[review_results['Status'] == 'good']

#create histogram for current version (fewer days I guess?)
dates = good_days['Date'].to_numpy()
dates = np.asarray([datetime.strptime(x, '%Y-%m-%d') for x in dates])

fig, ax = plt.subplots(figsize=(10, 1), layout='constrained', dpi=1200)
ax.hist(dates, bins=100, color='k')
ax.spines[['right','top','left']].set_visible(False)
ax.set(yticks=(0,5,10),title="Distribution of days included in v0.2",ylabel="# of days")
ax.set_axisbelow(True)
ax.grid(axis="y",zorder=10)

fig.savefig(os.path.join(config.outputdir,f'dataset_timeline_v02.pdf'))
