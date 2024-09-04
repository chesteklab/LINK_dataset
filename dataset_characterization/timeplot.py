import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

filepath = 'C:\Repos\\nhp_time_dataset_processing\preprocessing_results.csv'
df = pd.read_csv(filepath)

dates = df['Date'].to_numpy()
dates = np.asarray([datetime.strptime(x, '%Y-%m-%d') for x in dates])

fig, ax = plt.subplots(figsize=(8.8, 4), layout='constrained')
ax.vlines(dates, .5, '.')
ax.axhline(0, c='black')

fig, ax = plt.subplots(figsize=(10, 1), layout='constrained')
ax.hist(dates, bins=100, color='k')
ax.spines[['right','top','left']].set_visible(False)
plt.show()

