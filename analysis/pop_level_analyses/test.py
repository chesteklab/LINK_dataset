import os
from population_level_analyses import *
# import matplotlib.font_manager as fm
# font_path = "/home/riopar/.local/share/fonts/Atkinson-Hyperlegible-Regular-102.ttf"

# # Register it
# fm.fontManager.addfont(font_path)
# prop = fm.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = prop.get_name()
mpath = os.path.join("..", "AdaptiveAlignment", "data", "hisham_good_days")
results = load_all_datasets(mpath, 303)

### LOAD AND PREPROCESS DATA ###
df_tuning = prepare_tuning_data(results)
plot_avg_trajectories(df_tuning, type_of_data='sbps', group_by = 'year', trim_method = trim_neural_data_at_movement_onset_std_and_smooth, trim_pt = max_jerk, sigma = .5, years_to_skip=[2021, 2022], directions='ext_flex', remove_RT=False)
# plot_centroid_of_pca_data_across_time(df_tuning, group_by='quarter', remove_RT=False, normalization_method = 'all', years_to_skip = [], plot_centr_across_time=True)

plt.show()

