import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

########################################################
########################################################

# Load the evaluation results
results_folder = '/home/joey/University of Michigan Dropbox/Joseph Costello/Chestek Lab/Code/NeuralNet/Temmar2025BigDataAnalysis/LINK_dataset/analysis/bci_decoding/single_day_model_results'
# results_folder = '/Users/jcostello/University of Michigan Dropbox/Joseph Costello/Chestek Lab/Code/NeuralNet/Temmar2025BigDataAnalysis/LINK_dataset/analysis/bci_decoding/single_day_model_results'


# full dataset, training with noise
# rr_df = pd.read_csv(f'{results_folder}/rr_evaluation_20250507_gooddays.csv')
# lstm_df = pd.read_csv(f'{results_folder}/lstm_evaluation_20250507_gooddays.csv')
# out_fname = 'model_performance_plots_withnoise.png'

# # updated norm, full dataset
rr_df = pd.read_csv(f'{results_folder}/rr_evaluation_updatednorm_20250508_gooddays.csv')
lstm_df = pd.read_csv(f'{results_folder}/lstm_evaluation_updatednorm_20250507_gooddays.csv')
out_fname = 'model_performance_plots_withnoise_updatenorm.png'


XLIM_ONESIDE = 100

########################################################
########################################################

print("Data loaded, RR shape:", rr_df.shape, "LSTM shape:", lstm_df.shape)
print("\nFirst few rows of RR data:")
print(rr_df.head())
print("\nFirst few rows of LSTM data:")
print(lstm_df.head())

# Pre-aggregate the data
rr_grouped = rr_df.groupby('Day_diff').agg({
    'MSE': ['mean', 'std', 'count'],
    'Correlation': ['mean', 'std', 'count'],
    'R2': ['mean', 'std', 'count']
}).reset_index()

lstm_grouped = lstm_df.groupby('Day_diff').agg({
    'MSE': ['mean', 'std', 'count'],
    'Correlation': ['mean', 'std', 'count'],
    'R2': ['mean', 'std', 'count']
}).reset_index()

print("\nAggregated data shapes - RR:", rr_grouped.shape, "LSTM:", lstm_grouped.shape)

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Function to plot performance metric
def plot_performance(ax, metric, title, ylabel, xlim_oneside=XLIM_ONESIDE, error_type='sd'):
    # Plot RR data
    x_rr = rr_grouped['Day_diff']
    y_mean_rr = rr_grouped[(metric, 'mean')]
    y_std_rr = rr_grouped[(metric, 'std')]
    if error_type.lower() == 'se':
        # Standard error = standard deviation / sqrt(n)
        counts_rr = rr_grouped[(metric, 'count')]
        y_error_rr = y_std_rr / np.sqrt(counts_rr)
        error_label = 'Standard Error'
    else:  # Default to standard deviation
        y_error_rr = y_std_rr
        error_label = 'Standard Deviation'
    
    # Plot mean and shaded error region for RR
    ax.plot(x_rr, y_mean_rr, 'b-', linewidth=2, label='Ridge Regression')
    ax.fill_between(x_rr, y_mean_rr - y_error_rr, y_mean_rr + y_error_rr, 
                    color='blue', alpha=0.2)
    
    # Plot LSTM data
    x_lstm = lstm_grouped['Day_diff']
    y_mean_lstm = lstm_grouped[(metric, 'mean')]
    y_std_lstm = lstm_grouped[(metric, 'std')]
    if error_type.lower() == 'se':
        counts_lstm = lstm_grouped[(metric, 'count')]
        y_error_lstm = y_std_lstm / np.sqrt(counts_lstm)
    else:
        y_error_lstm = y_std_lstm
    
    # Plot mean and shaded error region for LSTM
    ax.plot(x_lstm, y_mean_lstm, 'r-', linewidth=2, label='LSTM')
    ax.fill_between(x_lstm, y_mean_lstm - y_error_lstm, y_mean_lstm + y_error_lstm, 
                    color='red', alpha=0.2)
    
    # set plot parameters
    ax.set_xlabel('Days from Training')
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n(±{error_label})")
    ax.set_xlim(-xlim_oneside, xlim_oneside)
    
    # Set y-axis limits based on the metric
    if metric == 'MSE':
        # log scale y
        # ax.set_yscale('log')
        # ax.set_ylim(10e-4, 10e1)
        ax.set_ylim(0, 0.06)
    elif metric == 'Correlation':
        ax.set_ylim(0, 0.9)
    elif metric == 'R2':
        ax.set_ylim(-1, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend()

# Choose error type: 'sd' for standard deviation or 'se' for standard error
error_type = 'se'  # Change to 'se' for standard error

plot_performance(ax1, 'MSE', 'Mean Squared Error vs Days from Training', 'MSE', error_type=error_type)
plot_performance(ax2, 'Correlation', 'Correlation vs Days from Training', 'Correlation', error_type=error_type)
plot_performance(ax3, 'R2', 'R² vs Days from Training', 'R²', error_type=error_type)
plt.tight_layout()
plt.savefig(f'{results_folder}/{out_fname}', dpi=300, bbox_inches='tight')
plt.close() 

print("\nPlot saved with both RR and LSTM data")