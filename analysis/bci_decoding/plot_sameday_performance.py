import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime


########################################################
########################################################
# Set path to results folder
results_folder = '/home/joey/University of Michigan Dropbox/Joseph Costello/Chestek Lab/Code/NeuralNet/Temmar2025BigDataAnalysis/LINK_dataset/analysis/bci_decoding/single_day_model_results'
# results_folder = '/Users/jcostello/University of Michigan Dropbox/Joseph Costello/Chestek Lab/Code/NeuralNet/Temmar2025BigDataAnalysis/LINK_dataset/analysis/bci_decoding/single_day_model_results'

# Load the evaluation results
rr_df = pd.read_csv(f'{results_folder}/rr_evaluation_20250507_gooddays.csv')
lstm_df = pd.read_csv(f'{results_folder}/lstm_evaluation_20250507_gooddays.csv')
out_fname_corr = 'sameday_performance_plots_corr.png'


# Define DOF labels
dof_labels = ["IDX pos", "MRS pos", "IDX vel", "MRS vel"]

# Define the window size for smoothing
smoothing_window_size = 20

########################################################


# Filter data for same-day evaluations (Day_diff = 0)
rr_sameday = rr_df[rr_df['Day_diff'] == 0].copy()
lstm_sameday = lstm_df[lstm_df['Day_diff'] == 0].copy()

# Convert date strings to datetime objects for proper ordering
rr_sameday['Date'] = pd.to_datetime(rr_sameday['Train_day'])
lstm_sameday['Date'] = pd.to_datetime(lstm_sameday['Train_day'])

# Calculate days since earliest date for proper time spacing
all_dates = pd.concat([rr_sameday['Date'], lstm_sameday['Date']]).unique()
earliest_date = min(all_dates)

# Add days_since_start to each dataframe
rr_sameday['days_since_start'] = (rr_sameday['Date'] - earliest_date).dt.days
lstm_sameday['days_since_start'] = (lstm_sameday['Date'] - earliest_date).dt.days

# Sort by date
rr_sameday = rr_sameday.sort_values('Date')
lstm_sameday = lstm_sameday.sort_values('Date')

# Create mapping between days_since_start and actual dates for x-axis tick labels
unique_dates = sorted(pd.unique(pd.concat([rr_sameday['Date'], lstm_sameday['Date']])))
days_since_start_values = [(date - earliest_date).days for date in unique_dates]
date_mapping = dict(zip(days_since_start_values, unique_dates))

# Apply rolling window smoothing to all metrics
for metric in ['R2', 'MSE', 'Correlation']:
    # Smooth average metrics
    rr_sameday[f'{metric}_smooth'] = rr_sameday[metric].rolling(window=smoothing_window_size, center=True, min_periods=1).mean()
    lstm_sameday[f'{metric}_smooth'] = lstm_sameday[metric].rolling(window=smoothing_window_size, center=True, min_periods=1).mean()
    
    # Smooth per-DOF metrics
    for i in range(4):
        rr_sameday[f'{metric}_DOF{i}_smooth'] = rr_sameday[f'{metric}_DOF{i}'].rolling(window=smoothing_window_size, center=True, min_periods=1).mean()
        lstm_sameday[f'{metric}_DOF{i}_smooth'] = lstm_sameday[f'{metric}_DOF{i}'].rolling(window=smoothing_window_size, center=True, min_periods=1).mean()

# Print summary
print(f"Number of same-day evaluations - RR: {len(rr_sameday)}, LSTM: {len(lstm_sameday)}")
print(f"Date range: {pd.Timestamp(earliest_date).strftime('%Y-%m-%d')} to {pd.Timestamp(max(unique_dates)).strftime('%Y-%m-%d')}, spanning {max(days_since_start_values)} days")

colors = ['r', 'g', 'b', 'purple']

# Function to format dates for ticks
def format_date_ticks(ax, date_range_days):
    """Format x-axis ticks to show dates at reasonable intervals"""
    # Calculate a reasonable number of ticks based on the range
    total_days = max(date_range_days)
    
    if total_days <= 30:  # If less than a month, show weekly ticks
        tick_interval = 7
    elif total_days <= 180:  # If less than 6 months, show monthly ticks
        tick_interval = 30
    elif total_days <= 730:  # If less than 2 years, show quarterly ticks
        tick_interval = 91
    else:  # Otherwise show yearly ticks
        tick_interval = 365
    
    # Create tick positions
    tick_positions = list(range(0, total_days + tick_interval, tick_interval))
    
    # Set the tick positions
    ax.set_xticks(tick_positions)
    
    # Format tick labels as dates
    tick_labels = []
    for pos in tick_positions:
        # Find the closest actual data date to this position
        closest_date_idx = min(range(len(date_range_days)), 
                             key=lambda i: abs(date_range_days[i] - pos))
        actual_date = unique_dates[closest_date_idx]
        tick_labels.append(pd.Timestamp(actual_date).strftime('%Y/%m'))
    
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Set the x-axis limits to include a bit of padding
    padding = total_days * 0.02  # 2% padding on each side
    ax.set_xlim(-padding, total_days + padding)

# Create Correlation plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

# Plot 1: RR vs LSTM average Correlation
ax1.plot(rr_sameday['days_since_start'], rr_sameday['Correlation'], 'r-', linewidth=1, alpha=0.3)
ax1.plot(lstm_sameday['days_since_start'], lstm_sameday['Correlation'], 'b-', linewidth=1, alpha=0.3)
ax1.plot(rr_sameday['days_since_start'], rr_sameday['Correlation_smooth'], 'r-', linewidth=2.5, label='Ridge Regression')
ax1.plot(lstm_sameday['days_since_start'], lstm_sameday['Correlation_smooth'], 'b-', linewidth=2.5, label='LSTM')

ax1.set_ylabel('Correlation')
ax1.set_title('Same-Day Performance: RR vs LSTM (Averaged across DOFs)')
ax1.grid(True, alpha=0.3)
ax1.legend()
format_date_ticks(ax1, days_since_start_values)

# Plot 2: LSTM Correlation by DOF
for i in range(4):
    ax2.plot(lstm_sameday['days_since_start'], lstm_sameday[f'Correlation_DOF{i}'], 
             color=colors[i], linewidth=1, alpha=0.3)
    ax2.plot(lstm_sameday['days_since_start'], lstm_sameday[f'Correlation_DOF{i}_smooth'], 
             color=colors[i], linewidth=2.5, label=f'{dof_labels[i]}')

ax2.set_ylabel('Correlation')
ax2.set_title('Same-Day LSTM Performance by DOF')
ax2.grid(True, alpha=0.3)
ax2.legend()
format_date_ticks(ax2, days_since_start_values)

# Plot 3: RR Correlation by DOF
for i in range(4):
    ax3.plot(rr_sameday['days_since_start'], rr_sameday[f'Correlation_DOF{i}'], 
             color=colors[i], linewidth=1, alpha=0.3)
    ax3.plot(rr_sameday['days_since_start'], rr_sameday[f'Correlation_DOF{i}_smooth'], 
             color=colors[i], linewidth=2.5, label=f'{dof_labels[i]}')

ax3.set_ylabel('Correlation')
ax3.set_title('Same-Day Ridge Regression Performance by DOF')
ax3.grid(True, alpha=0.3)
ax3.legend()
format_date_ticks(ax3, days_since_start_values)

plt.tight_layout()
plt.savefig(f'{results_folder}/{out_fname_corr}', dpi=300, bbox_inches='tight')
plt.close()

print(f"Correlation plots saved to {results_folder}/{out_fname_corr}")
