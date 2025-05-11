import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib as mpl
import seaborn as sns
from matplotlib.gridspec import GridSpec
from datetime import datetime
#some basic text parameters for figures
mpl.rcParams['font.family'] = "Atkinson Hyperlegible" # if installed but not showing up, rebuild mpl cache
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
########################################################
########################################################

# Load the evaluation results
results_folder = "C:\Repos\\big_nhp_dataset_code\\analysis\\bci_decoding\single_day_model_results"

# results_folder = '/Users/jcostello/University of Michigan Dropbox/Joseph Costello/Chestek Lab/Code/NeuralNet/Temmar2025BigDataAnalysis/LINK_dataset/analysis/bci_decoding/single_day_model_results'
XLIM_ONESIDE = 100

# Define DOF labels
dof_labels = ["IDX pos", "MRS pos", "IDX vel", "MRS vel"]

# Define the window size for smoothing
smoothing_window_size = 5

#colors
rr_color = np.asarray([180, 22, 88])/255
lstm_color = np.asarray([54,106,159])/255

########################################################

def create_decoding_figure(norm=False):
    rr_df, lstm_df = load_eval_results(norm)
    
    fig = plt.figure(figsize=(10,6), layout='constrained')
    sfs = fig.subfigures(2,1, height_ratios=[1,1.5])
    sffs = sfs[1].subfigures(1,2)
    subfigs = (sfs[0],sffs[0],sffs[1])
    overall_performance_ax = subfigs[0].add_subplot()
    (rr_performance_ax, lstm_performance_ax) = subfigs[1].subplots(2,1, sharex=True, sharey=True)

    plot_sameday_performance(rr_df, lstm_df, overall_performance_ax, lstm_performance_ax, rr_performance_ax)
    lstm_performance_ax.get_legend().remove()
    lstm_performance_ax.set(ylabel=None)
    # overall_performance_ax.set(yticks=[0,.2,.4,.6,.8,1])
    # rr_performance_ax.set(yticks=[0,.2,.4,.6,.8,1])
    ax = subfigs[2].subplots(1,1)
    plot_performance_across_days(rr_df, lstm_df, 'R2', ax)
    subfigs[2].suptitle('Performance on days before/after training (±Standard Error)')
    ax.get_legend().remove()
    out_fname = 'model_performance_plots_withnoise_updatenorm.png' if norm else 'model_performance_plots_withnoise.png'

def load_eval_results(norm=True):
    if norm:
        rr_df = pd.read_csv(f'{results_folder}/rr_evaluation_updatednorm_20250508_gooddays.csv')
        lstm_df = pd.read_csv(f'{results_folder}/lstm_evaluation_updatednorm_20250507_gooddays.csv')
        out_fname = 'model_performance_plots_withnoise_updatenorm.png'
    else:
        rr_df = pd.read_csv(f'{results_folder}/rr_evaluation_20250507_gooddays.csv')
        lstm_df = pd.read_csv(f'{results_folder}/lstm_evaluation_20250507_gooddays.csv')
        out_fname = 'model_performance_plots_withnoise.png'
    
    print("Data loaded, RR shape:", rr_df.shape, "LSTM shape:", lstm_df.shape)
    print("\nFirst few rows of RR data:")
    print(rr_df.head())
    print("\nFirst few rows of LSTM data:")
    print(lstm_df.head())
    return rr_df, lstm_df

def plot_sameday_performance(rr_df, lstm_df, ax1, ax2, ax3):
    
    def filter_same_day(df):
        df_sameday = df[df['Day_diff'] == 0].copy() # Filter data for same-day evaluations (Day_diff = 0)
        df_sameday['Date'] = pd.to_datetime(df_sameday['Train_day']) # Convert date strings to datetime objects for proper ordering
        return df_sameday

    rr_sameday = filter_same_day(rr_df)
    lstm_sameday = filter_same_day(lstm_df)

    # Calculate days since earliest date for proper time spacing
    all_dates = pd.concat([rr_sameday['Date'], lstm_sameday['Date']]).unique()
    earliest_date = min(all_dates)

    def more_processing(df_sameday):
        # Add days_since_start to each dataframe
        df_sameday['days_since_start'] = (df_sameday['Date'] - earliest_date).dt.days
        # Sort by date
        df_sameday = df_sameday.sort_values('Date')
        # Apply rolling window smoothing to all metrics
        for metric in ['R2', 'MSE', 'Correlation']:
            # Smooth average metrics
            df_sameday[f'{metric}_smooth'] = df_sameday[metric].rolling(window=smoothing_window_size, 
                                                                        center=True, min_periods=1).mean()
             # Smooth per-DOF metrics
            for i in range(4):
                df_sameday[f'{metric}_DOF{i}_smooth'] = df_sameday[f'{metric}_DOF{i}'].rolling(window=smoothing_window_size, 
                                                                                               center=True, min_periods=1).mean()
        return df_sameday
    
    rr_sameday = more_processing(rr_sameday)
    lstm_sameday = more_processing(lstm_sameday)

    # Create mapping between days_since_start and actual dates for x-axis tick labels
    unique_dates = sorted(pd.unique(pd.concat([rr_sameday['Date'], lstm_sameday['Date']])))
    days_since_start_values = [(date - earliest_date).days for date in unique_dates]
    date_mapping = dict(zip(days_since_start_values, unique_dates))

    # Print summary
    print(f"Number of same-day evaluations - RR: {len(rr_sameday)}, LSTM: {len(lstm_sameday)}")
    print(f"Date range: {pd.Timestamp(earliest_date).strftime('%Y-%m-%d')} to {pd.Timestamp(max(unique_dates)).strftime('%Y-%m-%d')}, spanning {max(days_since_start_values)} days")

    # Create Correlation plots
    met = 'R2'
    # Plot 1: RR vs LSTM average Correlation
    ax1.plot(rr_sameday['days_since_start'], rr_sameday[met], color=rr_color, linewidth=1, alpha=0.3)
    ax1.plot(lstm_sameday['days_since_start'], lstm_sameday[met], color=lstm_color, linewidth=1, alpha=0.3)
    ax1.plot(rr_sameday['days_since_start'], rr_sameday[f'{met}_smooth'], color=rr_color, linewidth=1.5, label='Ridge Regression')
    ax1.plot(lstm_sameday['days_since_start'], lstm_sameday[f'{met}_smooth'], color=lstm_color, linewidth=1.5, label='LSTM')

    ax1.set_ylabel(f'Predicted Accuracy ({met})')
    ax1.set_title('Same-Day Performance: RR vs LSTM (Averaged across DOFs)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    format_date_ticks(ax1, days_since_start_values, unique_dates)
    
    colors = sns.color_palette('colorblind',4)
    def same_day_by_dof(sameday, ax):
        for i in range(4):
            ax.plot(lstm_sameday['days_since_start'], lstm_sameday[f'{met}_DOF{i}'], 
                    color=colors[i], linewidth=1, alpha=0.3)
            ax.plot(lstm_sameday['days_since_start'], lstm_sameday[f'{met}_DOF{i}_smooth'], 
                    color=colors[i], linewidth=1.5, label=f'{dof_labels[i]}')
            ax.set_ylabel(met)
            ax.set_title('Same-Day LSTM Performance by DOF')
            ax.grid(True, alpha=0.3)
            ax.legend()
            format_date_ticks(ax, days_since_start_values, unique_dates)
    
    same_day_by_dof(lstm_sameday, ax2) # Plot 2: LSTM Correlation by DOF
    same_day_by_dof(rr_sameday, ax3) # Plot 3: RR Correlation by DOF

def plot_performance_across_days(rr_df, lstm_df, metric, ax):
    # Pre-aggregate the data
    def aggregate_data(df):
        grouped = df.groupby('Day_diff').agg({
            'MSE': ['mean', 'std', 'count'],
            'Correlation': ['mean', 'std', 'count'],
            'R2': ['mean', 'std', 'count']
        }).reset_index()
        return grouped

    rr_grouped = aggregate_data(rr_df)
    lstm_grouped = aggregate_data(lstm_df)

    print("\nAggregated data shapes - RR:", rr_grouped.shape, "LSTM:", lstm_grouped.shape)

    def plot_decoder(grouped, color, label):
        x = grouped['Day_diff']
        y_mean = grouped[(metric, 'mean')]
        y_std = grouped[(metric, 'std')]

        if error_type.lower() == 'se':
            # Standard error = standard deviation / sqrt(n)
            counts = grouped[(metric, 'count')]
            y_error = y_std / np.sqrt(counts)
            error_label = 'Standard Error'
        else:
            y_error = y_std
            error_label = 'Standard Deviation'   
        
        ax.plot(x, y_mean, color=color, linewidth=1.5, label=label)
        ax.fill_between(x, y_mean - y_error, y_mean + y_error, color=color, alpha=0.2)

    # Choose error type: 'sd' for standard deviation or 'se' for standard error
    error_type = 'se'  # Change to 'se' for standard error
    plot_decoder(rr_grouped, rr_color, 'Ridge Regression')
    plot_decoder(lstm_grouped, lstm_color, 'LSTM')

    # 'R2', 'R²', 'R²'
    
    # set plot parameters
    ax.set_xlabel('Days from Training')
    ax.set_ylabel(f'Predicted Accuracy ({metric})')
    # ax.set_title(f"{title}")
    ax.set_xlim(-XLIM_ONESIDE, XLIM_ONESIDE)
    
    # Set y-axis limits based on the metric
    if metric == 'MSE':
        # log scale y
        ax.set_yscale('log')
        # ax.set_ylim(10e-4, 10e1)
        ax.set_ylim(0, 1)
    elif metric == 'Correlation':
        ax.set_ylim(0, 0.9)
    elif metric == 'R2':
        ax.set_ylim(-0.25, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend()

def format_date_ticks(ax, date_range_days, unique_dates):
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
        tick_interval = 182
    
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
    
    ax.set_xticklabels(tick_labels, ha='center')
    
    # Set the x-axis limits to include a bit of padding
    padding = total_days * 0.02  # 2% padding on each side
    ax.set_xlim(-padding, total_days + padding)

if __name__=="__main__":
    create_decoding_figure(False)
    create_decoding_figure(True)
    plt.show()