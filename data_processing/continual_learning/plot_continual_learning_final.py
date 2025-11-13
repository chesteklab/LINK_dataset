import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# USER CONFIG
# ----------------------
results_folder = 'continual_learning_results'

# Specify the unified results file from the sweep
results_file = 'results_continual_learning_sweep_RDandCO_neuralnormupdated_20seeddays_200days_20msbins.csv'

# Metric to plot (column name in CSV)
metric = 'R2'  # Change to 'Correlation', 'MSE', etc. if desired
metric_ylabel = 'Prediction Accuracy (R²)'

xlim = (-1, 201)
ylim = (-0.1, 0.8)


legend_names = {
    30*50: '30 s',
    60*50: '1 min', 
    120*50: '2 min',
    300*50: '5 min',
}

# Create color mapping based on fine-tune samples
sample_color_map = {
    30*50: '#1f77b4',
    60*50: '#ff7f0e',
    120*50: '#2ca02c',
    300*50: '#d62728',
}


# Day binning for smoothing the mean line/errbars(set to 1 for no binning, 5 for 5-day bins, etc.)
day_binning = 3

# ----------------------
# ----------------------

def plot_continual_learning_sweep():
    """Main plotting function."""
    print(f"Loading data from: {results_file}")
    
    # Load data
    df = pd.read_csv(os.path.join(results_folder, results_file))
    
    if df.empty:
        raise ValueError("No data found!")
    
    print(f"Loaded {len(df)} rows of data")
    
    # Set up the plot
    sns.set(style='whitegrid', context='talk')
    plt.figure(figsize=(12, 8), dpi=300)
    
    # Group by fine-tune samples
    if 'Finetune_samples' not in df.columns:
        raise ValueError("Finetune_samples column not found in data!")
    
    unique_sample_sizes = sorted(df['Finetune_samples'].unique())
    
    # Plot each sample size group
    for sample_size in unique_sample_sizes:
        # Filter data for this sample size
        sample_df = df[df['Finetune_samples'] == sample_size].copy()
        
        if sample_df.empty:
            continue
        
        # Handle day 0 separately (no binning) and bin days > 0
        day_0_df = sample_df[sample_df['Rel_day'] == 0].copy()
        days_gt_0_df = sample_df[sample_df['Rel_day'] > 0].copy()
        
        # Create summary for day 0 (if it exists)
        summary_parts = []
        if not day_0_df.empty:
            day_0_summary = day_0_df.groupby('Rel_day')[metric].agg(['mean', 'std', 'count']).reset_index()
            day_0_summary['day_bin_center'] = day_0_summary['Rel_day']  # Keep day 0 at x=0
            day_0_summary['se'] = day_0_summary['std'] / np.sqrt(day_0_summary['count'])
            summary_parts.append(day_0_summary[['day_bin_center', 'mean', 'std', 'count', 'se']])
        
        # Create bins for days > 0
        if not days_gt_0_df.empty:
            # Bin days > 0: days 1-day_binning go to first bin, etc.
            days_gt_0_df['bin_number'] = (days_gt_0_df['Rel_day'] - 1) // day_binning
            days_gt_0_df['day_bin_start'] = days_gt_0_df['bin_number'] * day_binning + 1
            days_gt_0_df['day_bin_center'] = days_gt_0_df['day_bin_start'] + (day_binning - 1) / 2
            
            # Group by day_bin_center and calculate statistics
            binned_summary = days_gt_0_df.groupby('day_bin_center')[metric].agg(['mean', 'std', 'count']).reset_index()
            binned_summary['se'] = binned_summary['std'] / np.sqrt(binned_summary['count'])
            summary_parts.append(binned_summary)
        
        # Combine day 0 and binned summaries
        if summary_parts:
            summary = pd.concat(summary_parts, ignore_index=True).sort_values('day_bin_center')
        else:
            summary = pd.DataFrame(columns=['day_bin_center', 'mean', 'std', 'count', 'se'])
         
        # (optional, just for printing) Calculate average R2 across all day bins for this sample size
        avg_r2 = summary['mean'].mean()
        print(f"  Average {metric} across all day bins: {avg_r2:.4f}")
        print(f"  Number of unique starting days: {sample_df['Starting_day'].nunique()}")
        
        # Plot the mean line
        plt.plot(summary['day_bin_center'], summary['mean'], 
                label=legend_names[sample_size], color=sample_color_map[sample_size], linewidth=2, zorder=2)
        
        # Add shaded error bars
        if summary['se'].sum() > 0:
            plt.fill_between(summary['day_bin_center'], 
                           summary['mean'] - summary['se'], 
                           summary['mean'] + summary['se'], 
                           alpha=0.3, color=sample_color_map[sample_size])
        
        # Plot all individual data points with transparency
        plt.scatter(sample_df['Rel_day'], sample_df[metric], 
                   color=sample_color_map[sample_size], alpha=0.3, s=12, zorder=1, edgecolors='none')
    
    # Customize plot
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel('Days since start')
    plt.ylabel(metric_ylabel)
    plt.legend(title='Finetuning Data', loc='lower right')
    plt.tight_layout()
    
    # Save the plot
    output_filename = f'continual_learning_final_{metric.lower()}_20msbins.png'
    output_path = os.path.join(results_folder, output_filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()
    
    print(f"\nSaved plot to: {output_filename}")
    
    return df, unique_sample_sizes


if __name__ == "__main__":
    df, unique_sample_sizes = plot_continual_learning_sweep()
        