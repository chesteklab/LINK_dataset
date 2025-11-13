import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# USER CONFIGURATION
# ----------------------
results_folder = '/home/joey/University of Michigan Dropbox/Joseph Costello/Chestek Lab/Code/NeuralNet/Temmar2025BigDataAnalysis/LINK_dataset/analysis/bci_decoding/continual_learning_results'

# Specify the unified results file from the sweep
# results_file = 'continual_learning_sweep_CO_neuralnormupdated_final_20250725-143542.csv'
# results_file = 'continual_learning_sweep_RD_neuralnormupdated_final_20250725-141921.csv'
# results_file = 'continual_learning_sweep_BOTH_neuralnormupdated_final_20250725-152234.csv'
# results_file = 'continual_learning_sweep_BOTH_neuralnormupdated_final_20250725-155304.csv'

results_file = 'continual_learning_sweep_BOTH_neuralnormupdated_final_20250725-164502.csv'


# Metric to plot (column name in CSV)
metric = 'R2'  # Change to 'Correlation', 'MSE', etc. if desired
metric_ylabel = 'R² (mean ± SE)'

# Choose which parameter combinations to plot (set to None to plot all)
# Format: {'Starting_day': ['2020-02-04'], 'Finetune_samples': [960, 1920], 'Finetune_iterations': [250, 500]}
filter_params = None  # Plot all combinations
# filter_params = {
#     # 'Starting_day': ['2020-02-04'],  # Only plot this starting day
#     'Finetune_samples': [30*32],  # Only plot these sample sizes
#     # 'Finetune_iterations': [250, 500]  # Only plot these iteration counts
# }

xlim = (-1, 101)

# Color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Custom legend names (optional)
# legend_names = None  # Set to None to use default names
legend_names = {
    960: '30 s',
    1920: '1 min', 
    3840: '2 min',
    9600: '5 min',
}

# ----------------------
# END USER CONFIGURATION
# ----------------------
 
def load_and_filter_data(results_file, filter_params=None):
    """Load data and optionally filter by parameter combinations."""
    df = pd.read_csv(os.path.join(results_folder, results_file))
    
    if filter_params is not None:
        for param, values in filter_params.items():
            if param in df.columns:
                df = df[df[param].isin(values)]
    
    return df

def get_unique_parameter_combinations(df):
    """Get unique parameter combinations from the dataframe."""
    param_cols = ['Starting_day', 'Finetune_samples', 'Finetune_iterations', 'Update_neural_scaler']
    # Only include columns that exist in the dataframe
    param_cols = [col for col in param_cols if col in df.columns]
    
    unique_combos = df[param_cols].drop_duplicates().reset_index(drop=True)
    return unique_combos, param_cols

def create_group_label(row, param_cols):
    """Create a readable label for a parameter combination."""
    label_parts = []
    
    if 'Starting_day' in param_cols:
        label_parts.append(f"Start: {row['Starting_day']}")
    if 'Finetune_samples' in param_cols:
        label_parts.append(f"Samples: {row['Finetune_samples']}")
    if 'Finetune_iterations' in param_cols:
        label_parts.append(f"Iters: {row['Finetune_iterations']}")
    if 'Update_neural_scaler' in param_cols:
        scaler_str = "Update" if row['Update_neural_scaler'] else "Fixed"
        label_parts.append(f"Scaler: {scaler_str}")
    
    return ", ".join(label_parts)

def plot_continual_learning_sweep():
    """Main plotting function."""
    print(f"Loading data from: {results_file}")
    
    # Load and filter data
    df = load_and_filter_data(results_file, filter_params)
    
    if df.empty:
        raise ValueError("No data found after filtering!")
    
    print(f"Loaded {len(df)} rows of data")
    
    # Set up the plot
    sns.set(style='whitegrid', context='talk')
    plt.figure(figsize=(12, 8), dpi=300)
    
    # Group by fine-tune samples and calculate statistics across starting days
    if 'Finetune_samples' not in df.columns:
        raise ValueError("Finetune_samples column not found in data!")
    
    unique_sample_sizes = sorted(df['Finetune_samples'].unique())
    print(f"Found {len(unique_sample_sizes)} unique fine-tune sample sizes: {unique_sample_sizes}")
    
    # Create color mapping based on fine-tune samples
    sample_color_map = {sample_size: colors[i % len(colors)] for i, sample_size in enumerate(unique_sample_sizes)}
    
    # Plot each sample size group
    print(f"\n{'='*80}")
    print(f"PLOTTING {len(unique_sample_sizes)} SAMPLE SIZE GROUPS:")
    print(f"{'='*80}")
    
    for sample_size in unique_sample_sizes:
        # Filter data for this sample size
        sample_df = df[df['Finetune_samples'] == sample_size].copy()
        
        if sample_df.empty:
            continue
        
        print(f"\nSample size: {sample_size}")
        print(f"  Number of data points: {len(sample_df)}")
        
        # Group by Rel_day and calculate statistics across all starting days
        # This will give us error bars across different starting days
        summary = sample_df.groupby('Rel_day')[metric].agg(['mean', 'std', 'count']).reset_index()
        summary['se'] = summary['std'] / np.sqrt(summary['count'])
        
        # Handle NaN values (when only one data point)
        summary['se'] = summary['se'].fillna(0)
        
        # Calculate average R2 across all days for this sample size
        avg_r2 = summary['mean'].mean()
        print(f"  Average {metric} across all days: {avg_r2:.4f}")
        print(f"  Number of unique starting days: {sample_df['Starting_day'].nunique()}")
        
        # Get color for this sample size
        color = sample_color_map[sample_size]
        
        # Create label - use custom name if provided, otherwise use default
        if legend_names is not None and sample_size in legend_names:
            label = legend_names[sample_size]
        else:
            label = f"Samples: {sample_size}"
        
        # Plot the mean line
        plt.plot(summary['Rel_day'], summary['mean'], 
                label=label, color=color, linewidth=2, marker='o', markersize=4)
        
        # Add shaded error bars
        if summary['se'].sum() > 0:
            plt.fill_between(summary['Rel_day'], 
                           summary['mean'] - summary['se'], 
                           summary['mean'] + summary['se'], 
                           alpha=0.3, color=color)
    
    # Customize plot
    plt.xlim(*xlim)
    plt.xlabel('Days since start')
    plt.ylabel(metric_ylabel)
    # plt.title(f'Continual Learning Performance by Fine-tune Sample Size')
    
    # Set reasonable y-limits based on the metric
    if metric in ['R2', 'Correlation']:
        plt.ylim(-0.1, 0.8)
    
    # Add legend
    plt.legend(title='Finetuning Data') #, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    output_filename = f'continual_learning_sweep_{metric.lower()}_by_samples.png'
    output_path = os.path.join(results_folder, output_filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()
    
    print(f"\nSaved plot to: {output_filename}")
    
    return df, unique_sample_sizes


if __name__ == "__main__":

    df, unique_sample_sizes = plot_continual_learning_sweep()
        