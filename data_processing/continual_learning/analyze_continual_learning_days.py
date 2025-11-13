import os
import pandas as pd
import numpy as np

# ----------------------
# USER CONFIG
# ----------------------
results_folder = 'continual_learning_results'

# Specify the unified results file from the sweep
results_file = 'results_continual_learning_sweep_RDandCO_neuralnormupdated_20seeddays_200days.csv'

# Metric to analyze
metric = 'R2'

# Specific days to analyze
target_days = [0, 5, 10, 50, 100, 200]

# Legend names for different sample sizes (from plot script)
legend_names = {
    960: '30 s',
    1920: '1 min', 
    3840: '2 min',
    9600: '5 min',
}

# ----------------------
# ----------------------

def analyze_continual_learning_by_days():
    """Analyze continual learning performance at specific time points."""
    print(f"Loading data from: {results_file}")
    
    # Load data
    df = pd.read_csv(os.path.join(results_folder, results_file))
    
    # Get unique sample sizes and sort them
    unique_sample_sizes = sorted(df['Finetune_samples'].unique())
    
    # Create results table
    results_table = []
    
    # Analyze each sample size group
    for sample_size in unique_sample_sizes:
        sample_name = legend_names.get(sample_size, f"{sample_size} samples")
        print(f"{'='*60}")
        print(f"Finetuning samples: {sample_size} ({sample_name})")
        print(f"{'='*60}")
        
        # Filter data for this sample size
        sample_df = df[df['Finetune_samples'] == sample_size].copy()
        
        if sample_df.empty:
            print(f"No data found for sample size {sample_size}")
            continue
        
        print(f"Number of data points: {len(sample_df)}")
        print(f"Number of unique starting days: {sample_df['Starting_day'].nunique()}")
        print(f"Rel_day range: {sample_df['Rel_day'].min()} to {sample_df['Rel_day'].max()}")
        print()
        
        # Analyze each target day
        day_results = {'sample_size': sample_size, 'sample_name': sample_name}
        
        for target_day in target_days:
            # Filter data for this specific day
            day_df = sample_df[sample_df['Rel_day'] == target_day]
            
            if day_df.empty:
                print(f"Day {target_day:3d}: No data available")
                day_results[f'day_{target_day}'] = None
                day_results[f'day_{target_day}_std'] = None
                day_results[f'day_{target_day}_count'] = 0
            else:
                # Calculate statistics
                mean_r2 = day_df[metric].mean()
                std_r2 = day_df[metric].std()
                count = len(day_df)
                se_r2 = std_r2 / np.sqrt(count) if count > 0 else 0
                
                print(f"Day {target_day:3d}: {metric} = {mean_r2:.4f} ± {se_r2:.4f} (SE), std = {std_r2:.4f}, n = {count}")
                
                day_results[f'day_{target_day}'] = mean_r2
                day_results[f'day_{target_day}_std'] = std_r2
                day_results[f'day_{target_day}_count'] = count
        
        results_table.append(day_results)
        print()
    
    # Create summary table
    print(f"{'='*80}")
    print(f"SUMMARY TABLE - Average {metric} by Finetuning Samples and Day")
    print(f"{'='*80}")
    
    # Create DataFrame for easy viewing
    summary_df = pd.DataFrame(results_table)
    
    # Print header
    header = f"{'Sample Size':<12} {'Name':<8}"
    for day in target_days:
        header += f"{'Day ' + str(day):<12}"
    print(header)
    print("-" * len(header))
    
    # Print each row
    for _, row in summary_df.iterrows():
        line = f"{row['sample_size']:<12} {row['sample_name']:<8}"
        for day in target_days:
            day_val = row[f'day_{day}']
            if day_val is not None:
                line += f"{day_val:<12.4f}"
            else:
                line += f"{'N/A':<12}"
        print(line)
    
    print()
    
    # Additional analysis: look at performance degradation
    print(f"{'='*80}")
    print(f"PERFORMANCE CHANGE FROM DAY 0")
    print(f"{'='*80}")
    
    change_header = f"{'Sample Size':<12} {'Name':<8}"
    for day in target_days[1:]:  # Skip day 0
        change_header += f"{'Δ Day ' + str(day):<12}"
    print(change_header)
    print("-" * len(change_header))
    
    for _, row in summary_df.iterrows():
        day_0_val = row['day_0']
        if day_0_val is not None:
            line = f"{row['sample_size']:<12} {row['sample_name']:<8}"
            for day in target_days[1:]:
                day_val = row[f'day_{day}']
                if day_val is not None:
                    change = day_val - day_0_val
                    line += f"{change:<12.4f}"
                else:
                    line += f"{'N/A':<12}"
            print(line)
    
    print()
    
    # # Save summary to CSV
    # summary_csv_path = os.path.join(results_folder, f'continual_learning_summary_{metric.lower()}_by_days.csv')
    # summary_df.to_csv(summary_csv_path, index=False)
    # print(f"Summary saved to: {summary_csv_path}")
    
    return summary_df


if __name__ == "__main__":
    summary_df = analyze_continual_learning_by_days() 