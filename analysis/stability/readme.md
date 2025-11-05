# Multi-Day Training
## Prerequisites

### Dependencies

1. **LINK Dataset**
2. **Neural Decoding Library**: Install the LINK-dev branch from the [neuraldecoding repository](https://github.com/Neuro-core-hub/neuraldecoding/tree/LINK-dev)

## Setup

### 1. Data Directory Setup

Update the following paths in **both** `trainer_multiday.py` and `multiday_eval.py`:

```python
data_dir = "/path/to/your/LINK/dataset/sub-Monkey-N"
neural_decoding_dir = "/path/to/neuraldecoding/library"
```

### 2. Evaluation Configuration

In `multiday_eval.py`, specify the training start day:

```python
target_day = "20210101"  # Set to the first day the model was trained on
```

### 3. Config Files Setup

For both `config_LSTM` and `config_LSTM_cpu`:

1. Navigate to `trainer/trainer.yaml`
2. Update the `data.datapath` field:
   ```yaml
   data:
     datapath: "/path/to/your/LINK/dataset/sub-Monkey-N/sub-Monkey-N_ses-YYYYMMDD_ecephys.nwb" # change YYYYMMDD to some date existing in the folder
   ```
   
   **Note**: Use any NWB file from your data folder, but ensure the filename does **not** contain the pattern `objXXXXX`

## Workflow

### Step 1: Train Models
```bash
python trainer_multiday.py
```

### Step 2: Evaluate Models
```bash
python multiday_eval.py
```

### Step 3: Visualize Results

1. Open `multiday_plotter.ipynb`
2. Update the file path in the notebook (results should be saved by default to root directory):
   ```python
   df = pd.read_csv('path/to/your/results_YYYY-MM-DD-HH-MM_day_YYYYMMDD_XXX.csv')
   ```
3. Use the most recent results file from your root directory
4. Run the notebook to generate plots
