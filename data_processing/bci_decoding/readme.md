# Overview of how to run the decoding analysis

## 1. Train models on each day

Trains an LSTM and a RR model on each day.

Verify the parameters are correct at the top of the script `train_models_single_day.py`. Then run the script.

This saves the models as a .pkl file in the specified directory. If both CO and RD datasets are available, both models will be trained.

> Note that "with noise" means the LSTM was trained with neural noise, which bumps up performance on the order of 1-2%. Noise is not added during eval.

## 2. Eval models on every day

This step takes a long time since we eval ~400 models on about ~400 days each. This took about 2 days to run everything (1 day for lstms and 1 day for RRs).

Verify the parameters are correct at the top of the script `eval_singleday_models_multithread.py`. Then run the script. You can increase the number of threads, but I didn't find much of a speedup after like 16 (but this was a very coarse test).

There's an optional debug mode that runs a limited number of evaluations (only close days to the training day).

This saves results as a csv file with one row per model-day evaluation.

Looks like:
- Train_day
- Train_data_type (CO or RD)
- Test_day
- Test_data_type (CO or RD)
- Model_type (LSTM or RR)
- Day_diff (negative if test day is before train day)
- MSE
- Correlation
- R2
- MSE_DOF0, Correlation_DOF0, R2_DOF0, etc.



## 3. Eval models on every day (updated normalization)

This step is the same as the above, but it updates the normalization parameters for each evaluation day.

Verify the parameters are correct at the top of the script `eval_singleday_models_multithread_updatenorm.py`. Then run the script.


## 4. Make plots

`bci_decoding_plots.py` creates the plot seen in the publication. Make sure to choose the correct results file to plot.


