import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy.stats import linregress
from scipy.signal import savgol_filter
from scipy.signal import savgol_filter
from datetime import datetime, timedelta
import pickle
import pdb
import seaborn as sns
from datetime import datetime
#import config
import sys

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler


def create_bci_decoding_figure():
    # load data
    # create train and test splits if the need to be (and probably save)
    # train ridge regressions if they need to be
    # get test predictions for each decoder on each day
    # get metrics (correlation, mse, r^2) for each of these decodes and save it.
    # create the daily plot (other_stuff_toplevel)
        # from plotters.RidgeRegression import daily_ridge_regression_perf
        # daily_ridge_regression_perf(dates,preprocessingdir,characterizationdir)
    # create the across days plot (other_stuff_toplevel relative model performance over time)
        # from plotters.OldPerformanceoverTime import models_relative_perfs_over_time_old
        # models_relative_perfs_over_time_old(perf_dir, characterizationdir)
    # create kde plot
    # from plotters.MultidayStability import multiday_stability
    # multiday_stability(perf_dir,characterizationdir)
    pass

def daily_ridge_regression_perf(dates,preprocessingdir,characterizationdir):
    mses = np.zeros((1,4))
    r2s = np.zeros((1,1))
    rs = np.zeros((1,4))

    for date in dates:
        file = os.path.join(preprocessingdir,f'{date}_preprocess.pkl')

        with open(file, 'rb') as f:
            data_CO, data_RD = pickle.load(f)
        sys.stdout.write(f"\r Date Processing: {date}")
        sys.stdout.flush()

        if data_CO and data_RD:
            neural_train, neural_test, finger_train, finger_test = dataPrep(data_CO,20)
            reg = linear_model.Ridge(alpha = 0.001,fit_intercept=False)
            reg.fit(neural_train, finger_train)
            rr_prediction = reg.predict(neural_test)
            mses_co = mean_squared_error(finger_test, rr_prediction, multioutput = 'raw_values').reshape(1,4)
            r2s_co = np.array([r2_score(finger_test, rr_prediction)]).reshape(1,1)
            rs_co = np.array([pearsonr(finger_test, rr_prediction).statistic]).reshape(1,4)

            neural_train, neural_test, finger_train, finger_test = dataPrep(data_RD,20)
            reg = linear_model.Ridge(alpha = 0.001,fit_intercept=False)
            reg.fit(neural_train, finger_train)
            rr_prediction = reg.predict(neural_test)
            mses = np.concatenate((mses,(mean_squared_error(finger_test, rr_prediction, multioutput = 'raw_values').reshape(1,4) + mses_co)/2),axis=0)
            r2s = np.concatenate((r2s,(np.array([r2_score(finger_test, rr_prediction)]).reshape(1,1)+r2s_co)/2),axis=0)
            rs = np.concatenate((rs,(np.array([pearsonr(finger_test, rr_prediction).statistic]).reshape(1,4)+rs_co)/2),axis=0)

        elif data_CO:
            neural_train, neural_test, finger_train, finger_test = dataPrep(data_CO,20)
            reg = linear_model.Ridge(alpha = 0.001,fit_intercept=False)
            reg.fit(neural_train, finger_train)
            rr_prediction = reg.predict(neural_test)
            mses = np.concatenate((mses,mean_squared_error(finger_test, rr_prediction, multioutput = 'raw_values').reshape(1,4)),axis=0)
            r2s = np.concatenate((r2s,np.array([r2_score(finger_test, rr_prediction)]).reshape(1,1)),axis=0)
            rs = np.concatenate((rs,np.array([pearsonr(finger_test, rr_prediction).statistic]).reshape(1,4)),axis=0)

        elif data_RD:
            neural_train, neural_test, finger_train, finger_test = dataPrep(data_RD,20)
            reg = linear_model.Ridge(alpha = 0.001,fit_intercept=False)
            reg.fit(neural_train, finger_train)
            rr_prediction = reg.predict(neural_test)
            mses = np.concatenate((mses,mean_squared_error(finger_test, rr_prediction, multioutput = 'raw_values').reshape(1,4)),axis=0)
            r2s = np.concatenate((r2s,np.array([r2_score(finger_test, rr_prediction)]).reshape(1,1)),axis=0)
            rs = np.concatenate((rs,np.array([pearsonr(finger_test, rr_prediction).statistic]).reshape(1,4)),axis=0)

        date_objects = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in dates]
        first_date = date_objects[0]
        days_from_first = [(date - first_date).days for date in date_objects]
    scaler = MinMaxScaler()
    np.save(os.path.join(characterizationdir, 'RR_perf_mses.npy'), mses)
    np.save(os.path.join(characterizationdir, 'RR_perf_rs.npy'), rs)
    np.save(os.path.join(characterizationdir, 'RR_perf_r2s.npy'), r2s)
    r2s_reshaped = r2s[1:,:].reshape(-1, 1)
    r2s_normalized = scaler.fit_transform(r2s_reshaped)

    average_mses = mses[1:,:].mean(axis=-1)
    x = np.arange(len(average_mses))
    print(x.shape)
    print(len(days_from_first))
    x2 = np.arange(len(r2s_normalized))



    slope, intercept = np.polyfit(x, average_mses, 1)
    trend = slope * x + intercept
    plt.figure(figsize=(12, 4))
    plt.plot(days_from_first, average_mses, label="MSE over Time", alpha=0.4)
    plt.plot(days_from_first, trend, label="Trend", color="green")
    plt.plot(days_from_first, savgol_filter(average_mses.flatten(), 50, 3), label="Filtered MSE over Time", color="red")
    plt.xlabel('Trial #')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 4))
    print(r2s.shape)
    plt.plot(days_from_first, r2s[1:,:].flatten(), label="R2 over Time", alpha=0.4)
    plt.plot(days_from_first, savgol_filter(r2s[1:,:].flatten(), 70, 3), label="Filtered R2 over Time", color="red")
    plt.legend()
    plt.xlabel('Trial #')
    plt.ylabel('RR')
    plt.show()

    average_rs = rs[1:,:].mean(axis=-1)
    x3 = np.arange(len(average_rs))
    plt.figure(figsize=(12, 4))
    plt.plot(days_from_first, average_rs, label="Pearson R over Time", alpha=0.4)
    plt.plot(days_from_first, savgol_filter(average_rs, 50, 3), label="Filtered Pearson R over Time", color="red")
    plt.legend()
    plt.ylim((0,1))
    plt.xlabel('Trial #')
    plt.ylabel('Pearson R')
    plt.show()

    labels = ('IDX pos','MRS pos','IDX vel','MRS vel')
    colors = ('aqua','blue','lime','darkgreen')
    for i in range(rs.shape[-1]):
        plt.plot(days_from_first, rs[1:,i], c=colors[i], alpha=0.3)
        plt.plot(days_from_first, savgol_filter(rs[1:,i], 50, 3), label=f'{labels[i]}', c=colors[i], lw=2)
        plt.ylim(0,1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

def multiday_stability(perf_dir,characterizationdir):
    # this needs to be overhauled significantly - train on multiple days and test - not critical
    # LOAD DATA
    with open(os.path.join(perf_dir, 'x+n results', 'ridge_correlation_results.pkl'), 'rb') as f:
        rr_corr_results = pickle.load(f)

    with open(os.path.join('sfn_stuff', 'x+n results', 'lstm_correlation_results.pkl'), 'rb') as f:
        lstm_corr_results = pickle.load(f)

    def reformat_into_df(data):
        df_prep = {
            'num_training_days':[],
            'testing_day':[],
            'mean':[],
            'std':[]
        }   
        for training_days in data['sessions_means'].keys():
            for testing_days in data['sessions_means'][training_days].keys():
                df_prep['num_training_days'].append(training_days)
                df_prep['testing_day'].append(testing_days)
                df_prep['mean'].append(data['sessions_means'][training_days][testing_days])
                df_prep['std'].append(data['sessions_stds'][training_days][testing_days])
        df = pd.DataFrame.from_dict(df_prep)

        #hardcoded but it is what is is
        dodge_amounts = 0.1 * np.tile(np.asarray([-2,-1,0,1,2]), 10)
        df['num_training_days_dodged'] = df['num_training_days'] + dodge_amounts
        return df
    
    rr_df = reformat_into_df(rr_corr_results)
    lstm_df = reformat_into_df(lstm_corr_results)

    fig, ax = plt.subplots(1,1, figsize=(10,6), sharex=True, sharey=True)
    # sns.lineplot(rr_df, x='num_training_days_dodged', y='mean', hue='testing_day', ax=ax[0])
    # sns.lineplot(lstm_df, x='num_training_days_dodged', y='mean', hue='testing_day', ax=ax[1])


    # custom error bar
    rr_c = sns.color_palette('rocket', as_cmap=True)(np.linspace(0,1,10))[2:7,:]
    lstm_c = sns.color_palette('mako', as_cmap=True)(np.linspace(0,1,10))[2:7,:]
    def make_perf_plot(data, ax, colors):
        for i, (test_day, group) in enumerate(data.groupby('testing_day')):
            ax.errorbar(group['num_training_days_dodged'], group['mean'], group['std'], c=colors[i], label=test_day)
        ax.grid(True)
        ax.set(xticks=(1,2,3,4,5,6,7,8,9,10), xlabel='# of training days used')
        ax.legend()

    ax.set(ylabel='Correlation Coeff.')
    make_perf_plot(rr_df, ax, rr_c)
    make_perf_plot(lstm_df, ax, lstm_c)
    fig.suptitle('Stabilization through multi-day training')
    fig.savefig(os.path.join(characterizationdir, "stbailization_multiday.pdf"))

def exp_decay():
    # fit exponential decay
    nn_means = means_forexp['nn']
    nn_offset = np.min(nn_means) - 0.0000001 #can't have divide by zero

    rr_means = means_forexp['rr']
    rr_offset = np.min(rr_means) - 0.0000001 #can't have divide by zero

    days = x

    def fit_exp_linear(t, y, C=0):
        y = y - C
        y = np.log(y)
        K, A_log = np.polyfit(t, y, 1)
        A = np.exp(A_log)
        return A, K

    nn_A, nn_K = fit_exp_linear(days, nn_means, nn_offset)

    rr_A, rr_K = fit_exp_linear(days, rr_means, rr_offset)
    exp_decay_nn = nn_A * np.exp(nn_K * x) + nn_offset
    exp_decay_rr = rr_A * np.exp(rr_K * x) + rr_offset

    plt.plot(days, nn_means, label='nn means',color='r')
    plt.plot(days, exp_decay_nn, label=f'${{{round(nn_A,3)}}}e^{{{round(nn_K,3)}*day}} + {round(nn_offset,3)}$',color='purple')
    plt.plot(days, rr_means, label='rr means', color='b')
    plt.plot(days, exp_decay_rr, label=f'${{{round(rr_A,3)}}}e^{{{round(rr_K,3)}*day}} + {round(rr_offset,3)}$',color='g')
    plt.legend()
    plt.title('exp decay fit on mean correlation relative to day 0')
    plt.xlabel('days from day 0')
    plt.ylabel('drop in correlation rel. to day 0')