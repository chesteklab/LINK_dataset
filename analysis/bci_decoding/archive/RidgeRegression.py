import os
import pandas as pd
import numpy as np
from datetime import datetime
#import config
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pdb
import os 
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

from collections import defaultdict
def adjustfeats(X, Y, lag=0, hist=0, fillb4=None, out2d=False):
    '''
    This function takes in neural data X and behavior Y and returns two "adjusted" neural data and behavior matrices
    based on the optional params. Specifically the amount of lag between neural data and behavior can be set in units
    and the number of historical bins of neural data can be set. (BASED ON CODE BY SAM NASON)
    Inputs:
        - X (ndarray):
            The neural data, which should be [t, neu] in size, where t is the numebr of smaples and neu is the number
            of neurons.
        - y (ndarray):
            The behavioral data, which should be [t, dim] in size, where t is the number of samples and dim is the
            number of states.
        - lag (int, optional):
            Defaults to 0. The number of bins to lag the neural data relative to the behavioral data. For example,
            adjustFeats(X,Y, lag=1) will return X[0:-1] for adjX and Y[1:] for adjY.
        - hist (int, optional):
            Default 0. The number of bins to append to each sample of neural data from the previous 'hist' bins.
        - fillb4 (ndarray or scalar, optional):
            Default None, disabled. This fills previous neural data with values before the experiment began. A single
            scalar wil fill all previous neural data with that value. Otherwise, a [1,neu] ndarray equal to the first
            dimension of X (# of neurons) should represent the value to fill for each channel.
        - out2d (bool, optional):
            if history is added, will return the adjusted matrices either in 2d or 3d form (2d has the history appended
            as extra columns, 3d has history as a third dimension. For example, out2d true returns a sample as:
            [1, neu*hist+1] whereas out2d false returns: [1, neu, hist+1]. Default False
    Outputs:
        - adjX (ndarray):
            The adjusted neural data
        - adjY (ndarray):
            The adjusted behavioral data
    '''
    nNeu = X.shape[1]
    if fillb4 is not None:
        if isinstance(fillb4, np.ndarray):
            Xadd = np.tile(fillb4, hist)
            Yadd = np.zeros((hist, Y.shape[1]))
        else:
            Xadd = np.ones((hist, nNeu))*fillb4
            Yadd = np.zeros((hist, Y.shape[1]))
        X = np.concatenate((Xadd, X))
        Y = np.concatenate((Yadd, Y))

    #reshape data to include historical bins
    adjX = np.zeros((X.shape[0]-hist, nNeu, hist+1))
    for h in range(hist+1):
        adjX[:,:,h] = X[h:X.shape[0]-hist+h,:]
    adjY = Y[hist:,:]

    if lag != 0:
        adjX = adjX[0:-lag,:,:]
        adjY = adjY[lag:,:]

    if out2d:
        #NOTE: History will be succesive to each column (ie with history 5, columns 0-5 will be channel 1, 6-10
        # channel 2, etc..
        adjX = adjX.reshape(adjX.shape[0],-1)

    return adjX, adjY

def dataPrep(feats, hist, numChans=96):
    '''
    Get test and training data splits from a single session - also prep 2D and 3D versions

    '''
    TrialIndex = feats['trial_index']
    
    if len(TrialIndex) > 300:
        test_len = np.min((len(TrialIndex)-1, 399))

        # neural_training = feats['NeuralFeature'][:TrialIndex[300]]
        # neural_testing = feats['NeuralFeature'][TrialIndex[300]:TrialIndex[test_len]]

        # finger_training = feats['FingerAnglesTIMRL'][:TrialIndex[300]]
        # finger_testing = feats['FingerAnglesTIMRL'][TrialIndex[300]:TrialIndex[test_len]]

        neural_training = feats['sbp'][:TrialIndex[300]]
        neural_testing = feats['sbp'][TrialIndex[300]:TrialIndex[test_len]]

        finger_training = feats['finger_kinematics'][:TrialIndex[300]]
        finger_testing = feats['finger_kinematics'][TrialIndex[300]:TrialIndex[test_len]]

    else:
        pdb.set_trace()
        raise Exception('not enough trials')

        
    neural_training, finger_training = adjustfeats(neural_training, finger_training, hist = hist, out2d = True)
    neural_testing, finger_testing = adjustfeats(neural_testing, finger_testing, hist = hist, out2d = True)

    neural_testing = np.concatenate((neural_testing, np.ones((len(neural_testing), 1))), axis=1) # add a column of ones for RR
    neural_training = np.concatenate((neural_training, np.ones((len(neural_training), 1))), axis=1) # add a column of ones for RR

    return neural_training, neural_testing, finger_training, finger_testing

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