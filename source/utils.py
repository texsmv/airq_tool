import pandas as pd
import numpy as np
import random
from scipy import interpolate
import os


def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:  
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]

def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
    B = np.where(np.isfinite(A),A,f(inds))
    return B

def tryFillMissing(window, maxMissing = 0.2):
    missPercentage = np.isnan(window).sum() / window.size
    if missPercentage < maxMissing:
        return fill_nan(window)
    return window

def intersection(lst1, lst2): 
    temp = set(lst2) 
    lst3 = [value for value in lst1 if value in temp] 
    return lst3 

def dfMonthWindows(pol_df):
    months = [g for n, g in pol_df.groupby(pd.Grouper(freq='M'))]
    dates = [n for n, g in pol_df.groupby(pd.Grouper(freq='M'))]
    count = 0
    monthlyValues = []
    monthlyDates = []
    yearlyDates = []
    for j in range(len(months)):
        m_df = months[j]
        m_date = dates[j]
        
        
        temp = m_df.resample('D').mean()
        
        
        if np.count_nonzero(np.isnan(temp.to_numpy().squeeze())) == 0:
            if len(temp) < 28:
                continue
            monthlyValues.append(temp.to_numpy()[:28])
            monthlyDates.append(m_date)
        else:
            count = count + 1
    monthlyValues = np.array(monthlyValues)
    monthlyDates = np.array(monthlyDates)
    
    return monthlyValues, monthlyDates

def dfDailyWindows(pol_df):
    print('SHAPE')
    print(pol_df.shape)
    print(pol_df.head())
    months = [g for n, g in pol_df.groupby(pd.Grouper(freq='D'))]
    dates = [n for n, g in pol_df.groupby(pd.Grouper(freq='D'))]
    count = 0
    monthlyValues = []
    monthlyDates = []
    yearlyDates = []
    print(len(months))
    for j in range(len(months)):
        m_df = months[j]
        m_date = dates[j]
        # temp = m_df.resample('D').mean()
        temp = m_df
        
        if np.count_nonzero(np.isnan(temp.to_numpy().squeeze())) == 0:
            if len(temp) < 24:
                # print(temp)
                # print(len(temp))
                continue
            monthlyValues.append(temp.to_numpy())
            monthlyDates.append(m_date)
        else:
            count = count + 1
    monthlyValues = np.array(monthlyValues)
    monthlyDates = np.array(monthlyDates)
    
    return monthlyValues, monthlyDates

def dfYearWindows(pol_df):
    years = [g for n, g in pol_df.groupby(pd.Grouper(freq='Y'))]
    dates = [n for n, g in pol_df.groupby(pd.Grouper(freq='Y'))]
    count = 0
    
    values = []
    yearlyDates = []
    for j in range(len(years)):
        m_df = years[j]
        m_date = dates[j]
        
        temp = m_df.resample('D').mean()
        
        
        if np.count_nonzero(np.isnan(temp.to_numpy().squeeze())) == 0:
            if len(temp) < 365:
                continue
            values.append(temp.to_numpy()[:365])
            yearlyDates.append(m_date)
        else:
            count = count + 1
    values = np.array(values)
    yearlyDates = np.array(yearlyDates)
    
    return values, yearlyDates




def commonWindows(windows_map,  pollutants):
    # ---------------------------------Get list of filtered stations---------------------------
    stations = []
    for pol in pollutants:
        stations = stations + list(windows_map[pol].keys()) 
    stations = np.unique(np.array(stations))
    all_windows = []
    all_dates = []
    all_stations = []
    for st in range(len(stations)):
        station = stations[st]
        
        # ---------------------- Check if the station has all the pollutants--------------------
        skip = False
        for pol in pollutants:
            if station not in windows_map[pol]:
                skip = True
        if skip:
            continue
        
        # ------------------------- Get common dates of the station------------------------------
        common_dates = []
        is_first_time = True
        for pol in pollutants:
            stationDates = list(windows_map[pol][station].keys())
            # print(stationDates)
            if len(common_dates) == 0:
                if is_first_time:
                    common_dates = stationDates
                    is_first_time = False
                else:
                    break
            else:
                common_dates = intersection(common_dates, stationDates)
        if len(common_dates) == 0:
            continue
                
        # ------------------------ Get the windows from the common dates-------------------------
        station_windows = []
        station_dates = []
        for pol in pollutants:
            dim_windows = np.array([windows_map[pol][station][dstr][0] for dstr in common_dates])
            dim_dates = np.array([windows_map[pol][station][dstr][1] for dstr in common_dates])
            if len(station_windows) == 0:
                station_windows = dim_windows
                station_dates = dim_dates
            else:
                station_windows = np.concatenate([station_windows, dim_windows], axis=2)
        station_ids = np.ones(len(station_windows)) * st
        
        # ------------------------------ Add windows to all windows-----------------------------
        if len(station_windows) == 0:
            continue
        
        if len(all_windows) == 0:
            all_windows = station_windows
            all_dates = station_dates
            all_stations = station_ids
        else:
            all_windows = np.concatenate([all_windows, station_windows],  axis = 0)
            all_dates = np.concatenate([all_dates, station_dates],  axis = 0)
            all_stations = np.concatenate([all_stations, station_ids],  axis = 0)
    
    return all_windows, all_dates, all_stations.astype(int), stations


def rescale(val, in_min, in_max, out_min, out_max):
    return out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))

# For univariate time series
def decomposed_normalization(windows):
    norm_windows = np.copy(windows)

    mean = np.mean(norm_windows, axis=0)
    all_std = np.std(norm_windows)

    means = np.array([np.mean(e) for e in norm_windows])

    norm_windows = np.array([(e - np.mean(e))/all_std for e in norm_windows])

    min_val = np.min(norm_windows)
    max_val = np.max(norm_windows)

    norm_windows = np.array([ [rescale(i, min_val, max_val, 0, 1) for i in e]  for e in norm_windows]) 
    return norm_windows, mean, means


# For univariate time series only
def normal_normalization(windows, minv=None, maxv=None):
    norm_windows = np.copy(windows)

    if minv is None:
        min_val = np.min(norm_windows)
        max_val = np.max(norm_windows)
    else:
        min_val = minv
        max_val = maxv

    norm_windows = np.array([ [rescale(i, min_val, max_val, 0, 1) for i in e]  for e in norm_windows]) 

    return norm_windows, min_val, max_val

# Shape normalization for multivariate time series
def mts_shape_norm(X):
    X_tr = X.transpose([0, 2, 1])
    N, D, T = X_tr.shape
    print(X_tr.shape)
    shapes = [None] * D
    means = [None]* D
    magnitudes = [None] * D
    for d in range(D):
        shape, mean, magnitude = decomposed_normalization(X_tr[:,d,:])
        shapes[d] = shape
        means[d] = mean
        magnitudes[d] = magnitude
    return np.array(shapes).transpose(1,2,0), np.array(means), np.array(magnitudes).transpose()

# MinMax normalization for multivariate time series
def mts_norm(X, minl = [], maxl= []):
    norm_X = X.transpose([0, 2, 1])
    N, D, T = norm_X.shape
    min_l = []
    max_l = []
    for d in range(D):
        if len(minl) == 0:
            norm_windows, minv, maxv = normal_normalization(norm_X[:,d,:])
        else:
            norm_windows, minv, maxv = normal_normalization(norm_X[:,d,:], minv=minl[d], maxv=maxl[d])
        min_l.append(minv)
        max_l.append(maxv)
        norm_X[:,d, :] = norm_windows
    return norm_X.transpose([0, 2, 1]), min_l, max_l


# Smoothing to a mts
def mts_smooth(X, window_len = 20):
    X_smooth = X.transpose([0, 2, 1])
    N, D, T = X_smooth.shape
    for i in range(N):
        for k in range(D):
            X_smooth[i][k] = smooth(X_smooth[i][k], window_len = window_len)
    return X_smooth.transpose([0, 2, 1])



def create_dir(path):
    # Check whether the specified path exist or not
    isExist = os.path.exists(path)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(path)
      print("The new directory is created!")



# 3D tensor unfolding
# Collapses the first two dimensions into one e.g. (N, D, T) -> (N x D, T)
def unfolding_3D(X):
    N, D, T = X.shape
    X_u = np.zeros((N * D, T))
    for n in range(N):
        X_u[n * D:(n + 1) * D, :] = X[:, n, :].T
    return X_u

# Folds the an array into two a 2D array
def folding_2D(X, N, D):
    return X.reshape((N, D))
    