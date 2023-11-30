import pandas as pd
import numpy as np
import random
from scipy import interpolate
import os
from scipy import stats
from skfda.exploratory.visualization import MagnitudeShapePlot
import skfda
from skfda.representation.interpolation import SplineInterpolation
from skfda.exploratory.depth.multivariate import SimplicialDepth
import skfda.representation.basis as basis
import aqi

AVAILABLE_POLUTANTS = [
    'O3', 'PM25', 'PM10', 'FSP', 'NO2', 'SO2', 'CO',
    'MP25', 'MP10',
    'FSP', 'RSP'
]

CO_MOLECULAR_WEIGHT = 28.01
NO2_MOLECULAR_WEIGHT = 46.01
O3_MOLECULAR_WEIGHT = 48.00
SO2_MOLECULAR_WEIGHT = 64.06

def ppb_to_ug_per_m3(ppb, molecular_weight):
    """
    Convert parts per billion (ppb) to micrograms per cubic meter (µg/m³).
    
    Args:
    ppb (float): Concentration in parts per billion.
    molecular_weight (float): Molecular weight of the substance in g/mol.
    
    Returns:
    float: Concentration in µg/m³.
    """
    constant = 24.45
    micrograms_per_cubic_meter = (ppb * molecular_weight) / constant
    return micrograms_per_cubic_meter

def ppm_to_mg_per_m3(ppm, molecular_weight):
    """
    Convert parts per million (ppm) to milligrams per cubic meter (mg/m³).
    
    Args:
    ppm (float): Concentration in parts per million.
    molecular_weight (float): Molecular weight of the substance in g/mol.
    
    Returns:
    float: Concentration in mg/m³.
    """
    constant = 24.45
    mg_per_m3 = (ppm * molecular_weight) / constant
    return mg_per_m3

def daily_iaqi(pollutant, data):
    if pollutant == 'O3':
        N = 24
        subarrays = np.array([data[i:i + 8] for i in range(N - 8 + 1)])
        averages = [np.mean(e) for e in subarrays]
        max_val = max(averages)
        return aqi.to_iaqi('o3_8h', str(max_val), algo=aqi.ALGO_MEP)
    elif pollutant == 'PM25' or pollutant == 'FSP' or pollutant == 'MP25':
        d_mean = data.mean()
        return aqi.to_iaqi(aqi.POLLUTANT_PM25, str(d_mean), algo=aqi.ALGO_MEP)
    elif pollutant == 'PM10' or pollutant == 'RSP' or pollutant == 'MP10':
        d_mean = data.mean()
        return aqi.to_iaqi(aqi.POLLUTANT_PM10, str(d_mean), algo=aqi.ALGO_MEP)
    elif pollutant == 'NO2':
        d_mean = data.mean()
        return aqi.to_iaqi('no2_24h', str(d_mean), algo=aqi.ALGO_MEP)
    elif pollutant == 'SO2':
        d_mean = data.mean()
        return aqi.to_iaqi('so2_24h', str(d_mean), algo=aqi.ALGO_MEP)
    elif pollutant == 'CO':
        d_mean = data.mean()
        return aqi.to_iaqi('co_24h', str(d_mean), algo=aqi.ALGO_MEP)

def format_date(date, gran):
    if gran == 'daily':
        return '{}_{}_{}'.format( date.year, date.month, date.day)
    elif gran == 'monthly':
        return '{}_{}'.format( date.year, date.month)
    elif gran == 'annualy':
        return '{}'.format(date.year)

def get_annualy_iaqis_map(daily_iaqi):
    iaqi = {}
    for polK, polD in daily_iaqi.items():
        if  polK not in AVAILABLE_POLUTANTS:
            continue
        iaqi[polK] = {}
        for statK, statD in polD.items():
            iaqi[polK][statK] = {}
            for yearK, yearD in statD.items():
                values = []
                for monthK, monthD in yearD.items():
                    values = values + list(monthD.values())
                    
                values = np.array(values)
                iaqi[polK][statK][yearK] = np.mean(values)
                
    return iaqi

def get_monthly_iaqis_map(daily_iaqi):
    monthly_iaqi = {}
    for polK, polD in daily_iaqi.items():
        if  polK not in AVAILABLE_POLUTANTS:
            continue
        monthly_iaqi[polK] = {}
        for statK, statD in polD.items():
            monthly_iaqi[polK][statK] = {}
            for yearK, yearD in statD.items():
                monthly_iaqi[polK][statK][yearK] = {}
                for monthK, monthD in yearD.items():
                    # print(np.array(list(monthD.values())))
                    values = np.array(list(monthD.values()))
                    monthly_iaqi[polK][statK][yearK][monthK] = np.mean(values)
                
    return monthly_iaqi

def get_daily_iaqis_map(win_map):
    windows_aqis = {}
    for polK, polD in win_map.items():
        if  polK not in AVAILABLE_POLUTANTS:
            continue
        windows_aqis[polK] = {}
        for statK, statD in polD.items():
            windows_aqis[polK][statK] = {}
            for dateK, dateD in statD.items():
                date = dateD[1]
                data = dateD[0]
                if dateD[0].shape[0] != 24:
                    print('Error, not 24 days')
                else:
                    try:
                        if date.year not in windows_aqis[polK][statK]:
                            windows_aqis[polK][statK][date.year] = {}
                            
                        if date.month not in windows_aqis[polK][statK][date.year]:
                            windows_aqis[polK][statK][date.year][date.month] = {}

                        windows_aqis[polK][statK][date.year][date.month][date.day] = int(daily_iaqi(polK, data))
                    except:
                        print('Exception while getting daily_iaqi')
    return windows_aqis



class ValueLogger(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, epoch_freq = 5):
        self.name = name
        self.epoch_freq = epoch_freq
        self.reset()
    
    def reset(self):
        self.avgs = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0
        self.bestAvg = np.inf
        
        
    def end_epoch(self):
        
        self.avgs = self.avgs + [self.avg]
        self.val = 0
        self.sum = 0
        self.count = 0.0
        if len(self.avgs) == 1 or len(self.avgs) % self.epoch_freq == 0:
            print("Epoch[{}] {} {}: {}".format(len(self.avgs), self.name, "avg", self.avg))
    
        if self.bestAvg > self.avg:
            self.bestAvg = self.avg
            return True
        else:
            return False

    # Updates de value history
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



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
    # TODO fix
    # f = interpolate.interp1d(inds[good], A[good], kind='cubic', bounds_error=False)
    # B = np.where(np.isfinite(A),A,f(inds))
    f = np.zeros(A.shape)
    B = np.where(np.isfinite(A),A,f)
    B[B < 0] = 0
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

def dfMonthWindows(pol_df, fill_missing=False, maxMissing=0.1):
    months = [g for n, g in pol_df.groupby(pd.Grouper(freq='M'))]
    dates = [n for n, g in pol_df.groupby(pd.Grouper(freq='M'))]
    count = 0
    monthlyValues = []
    monthlyDates = []
    yearlyDates = []
    for j in range(len(months)):
        m_df = months[j]
        m_date = dates[j]
        
        
        temp = m_df.resample('D').mean().to_numpy()
        
        if fill_missing and len(temp) >= 28:
            # print(temp.shape)
            temp = tryFillMissing(temp.squeeze(), maxMissing=maxMissing)
            temp = np.expand_dims(temp, 1)
        
        if np.count_nonzero(np.isnan(temp.squeeze())) == 0:
            if len(temp) < 28:
                continue
            monthlyValues.append(temp[:28])
            monthlyDates.append(m_date)
        else:
            count = count + 1
    monthlyValues = np.array(monthlyValues)
    monthlyDates = np.array(monthlyDates)
    
    return monthlyValues, monthlyDates

def dfDailyWindows(pol_df, fill_missing=False, maxMissing=0.1):
    months = [g for n, g in pol_df.groupby(pd.Grouper(freq='D'))]
    dates = [n for n, g in pol_df.groupby(pd.Grouper(freq='D'))]
    count = 0
    monthlyValues = []
    monthlyDates = []
    yearlyDates = []
    for j in range(len(months)):
        m_df = months[j]
        m_date = dates[j]
        # temp = m_df.resample('D').mean()
        temp = m_df.to_numpy()
        
        if fill_missing:
            temp = tryFillMissing(temp.squeeze(), maxMissing=maxMissing)
            temp = np.expand_dims(temp, 1)
        
        if np.count_nonzero(np.isnan(temp.squeeze())) == 0:
            if len(temp) < 24:
                continue
            monthlyValues.append(temp)
            monthlyDates.append(m_date)
        else:
            count = count + 1
    monthlyValues = np.array(monthlyValues)
    monthlyDates = np.array(monthlyDates)
    
    return monthlyValues, monthlyDates

def dfYearWindows(pol_df, fill_missing=False, maxMissing=0.1):
    years = [g for n, g in pol_df.groupby(pd.Grouper(freq='Y'))]
    dates = [n for n, g in pol_df.groupby(pd.Grouper(freq='Y'))]
    count = 0
    
    values = []
    yearlyDates = []
    for j in range(len(years)):
        m_df = years[j]
        m_date = dates[j]
        
        temp = m_df.resample('D').mean().to_numpy()
        
        if fill_missing and len(temp) >= 365:
            # print(temp.shape)
            temp = tryFillMissing(temp.squeeze(), maxMissing=maxMissing)
            temp = np.expand_dims(temp, 1)
        
        if np.count_nonzero(np.isnan(temp.squeeze())) == 0:
            if len(temp) < 365:
                continue
            values.append(temp[:365])
            yearlyDates.append(m_date)
        else:
            count = count + 1
    values = np.array(values)
    yearlyDates = np.array(yearlyDates)
    
    # print('Missing values size: {}'.format(count))
    
    return values, yearlyDates


def getAllStations(windows_map, pollutants):
    stations = []
    for pol in pollutants:
        stations = stations + list(windows_map[pol].keys()) 
    stations = np.unique(np.array(stations))
    return stations


def getCommonDates(windows_map, station, pollutants):
    common_dates = []
    first = True
    
    for pol in pollutants:
        stationDates = list(windows_map[pol][station].keys())
        
        if len(common_dates) == 0 and first:
            common_dates = stationDates
            first = False
            
        else:
            common_dates = intersection(common_dates, stationDates)
    return common_dates

def commonWindows(windows_map,  pollutants, inStations):
    # ---------------------------------Get list of filtered stations---------------------------
    # Counting the number of times it was added to stations
    stations = []
    for pol in pollutants:
        stations = stations + list(windows_map[pol].keys()) 
    
    uniques_stations, station_counts = np.unique(stations, return_counts=True)
    stations = uniques_stations[station_counts == len(pollutants)]
    
    stations = np.unique(np.array(stations)).tolist()
    print('')
    print('Stations that have all pollutants: {}  -  {}'.format(len(stations), stations))
    
    
    # def intersection(lst1, lst2):
    #     lst3 = [value for value in lst1 if value in lst2]
    #     return lst3
    
    # Stations that do have the pollutants
    
    possible_stations = intersection(stations, inStations)
    print('Possible stations')
    print(possible_stations)
    present_stations = []
    
    all_windows = []
    all_dates = []
    all_stations_ids = []
    st = 0
    
    for k in range(len(possible_stations)):
        station = possible_stations[k]
        print(station)
        # ------------------------- Get common dates of the station------------------------------
        print(pollutants)
        common_dates = getCommonDates(windows_map, station, pollutants)
        if len(common_dates) == 0:
            continue
        else:
            present_stations.append(station)
        
        print(common_dates)
        # ------------------------ Get the windows from the common dates-------------------------
        station_windows = []
        station_dates = []
        for pol in pollutants:
            print(pol)
            print(windows_map[pol][station].keys())
            dim_windows = np.array([windows_map[pol][station][dstr][0] for dstr in common_dates])
            dim_dates = np.array([windows_map[pol][station][dstr][1] for dstr in common_dates])
            
            if len(station_windows) == 0:
                station_windows = dim_windows
                station_dates = dim_dates
            else:
                station_windows = np.concatenate([station_windows, dim_windows], axis=2)
        station_ids = np.ones(len(station_windows)) * st
        
        
        # ------------------------------ Add windows to all windows-----------------------------
        # if len(station_windows) == 0:
            # continue
        
        if len(all_windows) == 0:
            all_windows = station_windows
            all_dates = station_dates
            all_stations_ids = station_ids
        else:
            all_windows = np.concatenate([all_windows, station_windows],  axis = 0)
            all_dates = np.concatenate([all_dates, station_dates],  axis = 0)
            all_stations_ids = np.concatenate([all_stations_ids, station_ids],  axis = 0)
        st = st + 1
    
    if len(all_stations_ids) != 0:
        all_stations_ids = all_stations_ids.astype(int)
    return all_windows, all_dates, all_stations_ids, np.array(present_stations)


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


def sample_data(arr, n_samples):
    n = len(arr[0])
    indices = np.array(list(range(n)))
    np.random.shuffle(indices)
    
    sampled_ind = indices[:n_samples]
    
    return [ar[sampled_ind] for ar in arr]



def fdaOutlier(X):
    
    dts = np.transpose(X)
    median_vec = np.median(dts, axis=1)
    mad_vec = stats.median_abs_deviation(dts, axis=1)
    dir_out_matrix = np.subtract(dts.transpose(), median_vec) / mad_vec
    mean_dir_out = np.mean(dir_out_matrix, axis=1)
    var_dir_out = np.var(dir_out_matrix, axis=1)
    
    return mean_dir_out, var_dir_out
    
def magnitude_shape_plot(X):
    t = np.arange(X.shape[1])
    fd = skfda.FDataGrid(X, t)
    
    msplot = MagnitudeShapePlot(
        fd,
        multivariate_depth=SimplicialDepth(),
        # multivariate_depth=Outlyingness(),
    )
    return msplot.points[:,1], msplot.points[:,0], msplot.outliers

def ts_to_basis(X, n_basis):
    t = np.arange(X.shape[1])
    fd = skfda.FDataGrid(X, t)
    fd_basis = fd.to_basis(basis.FourierBasis(n_basis=n_basis))
    data = fd_basis.to_grid(grid_points=t).data_matrix.squeeze()
    return data
    

    # fd_basis = fd.to_basis(basis.BSplineBasis(n_basis=30))
#     dts = np.transpose(X)
#     # median_vec = np.median(dts, axis=1)
#     mad_vec = stats.median_abs_deviation(dts, axis=1)
#     dir_out_matrix = dts.transpose() / mad_vec
#     mean_dir_out = np.mean(dir_out_matrix, axis=1)
#     var_dir_out = np.var(dir_out_matrix, axis=1)
    
#     return mean_dir_out, var_dir_out

station_locations = {
    'Capão Redondo': (-23.6719026,-46.7794354),
    'Cerqueira César': (-23.0353135,-49.1650519),
    'Cid.Universitária-USP-Ipen': (-23.557594299316406,-46.71200180053711),
    'Congonhas': (-20.5015168,-43.8564586),
    'Ibirapuera': (-14.8428108,-40.8546285),
    'Interlagos': (-23.7019315,-46.6967078),
    'Itaim Paulista': (-23.5017648,-46.3996091),
    'Itaquera': (-23.5360799,-46.4555099),
    'Marg.Tietê-Pte Remédios': (-23.516924,-46.733631),
    'Mooca': (-23.5606808,-46.5971924),
    'N.Senhora do Ó': (-8.4720591,-35.0103062),
    'Osasco': (-8.399660110473633,-35.06126022338867),
    'Parelheiros': (-23.827312,-46.7277941),
    'Parque D.Pedro II': (-23.5508698,-46.6275136),
    'Pico do Jaraguá': (-23.4584254,-46.7670295),
    'Pinheiros': (-23.567249,-46.7019515),
    'Santana': (-12.979217,-44.05064),
    'Santo Amaro': (-12.5519686,-38.7060448),
}

# def getStationGps(station):
