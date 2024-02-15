import os
import numpy as np
from datetime import datetime
import pandas as pd
from .utils import CO_MOLECULAR_WEIGHT, dfMonthWindows, dfYearWindows, dfDailyWindows, ppm_to_ug_per_m3

DB_PATH = 'datasets/Brasil/'

all_fileNames = ['BEN', 'CO', 'DV', 'DVG', 'ERT', 'MP10', 'MP25', 'NO', 'NO2', 'NOx', 'O3', 'PRESS', 'RADG', 'RADUV', 'SO2', 'TEMP', 'TOL', 'UR']

# Returns a Dict with levels pollutant-station-date
# granularity can be years or months
def read_brasil(files=all_fileNames, granularity='years', cache=True, max_missing=0.1, fill_missing=True):
    if fill_missing:
        DB_CACHE_PATH = os.path.join(DB_PATH, 'data_{}_{}.npy'.format(granularity, max_missing))
        DB_CACHE_ORIG_PATH = os.path.join(DB_PATH, 'data_orig_{}_{}.npy'.format(granularity, max_missing))
    else:
        DB_CACHE_PATH = os.path.join(DB_PATH, 'data_{}_{}.npy'.format(granularity, 0))
        DB_CACHE_ORIG_PATH = os.path.join(DB_PATH, 'data_orig_{}_{}.npy'.format(granularity, 0))
    
    if cache and os.path.exists(DB_CACHE_PATH):
        windows_map = np.load(DB_CACHE_PATH, allow_pickle=True)
        windows_map = windows_map[()]
        
        windows_orig_map = np.load(DB_CACHE_ORIG_PATH, allow_pickle=True)
        windows_orig_map = windows_orig_map[()]
        return windows_map, windows_orig_map
    
    windows_map={}
    windows_original_map={}
    for fname in all_fileNames:
        df = pd.read_csv(os.path.join(DB_PATH, fname))
        stations = pd.unique(df['stationname'])
        conc_map = {}
        conc_original_map = {}
        
        for station in stations:
            station_map = {}
            station_original_map = {}
            df_station = df[df['stationname'] == station]
            
            days = df_station['date'].to_numpy()
            days = [datetime.strptime(day, '%Y-%m-%d').date() for day in days ]
            hours = df_station['hour'].to_numpy()
            
            # replace value equal to 24 with 0
            hours = np.array([hour if hour != 24 else 0 for hour in hours])
            
            pol_datetimes = [datetime(days[i].year, days[i].month, days[i].day, hours[i]) for i in range(len(days))]
            
            df_conc = pd.DataFrame({'date': pol_datetimes, 'value': df_station['conc']})
            df_conc = df_conc.set_index('date')
            
            if granularity == 'years':
                values, dates = dfYearWindows(df_conc, fill_missing=fill_missing, maxMissing=max_missing)    
            elif granularity == 'months':
                values, dates = dfMonthWindows(df_conc, fill_missing=fill_missing, maxMissing=max_missing)
            elif granularity == 'daily':
                values, dates = dfDailyWindows(df_conc, fill_missing=fill_missing, maxMissing=max_missing)
            
            for k in range(len(values)):
                # dKey = '{}-{}-{}'.format(dates[k].year, dates[k].month, dates[k].day)
                dKey = str(dates[k])
                pollutant = fname
                station_original_map[dKey] = (values[k], dates[k])
                if pollutant == 'CO':
                    values[k] = ppm_to_ug_per_m3(values[k], CO_MOLECULAR_WEIGHT)
                
                station_map[dKey] = (values[k], dates[k])
            
            conc_map[station] = station_map
            conc_original_map[station] = station_original_map
        windows_map[fname] = conc_map
        windows_original_map[fname] = conc_original_map
    
    np.save(DB_CACHE_PATH, windows_map)
    np.save(DB_CACHE_ORIG_PATH, windows_original_map)
    return windows_map, windows_original_map
    
