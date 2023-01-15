import os
import numpy as np
from datetime import datetime
import pandas as pd
from .utils import dfMonthWindows, dfYearWindows, dfDailyWindows

DB_PATH = 'datasets/brasil/Data_AirQuality/'

all_fileNames = ['BEN', 'CO', 'DV', 'DVG', 'ERT', 'MP10', 'MP25', 'NO', 'MP25', 'NOx', 'O3', 'PRESS', 'RADG', 'RADUV', 'SO2', 'TEMP', 'TOL', 'UR']

# Returns a Dict with levels pollutant-station-date
# granularity can be years or months
def read_brasil(files=all_fileNames, granularity='years', cache=True):
    DB_CACHE_PATH = os.path.join(DB_PATH, 'data_{}.npy'.format(granularity))
    
    if cache and os.path.exists(DB_CACHE_PATH):
        windows_map = np.load(DB_CACHE_PATH, allow_pickle=True)
        windows_map = windows_map[()]
        return windows_map
    
    windows_map={}
    for fname in all_fileNames:
        df = pd.read_csv(os.path.join(DB_PATH, fname))
        stations = pd.unique(df['stationname'])
        conc_map = {}
        
        for station in stations:
            station_map = {}
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
                values, dates = dfYearWindows(df_conc, fill_missing=True)    
            elif granularity == 'months':
                values, dates = dfMonthWindows(df_conc, fill_missing=True)
            elif granularity == 'daily':
                values, dates = dfDailyWindows(df_conc)
            
            for k in range(len(values)):
                # dKey = '{}-{}-{}'.format(dates[k].year, dates[k].month, dates[k].day)
                dKey = str(dates[k])
                station_map[dKey] = (values[k], dates[k])
            
            conc_map[station] = station_map
        windows_map[fname] = conc_map
    
    np.save(DB_CACHE_PATH, windows_map)
    return windows_map
    
