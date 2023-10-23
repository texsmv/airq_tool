import os
import numpy as np
from datetime import datetime
import pandas as pd
from .utils import dfMonthWindows, dfYearWindows, dfDailyWindows

DB_PATH = 'datasets/HongKong/'

def read_hongkong(granularity='years', cache=True, max_missing=0.1, fill_missing=True):
    if fill_missing:
        DB_CACHE_PATH = os.path.join(DB_PATH, 'data_{}_{}.npy'.format(granularity, max_missing))
    else:
        DB_CACHE_PATH = os.path.join(DB_PATH, 'data_{}_{}.npy'.format(granularity, 0))
    if cache and os.path.exists(DB_CACHE_PATH):
        windows_map = np.load(DB_CACHE_PATH, allow_pickle=True)
        windows_map = windows_map[()]
        return windows_map
    
    initial_year = 1990
    final_year = 2023
    # initial_year = 2009
    # final_year = 2019
    year_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
    ]
    print('[HONG KONG] - Reading ...')
    dframes = []
    index = 0
    for year in range(initial_year, final_year + 1):
        file_name = '{}_EN.xlsx'.format(year)
        df = pd.read_excel(DB_PATH + file_name, skiprows=11)
        
        df['HOUR'] = df['HOUR'].apply(lambda x: '{0:0>2}'.format(x - 1))
        df['datetime'] = pd.to_datetime(df['DATE'].astype(str) + ' ' +  df['HOUR'].astype(str) + ':00:00', format=year_formats[index])
        dframes.append(df)
        index = index + 1
    
    all_df = pd.concat(dframes)
    all_df = all_df.replace('N.A.', np.NaN)
    
    all_df['STATION'] = all_df['STATION'].str.upper()
    all_df['STATION'] = all_df['STATION'].str.strip()
    stations = all_df['STATION'].unique()
    
    
    
    # all_df = pd.concat(dframes)
    # all_df = all_df.replace('N.A.', np.NaN)
    
    # all_df['STATION'] = all_df['STATION'].str.upper()
    # all_df['STATION'] = all_df['STATION'].str.strip()
    # stations = all_df['STATION'].unique()
    
    # all_df['HOUR'] = all_df['HOUR'].apply(lambda x: '{0:0>2}'.format(x - 1))
    # all_df['datetime'] = pd.to_datetime(all_df['DATE'].astype(str) + ' ' +  all_df['HOUR'].astype(str) + ':00:00', format="%Y/%m/%d: ")
    
    
    windows_map = {}
    # stations = ['TAP MUN']
    for station in stations:
        df = all_df[all_df['STATION'] == station]
        df = df.drop(columns=['DATE', 'HOUR', 'STATION'])
        df = df.set_index('datetime')
        df = df.dropna(axis=1, how='all')
        columns = df.columns
        
        for pol in df.columns:
            if not pol in windows_map:
                windows_map[pol] = {}
            
            station_map = {}
            df_conc = pd.DataFrame({'date': df.index, 'value': df[pol]})
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
                # * Convert to uniform ranges
                if pol == 'CO':
                    values[k] = (values[k] * 10.0) * 0.001
                station_map[dKey] = (values[k], dates[k])
            windows_map[pol][station] = station_map
    print('[HONG KONG] - Reading done')
    
    np.save(DB_CACHE_PATH, windows_map)
    return windows_map
        