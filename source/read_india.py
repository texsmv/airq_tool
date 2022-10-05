import pandas as pd
import numpy as np
import os
from .utils import dfMonthWindows, dfYearWindows

DB_PATH = 'datasets/india/'

years = [2015, 2016, 2017, 2018, 2019, 2020]
pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']


def read_india(granularity='years'):
    data = pd.read_csv(os.path.join(DB_PATH, 'india_city_day.csv'))
    data_by_city = data.groupby('City')
    
    windows_map = {}
    for pollutant in pollutants:
        conc_map = {}
        for cityTuple in data_by_city:
            station_map = {}
            station, df = cityTuple
            df['Date'] = pd.to_datetime(df['Date'])
            
            df_conc = pd.DataFrame({'date': df['Date'], 'value': df[pollutant]})
            df_conc = df_conc.set_index('date')
            
            if granularity == 'years':
                values, dates = dfYearWindows(df_conc)    
            elif granularity == 'months':
                values, dates = dfMonthWindows(df_conc)
            
            for k in range(len(values)):
                dKey = '{}-{}'.format(dates[k].year, dates[k].month)
                station_map[dKey] = (values[k], dates[k])
            
            conc_map[station] = station_map
        windows_map[pollutant] = conc_map
    return windows_map

