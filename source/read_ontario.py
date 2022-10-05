import pandas as pd
import numpy as np
import os
import io


from .utils import dfMonthWindows, dfDailyWindows, dfYearWindows, tryFillMissing
years = ['2010', '2011', '2012', '2013', '2014', '2015','2016', '2017', '2018', '2019', '2020']
pollutants = ['NO', 'NOx', 'NO2', 'SO2', 'CO', 'O3', 'PM25']
# pollutants = ['CO', 'O3']

DB_PATH = 'datasets/ontario/'

def read_ontario(granularity='years', cache=True, maxMissing=0.2):
    DB_CACHE_PATH = os.path.join(DB_PATH, 'data_{}.npy'.format(granularity))
    
    if cache and os.path.exists(DB_CACHE_PATH):
        windows_map = np.load(DB_CACHE_PATH, allow_pickle=True)
        windows_map = windows_map[()]
        return windows_map
        
    
    windows_map = {}
    for pollutant in pollutants:
        conc_map = {}
        aux_map = {}
        for i in range(len(years)):
            with open(os.path.join(DB_PATH, "{}-{}.csv".format(pollutant, years[i])), "r") as f:
                data = f.read()
            
            slices = data.split("\n\n")
            for slice in slices:
                pos = slice.find('Station ID')
                if pos != -1:
                    content = slice[pos:]
                    content = content.replace(',\n', '\n')
                    content = content[:-1]
                    df = pd.read_csv(io.StringIO(content), sep=',', header =0)
                    station = df['Station ID'][0]
                    
                
                    hours = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
                    hoursD = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
                    values = np.zeros((365, 24))
                    dates = []
                    for index, row in df.iterrows():
                        if index < 365:
                            for k in range(len(hours)):
                                values[index][k] = (row['H{}'.format(hours[k])])
                                if granularity == 'daily':
                                    dateStr = '{}T{}:00:00.000000000'.format(row['Date'], hoursD[k])
                                    date = pd.to_datetime(dateStr, infer_datetime_format=True)
                                    dates.append(date)
                                elif k == 0:
                                    dateStr = '{}T{}:00:00.000000000'.format(row['Date'], hoursD[k])
                                    date = pd.to_datetime(dateStr, infer_datetime_format=True)
                                    dates.append(date)
                    values[values==9999]=np.nan
                    values[values==-999]=np.nan
                    if granularity != 'daily':
                        values = np.mean(values, axis=1)
                        values = tryFillMissing(values, maxMissing=maxMissing)
                    else:
                        values = values.flatten()
                    #     for t in range(len(values)):
                    #         values[t] = tryFillMissing(values[t], maxMissing=maxMissing)
                    
                    values = values[:len(dates)]
                    dates = np.array(dates)
                    # print(dates)
                    # print(values)
                    # print('---')
                    # print('Values shape: {}'.format(values.shape))
                    # print('Dates shape: {}'.format(dates.shape))
                    # print('---')
                    # return 
                    
                    if  station not in aux_map:
                        aux_map[station] = {}
                    
                    if pollutant not in aux_map[station]:
                        aux_map[station][pollutant] = (values, dates)
                    else:
                        currValues, currDates = aux_map[station][pollutant]
                        aux_map[station][pollutant] = (np.concatenate([currValues, values], axis=0), np.concatenate([currDates, dates], axis=0))

        stations = list(aux_map.keys())
        for station in stations:
            station_map = {}
            values, pol_datetimes = aux_map[station][pollutant]
            df_conc = pd.DataFrame({'date': pol_datetimes, 'value': values})
            df_conc = df_conc.set_index('date')
            if granularity == 'years':
                values, dates = dfYearWindows(df_conc)    
            elif granularity == 'months':
                values, dates = dfMonthWindows(df_conc)
            elif granularity == 'daily':
                values, dates = dfDailyWindows(df_conc)
            for k in range(len(values)):
                dKey = '{}-{}'.format(dates[k].year, dates[k].month)
                station_map[dKey] = (values[k], dates[k])
            conc_map[station] = station_map
        windows_map[pollutant] = conc_map
    
    np.save(DB_CACHE_PATH, windows_map)
    return windows_map
