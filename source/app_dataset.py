import numpy as np
from .read_brasil import read_brasil
from .utils import commonWindows, sample_data

class OntarioDataset:
    def __init__(self):
        a = 1
        

class BrasilDataset:
    all_pollutants = [
        'BEN', 'CO', 'DV', 'DVG', 'ERT', 
        'MP10', 'MP25', 'NO', 'MP25', 'NOx', 
        'O3', 'PRESS', 'RADG', 'RADUV', 'SO2',
        'TEMP', 'TOL', 'UR']
    
    def __init__(self, granularity='years', cache=True):
        self.windows_map = read_brasil(granularity='months', cache=True)
        self.pollutants = list(self.windows_map.keys())
        
        
        # Get stations and dates ranges
        all_stations = []
        for pol in self.pollutants:
            all_stations = all_stations + list(self.windows_map[pol].keys())
        all_stations = np.unique(np.array(all_stations))


        all_dates = []
        for pol in self.pollutants:
            stations = list(self.windows_map[pol].keys())
            for station in stations:
                all_dates = all_dates + list(self.windows_map[pol][station].keys())
                
        all_dates = np.unique(np.array(all_dates))
        
        all_dates = [np.datetime64(date) for date in all_dates]
        all_dates.sort()
        
        self.dates = all_dates
        self.stations = all_stations

    def common_windows(self, pollutants):
        self.windows, self.window_dates, self.window_station_ids, self.window_stations = commonWindows(self.windows_map, pollutants)
        self.window_pollutants = pollutants
        
        self.window_stations = self.window_stations[np.unique(self.window_station_ids)] 
    
    def dateRanges(self):
        return self.dates[0], self.dates[-1]


    