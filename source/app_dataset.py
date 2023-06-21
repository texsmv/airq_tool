import numpy as np
from .read_brasil import read_brasil
from .read_hongkong import read_hongkong
from .read_ontario import read_ontario, read_ontario_stations
from .utils import commonWindows, sample_data, getAllStations

class OntarioDataset:
    all_pollutants = ['NO', 'NOx', 'NO2', 'SO2', 'CO', 'O3', 'PM25']
    def __init__(self, granularity='years', cache=True, fill_missing=True, max_missing=0.1):
        self.windows_map = read_ontario(granularity=granularity, cache=cache, fill_missing=fill_missing, max_missing=max_missing)
        self.stations_map = read_ontario_stations()
        self.pollutants = list(self.windows_map.keys())
        
        
        n_map = {}
        for pol_k, pol_d in self.windows_map.items():
            # n_map[pol_k] = {}
            # Eg for 44008, _ in ...
            pol_dict = self.windows_map[pol_k]
            n_pol_dict = {}
            for sta_k, sta_d in pol_dict.items():
                real_name = self.stations_map[str(sta_k)]['name']
                stat_dict = pol_dict[sta_k]
                
                if real_name in n_pol_dict:
                    # print('-')
                    real_dict = n_pol_dict[real_name]
                    for win_k, win_d in stat_dict.items():
                        real_dict[win_k] = win_d
                else:
                    n_pol_dict[real_name] = stat_dict
            n_map[pol_k] = n_pol_dict
        self.windows_map = n_map
        
        # Get stations and dates ranges
        # all_stations = []
        # for pol in self.pollutants:
        #     all_stations = all_stations + list(self.windows_map[pol].keys())
        # all_stations = np.unique(np.array(all_stations))
        all_stations = getAllStations(self.windows_map, self.all_pollutants).tolist()
        all_stations = [str(station) for station in all_stations]
        
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
        # self.old_stations = all_stations
        
        # new_station_map = {}
        # self.old_stations_map  = self.stations_map
        # new_names = []
        # for station in self.stations:
        #     # print(station)
        #     new_names = new_names + [self.stations_map[station]['name']]
        #     self.stations_map[station]['old_name'] = station
        #     new_station_map[self.stations_map[station]['name']] = self.stations_map[station]
        #     # old_station_map[station] = self.stations_map[station]
        # self.stations_map = new_station_map
        # self.stations = new_names
        # # print('Stations type: {}'.format(type(self.stations)))
        # # print(self.old_stations_map)

    def common_windows(self, pollutants, stations, max_windows = 10000):
        # print('----------------------------------------------------------------')
        # print(stations)
        # print(type(stations))
        # print(self.windows_map.keys())
        # print(self.windows_map['O3'].keys())
        # print('----------------------------------------------------------------')
        # stations = [int(self.stations_map[st]['old_name']) for st in stations]
        # print(stations)
        # print(self.old_stations_map)
        
        # new_map = {}
        # for k, v in self.windows_map.items():
        #     new_map[k] = {}
        #     for st_k, data in v.items():
        #         new_k = self.old_stations_map[str(st_k)]['name']
        #         new_map[k][new_k] = data
                
        # self.windows_map = new_map
        
        self.windows, self.window_dates, self.window_station_ids, self.window_stations = commonWindows(
            self.windows_map, 
            pollutants, 
            stations
        )
        # print(self.window_station_ids)
        # print(self.window_stations)
        
        N = len(self.windows) 
        if N > max_windows:
            indices = np.arange(N)
            np.random.shuffle(indices)
            # print('INDICES : {}'.format(indices[:10]))
            idx = indices[:max_windows]
            idx = np.array(idx)
        
            print(idx)
            
            self.windows = self.windows[idx]
            self.window_dates = self.window_dates[idx]
            print(np.unique(self.window_station_ids, return_counts=True))
            
            self.window_station_ids = np.array(self.window_station_ids)
            self.window_station_ids = self.window_station_ids[idx]
            
            print(self.window_station_ids)
            print(np.unique(self.window_station_ids, return_counts=True))
        
        self.window_pollutants = pollutants
        
        # self.window_stations = self.old_stations
        self.window_stations_all = self.window_stations
        print('unique')
        print(len(np.unique(self.window_station_ids)))
        print(np.unique(self.window_station_ids))
        print('----')
        # print(len(self.window_stations))
        # print(self.window_stations)
        
        print(self.window_stations)
        self.window_stations = self.window_stations[np.unique(self.window_station_ids)] 
        
        # self.window_stations = [self.old_stations_map[str(k)]['name'] for k in self.window_stations]
        print(self.window_stations)
    
    def dateRanges(self):
        return self.dates[0], self.dates[-1]


        

class BrasilDataset:
    all_pollutants = [
        'BEN', 'CO', 'DV', 'DVG', 'ERT', 
        'MP10', 'MP25', 'NO', 'NOx', 
        'O3', 'PRESS', 'RADG', 'RADUV', 'SO2',
        'TEMP', 'TOL', 'UR']
    
    def __init__(self, granularity='years', cache=True, fill_missing=False, max_missing=0.1):
        self.windows_map = read_brasil(granularity=granularity, cache=cache, fill_missing=fill_missing, max_missing=max_missing)
        self.pollutants = list(self.windows_map.keys())
        
        
        # Get stations and dates ranges
        # all_stations = []
        # for pol in self.pollutants:
        #     all_stations = all_stations + list(self.windows_map[pol].keys())
        # all_stations = np.unique(np.array(all_stations))
        all_stations = getAllStations(self.windows_map, self.all_pollutants).tolist()
        


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
        

    def common_windows(self, pollutants, stations, max_windows = 10000):
        self.windows, self.window_dates, self.window_station_ids, self.window_stations = commonWindows(self.windows_map, pollutants, stations)
        
        N = len(self.windows) 
        if N > max_windows:
            indices = np.arange(N)
            np.random.shuffle(indices)
            idx = indices[:max_windows]
            
            self.windows = self.windows[idx]
            self.window_dates = self.window_dates[idx]
            self.window_station_ids = self.window_station_ids[idx]
        
        self.window_pollutants = pollutants
        
        self.window_stations_all = self.window_stations
        self.window_stations = self.window_stations[np.unique(self.window_station_ids)] 
    
    def dateRanges(self):
        return self.dates[0], self.dates[-1]


class HongKongDataset:
    all_pollutants = [
        'CO', 'FSP', 'NO2', 'NOX', 'O3', 'RSP', 'SO2']
    
    def __init__(self, granularity='years', cache=True, fill_missing=False, max_missing=0.1):
        self.windows_map = read_hongkong(granularity=granularity, cache=cache, fill_missing=fill_missing, max_missing=max_missing)
        self.pollutants = list(self.windows_map.keys())
        
        all_stations = getAllStations(self.windows_map, self.all_pollutants).tolist()

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
        print('Stations type: {}'.format(type(self.stations)))
        

    def common_windows(self, pollutants, stations, max_windows = 10000):
        print('----------------------------------------------------------------')
        print(stations)
        print(type(stations))
        print(self.windows_map.keys())
        print(self.windows_map['O3'].keys())
        print('----------------------------------------------------------------')
        self.windows, self.window_dates, self.window_station_ids, self.window_stations = commonWindows(
            self.windows_map, 
            pollutants, 
            stations
        )
        
        N = len(self.windows) 
        if N > max_windows:
            indices = np.arange(N)
            np.random.shuffle(indices)
            idx = indices[:max_windows]
            
            self.windows = self.windows[idx]
            self.window_dates = self.window_dates[idx]
            self.window_station_ids = self.window_station_ids[idx]
        
        self.window_pollutants = pollutants
        
        self.window_stations_all = self.window_stations
        self.window_stations = self.window_stations[np.unique(self.window_station_ids)] 
    
    def dateRanges(self):
        return self.dates[0], self.dates[-1]