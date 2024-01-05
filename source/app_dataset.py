import numpy as np
from .read_brasil import read_brasil
from .read_hongkong import read_hongkong
from .read_ontario import read_ontario, read_ontario_stations
from .utils import AVAILABLE_POLUTANTS, commonWindows, get_annualy_iaqis_map, get_monthly_iaqis_map, sample_data, getAllStations

class OntarioDataset:
    all_pollutants = ['NO', 'NOx', 'NO2', 'SO2', 'CO', 'O3', 'PM25']
    def __init__(self, granularity='years', cache=True, fill_missing=True, max_missing=0.1):
        self.windows_map, self.windows_original_map = read_ontario(granularity=granularity, cache=cache, fill_missing=fill_missing, max_missing=max_missing)
        self.stations_map = read_ontario_stations()
        self.pollutants = list(self.windows_map.keys())
        self.name='ontario'
        self.granularity = granularity
        
        
        n_map = {}
        for pol_k, pol_d in self.windows_map.items():
            pol_dict = self.windows_map[pol_k]
            n_pol_dict = {}
            for sta_k, sta_d in pol_dict.items():
                real_name = self.stations_map[str(sta_k)]['name']
                stat_dict = pol_dict[sta_k]
                
                if real_name in n_pol_dict:
                    real_dict = n_pol_dict[real_name]
                    for win_k, win_d in stat_dict.items():
                        real_dict[win_k] = win_d
                else:
                    n_pol_dict[real_name] = stat_dict
            n_map[pol_k] = n_pol_dict
        self.windows_map = n_map
        
            
        n_original_map = {}
        for pol_k, pol_d in self.windows_original_map.items():
            pol_dict = self.windows_original_map[pol_k]
            n_pol_dict = {}
            for sta_k, sta_d in pol_dict.items():
                real_name = self.stations_map[str(sta_k)]['name']
                stat_dict = pol_dict[sta_k]
                
                if real_name in n_pol_dict:
                    real_dict = n_pol_dict[real_name]
                    for win_k, win_d in stat_dict.items():
                        real_dict[win_k] = win_d
                else:
                    n_pol_dict[real_name] = stat_dict
            n_original_map[pol_k] = n_pol_dict
        
        self.windows_original_map = n_original_map
        
        
        n_stations_map = {}
        for sta_k, sta_v in self.stations_map.items():
            n_stations_map[sta_v['name']] = sta_v
        self.stations_map = n_stations_map
        
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
        

    def common_windows(self, pollutants, stations, max_windows = 10000):
        print('IN STATIONS: {} - {}'.format(len(stations), stations))
        # self.windows_prepro, _, _, _ = commonWindows(
        #     self.windows_map, 
        #     pollutants, 
        #     stations
        # )
        
        self.windows, self.window_dates, self.window_station_ids, self.window_stations = commonWindows(
            self.windows_original_map,
            pollutants, 
            stations
        )
        print('OUT STATIONS: {} - {}'.format(len(self.window_stations), self.window_stations))
        
        N = len(self.windows) 
        if N > max_windows:
            indices = np.arange(N)
            np.random.shuffle(indices)
            idx = indices[:max_windows]
            idx = np.array(idx)
        
            self.windows = self.windows[idx]
            self.window_dates = self.window_dates[idx]
            self.window_station_ids = self.window_station_ids[idx]
            
        
        self.window_pollutants = pollutants
        
        daily_iaqis = np.load("{}_daily_iaqis.npy".format(self.name), allow_pickle=True)[()]
        
        print('')
        print('Loaded stations: {}'.format(np.unique(self.window_stations)))
        print('')
        for stationId in np.unique(self.window_station_ids):
            station = self.window_stations[stationId]
            # print('STATION: {}'.format(station))
            
            stationDates = self.window_dates[self.window_station_ids == stationId]
            
            # print(np.unique(stationDates))
        print('')
        print('-----------------------------------------------------')
        
        
        if self.granularity == 'daily':
            self.iaqis_map = daily_iaqis
        elif self.granularity == 'months':
            self.iaqis_map = get_monthly_iaqis_map(daily_iaqis)
        else:
            self.iaqis_map = get_annualy_iaqis_map(daily_iaqis)
        
        wrong_ids = []
        self.window_iaqis = {}
        for poll in pollutants:
            print('Gettin IAQI poll: {}'.format(poll))
            if poll not in AVAILABLE_POLUTANTS:
                continue
            self.window_iaqis[poll] = []
            for i in range(len(self.windows)):
                station = self.window_stations[self.window_station_ids[i]]
                date = self.window_dates[i]
                # print(date)
                try:
                    if self.granularity == 'daily':
                        iaqi = self.iaqis_map[poll][station][date.year][date.month][date.day]
                    if self.granularity == 'months':
                        iaqi = self.iaqis_map[poll][station][date.year][date.month]
                    if self.granularity == 'years':
                        iaqi = self.iaqis_map[poll][station][date.year]
                except:
                    iaqi = -1
                    wrong_ids.append(i)
                    print('EXCEPT')
                    print('---------------------------------------------------')
                    print('Station: {} Data: {}'.format(station, date))
                    print(self.iaqis_map[poll].keys())
                    print(self.iaqis_map[poll][station].keys())
                    print('---------------------------------------------------')
                    # break
                    
                    
                self.window_iaqis[poll].append(iaqi)
                
            self.window_iaqis[poll] = np.array(self.window_iaqis[poll])
        
        wrong_ids = np.sort(np.unique(wrong_ids))
        if len(wrong_ids) != 0:
            print('removing wrong indexes')
            print(wrong_ids)
            print(type(wrong_ids[0]))
            print(wrong_ids.shape)
            print(self.windows.shape)
            
            mask = np.ones(len(self.windows), dtype=bool)
            mask[wrong_ids] = False
            self.windows = self.windows[mask]
            self.window_dates = self.window_dates[mask]
            self.window_station_ids = self.window_station_ids[mask]
            for poll in pollutants:
                if poll not in AVAILABLE_POLUTANTS:
                    continue
                self.window_iaqis[poll] = self.window_iaqis[poll][mask]


    
    def dateRanges(self):
        return self.dates[0], self.dates[-1]


        

class BrasilDataset:
    all_pollutants = [
        'BEN', 'CO', 'DV', 'DVG', 'ERT', 
        'MP10', 'MP25', 'NO', 'NO2', 'NOx', 
        'O3', 'PRESS', 'RADG', 'RADUV', 'SO2',
        'TEMP', 'TOL', 'UR']
    
    def __init__(self, granularity='years', cache=True, fill_missing=False, max_missing=0.1):
        self.windows_map, self.windows_original_map = read_brasil(granularity=granularity, cache=cache, fill_missing=fill_missing, max_missing=max_missing)
        self.pollutants = list(self.windows_map.keys())
        self.granularity = granularity
        
        self.name='brasil'
        
        all_stations = getAllStations(self.windows_map, self.all_pollutants).tolist()
        
        self.stations_map = {
            'Capão Redondo': {
                'name':'Capão Redondo',
                'latitude':-23.6719026, 
                'longitude':-46.7794354
            },
            'Cerqueira César': {
                'name':'Cerqueira César',
                'latitude':-23.0353135, 
                'longitude':-49.1650519
            },
            'Cid.Universitária-USP-Ipen': {
                'name':'Cid.Universitária-USP-Ipen',
                'latitude':-23.557594299316406,
                'longitude':-46.71200180053711
            },
            'Congonhas': {
                'name':'Congonhas',
                'latitude':-20.5015168,
                'longitude':-43.8564586
            },
            'Ibirapuera': {
                'name':'Ibirapuera',
                'latitude':-14.8428108,
                'longitude':-40.8546285
            },
            'Interlagos': {
                'name':'Interlagos',
                'latitude':-23.7019315,
                'longitude':-46.6967078
            },
            'Itaim Paulista': {
                'name':'Itaim Paulista',
                'latitude':-23.5017648,
                'longitude':-46.3996091
            },
            'Itaquera': {
                'name':'Itaquera',
                'latitude':-23.5360799,
                'longitude':-46.4555099
            },
            'Marg.Tietê-Pte Remédios': {
                'name':'Marg.Tietê-Pte Remédios',
                'latitude':-23.516924,
                'longitude':-46.733631
            },
            
            'Mooca': {
                'name': 'Mooca',
                'latitude':-23.5606808, 
                'longitude':-46.5971924},
            'N.Senhora do Ó': {
                'name': 'N.Senhora do Ó',
                'latitude':-8.4720591, 
                'longitude':-35.0103062},
            'Osasco': {
                'name': 'Osasco',
                'latitude':-8.399660110473633, 
                'longitude':-35.06126022338867},
            'Parelheiros': {
                'name': 'Parelheiros',
                'latitude':-23.827312, 
                'longitude':-46.7277941},
            'Parque D.Pedro II': {
                'name': 'Parque D.Pedro II',
                'latitude':-23.5508698, 
                'longitude':-46.6275136},
            'Pico do Jaraguá': {
                'name': 'Pico do Jaraguá',
                'latitude':-23.4584254, 
                'longitude':-46.7670295},
            'Pinheiros': {
                'name': 'Pinheiros',
                'latitude':-23.567249, 
                'longitude':-46.7019515},
            'Santana': {
                'name': 'Santana',
                'latitude':-12.979217, 
                'longitude':-44.05064},
            'Santo Amaro': {
                'name': 'Santo Amaro',
                'latitude':-12.5519686, 
                'longitude':-38.7060448},
            'Windsor Downtown': {
                'name': 'Windsor Downtown',
                'latitude':42.315778, 
                'longitude':-83.043667},
            'Windsor West': {
                'name': 'Windsor West',
                'latitude':42.292889, 
                'longitude':-83.073139},
            'Chatham': {
                'name': 'Chatham',
                'latitude':42.403694, 
                'longitude':-82.208306},
            'Sarnia': {
                'name': 'Sarnia',
                'latitude':42.990263, 
                'longitude':-82.395341},
            'Sarnia': {
                'name': 'Sarnia',
                'latitude':42.990263, 
                'longitude':-82.395341},
            'Grand Bend': {
                'name': 'Grand Bend',
                'latitude':43.333083, 
                'longitude':-81.742889},
            'London': {
                'name': 'London',
                'latitude':42.97446, 
                'longitude':-81.200858},
            'London': {
                'name': 'London',
                'latitude':42.97446, 
                'longitude':-81.200858},
            'Port Stanley': {
                'name': 'Port Stanley',
                'latitude':42.672083, 
                'longitude':-81.162889},
            'Tiverton': {
                'name': 'Tiverton',
                'latitude':44.314472, 
                'longitude':-81.549722},
            'Brantford': {
                'name': 'Brantford',
                'latitude':43.138611, 
                'longitude':-80.292639},
            'Kitchener': {
                'name': 'Kitchener',
                'latitude':43.443833, 
                'longitude':-80.503806},
            'St. Catharines': {
                'name': 'St. Catharines',
                'latitude':43.160056, 
                'longitude':-79.23475},
            'Guelph': {
                'name': 'Guelph',
                'latitude':43.551611, 
                'longitude':-80.264167},
            'Hamilton Downtown': {
                'name': 'Hamilton Downtown',
                'latitude':43.257778, 
                'longitude':-79.861667
            },
            'Hamilton Mountain': {
                'name': 'Hamilton Mountain',
                'latitude':43.24132, 
                'longitude':-79.88941
            },
            'Hamilton West': {
                'name': 'Hamilton West',
                'latitude':43.257444, 
                'longitude':-79.90775
            },
            'Hamilton Mountain': {
                'name': 'Hamilton Mountain',
                'latitude':43.24132, 
                'longitude':-79.88941
            },
            'Toronto Downtown': {
                'name': 'Toronto Downtown',
                'latitude':43.662972, 
                'longitude':-79.388111
            },
            'Toronto East': {
                'name': 'Toronto East',
                'latitude':43.747917, 
                'longitude':-79.274056
            },
            'Toronto North': {
                'name': 'Toronto North',
                'latitude':43.78047, 
                'longitude':-79.467372
            },
            'Toronto North': {
                'name': 'Toronto North',
                'latitude':43.78047, 
                'longitude':-79.467372
            },
            'Toronto West': {
                'name': 'Toronto West',
                'latitude':43.709444, 
                'longitude':-79.5435
            },
            'Burlington': {
                'name': 'Burlington',
                'latitude': 43.315111, 
                'longitude': -79.802639},
            'Oakville': {
                'name': 'Oakville',
                'latitude': 43.486917, 
                'longitude': -79.702278},
            'Milton': {
                'name': 'Milton',
                'latitude': 43.529650, 
                'longitude': -79.862449},
            'Oshawa': {
                'name': 'Oshawa',
                'latitude': 43.95222, 
                'longitude': -78.9125},
            'Oshawa': {
                'name': 'Oshawa',
                'latitude': 43.95222, 
                'longitude': -78.9125},
            'Brampton': {
                'name': 'Brampton',
                'latitude': 43.669911, 
                'longitude': -79.766589},
            'Brampton': {
                'name': 'Brampton',
                'latitude': 43.669911, 
                'longitude': -79.766589},
            'Mississauga': {
                'name': 'Mississauga',
                'latitude': 43.54697, 
                'longitude': -79.65869},
            'Barrie': {
                'name': 'Barrie',
                'latitude': 44.382361, 
                'longitude': -79.702306},
            'Newmarket': {
                'name': 'Newmarket',
                'latitude': 44.044306, 
                'longitude': -79.48325},
            'Parry Sound': {
                'name': 'Parry Sound',
                'latitude': 45.338261, 
                'longitude': -80.039269},
            'Dorset': {
                'name': 'Dorset',
                'latitude': 45.224278, 
                'longitude': -78.932944},
            'Ottawa Downtown': {
                'name': 'Ottawa Downtown',
                'latitude': 45.434333, 
                'longitude': -75.676},
            'Ottawa Central': {
                'name': 'Ottawa Central',
                'latitude': 45.382528, 
                'longitude': -75.714194},
            'Petawawa': {
                'name': 'Petawawa',
                'latitude': 45.996722, 
                'longitude': -77.441194},
            'Kingston': {
                'name': 'Kingston',
                'latitude': 44.219722, 
                'longitude': -76.521111},
            'Kingston': {
                'name': 'Kingston',
                'latitude': 44.219722, 
                'longitude': -76.521111},
            'Belleville': {
                'name': 'Belleville',
                'latitude': 44.150528, 
                'longitude': -77.3955},
            'Morrisburg': {
                'name': 'Morrisburg',
                'latitude': 44.89975, 
                'longitude': -75.189944},
            'Cornwall': {
                'name': 'Cornwall',
                'latitude': 45.017972, 
                'longitude': -74.735222},
            'Peterborough': {
                'name': 'Peterborough',
                'latitude': 44.301917, 
                'longitude': -78.346222},
            'Thunder Bay': {
                'name': 'Thunder Bay',
                'latitude': 48.379389, 
                'longitude': -89.290167},
            'Sault Ste. Marie': {
                'name': 'Sault Ste. Marie',
                'latitude': 46.533194, 
                'longitude': -84.309917},
            'North Bay': {
                'name': 'North Bay',
                'latitude': 46.322500, 
                'longitude': -79.4494444},
            'Sudbury': {
                'name': 'Sudbury',
                'latitude': 46.49194, 
                'longitude': -81.003105},
            'Sudbury': {
                'name': 'Sudbury',
                'latitude': 46.49194, 
                'longitude': -81.003105},
            'Merlin': {
                'name': 'Merlin',
                'latitude': 42.249526, 
                'longitude': -82.2180688},
            'Simcoe': {
                'name': 'Simcoe',
                'latitude': 42.856834, 
                'longitude': -80.269722},
            'Stouffville': {
                'name': 'Stouffville',
                'latitude': 43.964580, 
                'longitude': -79.266070},
        }


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
        self.windows, self.window_dates, self.window_station_ids, self.window_stations = commonWindows(self.windows_original_map, pollutants, stations)
        
        N = len(self.windows) 
        if N > max_windows:
            indices = np.arange(N)
            np.random.shuffle(indices)
            idx = indices[:max_windows]
            
            self.windows = self.windows[idx]
            self.window_dates = self.window_dates[idx]
            self.window_station_ids = self.window_station_ids[idx]
        
        self.window_pollutants = pollutants
        
        daily_iaqis = np.load("{}_daily_iaqis.npy".format(self.name), allow_pickle=True)[()]
        
        print('')
        print('Loaded stations: {}'.format(np.unique(self.window_stations)))
        print('')
        for stationId in np.unique(self.window_station_ids):
            station = self.window_stations[stationId]
            print('STATION: {}'.format(station))
            
            stationDates = self.window_dates[self.window_station_ids == stationId]
            
            print(np.unique(stationDates))
        print('')
        print('-----------------------------------------------------')
        
        
        if self.granularity == 'daily':
            self.iaqis_map = daily_iaqis
        elif self.granularity == 'months':
            self.iaqis_map = get_monthly_iaqis_map(daily_iaqis)
        else:
            self.iaqis_map = get_annualy_iaqis_map(daily_iaqis)
        
        wrong_ids = []
        self.window_iaqis = {}
        for poll in pollutants:
            print('Gettin IAQI poll: {}'.format(poll))
            if poll not in AVAILABLE_POLUTANTS:
                continue
            self.window_iaqis[poll] = []
            for i in range(len(self.windows)):
                station = self.window_stations[self.window_station_ids[i]]
                date = self.window_dates[i]
                # print(date)
                try:
                    if self.granularity == 'daily':
                        iaqi = self.iaqis_map[poll][station][date.year][date.month][date.day]
                    if self.granularity == 'months':
                        iaqi = self.iaqis_map[poll][station][date.year][date.month]
                    if self.granularity == 'years':
                        iaqi = self.iaqis_map[poll][station][date.year]
                except:
                    iaqi = -1
                    wrong_ids.append(i)
                    print('---------------------------------------------------')
                    print('Station: {} Data: {}'.format(station, date))
                    print(self.iaqis_map[poll].keys())
                    print(self.iaqis_map[poll][station].keys())
                    print('---------------------------------------------------')
                    # break
                    
                    
                self.window_iaqis[poll].append(iaqi)
                
            self.window_iaqis[poll] = np.array(self.window_iaqis[poll])
        
        wrong_ids = np.sort(np.unique(wrong_ids))
        if len(wrong_ids) != 0:
            print('removing wrong indexes')
            print(wrong_ids)
            print(type(wrong_ids[0]))
            print(wrong_ids.shape)
            print(self.windows.shape)
            
            mask = np.ones(len(self.windows), dtype=bool)
            mask[wrong_ids] = False
            self.windows = self.windows[mask]
            self.window_dates = self.window_dates[mask]
            self.window_station_ids = self.window_station_ids[mask]
            for poll in pollutants:
                if poll not in AVAILABLE_POLUTANTS:
                    continue
                self.window_iaqis[poll] = self.window_iaqis[poll][mask]


    
    def dateRanges(self):
        return self.dates[0], self.dates[-1]


class HongKongDataset:
    all_pollutants = [
        'CO', 'FSP', 'NO2', 'NOX', 'O3', 'RSP', 'SO2']
    
    def __init__(self, granularity='years', cache=True, fill_missing=False, max_missing=0.1):
        self.windows_map, self.windows_original_map = read_hongkong(granularity=granularity, cache=cache, fill_missing=fill_missing, max_missing=max_missing)
        self.pollutants = list(self.windows_map.keys())
        self.name='hongkong'
        self.granularity = granularity
        
        all_stations = getAllStations(self.windows_map, self.all_pollutants).tolist()

        all_dates = []
        for pol in self.pollutants:
            stations = list(self.windows_map[pol].keys())
            for station in stations:
                all_dates = all_dates + list(self.windows_map[pol][station].keys())
                
        all_dates = np.unique(np.array(all_dates))
        all_dates = [np.datetime64(date) for date in all_dates]
        all_dates.sort()
        
        self.stations_map = {
            'TAP MUN': {
                'name': 'TAP MUN',
                'latitude': 22.47739889541748, 
                'longitude': 114.36275530753716,
            },
            'TAI PO': {
                'name': 'TAI PO',
                'latitude':22.448113324316108, 
                'longitude':114.16586746973708,
            },
            'NORTH': {
                'name': 'NORTH',
                'latitude':22.50172748897659, 
                'longitude':114.1285540488637,
            },
            'YUEN LONG': {
                'name':'YUEN LONG',
                'latitude':22.467372785923885, 
                'longitude':114.02375890938951,
            },
            'TUEN MUN': {
                'name':'TUEN MUN',
                'latitude':22.39509752493444, 
                'longitude':113.97929821572177,
            },
            'TSUEN WAN': {
                'name':'TSUEN WAN',
                'latitude':22.380913023614593, 
                'longitude':114.10483640411181,
            },
            'SHATIN': {
                'name':'SHATIN',
                'latitude':22.39567098477031, 
                'longitude':114.18670913567028,
            },
            'SHA TIN': {
                'name':'SHA TIN',
                'latitude':22.39567098477031, 
                'longitude':114.18670913567028,
            },
            'KWAI CHUNG': {
                'name':'KWAI CHUNG',
                'latitude':22.37406877003002, 
                'longitude':114.11624896669262,
            },
            'SHAM SHUI PO': {
                'name':'SHAM SHUI PO',
                'latitude':22.324581323126527, 
                'longitude':114.15644103491213,
            },
            'MONG KOK': {
                'name':'MONG KOK',
                'latitude':22.31882674036733, 
                'longitude':114.15987674952666,
            },
            'TUNG CHUNG': {
                'name':'TUNG CHUNG',
                'latitude':22.282589394723665, 
                'longitude':113.94520448629142,
            },
            'CENTRAL': {
                'name':'CENTRAL',
                'latitude':22.2799590403546, 
                'longitude':114.16612350337158,
            },
            'CENTRAL/WESTERN': {
                'name':'CENTRAL/WESTERN',
                'latitude':22.28532988666009, 
                'longitude':114.14269817645378,
            },
            'SOUTHERN': {
                'name':'SOUTHERN',
                'latitude':22.247376119378032, 
                'longitude':114.16203384734007,
            },
            'EASTERN': {
                'name':'EASTERN',
                'latitude':22.283493002255888, 
                'longitude':114.22174605856657,
            },
            'CAUSEWAY BAY': {
                'name':'CAUSEWAY BAY',
                'latitude':22.2798588997558, 
                'longitude':114.1857213360911,
            },
            'KWUN TONG': {
                'name':'KWUN TONG',
                'latitude':22.305447811846236, 
                'longitude':114.23013537749924,
            },
            'TSEUNG KWAN O': {
                'name':'TSEUNG KWAN O',
                'latitude':22.306836517257125, 
                'longitude':114.26344590855533,
            },
        }
        
        self.dates = all_dates
        self.stations = all_stations
        

    def common_windows(self, pollutants, stations, max_windows = 10000):
        
        self.windows, self.window_dates, self.window_station_ids, self.window_stations = commonWindows(
            self.windows_original_map, 
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
        
        # self.window_stations_all = self.window_stations
        # self.window_stations = self.window_stations[np.unique(self.window_station_ids)] 
        
        daily_iaqis = np.load("{}_daily_iaqis.npy".format(self.name), allow_pickle=True)[()]
        
        print('')
        print('Loaded stations: {}'.format(np.unique(self.window_stations)))
        print('')
        for stationId in np.unique(self.window_station_ids):
            station = self.window_stations[stationId]
            print('STATION: {}'.format(station))
            
            stationDates = self.window_dates[self.window_station_ids == stationId]
            
            print(np.unique(stationDates))
        print('')
        print('-----------------------------------------------------')
        
        
        if self.granularity == 'daily':
            self.iaqis_map = daily_iaqis
        elif self.granularity == 'months':
            self.iaqis_map = get_monthly_iaqis_map(daily_iaqis)
        else:
            self.iaqis_map = get_annualy_iaqis_map(daily_iaqis)
        
        wrong_ids = []
        self.window_iaqis = {}
        for poll in pollutants:
            print('Gettin IAQI poll: {}'.format(poll))
            if poll not in AVAILABLE_POLUTANTS:
                continue
            self.window_iaqis[poll] = []
            for i in range(len(self.windows)):
                station = self.window_stations[self.window_station_ids[i]]
                date = self.window_dates[i]
                # print(date)
                try:
                    if self.granularity == 'daily':
                        iaqi = self.iaqis_map[poll][station][date.year][date.month][date.day]
                    if self.granularity == 'months':
                        iaqi = self.iaqis_map[poll][station][date.year][date.month]
                    if self.granularity == 'years':
                        iaqi = self.iaqis_map[poll][station][date.year]
                except:
                    iaqi = -1
                    wrong_ids.append(i)
                    print('---------------------------------------------------')
                    print('Station: {} Data: {}'.format(station, date))
                    print(self.iaqis_map[poll].keys())
                    print(self.iaqis_map[poll][station].keys())
                    print('---------------------------------------------------')
                    # break
                    
                    
                self.window_iaqis[poll].append(iaqi)
                
            self.window_iaqis[poll] = np.array(self.window_iaqis[poll])
        
        wrong_ids = np.sort(np.unique(wrong_ids))
        if len(wrong_ids) != 0:
            print('removing wrong indexes')
            print(wrong_ids)
            print(type(wrong_ids[0]))
            print(wrong_ids.shape)
            print(self.windows.shape)
            
            mask = np.ones(len(self.windows), dtype=bool)
            mask[wrong_ids] = False
            self.windows = self.windows[mask]
            self.window_dates = self.window_dates[mask]
            self.window_station_ids = self.window_station_ids[mask]
            for poll in pollutants:
                if poll not in AVAILABLE_POLUTANTS:
                    continue
                self.window_iaqis[poll] = self.window_iaqis[poll][mask]


    def dateRanges(self):
        return self.dates[0], self.dates[-1]