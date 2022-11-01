from .read_brasil import read_brasil

class OntarioDataset:
    def __init__(self):
        a = 1
        

class BrasilDataset:
    pollutants = [
        'BEN', 'CO', 'DV', 'DVG', 'ERT', 
        'MP10', 'MP25', 'NO', 'MP25', 'NOx', 
        'O3', 'PRESS', 'RADG', 'RADUV', 'SO2',
        'TEMP', 'TOL', 'UR']
    
    def __init__(self, granularity='years', cache=True):
        windows_map = read_brasil(granularity='months', cache=True)
    
    
    # def dateRanges(self):
    #     return 