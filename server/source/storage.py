import numpy as np
import os



class MTSStorage:
    def __init__(self, filename):
        self.filename = filename
        self.objects = {}
    
    # mts with shape N, T, D
    def add_mts(self, name, mts, dimensions = [], coords = {}, labels = {},  labelsNames = {}, sampling = False, n_samples = 100):
        
        print('mts shape: N: {} -  T: {} - D: {}'.format(mts.shape[0], mts.shape[1], mts.shape[2]))
        if sampling:
            n = mts.shape[0]
            indices = np.array(list(range(n)))
            np.random.shuffle(indices)
            
            sampled_ind = indices[:n_samples]
            
            # Getting sampled data
            imts = mts[sampled_ind]
            icoords = {}
            ilabels = {}
            for k, kcoords in coords.items():
                icoords[k] = kcoords[sampled_ind]
            
            for k, klabels in labels.items():
                ilabels[k] = klabels[sampled_ind]
        else:
            imts = mts
            icoords = coords
            ilabels = labels

        self.objects[name] = {'mts': imts}
        if len(icoords) != 0:
            self.objects[name]['coords'] = {}
            for k, v in icoords.items():
                self.objects[name]['coords'][k] = np.array(v)
        if len(ilabels) != 0:
            self.objects[name]['labels'] = {}
            for k, v in ilabels.items():
                self.objects[name]['labels'][k] = np.array(v).astype(int)
                
        if len(dimensions) != 0:
            self.objects[name]['dimensions'] = np.array(dimensions)
            
        if len(labelsNames) != 0:
            self.objects[name]['labelsNames'] = labelsNames
            # for k, v in labelsNames.items():
            #     self.objects[name]['labelsNames'][k] = v
    
    # saves objects dict as pickle file
    def save(self):
        np.save(self.filename, self.objects)
    
    def load(self):
        if os.path.exists(self.filename ):
            self.objects = np.load(self.filename , allow_pickle=True)
            self.objects = self.objects[()]
    
    def delete(self):
        if os.path.exists(self.filename ):
            os.remove(self.filename )
            
    
    
    