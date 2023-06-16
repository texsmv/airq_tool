import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from .utils import mts_norm, mts_shape_norm, mts_smooth, ts_to_basis


class TSerie:
    def __init__(self, X: np.array, y:np.array):
        self.N = X.shape[0]
        self.T = X.shape[1]
        self.D = X.shape[2]
        self.X_orig = np.copy(X)
        # self.X_orig = X
        self.X = X
        self.X_norm = X
        self.y = y
        self.time_features = None
        
        self.magnitudes = None
        
        print('Loaded mts - N: {}, T: {}, D: {} '.format(self.N, self.T, self.D))
        
    def folding_features_v1(self):
        N, T, D = self.X.shape
        self.features = np.zeros((N, D * T))
        for n in range(N):       
            for d in range(D):
                self.features[n, d * T : (d + 1) * T] = self.X[n, :, d]
        print('Features shape: {}'.format(self.features.shape))

    def folding_features_v2(self, isNorm = False):
        if isNorm:
            X = self.X_norm
        else:
            X = self.X
        N, T, D = X.shape
        self.features = np.zeros((N, D * T))
        for n in range(N):
            for t in range(T):
                for d in range(D):
                    self.features[n, D * t + d] = X[n, t, d]
        print('Features shape: {}'.format(self.features.shape))
    
    def shapeNormalizization(self, returnValues = False):
        if returnValues:
            X_c, _, magnitudes = mts_shape_norm(self.X)
            return X_c, magnitudes
        self.X, _, self.magnitudes = mts_shape_norm(self.X)
    
    def minMaxNormalizization(self, minl=[], maxl=[], returnValues = False):
        _, _, self.magnitudes = mts_shape_norm(self.X)
        if returnValues:
            if len(minl) == 0:
                Xc, min_l, max_l = mts_norm(self.X)
            else:
                Xc, min_l, max_l = mts_norm(self.X, minl=minl, maxl=maxl)
                
            return Xc, min_l, max_l
        else:
            if len(minl) == 0:
                self.X_norm, min_l, max_l = mts_norm(self.X)
            else:
                self.X_norm, min_l, max_l = mts_norm(self.X, minl=minl, maxl=maxl)
            
        return min_l, max_l
        
    def smooth(self, window_size = 5, returnValues = False):
        if returnValues:
            return mts_smooth(self.X, window_len=window_size)
        self.X = mts_smooth(self.X, window_len=window_size)
        
    
    def to_basis(self, n_basis=30):
        new_X = []
        for d in range(self.D):
            new_X.append(ts_to_basis(self.X[:, :, d], n_basis=n_basis))
        self.X = np.array(new_X)
        self.X = self.X.transpose(1, 2, 0)
        self.T = self.X.shape[1]
    
        