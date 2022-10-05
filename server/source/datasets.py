import pandas as pd
import numpy as np
import os

from sktime.datasets import load_from_tsfile

def loadFuncionalModel(n):
    dirname = os.path.dirname(__file__)
    dirname = os.path.abspath(os.path.join(dirname, os.pardir))
    df = pd.read_csv(os.path.join(dirname,'datasets/outliers/model{}.csv'.format(n)))
    data = df.to_numpy()
    return data
    
def loadWafer():
    dirname = os.path.dirname(__file__)
    dirname = os.path.abspath(os.path.join(dirname, os.pardir))
    X_train, y_train = load_from_tsfile(os.path.join(dirname, 'datasets/Wafer/Wafer_TRAIN.ts'), return_data_type="numpy3d")
    X_test, y_test = load_from_tsfile(os.path.join(dirname,'datasets/Wafer/Wafer_TEST.ts'), return_data_type="numpy3d")
    
    y_train = np.array([int(y_train[i]) for i in range(len(y_train))])
    y_test = np.array([int(y_test[i]) for i in range(len(y_test))])
    
    return X_train, y_train, X_test, y_test

def loadWeather():
    dirname = os.path.dirname(__file__)
    dirname = os.path.abspath(os.path.join(dirname, os.pardir))
    X_train, y_train = load_from_tsfile(os.path.join(dirname, 'datasets/Wafer/Wafer_TRAIN.ts'), return_data_type="numpy3d")
    X_test, y_test = load_from_tsfile(os.path.join(dirname,'datasets/Wafer/Wafer_TEST.ts'), return_data_type="numpy3d")
    
    return X_train, y_train, X_test, y_test


# Returns data with the shape NxDxT
def loadNatops():
    dirname = os.path.dirname(__file__)
    dirname = os.path.abspath(os.path.join(dirname, os.pardir))
    X_train, y_train = load_from_tsfile(os.path.join(dirname, 'datasets/NATOPS/NATOPS_TRAIN.ts'), return_data_type="numpy3d")
    X_test, y_test = load_from_tsfile(os.path.join(dirname,'datasets/NATOPS/NATOPS_TEST.ts'), return_data_type="numpy3d")
    
    return X_train, y_train, X_test, y_test

def loadSelfRegulationSCP2():
    dirname = os.path.dirname(__file__)
    dirname = os.path.abspath(os.path.join(dirname, os.pardir))
    X_train, y_train = load_from_tsfile(os.path.join(dirname, 'datasets/SelfRegulationSCP2/SelfRegulationSCP2_TRAIN.ts'), return_data_type="numpy3d")
    X_test, y_test = load_from_tsfile(os.path.join(dirname,'datasets/SelfRegulationSCP2/SelfRegulationSCP2_TEST.ts'), return_data_type="numpy3d")
    
    return X_train, y_train, X_test, y_test