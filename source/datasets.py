import pandas as pd
import numpy as np
import os

from sklearn import preprocessing
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
    
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    
    classLabels = {
        0: 'I have command',
        1: 'All clear',
        2: 'Not clear',
        3: 'Spread wings',
        4: 'Fold wings',
        5: 'Lock wings',
    }
    
    return X_train, le.transform(y_train), X_test, le.transform(y_test), classLabels

def loadSelfRegulationSCP2():
    dirname = os.path.dirname(__file__)
    dirname = os.path.abspath(os.path.join(dirname, os.pardir))
    X_train, y_train = load_from_tsfile(os.path.join(dirname, 'datasets/SelfRegulationSCP2/SelfRegulationSCP2_TRAIN.ts'), return_data_type="numpy3d")
    X_test, y_test = load_from_tsfile(os.path.join(dirname,'datasets/SelfRegulationSCP2/SelfRegulationSCP2_TEST.ts'), return_data_type="numpy3d")
    
    print(np.unique(y_train))
    
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    
    
    return X_train, le.transform(y_train), X_test, le.transform(y_test), le.classes_


def loadBasicMotions():
    dirname = os.path.dirname(__file__)
    dirname = os.path.abspath(os.path.join(dirname, os.pardir))
    X_train, y_train = load_from_tsfile(os.path.join(dirname, 'datasets/BasicMotions/BasicMotions_TRAIN.ts'), return_data_type="numpy3d")
    X_test, y_test = load_from_tsfile(os.path.join(dirname,'datasets/BasicMotions/BasicMotions_TEST.ts'), return_data_type="numpy3d")
    
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    
    # print(le.classes_)
    # print(np.unique(le.transform(y_train)))
    # print(le.inverse_transform([0, 1, 2, 3]))
    
    classes = np.unique(le.transform(y_train))
    decClasses = le.inverse_transform(classes)
    
    classLabels = {int(classes[i]): decClasses[i] for i in range(len(classes))}
    
    return X_train, le.transform(y_train), X_test, le.transform(y_test), classLabels

def loadEarthquakes():
    dirname = os.path.dirname(__file__)
    dirname = os.path.abspath(os.path.join(dirname, os.pardir))
    X_train, y_train = load_from_tsfile(os.path.join(dirname, 'datasets/Earthquakes/Earthquakes_TRAIN.ts'), return_data_type="numpy3d")
    X_test, y_test = load_from_tsfile(os.path.join(dirname,'datasets/Earthquakes/Earthquakes_TEST.ts'), return_data_type="numpy3d")
    
    le = preprocessing.LabelEncoder()
    # print(le.inverse_transform([0, 1, 2, 3]))
    
    
    
    return X_train, le.transform(y_train), X_test, le.transform(y_test)

def loadItalyPowerDemand():
    dirname = os.path.dirname(__file__)
    dirname = os.path.abspath(os.path.join(dirname, os.pardir))
    X_train, y_train = load_from_tsfile(os.path.join(dirname, 'datasets/ItalyPowerDemand/ItalyPowerDemand_TRAIN.ts'), return_data_type="numpy3d")
    X_test, y_test = load_from_tsfile(os.path.join(dirname,'datasets/ItalyPowerDemand/ItalyPowerDemand_TEST.ts'), return_data_type="numpy3d")
    
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    
    classes = np.unique(le.transform(y_train))
    decClasses = le.inverse_transform(classes)
    
    classLabels = {int(classes[i]): decClasses[i] for i in range(len(classes))}
    
    return X_train, le.transform(y_train), X_test, le.transform(y_test), classLabels

def loadEigenWorms():
    dirname = os.path.dirname(__file__)
    dirname = os.path.abspath(os.path.join(dirname, os.pardir))
    X_train, y_train = load_from_tsfile(os.path.join(dirname, 'datasets/EigenWorms/EigenWorms_TRAIN.ts'), return_data_type="numpy3d")
    X_test, y_test = load_from_tsfile(os.path.join(dirname,'datasets/EigenWorms/EigenWorms_TEST.ts'), return_data_type="numpy3d")
    
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    
    return X_train, le.transform(y_train), X_test, le.transform(y_test)


