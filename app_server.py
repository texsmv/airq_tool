from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from flask import jsonify
import json
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from source.tserie import TSerie
from sklearn.decomposition import PCA
from source.utils import AVAILABLE_POLUTANTS, folding_2D, magnitude_shape_plot
from source.app_dataset import OntarioDataset, BrasilDataset, HongKongDataset
from source.read_ontario import read_ontario_stations
from source.utils import fdaOutlier
import umap
import pacmap
from source.featlearn.autoencoder_lr import AutoencoderFL

from dist_matrix.cuda_dist_matrix_full import dist_matrix as gpu_dist_matrix
import aqi
from fast_pytorch_kmeans import KMeans
import torch
# from fastdist import fastdist


import datetime
epoch = datetime.datetime.utcfromtimestamp(0)

def unix_time_millis(dt):
    return (dt - epoch).total_seconds() * 1000.0

# PATH to ST2vec repository if you want to use it
# sys.path.append('/home/texs/Documentos/Repositories/ts2vec')
# from ts2vec import TS2Vec

MAX_MISSING = 0.1
FILL_MISSING = True

MODE = 2 # 0 for umap, 1 for ts2vec, 2 for CAE

USE_TS2VEC = True
MAX_WINDOWS = 40000
UMAP_METRIC = 'braycurtis'
BATCH_SIZE = 800

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True




def filter_iaqi(pollutants, values): # values shape NxD
    out_index = []
    for i in range(len(pollutants)):
        if pollutants[i] in AVAILABLE_POLUTANTS:
            out_index.append(i)
    return np.array(pollutants)[out_index], values[:, :, out_index]


def get_aqi(pollutants, values): # values shape NxD
    iaqis = []
    for i in range(len(pollutants)):
        pollutant = pollutants[i]
        data = values[:, i]
        if pollutant == 'O3':
            d_mean = data.mean()
            iaqis.append(('o3_8h', str(d_mean)))
        elif pollutant == 'PM25' or pollutant == 'FSP' or pollutant == 'MP25':
            d_mean = data.mean()
            iaqis.append(('pm25', str(d_mean)))
        elif pollutant == 'PM10' or pollutant == 'RSP' or pollutant == 'MP10':
            d_mean = data.mean()
            # return aqi.to_iaqi(aqi.POLLUTANT_PM10, str(d_mean), algo=aqi.ALGO_EPA)
            iaqis.append(('pm10', str(d_mean)))
        elif pollutant == 'NO2':
            d_mean = data.mean()
            iaqis.append(('no2_24h', str(d_mean)))
        elif pollutant == 'SO2':
            d_mean = data.mean()
            iaqis.append(('so2_24h', str(d_mean)))
        elif pollutant == 'CO':
            d_mean = data.mean()
            iaqis.append(('co_24h', str(d_mean)))
    return aqi.to_aqi(iaqis, algo=aqi.ALGO_MEP)
        


dataset = None
mts = None
g_coords = None
space_DM = None
time_DM = None
latlong = None
timeInMs = None

app = Flask(__name__)
CORS(app)

@app.route("/datasets", methods=['POST'])
def datasetsInfo():
    global dataset
    ontarioDataset = OntarioDataset(fill_missing=FILL_MISSING, max_missing=MAX_MISSING, granularity='years')
    brasilDataset = BrasilDataset(fill_missing=FILL_MISSING, max_missing=MAX_MISSING, granularity='years')
    hongkongDataset = HongKongDataset(fill_missing=FILL_MISSING, max_missing=MAX_MISSING, granularity='years')
    
    resp_map = {
        'ontario' : {
            'pollutants': ontarioDataset.all_pollutants,
            'stations': ontarioDataset.stations,
        },
        'brasil' : {
            'pollutants': brasilDataset.all_pollutants,
            'stations': brasilDataset.stations,
        },
        'hongkong':{
            'pollutants': hongkongDataset.all_pollutants,
            'stations': hongkongDataset.stations,
        }
    }
    # dataset = hongkongDataset
    return jsonify(resp_map)

@app.route("/correlation", methods=["POST"])
def correlation():
    global dataset
    global mts
    
    positions = np.array(json.loads(request.form['positions']))
    
    n = len(mts.X_orig)
    back_positions = []
    all_positions = np.arange(n)
    for pos in all_positions:
        if not pos in positions:
            back_positions.append(pos)
    back_positions = np.array(back_positions, dtype=int)
    
    # back_mts  =TSerie(mts.X_orig[back_positions], None)
    # fore_mts  =TSerie(mts.X_orig[positions], None)
    
    
    # back_mts.folding_features_v1()
    # fore_mts.folding_features_v1()
    
    # cpca = CPCA(standardize=False)
    # result = cpca.fit_transform(background=back_mts.features, foreground=fore_mts.features, plot=False)
    # result = cpca.fit_transform(background=mts.features[back_positions], foreground=mts.features[positions], plot=False)
    
    # allCoords = np.zeros((n, 2))
    
    # allCoords[positions] = result[1]
    
    
    X = mts.X_orig[positions]
    
    
    
    X = np.vstack(X) # To shape: N * T, D
    
    df = pd.DataFrame(data=X)
    corr_matrix = df.corr().to_numpy()
    
    resp_map = {}
    resp_map['correlation_matrix'] = corr_matrix.flatten().tolist()
    print(corr_matrix)
    # resp_map['coords'] = allCoords.flatten().tolist()
    
    
    min_values = []
    max_values = []
    mean_values = []
    std_values = []
    for d in range(mts.D):
        min_values.append(mts.X_orig[positions,:,d].min())
        max_values.append(mts.X_orig[positions,:,d].max())
        mean_values.append(mts.X_orig[positions,:,d].mean())
        std_values.append(mts.X_orig[positions,:,d].std())
    
    
    resp_map['minv'] = min_values
    resp_map['maxv'] = max_values
    resp_map['meanv'] = mean_values
    resp_map['stdv'] = std_values
    return jsonify(resp_map)
    


@app.route("/getIaqis", methods=['POST'])
def getIaqis():
    global dataset
    global mts
    pollutants = np.array(json.loads(request.form['pollutants']))
    
    filtered_pollutans, filtered_windows = filter_iaqi(pollutants, mts.X_orig)
    # print(filtered_pollutans)
    # print(filtered_windows.shape) 
    resp_map = {}
    if len(filtered_pollutans) != 0:
        resp_map['status']= 'DONE'
        aqi = [int(get_aqi(filtered_pollutans, filtered_windows[i])) for i in range(len(filtered_windows))]
        resp_map['aqi'] = aqi
        # print(aqi)
        for k in range(len(filtered_pollutans)):
            pollutant = filtered_pollutans[k]
            # iaqi = [int(daily_iaqi(pollutant, filtered_windows[i,:,k])) for i in range(len(filtered_windows))]
            iaqi = [int(mts.iaqi[pollutant][i]) for i in range(len(filtered_windows))]
            # iaqi
            resp_map[pollutant] = iaqi
            # print(iaqi)
    else:
        resp_map['status']= 'ERROR'
    return jsonify(resp_map)

# def 

def groupRank(cluster_mean, series):
    # ranks = np.array([np.abs(cluster_mean - serie).sum() for serie in series]).sum()
    rank = np.array([np.linalg.norm(cluster_mean - serie) for serie in series]).sum()
    return rank
    

@app.route("/pollutantRanking", methods=['POST'])
def pollutantRanking():
    global dataset
    global mts
    
    selection = np.array(json.loads(request.form['selectionIds']))
    
    ranks={}
    for polPos in range(mts.D):
        ranks[polPos] = groupRank(mts.X.mean(axis=2), mts.X[selection, :, polPos])
        # print(mts.X[selection, :, polPos].shape)
    
    
    # filtered_pollutans, filtered_windows = filter_iaqi(pollutants, mts.X_orig)
    # # print(filtered_pollutans)
    # # print(filtered_windows.shape) 
    resp_map = {
        'ranks':ranks
    }
    # if len(filtered_pollutans) != 0:
    #     resp_map['status']= 'DONE'
    #     aqi = [int(get_aqi(filtered_pollutans, filtered_windows[i])) for i in range(len(filtered_windows))]
    #     resp_map['aqi'] = aqi
    #     # print(aqi)
    #     for k in range(len(filtered_pollutans)):
    #         pollutant = filtered_pollutans[k]
    #         # iaqi = [int(daily_iaqi(pollutant, filtered_windows[i,:,k])) for i in range(len(filtered_windows))]
    #         iaqi = [int(mts.iaqi[pollutant][i]) for i in range(len(filtered_windows))]
    #         # iaqi
    #         resp_map[pollutant] = iaqi
    #         # print(iaqi)
    # else:
    #     resp_map['status']= 'ERROR'
    return jsonify(resp_map)
    
@app.route("/getProjection", methods=['POST'])
def getProjection():
    global dataset
    global granularity
    global mts
    global g_coords
    
    pollPositions = np.array(json.loads(request.form['pollutantsPositions']))
    
    N_NEIGHBORS = int(request.form['neighbors'])
    delta = float(request.form['delta'])
    beta = float(request.form['beta'])
    
    
    if granularity == 'months':
        EPOCHS = 20
        # EPOCHS_CAE = 100
        EPOCHS_CAE = 1000
        FEATURE_SIZE_CAE = 10
        N_NEIGHBORS = 15
    elif granularity == 'years':
        EPOCHS = 100
        # EPOCHS_CAE = 2000
        EPOCHS_CAE = 300
        FEATURE_SIZE_CAE = 30
        N_NEIGHBORS = 15
    elif granularity == 'daily':
        EPOCHS = 20
        # EPOCHS_CAE = 200
        EPOCHS_CAE = 200
        FEATURE_SIZE_CAE = 8
        N_NEIGHBORS = 15
    # _, _ = mts.minMaxNormalizization(returnValues=False)
    # mts.robustScaler(returnValues=False)
    
    if MODE == 0:    
        X_filtered = mts.X[:, :, pollPositions]
        mts_filtered = TSerie(X_filtered, mts.y)
        mts_filtered.folding_features_v1()
        
        model = UMAP_FL(n_components=32, n_neighbors=N_NEIGHBORS, metric=UMAP_METRIC, n_epochs = 15000)
        # model = PCA(n_components=16)
        mts.features = model.fit_transform(mts_filtered.features)
    elif MODE == 1:
        model = TS2Vec(
            input_dims=mts.D,
            device=0,
            output_dims=32,
            batch_size=BATCH_SIZE,
            depth=10,
            hidden_dims=128,
        )
        model.fit(mts.X, verbose=True,n_epochs = EPOCHS)
        mts.time_features = model.encode(mts.X, batch_size=BATCH_SIZE)
        mts.features = model.encode(mts.X, encoding_window='full_series', batch_size=BATCH_SIZE)
    else:
        cae = AutoencoderFL(mts.D, mts.T, feature_size=FEATURE_SIZE_CAE)
        
        cae.fit(mts.X, epochs=EPOCHS_CAE, batch_size=200)
        _, mts.features = cae.encode(mts.X)
        
        # cae = MEC_FL(mts.D, mts.T, feature_size=FEATURE_SIZE_CAE)
        # cae.fit(mts.X.transpose([0, 2, 1]), epochs=EPOCHS_CAE, batch_size=200, freq=10)
        # mts.features = cae.encode(mts.X.transpose([0, 2, 1]))
        # X_train, X_val = train_test_split(mts.X.transpose([0, 2, 1]))
        
    
    print('[ROOT]: Computing distance matrix')    
    feature_DM = gpu_dist_matrix(mts.features)
    feature_DM = feature_DM / np.max(feature_DM)
    print('[ROOT]: done')    
    
    
    # feature_DM = fastdist.matrix_to_matrix_distance(mts.features, mts.features, fastdist.cosine, "cosine")
    # print(feature_DM.shape) 
    # reducer = umap.UMAP(n_components=2)
    reducer = umap.UMAP(n_components=2, metric='precomputed', n_neighbors=10)
    # reducer = pacmap.PaCMAP(n_components=2)
    
    coords = reducer.fit_transform(feature_DM)
    # coords = reducer.fit_transform(mts.features)
    g_coords = coords
    reducer = None
    
    
    resp_map = {}
    resp_map['coords'] = coords.flatten().tolist()
    
    return jsonify(resp_map)


@app.route("/getCustomProjection", methods=['POST'])
def getCustomProjection():
    global dataset
    global granularity
    global mts
    global timeInMs
    global latlong
    global g_coords
    
    # pollutantPosition = request.form['pollutantPosition']
    # pollutantPosition = int(pollutantPosition)
    filtered = np.array(json.loads(request.form['itemsPositions']))
    itemPositions = np.argwhere(filtered == True).squeeze()
    n_neightbors = int(request.form['neighbors'])
    # delta = float(request.form['delta'])
    beta = float(request.form['beta'])


    feature_DM = gpu_dist_matrix(mts.features[itemPositions])
    feature_DM = feature_DM / np.max(feature_DM)
    
    space_DM = gpu_dist_matrix(latlong[itemPositions])
    space_DM = space_DM / (np.max(space_DM) + 0.00000001)
    
    # time_DM = gpu_dist_matrix(timeInMs[itemPositions])
    # time_DM = time_DM / np.max(time_DM)
    
    
    distM = feature_DM * (1 - beta) + space_DM * beta
    # distM = feature_DM * (1 - (delta + beta)) + space_DM * delta + time_DM * beta
    
    reducer = umap.UMAP(n_components=2, metric='precomputed',n_neighbors=n_neightbors)
    # reducer = pacmap.PaCMAP(n_components=2)
    coords = reducer.fit_transform(distM)
    # coords = reducer.fit_transform(mts.features[itemPositions])
    
    g_coords[itemPositions] = coords
    reducer = None
    

    
    # Outliers
    ts = mts.X[itemPositions, :, 0]
    
    cmean, cvar, outliers = magnitude_shape_plot(ts)
    
    mmean = cvar.mean()
    lower_o = np.bitwise_and(outliers == 1, cvar < mmean)
    upper_o = np.bitwise_and(outliers == 1, cvar > mmean)
    
    n_outliers = np.zeros(outliers.shape[0]).astype(int)
    n_outliers[lower_o] = 1
    n_outliers[upper_o] = 2
    n_outliers[outliers == 0] = 0
    
    
    resp_map = {}
    resp_map['coords'] = coords.flatten().tolist()
    resp_map['cmean'] = cmean.tolist()
    resp_map['cvar'] = cvar.tolist()
    resp_map['outliers'] = n_outliers.tolist()
    
    print(resp_map['outliers'])
    
    return jsonify(resp_map)





@app.route("/spatioTemporalProjection", methods=['POST'])
def spatioTemporalProjection():
    global dataset
    global granularity
    global mts
    global g_coords
    
    
    # EPOCHS = 5
    N_NEIGHBORS = int(request.form['neighbors'])
    
    delta = float(request.form['delta'])
    beta = float(request.form['beta'])
        
    
    feature_DM = pairwise_distances(mts.features, metric='cosine')
    feature_DM = feature_DM / np.max(feature_DM)
    
    distM = feature_DM * (1 - (delta + beta)) + space_DM * delta + time_DM * beta
    reducer = umap.UMAP(n_components=2, metric='precomputed', n_neighbors=N_NEIGHBORS)
    
    # reducer = umap.UMAP(n_components=2, metric='euclidean')
    # reducer = umap.UMAP(n_components=2, metric='cosine')
    #  reducer.transform(mts.features)
    coords = reducer.fit_transform(distM)
    # g_coords = coords
    reducer = None
    
    
    resp_map = {}
    resp_map['coords'] = coords.flatten().tolist()
    
    return jsonify(resp_map)



@app.route("/getFdaOutliers", methods=['POST'])
def getFdaOutliers():
    global dataset
    global granularity
    global mts
    
    pollutantPosition = request.form['pollutantPosition']
    pollutantPosition = int(pollutantPosition)
    
    ts = mts.X[:, :, pollutantPosition]
    # print(ts.shape)
    
    # cmean, cvar = fdaOutlier(ts)
    cmean, cvar, outliers = magnitude_shape_plot(ts)
    
    mmean = cvar.mean()

    
    
    lower_o = np.bitwise_and(outliers == 1, cvar < mmean)
    upper_o = np.bitwise_and(outliers == 1, cvar > mmean)
    
    
    n_outliers = np.zeros(outliers.shape[0]).astype(int)
    n_outliers[lower_o] = 1
    n_outliers[upper_o] = 2
    n_outliers[outliers == 0] = 0
    
    
    resp_map = {}
    resp_map['cmean'] = cmean.tolist()
    resp_map['cvar'] = cvar.tolist()
    resp_map['outliers'] = n_outliers.tolist()
    
    return jsonify(resp_map)

@app.route("/kmeans", methods=['POST'])
def kmeans():
    global dataset
    global granularity
    global mts
    global g_coords
    
    n_clusters = request.form['n_clusters']
    n_clusters = int(n_clusters)
    
    # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(mts.features)
    # feat = torch.from_numpy(mts.features).to('cuda')
    
    itemPositions = np.array(json.loads(request.form['itemsPositions']))
    # itemPositions = np.argwhere(filtered == True).squeeze()
    
    feat = torch.from_numpy(g_coords[itemPositions]).to('cuda')
    
    
    kmeans = KMeans(n_clusters, max_iter = 300, mode='euclidean')
    labels = kmeans.fit_predict(feat)
    classes = labels.cpu().numpy()
    
    
    # clustering = DBSCAN(eps=0.2, min_samples=5)
    # clustering.fit(g_coords)
    # classes = clustering.labels_
    
    # n_classes = len(np.unique(classes))
    
    
    # classes[classes == -1] = n_classes
    # classes = np.unique(labels)
    
    
    
    resp_map = {}
    resp_map['classes'] = classes.tolist()
    
    return jsonify(resp_map)


@app.route("/dbscan", methods=['POST'])
def dbscan():
    global dataset
    global granularity
    global mts
    global g_coords

    itemPositions = np.array(json.loads(request.form['itemsPositions']))
    
    eps = request.form['eps']
    eps = float(eps)
    
    clustering = DBSCAN(eps=eps, min_samples=5)
    clustering.fit(g_coords[itemPositions])
    classes = clustering.labels_
    n_classes = len(np.unique(classes))
    
    if (classes == -1).any():
        classes[classes == -1] = n_classes - 1
        
    print(classes)
    
    print(np.unique(classes))
    resp_map = {}
    resp_map['classes'] = classes.tolist()
    resp_map['n_classes'] = len(np.unique(classes))
    
    return jsonify(resp_map)


@app.route("/getContrastiveFeatures", methods=["POST"])
def getContrastiveFeatures():
    global dataset
    global granularity
    global mts
    
    classes = json.loads(request.form['classes'])
    classes = np.array(classes)
    clusters = np.unique(classes)
    
    print('Clusters unique IDs: {}'.format(clusters))
    
    modelCL = UMAP_CFL()
    if USE_TS2VEC:
        
        print(mts.time_features.shape)
        modelCL.fit_transform(mts.time_features)
        fcs = modelCL.contrast(classes, clusters)
    else:
        modelCL.fit_transform(mts.X)
        fcs = modelCL.contrast(classes, clusters)
    fcs = np.array(fcs)
    
    resp_map = {}
    
    for i in range(len(clusters)):
        resp_map[int(clusters[i])] = fcs[i].tolist()
    
    return jsonify(resp_map)

@app.route("/loadWindows", methods=['POST'])
def loadWindows():
    global dataset
    global granularity
    global mts
    global space_DM
    global time_DM
    
    granularity = request.form['granularity']
    datasetName = request.form['dataset']
    pollutants = json.loads(request.form['pollutants'])
    stations = json.loads(request.form['stations'])
    smoothWindow = int(request.form['smoothWindow'])
    shapeNorm = request.form['shapeNorm'] == 'true'
    
    
    print('Loading dataset: {} with granularity {}'.format(datasetName, granularity))
    print('Pollutants: {}'.format(pollutants))
    
    
    if datasetName=='brasil':
        dataset = BrasilDataset(granularity=granularity, fill_missing=FILL_MISSING, max_missing=MAX_MISSING)
    elif datasetName =='ontario':
        dataset = OntarioDataset(granularity=granularity, fill_missing=FILL_MISSING, max_missing=MAX_MISSING)
    if datasetName =='hongkong':
        dataset = HongKongDataset(granularity=granularity, fill_missing=FILL_MISSING, max_missing=MAX_MISSING)
    
    print('Reading stations {}'.format(stations))
    
    resp_map = {}   
    
    dataset.common_windows(pollutants, stations, max_windows=MAX_WINDOWS)
    
    if len(dataset.windows) == 0:
        resp_map['STATUS'] = 'ERROR'
        resp_map['message'] = 'No windows found'
        return jsonify(resp_map)
    
    
    
    resp_map['pollutants'] = dataset.window_pollutants
    resp_map['stations'] = {}
    
    
    for i in range(len(dataset.window_stations)):
        resp_map['stations'][i] = {
            'name': dataset.window_stations[i]
        }
    
    resp_map['windows'] = {}
    
    
    mts = TSerie(X=dataset.windows, y=dataset.window_station_ids, iaqi = dataset.window_iaqis)
    
    
    if mts.T > 40:
        n_basis = 40
    else:
        n_basis = 15
    
    mts.to_basis(n_basis=n_basis)
    mts.smooth(window_size=smoothWindow)
    mts.robustScaler(returnValues=False)
    
    data={
        'X':mts.X,
        'dates': dataset.window_dates.tolist(),
        'stations_ids': dataset.window_station_ids.tolist(),
        'station_map':dataset.stations_map,
        'stations': dataset.stations,
    }
    np.save( 'preprocessed_data', data)
    
    resp_map['windows_labels'] = {
        'dates' : dataset.window_dates.tolist(),
        'stations': dataset.window_station_ids.tolist(),
    }
    
    
    for i in range(len(dataset.window_pollutants)):
        pol = dataset.window_pollutants[i]
        resp_map['windows'][pol] = mts.X_orig[:,:,i].flatten().tolist()
    
    resp_map['windows_labels'] = {
        'dates' : dataset.window_dates.tolist(),
        'stations': dataset.window_station_ids.tolist(),
    }
    resp_map['STATUS'] = 'DONE'
    
    
    # if shapeNorm:
    #     print('Normalizing')
    #     X_norm, _ = mts.shapeNormalizization(returnValues=True)
    #     print('Normalizing done')
    # else:
    #     X_norm, _, _ = mts.minMaxNormalizization(returnValues=True)
    
    # X_norm = mts.X
    resp_map['proc_windows'] = {}
    for i in range(len(dataset.window_pollutants)):
        pol = dataset.window_pollutants[i]
        resp_map['proc_windows'][pol] = mts.X[:,:,i].flatten().tolist()
        # resp_map['orig_windows'][pol] = X_norm[:,:,i].flatten().tolist()
        
    global timeInMs
    global latlong
    
    # Dates
    dates = dataset.window_dates
    timeInMs = np.array([unix_time_millis(d) for d in dates])
    timeInMs = np.expand_dims(timeInMs, axis=1)
    
    # Space coordinates
    stations = dataset.window_stations
    data_info = dataset.stations
    station_ids = dataset.window_station_ids
    
    latlong = np.array([
        [
            float(dataset.stations_map[stations[station_ids[i]]]['latitude']), 
            float(dataset.stations_map[stations[station_ids[i]]]['longitude'])
        ]
        for i in range(len(dates))
    ])
    
    return jsonify(resp_map)
        

# @app.route("/reloadFile", methods=['POST'])
# def reloadFile():
    
    
#     file_exists = exists(path)
#     if not file_exists:
#         print('File does not exist')
#         return
    
#     global storage
#     storage = MTSStorage(path)
#     storage.load()
    
#     global objects
#     objects = storage.objects

@app.route("/loadFile", methods=['POST'])
def reloadFile():
    # filePath = request.get_json()["path"]
    global global_path
    filePath = global_path  
    
    global storage
    storage = MTSStorage(filePath)
    storage.load()
    global objects
    objects = storage.objects[()]
    return jsonify({'status': 'Done'})

@app.route("/objectsInfo", methods=['POST'])
def objectsInfo():
    return jsonify({'names': list(objects.keys())})


@app.route("/object", methods=['POST'])
def object():
    objectName = request.get_json()["name"]
    
    if not objectName in objects:
        return jsonify({'status': 'Error'})
    else:
        N, T, D = objects[objectName]['mts'].shape
        resp_map = {}
        resp_map['data'] = objects[objectName]['mts'].flatten().tolist()
        resp_map['shape'] = objects[objectName]['mts'].shape
        
        
        if 'coords' in objects[objectName]:
            resp_map['coords'] = {}
            for k, v in objects[objectName]['coords'].items():
                resp_map['coords'][k] = v.flatten().tolist()
        if 'labels' in objects[objectName]:
            resp_map['labels'] = {}
            for k, v in objects[objectName]['labels'].items():
                resp_map['labels'][k] = v.flatten().tolist()
                
        if 'labelsNames' in objects[objectName]:
            resp_map['labelsNames'] = objects[objectName]['labelsNames']
        else:
            resp_map['labelsNames'] = {}
            for k, v in objects[objectName]['labels'].items():
                labls = np.unique(v.flatten())
                resp_map['labelsNames'][k] = { str(l):int(l) for l in labls }
            
        if 'dimensions' in objects[objectName]:
            resp_map['dimensions'] = objects[objectName]['dimensions'].flatten().tolist()
        else:
            resp_map['dimensions'] = [str(i) for i in range (D)]
            
    return jsonify(resp_map)

    
    

# def initServer(path, host = "127.0.0.1", port=5000):
    
    
#     file_exists = exists(path)
#     if not file_exists:
#         print('File does not exist')
#         return
    
#     global storage
#     storage = MTSStorage(path)
#     storage.load()
    
#     global objects
#     objects = storage.objects
    
#     global global_path
#     global_path = path
    
#     CORS(app)
#     app.run(host=host, port=port, debug=False)
if __name__ == "__main__":
    host = "0.0.0.0"
    # app.run(host='0.0.0.0', port=port)
    port=5000
    CORS(app)
    # jmap = loadWindows()
    app.run(host=host, port=port, debug=False)
    