from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from flask import jsonify
import json
import sys
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from contrastive import CPCA
import matplotlib.pyplot as plt

from source.tserie import TSerie
from sklearn.decomposition import PCA
from cuml.neighbors import NearestNeighbors
from cuml.manifold import UMAP
# from ccpca import CCPCA
from source.utils import folding_2D
from source.app_dataset import OntarioDataset, BrasilDataset
from source.read_ontario import read_ontario_stations
from source.utils import fdaOutlier
import umap
from source.featlearn.autoencoder_lr import AutoencoderFL, VAE_FL, DCEC
from source.featlearn.byol import BYOL, BarlowTwins
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split


import datetime

epoch = datetime.datetime.utcfromtimestamp(0)

def unix_time_millis(dt):
    return (dt - epoch).total_seconds() * 1000.0


sys.path.append('/home/texs/Documentos/Repositories/ts2vec')
from ts2vec import TS2Vec


MODE = 2 # 0 for umap, 1 for ts2vec, 2 for CAE

USE_TS2VEC = False
MAX_WINDOWS = 40000
UMAP_METRIC = 'braycurtis'
BATCH_SIZE = 240

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

class UMAP_FL:
    def __init__(self, n_components, n_neighbors, metric = 'braycurtis', n_epochs = 1000):
        self.reducer = UMAP(n_components=n_components, n_neighbors=n_neighbors, n_epochs=n_epochs)
        self.nearNeigh = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)

    def fit_transform(self, X, y=None):
        self.nearNeigh.fit(X)
        knn_graph = self.nearNeigh.kneighbors_graph(X, mode="distance")
        embeddings =  self.reducer.fit_transform(X, y=y, knn_graph=knn_graph.tocsr(), convert_dtype=True)
        return embeddings
    
    def transform(self, X):
        knn_graph = self.nearNeigh.kneighbors_graph(X, mode="distance")
        embeddings =  self.reducer.transform(X, knn_graph=knn_graph.tocsr(), convert_dtype=True)
        return embeddings

class UMAP_CFL:
    def __init__(self):
        self.reducer = PCA(n_components=1)
    def fit_transform(self, X, y=None):
        N, T, D = X.shape
        dim_compressed = np.zeros([N, T])
        
        # Compressing 
        dim_data = []
        for i in range(N):
            for j in range(T):
                dim_data.append(mts.X[i, j])
        dim_data = np.array(dim_data)
        dim_features = self.reducer.fit_transform(dim_data)
        dim_features = np.reshape(dim_features, [N, T])
        for i in range(N):
            for j in range(T):
                dim_compressed[i, j] = dim_features[i, j]
        self.features = dim_compressed

    def contrast(self, labels, clusters):
        fcs_time = []
        for target in clusters:
            ccpca = CCPCA(n_components=1)
            
            Xf = self.features[labels==target]
            Xb = self.features[labels!=target]
            
            ccpca.fit(
                Xf,
                Xb,
                var_thres_ratio=0.5,
                n_alphas=40,
                max_log_alpha=0.5,
            )
            _ = ccpca.transform(Xf)
            best_alpha = ccpca.get_best_alpha()
            cpca_fcs = ccpca.get_feat_contribs()
            fcs_time.append(cpca_fcs)        
        return fcs_time




dataset = None
mts = None
g_coords = None
space_DM = None
time_DM = None
feature_DM = None

app = Flask(__name__)
CORS(app)

@app.route("/datasets", methods=['POST'])
def datasetsInfo():
    ontarioDataset = OntarioDataset()
    brasilDataset = BrasilDataset()
    
    resp_map = {
        'ontario' : {
            'pollutants': ontarioDataset.all_pollutants
        },
        'brasil' : {
            'pollutants': brasilDataset.all_pollutants
        },
    }
    
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
    
    print(all_positions.shape)
    print(back_positions.shape)
    print(positions.shape)
    # back_mts  =TSerie(mts.X_orig[back_positions], None)
    # fore_mts  =TSerie(mts.X_orig[positions], None)
    
    
    # back_mts.folding_features_v1()
    # fore_mts.folding_features_v1()
    
    cpca = CPCA(standardize=False)
    # result = cpca.fit_transform(background=back_mts.features, foreground=fore_mts.features, plot=False)
    result = cpca.fit_transform(background=mts.features[back_positions], foreground=mts.features[positions], plot=False)
    
    allCoords = np.zeros((n, 2))
    
    allCoords[positions] = result[1]
    
    
    coords = result[0]
    plt.close()
    plt.cla()
    plt.clf()
    print(coords.shape)
    plt.scatter(coords[:,0],coords[:,1])
    plt.savefig('imagetest0.png')
    
    coords = result[1]
    plt.close()
    plt.cla()
    plt.clf()
    print(coords.shape)
    plt.scatter(coords[:,0],coords[:,1])
    plt.savefig('imagetest1.png')
    
    coords = result[2]
    plt.close()
    plt.cla()
    plt.clf()
    print(coords.shape)
    plt.scatter(coords[:,0],coords[:,1])
    plt.savefig('imagetest2.png')
    
    coords = result[3]
    plt.close()
    plt.cla()
    plt.clf()
    print(coords.shape)
    plt.scatter(coords[:,0],coords[:,1])
    plt.savefig('imagetest3.png')
    
    X = mts.X_orig[positions]
    
    
    
    X = np.vstack(X) # To shape: N * T, D
    
    df = pd.DataFrame(data=X)
    corr_matrix = df.corr().to_numpy()
    
    resp_map = {}
    resp_map['correlation_matrix'] = corr_matrix.flatten().tolist()
    resp_map['coords'] = allCoords.flatten().tolist()
    return jsonify(resp_map)
    
    
@app.route("/getProjection", methods=['POST'])
def getProjection():
    global dataset
    global granularity
    global mts
    global g_coords
    
    pollPositions = np.array(json.loads(request.form['pollutantsPositions']))
    
    
    # EPOCHS = 5
    N_NEIGHBORS = int(request.form['neighbors'])
    delta = float(request.form['delta'])
    beta = float(request.form['beta'])
    
    if granularity == 'months':
        EPOCHS = 10
        EPOCHS_CAE = 800
        FEATURE_SIZE_CAE = 12 
        # N_NEIGHBORS = 30
    elif granularity == 'years':
        EPOCHS = 25
        EPOCHS_CAE = 2000
        FEATURE_SIZE_CAE = 30
        # N_NEIGHBORS = 5
    elif granularity == 'daily':
        EPOCHS = 40
        EPOCHS_CAE = 800
        FEATURE_SIZE_CAE = 8
        # N_NEIGHBORS = 15
    
    if MODE == 0:
        
        
        X_filtered = mts.X[:, :, pollPositions]
        mts_filtered = TSerie(X_filtered, mts.y)
        mts_filtered.folding_features_v1()
        
        # model = UMAP_FL(n_components=32, n_neighbors=N_NEIGHBORS, metric=UMAP_METRIC, n_epochs = 15000)s
        model = PCA(n_components=16)
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
        # cae = AutoencoderFL(mts.D, mts.T, feature_size=FEATURE_SIZE_CAE)
        # cae = VAE_FL(mts.D, mts.T, feature_size=FEATURE_SIZE_CAE)
        # cae = DCEC(mts.D, mts.T, feature_size=FEATURE_SIZE_CAE, n_clusters=5)
        
        X_train, X_val = train_test_split(mts.X.transpose([0, 2, 1]))
        # cae = BYOL(mts.D, mts.T, feature_size=FEATURE_SIZE_CAE, aug_type='noise')
        cae = BarlowTwins(mts.D, mts.T, feature_size=FEATURE_SIZE_CAE, aug_type='noise')
        
        
        cae.fit(X_train, epochs=100, batch_size=320, X_val=X_val)
        # cae.fit(mts.X, epochs=500, batch_size=400)
        # cae.fit(mts.X, epochs=500, batch_size=400, gamma=100)
        # _, mts.features = cae.encode(mts.X)
        mts.features = cae.encode(mts.X.transpose([0, 2, 1]))
        # _, mts.features, clusters = cae.encode(mts.X)
        # preds = np.argmax(clusters, axis=1)
        # print(np.unique(preds, return_counts=True))
        
    
    feature_DM = pairwise_distances(mts.features, metric='cosine')
    feature_DM = feature_DM / np.max(feature_DM)
    
    delta = 0.0
    # delta = 0.001
    beta = 0.0
    # beta = 0.00
    
    distM = feature_DM * (1 - (delta + beta)) + space_DM * delta + time_DM * beta
    reducer = umap.UMAP(n_components=2, metric='precomputed')
    
    
    # reducer = umap.UMAP(n_components=2, metric='euclidean')
    # reducer = umap.UMAP(n_components=2, metric='cosine')
    #  reducer.transform(mts.features)
    coords = reducer.fit_transform(distM)
    g_coords = coords
    reducer = None
    
    
    resp_map = {}
    resp_map['coords'] = coords.flatten().tolist()
    
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
    g_coords = coords
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
    
    cmean, cvar = fdaOutlier(ts)
    
    resp_map = {}
    resp_map['cmean'] = cmean.tolist()
    resp_map['cvar'] = cvar.tolist()
    
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
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(g_coords)
    classes = kmeans.labels_
        
    resp_map = {}
    resp_map['classes'] = classes.tolist()
    
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
    smoothWindow = int(request.form['smoothWindow'])
    shapeNorm = request.form['shapeNorm'] == 'true'
    
    
    print('Loading dataset: {} with granularity {}'.format(datasetName, granularity))
    print('Pollutants: {}'.format(pollutants))
    
    
    if datasetName=='brasil':
        dataset = BrasilDataset(granularity=granularity)
    elif datasetName =='ontario':
        dataset = OntarioDataset(granularity=granularity)
    
    
    
    dataset.common_windows(pollutants, max_windows=MAX_WINDOWS)
    data_info = read_ontario_stations()
    
    # Dates matrix stuff
    dates = dataset.window_dates
    timeInMs = np.array([unix_time_millis(d) for d in dates])
    timeInMs = np.expand_dims(timeInMs, axis=1)
    
    time_DM = pairwise_distances(timeInMs)
    time_DM = time_DM / np.max(time_DM)
    
    
    # Location matrix stuff  
    stations = dataset.window_stations_all
    station_ids = dataset.window_station_ids
    print(data_info)
    print(station_ids)
    print(stations)
    coords = np.array([
        [
            float(data_info[str(stations[station_ids[i]])]['latitude']), 
            float(data_info[str(stations[station_ids[i]])]['longitude'])
        ]
        for i in range(len(dates))
    ])
    
    space_DM = pairwise_distances(coords)
    space_DM = space_DM / np.max(space_DM)


    
    
    resp_map = {}
    
    
    resp_map['pollutants'] = dataset.window_pollutants
    resp_map['stations'] = {}
    
    
    for i in range(len(dataset.stations)):
        resp_map['stations'][i] = {
            'name': dataset.stations[i]
        }
    
    resp_map['windows'] = {}
    
    
    mts = TSerie(X=dataset.windows, y=dataset.window_station_ids)
    mts.smooth(window_size=smoothWindow)
    
    resp_map['windows_labels'] = {
        'dates' : dataset.window_dates.tolist(),
        'stations': dataset.window_station_ids.tolist(),
    }
    
    
    for i in range(len(dataset.window_pollutants)):
        pol = dataset.window_pollutants[i]
        resp_map['windows'][pol] = mts.X[:,:,i].flatten().tolist()
    
    resp_map['windows_labels'] = {
        'dates' : dataset.window_dates.tolist(),
        'stations': dataset.window_station_ids.tolist(),
    }
    
    
    if shapeNorm:
        print('Normalizing')
        X_norm, _ = mts.shapeNormalizization(returnValues=True)
        print('Normalizing done')
    else:
        X_norm, _, _ = mts.minMaxNormalizization(returnValues=True)

    resp_map['proc_windows'] = {}
    for i in range(len(dataset.window_pollutants)):
        pol = dataset.window_pollutants[i]
        resp_map['proc_windows'][pol] = X_norm[:,:,i].flatten().tolist()
    
    
    
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
            # print('KHE?')
            resp_map['labelsNames'] = objects[objectName]['labelsNames']
        else:
            resp_map['labelsNames'] = {}
            for k, v in objects[objectName]['labels'].items():
                labls = np.unique(v.flatten())
                print('keys')
                print(k)
                print(labls)
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
    