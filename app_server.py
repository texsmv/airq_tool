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
from source.utils import fdaOutlier
import umap

sys.path.append('/home/amendoza/Documentos/Repositories/ts2vec')
from ts2vec import TS2Vec

USE_TS2VEC = False
MAX_WINDOWS = 40000
UMAP_METRIC = 'braycurtis'
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
    
    back_mts  =TSerie(mts.X_orig, None)
    fore_mts  =TSerie(mts.X_orig[positions], None)
    back_mts.folding_features_v1()
    fore_mts.folding_features_v1()
    
    cpca = CPCA(standardize=False)
    result = cpca.fit_transform(background=back_mts.features, foreground=fore_mts.features, plot=False)
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
    return jsonify(resp_map)
    
    
@app.route("/getProjection", methods=['POST'])
def getProjection():
    global dataset
    global granularity
    global mts
    
    pollPositions = np.array(json.loads(request.form['pollutantsPositions']))
    
    
    # EPOCHS = 5
    N_NEIGHBORS = int(request.form['neighbors'])
    
    if granularity == 'months':
        EPOCHS = 5
        # N_NEIGHBORS = 30
    elif granularity == 'years':
        EPOCHS = 25
        # N_NEIGHBORS = 5
    elif granularity == 'daily':
        EPOCHS = 40
        # N_NEIGHBORS = 15
    
    if not USE_TS2VEC:
        
        model = UMAP_FL(n_components=32, n_neighbors=N_NEIGHBORS, metric=UMAP_METRIC, n_epochs = 15000)
        
        X_filtered = mts.X[:, :, pollPositions]
        mts_filtered = TSerie(X_filtered, mts.y)
        mts_filtered.folding_features_v1()
        mts.features = model.fit_transform(mts_filtered.features)
    else:
        model = TS2Vec(
            input_dims=mts.D,
            device=0,
            output_dims=32,
            batch_size=4,
            depth=8,
            hidden_dims=32,
        )
        model.fit(mts.X, verbose=True,n_epochs = EPOCHS)
        mts.time_features = model.encode(mts.X, batch_size=4)
        mts.features = model.encode(mts.X, encoding_window='full_series', batch_size=4)
    
    reducer = umap.UMAP(n_components=2, metric='cosine')
    reducer.fit(mts.features)
    coords = reducer.transform(mts.features)
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
    
    n_clusters = request.form['n_clusters']
    n_clusters = int(n_clusters)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(mts.features)
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
    host = "127.0.0.1"
    port=5000
    CORS(app)
    # jmap = loadWindows()
    app.run(host=host, port=port, debug=False)
    