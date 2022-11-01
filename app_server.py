from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from flask import jsonify

from source.app_dataset import OntarioDataset, BrasilDataset

granularity = 'months'
dataset = BrasilDataset(granularity=granularity)

app = Flask(__name__)
CORS(app)

@app.route("/datasetInfo", methods=['POST'])
def datasetInfo():
    global dataset


@app.route("/loadWindows", methods=['POST'])
def loadWindows():
    global dataset
    dataset.common_windows(['CO', 'MP25'])
    
    resp_map = {}
    
    resp_map['pollutants'] = dataset.pollutants
    resp_map['stations'] = {}
    for i in range(len(dataset.window_stations)):
        resp_map['stations'][dataset.window_station_ids[i]] = {
            'name': dataset.window_stations[i]
        }
    resp_map['windows'] = {}
    for i in range(len(dataset.window_pollutants)):
        pol = dataset.window_pollutants[i]
        resp_map['windows'][pol] = dataset.windows[:,:,i].flatten().tolist()
    
    print(resp_map)
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
    jmap = loadWindows()
    app.run(host=host, port=port, debug=False)
    