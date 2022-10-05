import re
from flask import Flask, jsonify, request
from flask_cors import CORS
from source.storage import MTSStorage
import numpy as np
from os.path import exists

from flask import jsonify

# storage = MTSStorage('mts_datasets')
# storage = MTSStorage('basa_supervised')
# storage = MTSStorage('/home/texs/Documents/AirQuality/air_quality/airq')

global_path ='mts_datasets'
# path = '/home/texs/Documents/VivaBem/repositories/sedentary_datasets_visualizations/basa_supervised'
# path = '/home/texs/Documents/AirQuality/air_quality/airq'


storage = None
# storage = MTSStorage(path)
# storage.load()

objects = None
# objects = storage.objects[()]

# app = Flask(__name__)
# CORS(app)
app = Flask(__name__)
CORS(app)


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

    
    

def initServer(path, host = "127.0.0.1", port=5000):
    # path = '/home/texs/Documents/VivaBem/repositories/sedentary_datasets_visualizations/basa_metrics'
    # path = '/home/texs/Documents/VivaBem/repositories/sedentary_datasets_visualizations/har_metrics'
    # path = '/home/texs/Documents/VivaBem/repositories/sedentary_datasets_visualizations/open_metrics'
    
    # path = '/home/texs/Documents/VivaBem/repositories/sedentary_datasets_visualizations/basa_supervised'
    # path = '/home/texs/Documents/VivaBem/repositories/sedentary_datasets_visualizations/har'
    # path = '/home/texs/Documents/VivaBem/repositories/sedentary_datasets_visualizations/basa_metrics_test'
    # path = '/home/texs/Documents/VivaBem/repositories/sedentary_datasets_visualizations/har_metrics_exp'
    
    # path = 'mts_datasets'
    
    # path = '/home/texs/Documentos/AirQuality/air_quality/madrid_vae'
    # path = '/home/texs/Documentos/AirQuality/air_quality/madrid_UNI'
    # path = '/home/texs/Documentos/AirQuality/air_quality/madrid_peax'
    # path = '/home/texs/Documentos/AirQuality/air_quality/brasil_peax'
    # path = '/home/texs/Documentos/AirQuality/airqviz/ontario'
    # path = '/home/texs/Documentos/AirQuality/airqviz/mts_comparison'
    # path = '/home/texs/Documentos/AirQuality/airqviz/mts_test'
    
    # path = '/home/texs/Documentos/Repositories/mts_feature_learning/extraSensory'
    # path = '/home/texs/Documentos/Repositories/mts_feature_learning/pamap2'
    # path = '/home/texs/Documentos/Repositories/mts_feature_learning/har_kfold'
    # path = '/home/texs/Documentos/Repositories/mts_feature_learning/har_augmentations_cont'
    # path = '/home/texs/Documentos/Repositories/mts_feature_learning/har_augmentations'
    # path = '/home/texs/Documentos/Repositories/mts_feature_learning/har_augmentations_umap'
    # path = '/home/texs/Documentos/Repositories/mts_feature_learning/har_norm_viz'
    
    
    file_exists = exists(path)
    if not file_exists:
        print('File does not exist')
        return
    
    global storage
    storage = MTSStorage(path)
    storage.load()
    
    global objects
    objects = storage.objects
    
    global global_path
    global_path = path
    
    CORS(app)
    app.run(host=host, port=port, debug=False)
# if __name__ == "__main__":