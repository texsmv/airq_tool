import numpy as np
from source.storage import MTSStorage
from source.datasets import loadFuncionalModel, loadNatops, loadWafer
from sklearn.decomposition import PCA

storage = MTSStorage('mts_datasets')
# NxDxT
X_train, y_train, X_test, y_test = loadWafer()

pca = PCA(n_components=2)
pca.fit(X_train[:, 0, :])
train_coords = pca.transform(X_train[:, 0, :])
test_coords = pca.transform(X_test[:, 0, :])

print(np.unique(y_train))

labelsMap = {
    -1: 'class 1',
    1: 'class 2',
}
print(y_train)
storage.add_mts('wafer_train', np.transpose(X_train, (0, 2, 1)), labels = {'class': y_train},)
# storage.add_mts('wafer_train', X_train, labels = {'class': y_train}, coords={'train': train_coords}, labelsNames={'class':labelsMap} )
# storage.add_mts('wafer_train', X_train, labels = {'class': y_train}, coords={'train': train_coords}, labelsNames={'class':labelsMap} )
# storage.add_mts('wafer_test', X_test, labels = {'class':y_test}, coords={'test': test_coords}, labelsNames={'class':labelsMap})



# X_train, y_train, X_test, y_test = loadNatops()
# pca = PCA(n_components=2)
# pca.fit(X_train[:, 0, :])
# train_coords = pca.transform(X_train[:, 0, :])
# test_coords = pca.transform(X_test[:, 0, :])

# storage.add_mts('natops_train', X_train, labels = y_train, coords=train_coords)
# storage.add_mts('natops_test', X_test, labels = y_test, coords=test_coords)

storage.save()

# storage.load()
