import numpy as np
from .models.cvae import createAutoencoder

# X_norm of shape NxDxT
def getPeaxFeatures(X_norm, conv_filters, conv_kernels, feat_size = 15, epochs =100, batch_size = 64, X_test=[]):
    X_rec = np.zeros(X_norm.shape)
    feat = np.zeros((X_norm.shape[0], X_norm.shape[1], feat_size))
    if len(X_test) != 0:
        feat_test = np.zeros((X_test.shape[0], X_norm.shape[1], feat_size))
    D = X_norm.shape[1]
    
    for d in range(D):
        series = X_norm[:,d,:]
        series = series.reshape(series.shape[0], series.shape[1], 1)
        
        if len(X_test) != 0:
            series_test = X_test[:,d,:]
            series_test = series_test.reshape(series_test.shape[0], series_test.shape[1], 1)
            
        crop_end = (series.shape[1] % 2) == 1
        aencoder, adecoder, aautoencoder, ahistory = createAutoencoder(
            series,
            series,
            series,
            str(d),
            series.shape[1],
            batch_size,
            epochs,
            True,
            {
                'crop_end':crop_end,
                'embedding': feat_size,
                'summary': True,
                'conv_filters': conv_filters,
                'conv_kernels': conv_kernels,
            },
        )
        X_rec[:, d, :] = aautoencoder.predict(series).squeeze()
        feat[:, d, :] = aencoder.predict(series).squeeze()
        
        if len(X_test) != 0:
            feat_test[:, d, :] = aencoder.predict(series_test).squeeze()

    if len(X_test) != 0:
        return feat, feat_test            
    return feat
    
