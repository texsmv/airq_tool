from .ae.cnn import create_model
import numpy as np

def createAutoencoder(
         data_train: np.array, 
         data_test: np.array, 
         data_val: np.array,
         model_name:str,
         window_size: int,
         batch_size: int,
         n_epochs: int,
         crop_end: bool,
         settings: dict,
):
    encoder, decoder, autoencoder = create_model(window_size, **settings)
    # autoencoder.summary()
    history = autoencoder.fit(
        data_train,
        data_train,
        epochs=n_epochs,
        batch_size=batch_size,
        shuffle=True,
        # validation_data=(data_val, data_val),
        # sample_weight=sample_weight_train,
        verbose=True,
        # summary=True,
        # callbacks=callbacks,
    )
    # autoencoder.save(model_name)
    return encoder, decoder, autoencoder, history
