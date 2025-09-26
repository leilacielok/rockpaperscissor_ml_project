import time
import tensorflow as tf

def make_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-6),
    ]

def train(model, train_ds, val_ds, epochs=30, callbacks=None):
    callbacks = callbacks or make_callbacks()
    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    runtime = time.time() - t0
    return history, runtime
