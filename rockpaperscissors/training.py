import time
import tensorflow as tf
from pathlib import Path

def make_callbacks(checkpoint_path="models/best.keras"):
    Path("models").mkdir(exist_ok=True, parents=True)
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint("models/best.keras", monitor="val_loss", save_best_only=True)
    ]

def train(model, train_ds, val_ds, epochs=30, callbacks=None):
    callbacks = callbacks or make_callbacks()
    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)
    runtime = time.time() - t0
    return history, runtime
