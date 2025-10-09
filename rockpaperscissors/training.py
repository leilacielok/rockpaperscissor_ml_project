import tensorflow as tf
from time import perf_counter
from pathlib import Path

def make_callbacks(checkpoint_path=None):
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max",
            patience=10, min_delta=1e-3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5, cooldown=1
        ),
    ]
    if checkpoint_path:  # <— aggiunto: crea dir e checkpoint solo se richiesto
        Path(checkpoint_path).parent.mkdir(exist_ok=True, parents=True)
        cbs.append(tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", mode="max", save_best_only=True
        ))
    return cbs

def train(model, train_ds, val_ds, epochs=30, callbacks=None, learning_rate=3e-4):
    if getattr(model, "optimizer", None) is None:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=["accuracy"],
        )
    t0 = perf_counter()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs, callbacks=callbacks, verbose=1,
    )
    runtime = perf_counter() - t0
    return history, runtime
