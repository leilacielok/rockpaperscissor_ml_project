from pathlib import Path
from time import perf_counter

import tensorflow as tf


def make_callbacks(checkpoint_path=None):
    """
    Build and return a list of Keras callbacks controlling the training process.

    The callback set includes:
    - Early stopping based on validation accuracy, with restoration of the
      best-performing model weights.
    - Adaptive learning-rate reduction on validation-loss plateaus.
    - Optional checkpointing of the best model to disk.

    Parameters
        checkpoint_path : str or None, optional
            File path where the best model checkpoint is saved. If None, model
            checkpointing is disabled. Parent directories are created automatically
            if they do not exist.

    Returns
        list of tf.keras.callbacks.Callback
            List of callbacks to be passed to ``model.fit()``.
    """
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max",
            patience=10, min_delta=1e-3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5, cooldown=1
        ),
    ]
    if checkpoint_path:
        Path(checkpoint_path).parent.mkdir(exist_ok=True, parents=True)
        cbs.append(tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", mode="max", save_best_only=True
        ))
    return cbs

def train(model, train_ds, val_ds, epochs=50, callbacks=None, learning_rate=3e-4):
    """
    Train a Keras model on the given training and validation datasets.

    The model is compiled automatically if not already compiled, using the Adam
    optimizer and categorical cross-entropy loss with label smoothing. Training
    time is measured and returned together with the training history.

    Parameters
        model as tf.keras.Model: The Keras model to be trained.
        train_ds as tf.data.Dataset: Training dataset.
        val_ds as tf.data.Dataset: Validation dataset used for performance monitoring and callbacks.
        epochs (int, optional): maximum number of training epochs. Training may terminate earlier if
            early stopping is enabled via callbacks.
        callbacks : list of tf.keras.callbacks.Callback or None, optional
            Callbacks controlling training behavior (e.g., early stopping, learning
            rate scheduling, checkpointing). If None, no callbacks are applied.
        learning_rate (float, optional): Learning rate used for the Adam optimizer when compiling the model.

    Returns
        history : tf.keras.callbacks.History
            Object containing the training and validation metrics recorded at each
            epoch.
        runtime : float
            Total wall-clock training time in seconds.
    """
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
