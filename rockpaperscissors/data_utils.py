# rockpaperscissors/data_utils.py
import tensorflow as tf
from . import config

def load_train_val(validation_split: float = 0.2, augment: bool = True):
    """
    Returns (train_ds, val_ds) from the folder structure:
      data/
        rock/
        paper/
        scissors/
    Applies resize to (IMG_SIZE, IMG_SIZE) and normalization to [0,1].
    If augment=True, applies lightweight data augmentation to the training set.
    """
    data_dir = config.DATA_ROOT

    # Base datasets (Keras will stratify by folder)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=config.SEED,
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        class_names=config.CLASSES,  # <- force only 3 classes
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=config.SEED,
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
        class_names=config.CLASSES,
    )

    # Normalize pixels to [0,1]
    normalizer = tf.keras.layers.Rescaling(1.0 / 255.0)

    # Optional augmentation (train only)
    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])
        train_ds = train_ds.map(lambda x, y: (aug(normalizer(x)), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        train_ds = train_ds.map(lambda x, y: (normalizer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    val_ds = val_ds.map(lambda x, y: (normalizer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

