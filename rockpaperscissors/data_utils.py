import tensorflow as tf
from . import config

def load_train_val(validation_split: float = 0.2, augment: bool = True, cache: bool = True):
    """
    Returns (train_ds, val_ds) from the folder data
    Applies resize to (IMG_SIZE, IMG_SIZE) and normalization to [0,1].
    If augment=True, applies lightweight data augmentation to the training set.
    """
    data_dir = config.DATA_ROOT
    bs = batch_size or config.BATCH_SIZE

    val_raw = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, validation_split=validation_split, subset="validation",
        seed=config.SEED, image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE, label_mode="categorical",
        shuffle=False, class_names=config.CLASSES,
    )
    file_paths_val = val_raw.file_paths

    # Preprocessing layers
    normalizer = tf.keras.layers.Rescaling(1.0 / 255.0)
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ]) if augment else None

    # Validation raw
    val_raw = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, validation_split=validation_split, subset="validation",
        seed=config.SEED, image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=bs, label_mode="categorical",
        shuffle=False, class_names=config.CLASSES,
    )
    file_paths_val = val_raw.file_paths

    # Map
    val_ds = val_raw.map(lambda x, y: (normalizer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Training
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=config.SEED,
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=bs,
        label_mode="categorical",
        shuffle=True,
        class_names=config.CLASSES,
    )

    if augment is not None:
        train_ds = train_ds.map(lambda x, y: (aug(normalizer(x), training=True), y),
                                num_parallel_calls=tf.data.AUTOTUNE)
    else:
        train_ds = train_ds.map(lambda x, y: (normalizer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, file_paths_val


def load_external_test(cache: bool = True, batch_size:int=None):
    test_dir = "data/rps-cv-images"
    bs = batch_size or config.BATCH_SIZE
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=bs,
        label_mode="categorical",
        shuffle=False,
        class_names=config.CLASSES,
    )
    norm = tf.keras.layers.Rescaling(1.0/255.0)
    ds = ds.map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    return ds.prefetch(tf.data.AUTOTUNE)

