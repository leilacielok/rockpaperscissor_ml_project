import tensorflow as tf
from . import config

def load_train_val(
        validation_split: float = 0.2,
        augment: bool = True,
        cache: bool = True,
        batch_size: int = None,
        normalize: bool = True,
):
    """
    Returns (train_ds, val_ds, file_paths_val) from the folder 'data/' structured as:
        data/rock, data/paper, data/scissors

    - Resizes to (IMG_SIZE, IMG_SIZE)
    - Normalizes to [0,1] if normalize=True
    - Applies lightweight data augmentation to the training set if augment=True
    - If batch_size is None, uses config.BATCH_SIZE
    """
    data_dir = config.DATA_ROOT
    bs = batch_size or config.BATCH_SIZE

    # ---- Validation (single creation; keep order aligned with file paths)
    val_raw = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, validation_split=validation_split, subset="validation",
        seed=config.SEED, image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=bs, label_mode="categorical",
        shuffle=False, class_names=config.CLASSES,
    )
    file_paths_val = val_raw.file_paths

    # ---- Training (single creation)
    train_raw = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=config.SEED,
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=bs,
        label_mode="categororical",
        shuffle=True,
        class_names=config.CLASSES,
    )

    # ---- Preprocessing layers
    normalizer = tf.keras.layers.Rescaling(1.0 / 255.0)
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ]) if augment else None

    # ---- Map TRAIN
    if aug is not None and normalize:
        train_ds = train_raw.map(lambda x, y: (aug(normalizer(x), training=True), y),
                                 num_parallel_calls=tf.data.AUTOTUNE)
    elif aug is not None and not normalize:
        train_ds = train_raw.map(lambda x, y: (aug(x, training=True), y),
                                 num_parallel_calls=tf.data.AUTOTUNE)
    elif aug is None and normalize:
        train_ds = train_raw.map(lambda x, y: (normalizer(x), y),
                                 num_parallel_calls=tf.data.AUTOTUNE)
    else:  # no aug, no normalize
        train_ds = train_raw

    # ---- Map VAL
    if normalize:
        val_ds = val_raw.map(lambda x, y: (normalizer(x), y),
                             num_parallel_calls=tf.data.AUTOTUNE)
    else:
        val_ds = val_raw

    # ---- Performance
    if cache:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, file_paths_val



def load_external_test(cache: bool = True, batch_size: int = None, normalize: bool = True):
    test_dir = "data/rps-cv-images"
    bs = batch_size or config.BATCH_SIZE
    ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=bs,
        label_mode="categorical",
        shuffle=False,
        class_names=config.CLASSES,
    )
    if normalize:
        norm = tf.keras.layers.Rescaling(1.0/255.0)
        ds = ds_raw.map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds_raw

    if cache:
        ds = ds.cache()
    return ds.prefetch(tf.data.AUTOTUNE)


