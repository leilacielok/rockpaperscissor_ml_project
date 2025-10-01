import os, glob, math, random
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from . import config

# stratified split

def load_train_val_stratified(
    validation_split: float = 0.2,
    augment: bool = True,
    cache: bool = True,
    batch_size: int = None,
    normalize: bool = True,
):
    """
    Stratifica veramente per classe leggendo i file dal filesystem.
    Ritorna (train_ds, val_ds, file_paths_val).
    """
    data_dir = config.DATA_ROOT
    bs = batch_size or config.BATCH_SIZE
    class_to_idx = {c: i for i, c in enumerate(config.CLASSES)}

    # 1) Raccogli file e label
    files, labels = [], []
    for c in config.CLASSES:
        pattern = os.path.join(data_dir, c, "*")
        paths = [p for p in glob.glob(pattern) if os.path.isfile(p)]
        files.extend(paths)
        labels.extend([class_to_idx[c]] * len(paths))

    files = np.array(files)
    labels = np.array(labels)

    # 2) Split stratificato
    f_train, f_val, y_train, y_val = train_test_split(
        files, labels,
        test_size=validation_split,
        random_state=config.SEED,
        stratify=labels,
        shuffle=True,
    )

    # 3) Parser immagine
    def _load_img(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1] float32
        img = tf.image.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        return img

    def _parse(path, label):
        x = _load_img(path)
        if not normalize:
            # riportiamo a [0,255] per modelli che useranno preprocess_input interno
            x = x * 255.0
        return x, tf.one_hot(label, depth=len(config.CLASSES))

    # 4) Dataset TF
    ds_train = tf.data.Dataset.from_tensor_slices((f_train, y_train)).map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val   = tf.data.Dataset.from_tensor_slices((f_val,   y_val)).map(_parse,   num_parallel_calls=tf.data.AUTOTUNE)

    # 5) Augment (solo train)
    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomContrast(0.05),
        ])
        ds_train = ds_train.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    # 6) Batch + performance
    ds_train = ds_train.shuffle(4 * bs, seed=config.SEED, reshuffle_each_iteration=True).batch(bs)
    ds_val   = ds_val.batch(bs)

    if cache:
        ds_train = ds_train.cache()
        ds_val   = ds_val.cache()

    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    ds_val   = ds_val.prefetch(tf.data.AUTOTUNE)

    # file_paths_val nella stessa order dei batch
    file_paths_val = list(f_val)

    return ds_train, ds_val, file_paths_val


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

def compute_class_priors(ds, n_classes):
    import numpy as np
    counts = np.zeros(n_classes, dtype=np.int64)
    for _, y in ds:
        counts += y.numpy().sum(axis=0).astype(np.int64)
    priors = counts / counts.sum()
    return priors
