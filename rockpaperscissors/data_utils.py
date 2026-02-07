import os, glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from . import config

AUTOTUNE = tf.data.AUTOTUNE

def _decode_image(path):
    img_bytes = tf.io.read_file(path)
    # decode_image manages png/jpg
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    return img

def _load_img(path, img_size):
    img = _decode_image(path)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (img_size, img_size), method="bilinear", antialias=True)
    return img

def _parse(path, label, img_size, n_classes):
    x = _load_img(path, img_size)
    y = tf.one_hot(label, depth=n_classes)
    return x, y

def _augment_fn(x, y):
    # stable and light Augment (active on training)
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
        tf.keras.layers.RandomContrast(0.05),
    ])
    return aug(x, training=True), y

# count number of images per class in the first 20 batches
def _class_hist(ds, n_batches=20):
    tot = None
    for i, (_, y) in enumerate(ds.take(n_batches)):
        s = tf.reduce_sum(y, axis=0)
        tot = s if tot is None else tot + s
    return None if tot is None else tot.numpy()

# public API
def load_train_val_stratified(    
    validation_split: float = 0.2,
    augment: bool = True,
    cache_train: bool = False,
    cache_val: bool = True,
    batch_size: int | None = None,
):
    """
    Load a stratified train/validation split from the image dataset stored on disk.

    The function builds efficient ``tf.data`` pipelines including image decoding,
    resizing, normalization to the [0, 1] range, optional on-the-fly data
    augmentation (applied only to the training set), batching, caching, and
    prefetching.

    Returns:
    # train_ds and val_ds as tf.data.Dataset (dataset yielding batches of (image, one-hot label) pairs).
    # file_paths_val as list of str (List of file paths corresponding to the validation samples, useful 
    for external analysis or qualitative inspection).
    """
    # Stratified split 
    data_dir = config.DATA_ROOT
    img_size = getattr(config, "IMG_SIZE", 96)
    bs = batch_size or config.BATCH_SIZE
    class_names = list(config.CLASSES)
    n_classes = len(class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # 1) get rows per class (order from config.CLASSES)
    files, labels = [], []
    for c in class_names:
        pattern = os.path.join(data_dir, c, "*")
        paths = [p for p in glob.glob(pattern) if os.path.isfile(p)]
        files.extend(paths)
        labels.extend([class_to_idx[c]] * len(paths))

    files = np.array(files)
    labels = np.array(labels)

    if len(files) == 0:
        raise RuntimeError(f"No image found in {data_dir} in the subfolders {class_names}")

    # 2) stratified split
    f_train, f_val, y_train, y_val = train_test_split(
        files, labels,
        test_size=validation_split,
        random_state=config.SEED,
        stratify=labels,
        shuffle=True,
    )

    # diagnostic: all classes in both splits
    train_present = np.unique(y_train)
    val_present = np.unique(y_val)
    assert set(train_present) == set(range(n_classes)), f"Missing classes in train: {train_present}"
    assert set(val_present)   == set(range(n_classes)), f"Missing classes in val: {val_present}"

    # 3) dataset tf.data with normalization
    parse_train = lambda p, y: _parse(p, y, img_size, n_classes)
    parse_val   = lambda p, y: _parse(p, y, img_size, n_classes)

    ds_train = tf.data.Dataset.from_tensor_slices((f_train, y_train)).map(parse_train, num_parallel_calls=AUTOTUNE)
    ds_val   = tf.data.Dataset.from_tensor_slices((f_val,   y_val  )).map(parse_val,   num_parallel_calls=AUTOTUNE)

    # 4) augment only on train
    if augment:
        aug_layer = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomContrast(0.05),
        ], name="aug")

        def _apply_aug(x, y):
            return aug_layer(x, training=True), y

        ds_train = ds_train.map(_apply_aug, num_parallel_calls=AUTOTUNE)

    # 5) batching / shuffle / cache / prefetch
    ds_train = ds_train.shuffle(buffer_size=bs*4, seed=config.SEED, reshuffle_each_iteration=True).batch(bs)
    ds_val   = ds_val.batch(bs)

    if cache_train:
        ds_train = ds_train.cache()
    if cache_val:
        ds_val = ds_val.cache()

    ds_train = ds_train.prefetch(AUTOTUNE)
    ds_val   = ds_val.prefetch(AUTOTUNE)

    # 6) diagnostic: mapping classes and distributions (first ~20 batch)
    print("class_names (train/val):", class_names)
    tr_hist = _class_hist(ds_train, n_batches=20)
    va_hist = _class_hist(ds_val,   n_batches=20)
    if tr_hist is not None and va_hist is not None:
        print("approx train label sums:", tr_hist.astype(int))
        print("approx val   label sums:", va_hist.astype(int))

    # file_paths_val for external analysis
    file_paths_val = list(f_val)
    return ds_train, ds_val, file_paths_val

def load_external_test(cache: bool = True, batch_size: int | None = None):
    """
    External test, same normalization of training [0,1], same class order.
    """
    test_dir = "data/rps-cv-images"
    img_size = getattr(config, "IMG_SIZE", 96)
    bs = batch_size or config.BATCH_SIZE

    ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=(img_size, img_size),
        batch_size=bs,
        label_mode="categorical",
        shuffle=False,
        class_names=list(config.CLASSES),
        interpolation="bilinear",
    )

    norm = tf.keras.layers.Rescaling(1.0/255.0)
    ds = ds_raw.map(lambda x, y: (norm(x), y), num_parallel_calls=AUTOTUNE)

    if cache:
        ds = ds.cache()
    return ds.prefetch(AUTOTUNE)

def compute_class_priors(ds, n_classes):
    """
    Compute empirical class priors from a dataset with one-hot labels.

    Parameters
    ----------
    ds : tf.data.Dataset
        Dataset yielding (x, y) pairs with one-hot encoded labels.
    n_classes : int
        Number of classes.

    Returns
    -------
    np.ndarray
        Array of class prior probabilities summing to 1.
    """
    counts = np.zeros(n_classes, dtype=np.int64)
    for _, y in ds:
        counts += y.numpy().sum(axis=0).astype(np.int64)
    priors = counts / max(1, counts.sum())
    return priors
