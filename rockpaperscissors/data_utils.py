import tensorflow as tf
from . import config

def load_data(validation_split=0.2):
    """Load train/val datasets from the data/ folder."""
    data_dir = config.DATA_ROOT

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=config.SEED,
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        label_mode="categorical"
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=config.SEED,
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        label_mode="categorical"
    )

    # normalize pixels to [0,1]
    normalizer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalizer(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.map(lambda x, y: (normalizer(x), y)).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

if __name__ == "__main__":
    train_ds, val_ds = load_data()
    print("Train batches:", len(list(train_ds)))
    print("Val batches:", len(list(val_ds)))

