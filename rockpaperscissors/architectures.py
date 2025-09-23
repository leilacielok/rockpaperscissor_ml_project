import tensorflow as tf
from . import config

def model_a():
    """Baseline CNN (small)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input((config.IMG_SIZE, config.IMG_SIZE, 3)),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(config.CLASSES), activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model
