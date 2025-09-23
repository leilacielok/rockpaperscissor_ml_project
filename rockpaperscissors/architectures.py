import tensorflow as tf
from . import config

def model_a():
    """Baseline CNN (small)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input((config.IMG_SIZE, config.IMG_SIZE, 3)),
        tf.keras.layers.Conv2D(8, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(len(config.CLASSES), activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def model_tiny():
    """Very small, fast CNN (good baseline)."""
    inputs = tf.keras.Input((config.IMG_SIZE, config.IMG_SIZE, 3))

    x = tf.keras.layers.SeparableConv2D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.SeparableConv2D(24, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.SeparableConv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)          # <- no Flatten
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)      # small head

    outputs = tf.keras.layers.Dense(len(config.CLASSES), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model