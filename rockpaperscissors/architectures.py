import tensorflow as tf
from . import config
import matplotlib.pyplot as plt
from PIL import Image


def sep_block(x, filters):
    y = tf.keras.layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    return y

def residual_sep_block(x, filters, pool_first=False):
    """
    Residual block with two SeparableConv2D.
    - if pool_first=True, MaxPool on main and on shortcut before the match.
    - Projects the shortcut to 'filters'
    """
    shortcut = x

    if pool_first:
        x = tf.keras.layers.MaxPool2D()(x)
        shortcut = tf.keras.layers.MaxPool2D()(shortcut)

    y = sep_block(x, filters)
    y = sep_block(y, filters)

    # Proiezione dello shortcut se #canali non combacia
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    out = tf.keras.layers.Add()([shortcut, y])
    out = tf.keras.layers.ReLU()(out)
    return out


def model_a():
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

def model_b():
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


def model_c():
    inputs = tf.keras.Input((config.IMG_SIZE, config.IMG_SIZE, 3))

    # stem
    x = sep_block(inputs, 32)
    x = sep_block(x, 32)
    x = tf.keras.layers.MaxPool2D()(x)  # -> (H/2, W/2, 32)

    # block with 64 ch + residual
    y = sep_block(x, 64)
    y = sep_block(y, 64)  # -> (H/2, W/2, 64)
    x_proj = tf.keras.layers.Conv2D(64, 1, padding="same")(x)  # projection skip to 64 layers
    x = tf.keras.layers.Add()([x_proj, y])  # sums up skip - main
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)  # -> (H/4, W/4, 64)

    # head
    x = sep_block(x, 96)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(96, activation="relu")(x)
    outputs = tf.keras.layers.Dense(len(config.CLASSES), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(8e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def model_d():
    inputs = tf.keras.Input((config.IMG_SIZE, config.IMG_SIZE, 3))

    # stem
    x = sep_block(inputs, 32)
    x = sep_block(x, 32)
    x = tf.keras.layers.MaxPool2D()(x)

    # residual block
    x = residual_sep_block(x, 64, pool_first=False)  # match canali via 1x1 sullo shortcut
    x = tf.keras.layers.MaxPool2D()(x)  # -> (H/4, W/4, 64)

    # Head
    x = sep_block(x, 96)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(96, activation="relu")(x)
    outputs = tf.keras.layers.Dense(len(config.CLASSES), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(8e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model