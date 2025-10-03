import tensorflow as tf
from . import config
import matplotlib.pyplot as plt
from PIL import Image

def ln_relu(x, axis=(-1,)):
    x = tf.keras.layers.LayerNormalization(axis=axis)(x)
    return tf.keras.layers.ReLU()(x)

def sep_block_ln(x, filters, sd_rate=0.0):
    y = ln_relu(x)
    y = tf.keras.layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(y)
    if sd_rate > 0:
        y = tf.keras.layers.SpatialDropout2D(sd_rate)(y)
    return y

def residual_sep_block_ln(x, filters, downsample=False, sd_rate=0.1):
    shortcut = x
    y = sep_block_ln(x, filters, sd_rate=sd_rate)
    y = sep_block_ln(y, filters, sd_rate=sd_rate)

    if downsample:
        y = tf.keras.layers.MaxPool2D()(y)
        shortcut = tf.keras.layers.MaxPool2D()(shortcut)
    if shortcut.shape[-1] != filters:
        s = ln_relu(shortcut)
        shortcut = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False)(s)

    out = tf.keras.layers.Add()([shortcut, y])
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
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss,
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
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(optimizer="adam",
                  loss=loss,
                  metrics=["accuracy"])
    return model


def model_c(log_priors=None):
    inputs = tf.keras.Input((config.IMG_SIZE, config.IMG_SIZE, 3))

    # stem
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = residual_sep_block_ln(x, 32, downsample=True,  sd_rate=0.05)
    x = residual_sep_block_ln(x, 64, downsample=True,  sd_rate=0.10)

    # head
    x = ln_relu(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(96, activation="relu")(x)

    # probabilities
    bias_init = None
    if log_priors is not None:
        bias_init = tf.keras.initializers.Constant(log_priors.tolist())
    outputs = tf.keras.layers.Dense(
        len(config.CLASSES),
        activation="softmax",
        bias_initializer=bias_init,
        name="probs"
    )(x)

    model = tf.keras.Model(inputs, outputs)
    opt  = tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return model


def model_d(log_priors=None):
    inputs = tf.keras.Input((config.IMG_SIZE, config.IMG_SIZE, 3))

    # stem
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = residual_sep_block_ln(x, 32, downsample=True,  sd_rate=0.05)
    x = residual_sep_block_ln(x, 64, downsample=True,  sd_rate=0.10)
    x = residual_sep_block_ln(x, 64, downsample=False, sd_rate=0.10)

    # head
    x = ln_relu(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    # peobabilities
    bias_init = None
    if log_priors is not None:
        bias_init = tf.keras.initializers.Constant(log_priors.tolist())
    outputs = tf.keras.layers.Dense(
        len(config.CLASSES),
        activation="softmax",
        bias_initializer=bias_init,
        name="probs"
    )(x)

    model = tf.keras.Model(inputs, outputs)
    opt  = tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return model
