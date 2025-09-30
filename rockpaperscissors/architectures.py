import tensorflow as tf
from . import config

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

def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def model_c():
    inputs = tf.keras.Input((config.IMG_SIZE, config.IMG_SIZE, 3))
    x = conv_block(inputs, 32);  x = conv_block(x, 32);  x = tf.keras.layers.MaxPool2D()(x);  x = tf.keras.layers.Dropout(0.15)(x)
    x = conv_block(x, 64);       x = conv_block(x, 64);  x = tf.keras.layers.MaxPool2D()(x);  x = tf.keras.layers.Dropout(0.25)(x)
    x = conv_block(x, 96);       x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(len(config.CLASSES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def sep_block(x, filters):
    y = tf.keras.layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    return y

def model_d():
    inputs = tf.keras.Input((config.IMG_SIZE, config.IMG_SIZE, 3))
    x = sep_block(inputs, 32)
    x = sep_block(x, 32)
    x = tf.keras.layers.MaxPool2D()(x)

    y = sep_block(x, 64)
    y = sep_block(y, 64)
    x = tf.keras.layers.Add()([x, tf.keras.layers.Conv2D(64, 1, padding="same")(x)])  # skip 1x1
    x = tf.keras.layers.Add()([x, y])
    x = tf.keras.layers.MaxPool2D()(x)

    x = sep_block(x, 96)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(96, activation="relu")(x)
    outputs = tf.keras.layers.Dense(len(config.CLASSES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(8e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model
