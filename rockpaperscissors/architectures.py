import tensorflow as tf
from rockpaperscissors import config

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

## light helpers: BN + ReLU6 + SeparableConv
def bn_relu(x):
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU(max_value=6.0)(x)

def sep_block_bn(x, filters, stride=1, sd_rate=0.0):
    y = bn_relu(x)
    y = tf.keras.layers.DepthwiseConv2D(
        3, strides=stride, padding="same", use_bias=False
    )(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU(max_value=6.0)(y)
    y = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False)(y)
    if sd_rate > 0:
        y = tf.keras.layers.SpatialDropout2D(sd_rate)(y)
    return y

def residual_sep_block_fast(x, filters, downsample=False, sd_rate=0.0):
    stride = 2 if downsample else 1
    y = sep_block_bn(x, filters, stride=stride, sd_rate=sd_rate)
    shortcut = x
    if downsample or x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, 1, strides=stride, padding="same", use_bias=False
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    return tf.keras.layers.Add()([shortcut, y])

# ------------------ MODELS --------------------- #
def model_a():
    """Very small, fast CNN (good baseline)."""
    inputs = tf.keras.Input((config.IMG_SIZE, config.IMG_SIZE, 3))

    x = tf.keras.layers.SeparableConv2D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.SeparableConv2D(24, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.SeparableConv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)

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
    img_size    = getattr(config, "IMG_SIZE", 96)
    input_shape = getattr(config, "IMG_SHAPE", (img_size, img_size, 3))
    n_classes   = len(getattr(config, "CLASSES", ["rock", "paper", "scissors"]))

    width_mult = 1.0
    inputs = tf.keras.Input(shape=input_shape)

    # Normalization + Augment
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
    ], name="aug")
    x = aug(x)

    # stem
    x = tf.keras.layers.Conv2D(int(16 * width_mult), 3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)

    # 3 steps
    for filters, down in [(24, False), (32, True), (64, True)]:
        f = int(filters * width_mult)
        x = residual_sep_block_fast(x, f, downsample=down, sd_rate=0.0)
        x = residual_sep_block_fast(x, f, downsample=False, sd_rate=0.0)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="model_c")
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0),
                  metrics=["accuracy"])
    return model





def model_d(log_priors=None):
    img_size    = getattr(config, "IMG_SIZE", 96)
    input_shape = getattr(config, "IMG_SHAPE", (img_size, img_size, 3))
    n_classes   = len(getattr(config, "CLASSES", ["rock", "paper", "scissors"]))

    width_mult = 1.0
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Rescaling(1./255)(inputs)
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
    ], name="aug")
    x = aug(x)

    x = tf.keras.layers.Conv2D(int(24 * width_mult), 3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)

    for filters, stride in [
        (int(24 * width_mult), 1),
        (int(32 * width_mult), 2),
        (int(48 * width_mult), 1),
        (int(64 * width_mult), 2),
        (int(64 * width_mult), 1),
    ]:
        x = sep_block_bn(x, filters, stride=stride, sd_rate=0.0)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="model_d")
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0),
                  metrics=["accuracy"])
    return model



