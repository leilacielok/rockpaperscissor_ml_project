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

def mb_sep_block(x, filters, stride=1):
    """Depthwise separable conv (MobileNet v1): DW→BN→ReLU6→PW→BN→ReLU6."""
    y = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU(max_value=6.0)(y)

    y = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU(max_value=6.0)(y)
    return y

def mb_res_block(x, filters, stride=1):
    """Residual con proiezione quando serve (stride!=1 o canali diversi)."""
    y = mb_sep_block(x, filters, stride=stride)
    if stride != 1 or x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(x)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    else:
        shortcut = x
    out = tf.keras.layers.Add()([shortcut, y])
    out = tf.keras.layers.ReLU(max_value=6.0)(out)
    return out


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

    # stem
    x = tf.keras.layers.Conv2D(int(16 * width_mult), 3, strides=2, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)

    # stadi MobileNet v1 con residuo (2 blocchi per stadio)
    for f, stride in [(24, 1), (24, 1), (32, 2), (32, 1), (64, 2), (64, 1)]:
        x = mb_res_block(x, int(f * width_mult), stride=stride)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    bias_init = (tf.keras.initializers.Constant(log_priors) if log_priors is not None else "zeros")
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", bias_initializer=bias_init)(x)

    model = tf.keras.Model(inputs, outputs, name="model_c")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),  # LR un po' più alto aiuta con BN
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0),
        metrics=["accuracy"],
    )
    return model



def model_d(log_priors=None):
    img_size    = getattr(config, "IMG_SIZE", 96)
    input_shape = getattr(config, "IMG_SHAPE", (img_size, img_size, 3))
    n_classes   = len(getattr(config, "CLASSES", ["rock", "paper", "scissors"]))

    width_mult = 1.0
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(int(24 * width_mult), 3, strides=2, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)

    # blocchi MobileNet v1 "plain" (niente residuo)
    for f, s in [(24, 1), (32, 2), (48, 1), (64, 2), (64, 1), (96, 1)]:
        x = mb_sep_block(x, int(f * width_mult), stride=s)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    bias_init = (tf.keras.initializers.Constant(log_priors) if log_priors is not None else "zeros")
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", bias_initializer=bias_init)(x)

    model = tf.keras.Model(inputs, outputs, name="model_d")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0),
        metrics=["accuracy"],
    )
    return model



