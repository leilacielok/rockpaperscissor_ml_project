import tensorflow as tf
from rockpaperscissors import config

def ln_relu(x, axis=(-1,)):
    x = tf.keras.layers.LayerNormalization(axis=axis)(x)
    return tf.keras.layers.ReLU()(x)

def sep_block_ln(x, filters, sd_rate=0.0):
    y = ln_relu(x)
    y = tf.keras.layers.SeparableConv2D(filters, 3, padding="same", use_bias=False, depthwise_initializer=tf.keras.initializers.HeNormal(), pointwise_initializer=tf.keras.initializers.HeNormal())(y)  # inizializzazione HeNormal dei pesi
    if sd_rate > 0:
        y = tf.keras.layers.SpatialDropout2D(sd_rate)(y)  # aggiunta di dropout spaziale per ridurre overfitting e collasso
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
        shortcut = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeNormal())(s)

    out = tf.keras.layers.Add()([shortcut, y])
    return out

## Helper functions and blocks (using LayerNorm, HeNormal init, etc.)
def mb_sep_block(x, filters, stride=1, sd_rate=0.0):
    """Depthwise separable conv block con LayerNorm e ReLU.
    (DepthwiseConv → LayerNorm → ReLU → PointwiseConv → LayerNorm → ReLU, con SpatialDropout2D opzionale)"""
    y = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False, depthwise_initializer=tf.keras.initializers.HeNormal())(x)  # inizializzazione HeNormal dei pesi
    y = tf.keras.layers.LayerNormalization(axis=-1)(y)  # LayerNorm al posto di BatchNorm
    y = tf.keras.layers.ReLU()(y)  # ReLU standard (invece di ReLU6)
    y = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeNormal())(y)  # inizializzazione HeNormal dei pesi
    y = tf.keras.layers.LayerNormalization(axis=-1)(y)  # LayerNorm al posto di BatchNorm
    y = tf.keras.layers.ReLU()(y)  # ReLU standard (invece di ReLU6)
    if sd_rate > 0:
        y = tf.keras.layers.SpatialDropout2D(sd_rate)(y)  # aggiunta di dropout spaziale per ridurre overfitting e collasso
    return y

def mb_res_block(x, filters, stride=1, sd_rate=0.0):
    """Residual con proiezione quando serve (stride!=1 o canali diversi). Usa LayerNorm invece di BatchNorm,
    supporta SpatialDropout2D per regolarizzazione."""
    y = mb_sep_block(x, filters, stride=stride, sd_rate=sd_rate)
    if stride != 1 or x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeNormal())(x)  # inizializzazione HeNormal dei pesi
        shortcut = tf.keras.layers.LayerNormalization(axis=-1)(shortcut)  # LayerNorm sul shortcut
    else:
        shortcut = x
    out = tf.keras.layers.Add()([shortcut, y])
    out = tf.keras.layers.ReLU()(out)  # ReLU standard (invece di ReLU6 per maggiore dinamica)
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

    x = tf.keras.layers.Conv2D(int(24 * width_mult), 3, strides=2, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeNormal())(inputs)  # inizializzazione HeNormal dei pesi
    x = tf.keras.layers.LayerNormalization(axis=-1)(x)  # LayerNorm al posto di BatchNorm
    x = tf.keras.layers.ReLU()(x)  # ReLU standard (invece di ReLU6 per maggiore dinamica)

    # blocchi MobileNet v1 "plain" (niente residuo)
    for f, s in [(24, 1), (32, 2), (48, 1), (64, 2), (64, 1), (96, 1)]:
        x = mb_sep_block(x, int(f * width_mult), stride=s, sd_rate=0.1)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(x)  # inizializzazione HeNormal dei pesi
    x = tf.keras.layers.Dropout(0.2)(x)

    bias_init = (tf.keras.initializers.Constant(log_priors) if log_priors is not None else "zeros")
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", bias_initializer=bias_init, kernel_initializer=tf.keras.initializers.HeNormal())(x)  # inizializzazione HeNormal dei pesi

    model = tf.keras.Model(inputs, outputs, name="model_d")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),  # learning rate ridotto a 3e-4
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),  # abilitato label smoothing 5%
        metrics=["accuracy"],
    )
    return model

