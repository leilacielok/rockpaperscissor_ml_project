import tensorflow as tf
from rockpaperscissors import config

def conv_ln_act(x, filters, k=3, s=1, act=True):
    x = tf.keras.layers.Conv2D(filters, k, strides=s, padding="same", use_bias=False,
                               kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)(x)  # LN al posto di BN
    if act:
        x = tf.keras.layers.ReLU()(x)
    return x

def dws_ln_act(x, filters, s=1):
    # Depthwise 3x3 + LN + ReLU → Pointwise 1x1 + LN + ReLU
    y = tf.keras.layers.DepthwiseConv2D(3, strides=s, padding="same", use_bias=False,
                                        depthwise_initializer=tf.keras.initializers.HeNormal())(x)
    y = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False,
                               kernel_initializer=tf.keras.initializers.HeNormal())(y)
    y = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)(y)
    y = tf.keras.layers.ReLU()(y)
    return y

def dws_res_block_ln(x, filters, s=1):
    y = dws_ln_act(x, filters, s=s)
    if s == 1 and x.shape[-1] == filters:
        y = tf.keras.layers.Add()([x, y])
    return y

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

    inputs = tf.keras.Input(shape=input_shape)

    # Stem (↓/2 → 48x48)
    x = conv_ln_act(inputs, 24, k=3, s=2)

    # Stage 1 (48x48)
    x = dws_res_block_ln(x, 32, s=1)
    x = dws_res_block_ln(x, 32, s=1)

    # Downsample (↓/2 → 24x24)
    x = dws_ln_act(x, 48, s=2)
    x = dws_res_block_ln(x, 48, s=1)
    x = dws_res_block_ln(x, 48, s=1)

    # Downsample (↓/2 → 12x12)
    x = dws_ln_act(x, 64, s=2)
    x = dws_res_block_ln(x, 64, s=1)
    x = dws_res_block_ln(x, 64, s=1)
    x = dws_res_block_ln(x, 64, s=1)

    # Bottleneck
    x = conv_ln_act(x, 96, k=1, s=1)

    # Head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(96, activation="relu",
                              kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    bias_init = (tf.keras.initializers.Constant(log_priors) if log_priors is not None else "zeros")
    outputs = tf.keras.layers.Dense(
        n_classes, activation="softmax",
        bias_initializer=bias_init,
        kernel_initializer=tf.keras.initializers.HeNormal()
    )(x)

    model = tf.keras.Model(inputs, outputs, name="model_c")

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    # LR più basso per evitare “saturazione” precoce
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                  loss=loss, metrics=["accuracy"])
    return model