import tensorflow as tf
from rockpaperscissors import config

def conv_ln_act(x, filters, k=3, s=1, act=True):
    """
    Applies a Conv2D → Layer Normalization → ReLU block.
    
    The block is designed for stable training with small batch sizes and is used as a
    building component in the deeper architecture (model c).

    Args:
        x as tf.Tensor: input tensor of shape (batch, height, width, channels).
        filters (int): Number of output channels.
        k (int, optional): Convolution kernel size. Defaults to 3.
        s (int, optional): Convolution stride. Defaults to 1.
        act (bool): Whether to apply ReLU activation.

    Returns:
        tf.Tensor: Output tensor after convolution, normalization,
        and optional activation.

    Notes:
        - The convolution is created with use_bias=False since layer
          normalization makes an explicit bias term unnecessary.
        - Layer normalization is applied along the channel axis (axis=-1).
    """
    x = tf.keras.layers.Conv2D(filters, k, strides=s, padding="same", use_bias=False,
                               kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)(x)
    if act:
        x = tf.keras.layers.ReLU()(x)
    return x

def dws_ln_act(x, filters, s=1):
    """
    Applies a depthwise separable convolution block with layer normalization.

    The block consists of a depthwise 3×3 convolution followed by layer
    normalization and ReLU activation, and a subsequent pointwise (1×1)
    convolution to project features to the desired number of channels.
    This design significantly reduces parameter count and computation
    compared to a full 3×3 convolution.

    Args:
        x (tf.Tensor): Input tensor of shape (batch, height, width, channels).
        filters (int): Number of output channels after the pointwise convolution.
        s (int, optional): Stride for the depthwise convolution. A stride
            greater than 1 performs spatial downsampling. Defaults to 1.

    Returns:
        tf.Tensor: Output tensor after depthwise and pointwise convolutions,
        normalization, and activation.
    """
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
    """
    Applies a residual depthwise separable convolution block with layer normalization.

    This helper wraps a depthwise separable convolution block and adds a
    residual (skip) connection when input and output tensors have matching
    spatial resolution and channel dimensionality. The residual connection
    improves gradient flow and training stability without introducing
    additional parameters.

    Args:
        x (tf.Tensor): Input tensor of shape (batch, height, width, channels).
        filters (int): Number of output channels.
        s (int, optional): Stride for the depthwise convolution. Defaults to 1.

    Returns:
        tf.Tensor: Output tensor after the residual depthwise separable block.

    Notes:
        - The residual connection is applied only if:
            * s == 1, and
            * the number of input channels equals `filters`.
        - No projection layer is used when dimensions do not match, in order
          to keep the block lightweight.
    """
    y = dws_ln_act(x, filters, s=s)
    if s == 1 and x.shape[-1] == filters:
        y = tf.keras.layers.Add()([x, y])
    return y

# ------------------ MODELS --------------------- #
def model_a():
    """
    Builds Model A, a lightweight baseline CNN using separable convolutions.

    Model A relies on Keras `SeparableConv2D` layers to reduce parameter count
    and computational cost, combined with max pooling for spatial
    downsampling and global average pooling to limit overfitting.

    The model is intended to assess how far a low-capacity architecture
    with strong inductive bias can perform on the classification task.

    Architecture overview:
        - Input layer with fixed image size.
        - Three SeparableConv2D blocks with increasing channel depth.
        - Max pooling for spatial downsampling.
        - Global average pooling instead of flattening.
        - Small fully connected head with dropout.
        - Softmax output over the target classes.

    Compilation:
        - Optimizer: Adam with learning rate 1e-3.
        - Loss: Categorical cross-entropy with label smoothing.
        - Metric: Categorical accuracy.

    Returns:
        tf.keras.Model: Compiled Keras model ready for training.
    """
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
    """
    Builds Model B, a simple convolutional neural network baseline.

    Model B implements a conventional CNN architecture composed of
    standard Conv2D and MaxPooling layers followed by a fully connected
    classification head. Unlike Model A, this model uses flattening
    before the dense layers, resulting in a higher parameter count for
    comparable input resolutions.

    Architecture overview:
        - Input layer with fixed image size.
        - Two Conv2D + MaxPooling blocks with increasing filters.
        - Flattening of spatial feature maps.
        - Dense hidden layer.
        - Softmax output layer.

    Compilation:
        - Optimizer: Adam (default parameters).
        - Loss: Categorical cross-entropy with label smoothing.
        - Metric: Categorical accuracy.

    Returns:
        tf.keras.Model: Compiled Keras model ready for training.
    """
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
    """
    Builds Model C, a deeper CNN composed of modular convolutional blocks 
    based on explicit depthwise separable convolutions, layer normalization, 
    and conditional residual connections. The architecture follows a staged
    design with progressive spatial downsampling and increasing channel depth.

    An optional bias initialization based on log-prior class probabilities
    can be applied to the final classification layer to improve training
    stability in the presence of class imbalance.

    Architecture overview:
        - Initial strided convolutional stem.
        - Multiple stages of depthwise separable residual blocks.
        - Progressive spatial downsampling via strided depthwise convolutions.
        - Bottleneck 1×1 convolution.
        - Global average pooling.
        - Fully connected classification head with dropout.
        - Softmax output layer.

    Args:
        log_priors (np.ndarray or tf.Tensor, optional): Logarithm of class
            prior probabilities, used to initialize the bias of the final
            Dense layer. If None, the bias is initialized to zeros.
            Defaults to None.

    Compilation:
        - Optimizer: Adam with reduced learning rate (3e-4).
        - Loss: Categorical cross-entropy with label smoothing.
        - Metric: Categorical accuracy.

    Returns:
        tf.keras.Model: Compiled Keras model ready for training.
    """
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
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                  loss=loss, metrics=["accuracy"])
    return model