import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv2D, Add, Concatenate, ReLU, Lambda, Multiply
from tensorflow.keras.models import Model

# Custom Dense Layer
class DenseLayer(Layer):
    def __init__(self, growth_rate, **kwargs):
        super(DenseLayer, self).__init__(**kwargs)
        self.conv = Conv2D(growth_rate, kernel_size=3, padding='same')
        self.relu = ReLU()

    def call(self, x):
        y = self.conv(x)
        y = self.relu(y)
        return Concatenate()([x, y])

    def get_config(self):
        config = super(DenseLayer, self).get_config()
        config.update({
            'growth_rate': self.conv.filters,
            'kernel_size': self.conv.kernel_size,
            'padding': self.conv.padding
        })
        return config

# Custom Residual Dense Block
class ResidualDenseBlock(Layer):
    def __init__(self, input_channels, growth_rate, num_layers, **kwargs):
        super(ResidualDenseBlock, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.input_channels = input_channels

        # Initialize Dense Layers
        self.dense_layers = [DenseLayer(growth_rate) for _ in range(num_layers)]

        # Local Feature Fusion
        self.lff = Conv2D(input_channels, kernel_size=1, padding='same')
        self.add = Add()

    def call(self, x):
        inputs = x
        for layer in self.dense_layers:
            x = layer(x)
        x = self.lff(x)
        return self.add([inputs, x])

    def get_config(self):
        config = super(ResidualDenseBlock, self).get_config()
        config.update({
            'input_channels': self.input_channels,
            'growth_rate': self.growth_rate,
            'num_layers': self.num_layers
        })
        return config

# Modified RDN with Optional Up-Sampling for Denoising
def build_rdn(scale_factor, input_shape=(None, None, 1), num_features=64, growth_rate=64, num_blocks=6, num_layers=6):
    """
    Builds a Residual Dense Network (RDN) model.

    Parameters:
    - scale_factor: Integer. If 1, the model performs denoising or enhancement without up-sampling.
                    If 2, 3, or 4, the model performs super-resolution with the specified scaling.
    - input_shape: Tuple. Shape of the input image (height, width, channels).
    - num_features: Integer. Number of feature maps in convolutional layers.
    - growth_rate: Integer. Growth rate for DenseLayers.
    - num_blocks: Integer. Number of Residual Dense Blocks.
    - num_layers: Integer. Number of DenseLayers within each Residual Dense Block.

    Returns:
    - model: Keras Model instance.
    """

    # Scale factor can be set to 1 hence no upscaling will occur
    if scale_factor not in [1, 2, 3, 4]:
        raise ValueError("scale_factor must be 1, 2, 3, or 4")

    inputs = Input(shape=input_shape)

    # Shallow Feature Extraction
    sfe1 = Conv2D(num_features, kernel_size=3, padding='same', activation='relu')(inputs)
    sfe2 = Conv2D(num_features, kernel_size=3, padding='same', activation='relu')(sfe1)

    # Residual Dense Blocks
    x = sfe2
    rdb_outputs = []
    for i in range(num_blocks):
        if i == 0:
            # First RDB with input_channels = num_features
            rdb = ResidualDenseBlock(input_channels=num_features, growth_rate=growth_rate, num_layers=num_layers)
        else:
            # Subsequent RDBs with input_channels = num_features (consistent feature size)
            rdb = ResidualDenseBlock(input_channels=num_features, growth_rate=growth_rate, num_layers=num_layers)
        x = rdb(x)
        rdb_outputs.append(x)

    # Global Feature Fusion
    x = Concatenate()(rdb_outputs)
    x = Conv2D(num_features, kernel_size=1, padding='same', activation='relu')(x)
    x = Conv2D(num_features, kernel_size=3, padding='same', activation='relu')(x)
    x = Add()([x, sfe1])  # Global Residual Learning

    # Up-sampling or Identity Mapping for Denoising
    if scale_factor > 1:
        if scale_factor in [2, 4]:
            for _ in range(scale_factor // 2):
                x = Conv2D(num_features * 4, kernel_size=3, padding='same', activation='relu')(x)
                x = Lambda(lambda z: tf.nn.depth_to_space(z, block_size=2))(x)
        elif scale_factor == 3:
            x = Conv2D(num_features * 9, kernel_size=3, padding='same', activation='relu')(x)
            x = Lambda(lambda z: tf.nn.depth_to_space(z, block_size=3))(x)
        # Final output after up-sampling
        outputs = Conv2D(input_shape[-1], kernel_size=3, padding='same', activation='sigmoid')(x)
    else:
        # For scale_factor = 1, skip up-sampling
        outputs = Conv2D(input_shape[-1], kernel_size=3, padding='same', activation='sigmoid')(x)

    return Model(inputs, outputs)
