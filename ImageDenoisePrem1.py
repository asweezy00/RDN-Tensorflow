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
        return self.add([inputs, x])  # Local Residual Learning

    def get_config(self):
        config = super(ResidualDenseBlock, self).get_config()
        config.update({
            'input_channels': self.input_channels,
            'growth_rate': self.growth_rate,
            'num_layers': self.num_layers
        })
        return config

# RDN for Denoising
def build_rdn_denoising(input_shape=(None, None, 1), num_features=64, growth_rate=64, num_blocks=6, num_layers=6):
    """
    Builds a Residual Dense Network (RDN) model for image denoising.

    Parameters:
    - input_shape: Tuple. Shape of the input image (height, width, channels).
    - num_features: Integer. Number of feature maps in convolutional layers.
    - growth_rate: Integer. Growth rate for DenseLayers.
    - num_blocks: Integer. Number of Residual Dense Blocks.
    - num_layers: Integer. Number of DenseLayers within each Residual Dense Block.

    Returns:
    - model: Keras Model instance.
    """

    inputs = Input(shape=input_shape)

    # Shallow Feature Extraction
    sfe1 = Conv2D(num_features, kernel_size=3, padding='same', activation='relu')(inputs)
    sfe2 = Conv2D(num_features, kernel_size=3, padding='same', activation='relu')(sfe1)

    # Residual Dense Blocks
    x = sfe2
    rdb_outputs = []
    for i in range(num_blocks):
        rdb = ResidualDenseBlock(input_channels=num_features, growth_rate=growth_rate, num_layers=num_layers)
        x = rdb(x)
        rdb_outputs.append(x)

    # Global Feature Fusion
    x = Concatenate()(rdb_outputs)
    x = Conv2D(num_features, kernel_size=1, padding='same', activation='relu')(x)
    x = Conv2D(num_features, kernel_size=3, padding='same', activation='relu')(x)

    # First Global Residual Learning Addition
    x = Add()([x, sfe1])  # Adding the shallow feature map to fused features

    # Second Element-Wise Addition (Final Residual Learning)
    outputs = Add()([x, sfe2])  # Adding shallow feature map again for denoising

    # Final Reconstruction
    outputs = Conv2D(input_shape[-1], kernel_size=3, padding='same', activation='sigmoid')(outputs)

    return Model(inputs, outputs)
