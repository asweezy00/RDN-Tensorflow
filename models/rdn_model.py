"""
Residual Dense Network (RDN) Model Implementation

This module contains the core RDN architecture components including:
- DenseLayer: Dense connected convolutional layer
- ResidualDenseBlock: Building block with local feature fusion and residual learning
- build_rdn: Generic RDN for both denoising and super-resolution
- build_rdn_denoising: Specialized RDN for denoising tasks

Based on the paper:
"Residual Dense Network for Image Super-Resolution" (CVPR 2018)
by Zhang et al.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv2D, Add, Concatenate, ReLU, Lambda
from tensorflow.keras.models import Model


class DenseLayer(Layer):
    """
    Dense Layer with growth rate for feature expansion.
    
    Implements dense connectivity where each layer receives inputs from
    all preceding layers within the same dense block.
    
    Args:
        growth_rate (int): Number of feature maps to add
    """
    
    def __init__(self, growth_rate, **kwargs):
        super(DenseLayer, self).__init__(**kwargs)
        self.growth_rate = growth_rate
        self.conv = Conv2D(growth_rate, kernel_size=3, padding='same')
        self.relu = ReLU()

    def call(self, x):
        """Forward pass with dense connectivity."""
        y = self.conv(x)
        y = self.relu(y)
        return Concatenate()([x, y])

    def get_config(self):
        """Get layer configuration for serialization."""
        config = super(DenseLayer, self).get_config()
        config.update({
            'growth_rate': self.growth_rate,
        })
        return config


class ResidualDenseBlock(Layer):
    """
    Residual Dense Block (RDB) - Core building block of RDN.
    
    Features:
    - Dense connectivity within the block
    - Local feature fusion (LFF) with 1x1 convolution
    - Local residual learning (LRL)
    - Contiguous memory mechanism
    
    Args:
        input_channels (int): Number of input feature channels
        growth_rate (int): Growth rate for dense layers
        num_layers (int): Number of dense layers in the block
    """
    
    def __init__(self, input_channels, growth_rate, num_layers, **kwargs):
        super(ResidualDenseBlock, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.growth_rate = growth_rate
        self.num_layers = num_layers

        # Initialize Dense Layers
        self.dense_layers = [DenseLayer(growth_rate) for _ in range(num_layers)]

        # Local Feature Fusion (LFF)
        self.lff = Conv2D(input_channels, kernel_size=1, padding='same')
        
        # Local Residual Learning (LRL)
        self.add = Add()

    def call(self, x):
        """
        Forward pass implementing:
        1. Dense connectivity through multiple dense layers
        2. Local feature fusion
        3. Local residual learning
        """
        inputs = x
        
        # Pass through dense layers with concatenation
        for layer in self.dense_layers:
            x = layer(x)
        
        # Local feature fusion
        x = self.lff(x)
        
        # Local residual learning
        return self.add([inputs, x])

    def get_config(self):
        """Get layer configuration for serialization."""
        config = super(ResidualDenseBlock, self).get_config()
        config.update({
            'input_channels': self.input_channels,
            'growth_rate': self.growth_rate,
            'num_layers': self.num_layers
        })
        return config


def build_rdn(scale_factor, input_shape=(None, None, 1), num_features=64, 
              growth_rate=64, num_blocks=6, num_layers=6):
    """
    Build a Residual Dense Network (RDN) model.
    
    This is the generic RDN that can handle both denoising (scale_factor=1)
    and super-resolution (scale_factor=2,3,4) tasks.
    
    Architecture:
    1. Shallow Feature Extraction (2 Conv layers)
    2. Residual Dense Blocks with contiguous memory
    3. Global Feature Fusion
    4. Global Residual Learning
    5. Upsampling (if scale_factor > 1)
    
    Args:
        scale_factor (int): Scaling factor (1 for denoising, 2/3/4 for SR)
        input_shape (tuple): Input image shape (height, width, channels)
        num_features (int): Number of feature maps in conv layers
        growth_rate (int): Growth rate for dense layers
        num_blocks (int): Number of residual dense blocks
        num_layers (int): Number of dense layers per RDB
        
    Returns:
        tf.keras.Model: Compiled RDN model
        
    Raises:
        ValueError: If scale_factor is not in [1, 2, 3, 4]
    """
    
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
        rdb = ResidualDenseBlock(
            input_channels=num_features, 
            growth_rate=growth_rate, 
            num_layers=num_layers
        )
        x = rdb(x)
        rdb_outputs.append(x)

    # Global Feature Fusion (GFF)
    x = Concatenate()(rdb_outputs)
    x = Conv2D(num_features, kernel_size=1, padding='same', activation='relu')(x)
    x = Conv2D(num_features, kernel_size=3, padding='same', activation='relu')(x)
    
    # Global Residual Learning (GRL)
    x = Add()([x, sfe1])

    # Upsampling or Identity Mapping
    if scale_factor > 1:
        # Sub-pixel convolution for upsampling
        if scale_factor in [2, 4]:
            # For 2x and 4x, use iterative 2x upsampling
            for _ in range(scale_factor // 2):
                x = Conv2D(num_features * 4, kernel_size=3, padding='same', activation='relu')(x)
                x = Lambda(lambda z: tf.nn.depth_to_space(z, block_size=2))(x)
        elif scale_factor == 3:
            # For 3x, use single 3x upsampling
            x = Conv2D(num_features * 9, kernel_size=3, padding='same', activation='relu')(x)
            x = Lambda(lambda z: tf.nn.depth_to_space(z, block_size=3))(x)
        
        # Final output layer after upsampling
        outputs = Conv2D(input_shape[-1], kernel_size=3, padding='same', activation='sigmoid')(x)
    else:
        # For scale_factor = 1 (denoising), skip upsampling
        outputs = Conv2D(input_shape[-1], kernel_size=3, padding='same', activation='sigmoid')(x)

    return Model(inputs, outputs)


def build_rdn_denoising(input_shape=(None, None, 1), num_features=64, 
                       growth_rate=64, num_blocks=6, num_layers=6):
    """
    Build a specialized RDN model for image denoising.
    
    This version is optimized specifically for denoising tasks with
    enhanced residual connections for better noise removal.
    
    Architecture:
    1. Shallow Feature Extraction (2 Conv layers)
    2. Residual Dense Blocks
    3. Global Feature Fusion
    4. Dual Global Residual Learning (enhanced for denoising)
    5. Final reconstruction layer
    
    Args:
        input_shape (tuple): Input image shape (height, width, channels)
        num_features (int): Number of feature maps in conv layers
        growth_rate (int): Growth rate for dense layers
        num_blocks (int): Number of residual dense blocks
        num_layers (int): Number of dense layers per RDB
        
    Returns:
        tf.keras.Model: Compiled RDN denoising model
    """
    
    inputs = Input(shape=input_shape)

    # Shallow Feature Extraction
    sfe1 = Conv2D(num_features, kernel_size=3, padding='same', activation='relu')(inputs)
    sfe2 = Conv2D(num_features, kernel_size=3, padding='same', activation='relu')(sfe1)

    # Residual Dense Blocks
    x = sfe2
    rdb_outputs = []
    
    for i in range(num_blocks):
        rdb = ResidualDenseBlock(
            input_channels=num_features, 
            growth_rate=growth_rate, 
            num_layers=num_layers
        )
        x = rdb(x)
        rdb_outputs.append(x)

    # Global Feature Fusion
    x = Concatenate()(rdb_outputs)
    x = Conv2D(num_features, kernel_size=1, padding='same', activation='relu')(x)
    x = Conv2D(num_features, kernel_size=3, padding='same', activation='relu')(x)

    # First Global Residual Learning Addition
    x = Add()([x, sfe1])  # Adding shallow feature map to fused features

    # Second Element-Wise Addition (Enhanced for Denoising)
    x = Add()([x, sfe2])  # Adding second shallow feature map for better denoising

    # Final Reconstruction
    outputs = Conv2D(input_shape[-1], kernel_size=3, padding='same', activation='sigmoid')(x)

    return Model(inputs, outputs)
