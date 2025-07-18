"""
Model Compilation Utilities

This module contains custom metrics and compilation functions for RDN models.
Includes PSNR and SSIM metrics for image quality evaluation.
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError as MAE


class PSNRMetric(tf.keras.metrics.Metric):
    """
    Peak Signal-to-Noise Ratio (PSNR) metric for image quality evaluation.
    
    PSNR measures the ratio between the maximum possible power of a signal
    and the power of corrupting noise. Higher values indicate better quality.
    
    Args:
        name (str): Name of the metric
        max_val (float): Maximum pixel value (default: 1.0 for normalized images)
    """
    
    def __init__(self, name="psnr", max_val=1.0, **kwargs):
        super(PSNRMetric, self).__init__(name=name, **kwargs)
        self.max_val = max_val
        self.psnr = self.add_weight(name="psnr", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the metric state with new predictions."""
        psnr = tf.image.psnr(y_true, y_pred, max_val=self.max_val)
        self.psnr.assign(tf.reduce_mean(psnr))

    def result(self):
        """Return the current metric value."""
        return self.psnr

    def reset_states(self):
        """Reset the metric state."""
        self.psnr.assign(0.0)

    def get_config(self):
        """Get metric configuration for serialization."""
        config = super(PSNRMetric, self).get_config()
        config.update({
            'max_val': self.max_val,
        })
        return config


class SSIMMetric(tf.keras.metrics.Metric):
    """
    Structural Similarity Index Measure (SSIM) metric.
    
    SSIM measures the structural similarity between two images,
    considering luminance, contrast, and structure. Values range from 0 to 1,
    with 1 indicating perfect similarity.
    
    Args:
        name (str): Name of the metric
        max_val (float): Maximum pixel value (default: 1.0 for normalized images)
    """
    
    def __init__(self, name="ssim", max_val=1.0, **kwargs):
        super(SSIMMetric, self).__init__(name=name, **kwargs)
        self.max_val = max_val
        self.ssim = self.add_weight(name="ssim", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the metric state with new predictions."""
        ssim = tf.image.ssim(y_true, y_pred, max_val=self.max_val)
        self.ssim.assign(tf.reduce_mean(ssim))

    def result(self):
        """Return the current metric value."""
        return self.ssim

    def reset_states(self):
        """Reset the metric state."""
        self.ssim.assign(0.0)

    def get_config(self):
        """Get metric configuration for serialization."""
        config = super(SSIMMetric, self).get_config()
        config.update({
            'max_val': self.max_val,
        })
        return config


def compile_model(model, learning_rate=1e-4, loss='mae', metrics=None, max_val=1.0):
    """
    Compile an RDN model with appropriate optimizer, loss, and metrics.
    
    Args:
        model (tf.keras.Model): The model to compile
        learning_rate (float): Learning rate for Adam optimizer
        loss (str or callable): Loss function ('mae', 'mse', or custom)
        metrics (list): Additional metrics to include
        max_val (float): Maximum pixel value for PSNR/SSIM calculation
        
    Returns:
        tf.keras.Model: The compiled model
    """
    
    # Set up optimizer
    optimizer = Adam(learning_rate=learning_rate)
    
    # Set up loss function
    if loss == 'mae':
        loss_fn = MeanAbsoluteError()
    elif loss == 'mse':
        loss_fn = MeanSquaredError()
    else:
        loss_fn = loss  # Custom loss function
    
    # Set up default metrics
    default_metrics = [
        PSNRMetric(max_val=max_val),
        SSIMMetric(max_val=max_val),
        MAE(name="mae"),
        MeanSquaredError(name="mse")
    ]
    
    # Add any additional metrics
    if metrics:
        default_metrics.extend(metrics)
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=default_metrics
    )
    
    return model


def get_callbacks(checkpoint_path=None, tensorboard_dir=None, early_stopping=True,
                 patience=10, monitor='val_psnr', mode='max'):
    """
    Get a list of useful callbacks for training.
    
    Args:
        checkpoint_path (str): Path to save model checkpoints
        tensorboard_dir (str): Directory for TensorBoard logs
        early_stopping (bool): Whether to use early stopping
        patience (int): Patience for early stopping
        monitor (str): Metric to monitor for callbacks
        mode (str): 'min' or 'max' for the monitored metric
        
    Returns:
        list: List of Keras callbacks
    """
    
    callbacks = []
    
    # Model checkpoint callback
    if checkpoint_path:
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
    
    # TensorBoard callback
    if tensorboard_dir:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
    
    # Early stopping callback
    if early_stopping:
        early_stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate reduction callback
    lr_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        mode=mode,
        factor=0.5,
        patience=patience//2,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(lr_reduce_callback)
    
    return callbacks


def create_lr_schedule(initial_lr=1e-4, decay_steps=200, decay_rate=0.5):
    """
    Create a learning rate schedule function.
    
    Args:
        initial_lr (float): Initial learning rate
        decay_steps (int): Number of steps between decay
        decay_rate (float): Decay rate
        
    Returns:
        callable: Learning rate schedule function
    """
    
    def lr_schedule(epoch):
        """Learning rate schedule function."""
        return initial_lr * (decay_rate ** (epoch // decay_steps))
    
    return lr_schedule
