"""
RDN Trainer Class

This module provides the core training functionality for RDN models,
including training loop, callbacks, and model management.
"""

import os
import time
import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_rdn, build_rdn_denoising, compile_model
from utils.config import Config
from utils.image_utils import calculate_psnr, calculate_ssim
from .data_loader import create_dataset


class RDNTrainer:
    """
    RDN Trainer class for handling model training and evaluation.
    
    This class provides a complete training pipeline including:
    - Model building and compilation
    - Data loading and preprocessing
    - Training loop with callbacks
    - Model checkpointing and resuming
    - Validation and metrics tracking
    """
    
    def __init__(self, config: Config):
        """
        Initialize RDN trainer.
        
        Args:
            config: Configuration object containing all training parameters
        """
        self.config = config
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        
        # Create output directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.log_dir = Path(config.log_dir)
        self.output_dir = Path(config.output_dir)
        
        for directory in [self.checkpoint_dir, self.log_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_psnr = 0.0
        self.training_history = {
            'loss': [], 'val_loss': [],
            'psnr': [], 'val_psnr': [],
            'ssim': [], 'val_ssim': []
        }
        
        print(f"RDN Trainer initialized for {config.model.model_type}")
        print(f"Output directories created: {self.checkpoint_dir}, {self.log_dir}, {self.output_dir}")
    
    def build_model(self) -> tf.keras.Model:
        """
        Build and compile the RDN model based on configuration.
        
        Returns:
            Compiled TensorFlow model
        """
        print("Building RDN model...")
        
        if self.config.model.model_type == "rdn_denoising":
            model = build_rdn_denoising(
                input_shape=self.config.model.input_shape,
                num_features=self.config.model.num_features,
                growth_rate=self.config.model.growth_rate,
                num_blocks=self.config.model.num_blocks,
                num_layers=self.config.model.num_layers
            )
        else:
            model = build_rdn(
                scale_factor=self.config.model.scale_factor,
                input_shape=self.config.model.input_shape,
                num_features=self.config.model.num_features,
                growth_rate=self.config.model.growth_rate,
                num_blocks=self.config.model.num_blocks,
                num_layers=self.config.model.num_layers
            )
        
        # Compile model
        model = compile_model(
            model,
            learning_rate=self.config.training.learning_rate,
            loss=self.config.training.loss,
            max_val=self.config.training.max_val
        )
        
        self.model = model
        
        print(f"✓ Model built successfully!")
        print(f"  - Input shape: {model.input_shape}")
        print(f"  - Output shape: {model.output_shape}")
        print(f"  - Total parameters: {model.count_params():,}")
        
        return model
    
    def load_data(self, data_path: str, val_data_path: Optional[str] = None):
        """
        Load training and validation datasets.
        
        Args:
            data_path: Path to training data directory
            val_data_path: Path to validation data directory (optional)
        """
        print("Loading datasets...")
        
        # Load training dataset
        self.train_dataset = create_dataset(self.config, data_path, mode='train')
        print(f"✓ Training dataset loaded: {len(self.train_dataset)} images")
        
        # Load validation dataset
        if val_data_path:
            self.val_dataset = create_dataset(self.config, val_data_path, mode='val')
        else:
            # Use a subset of training data for validation
            val_split = self.config.training.validation_split
            if val_split > 0:
                total_images = len(self.train_dataset.image_files)
                val_size = int(total_images * val_split)
                
                # Split image files
                val_files = self.train_dataset.image_files[-val_size:]
                train_files = self.train_dataset.image_files[:-val_size]
                
                # Update training dataset
                self.train_dataset.image_files = train_files
                
                # Create validation dataset with same data path but different mode
                # We'll manually set the image files after creation
                try:
                    self.val_dataset = create_dataset(self.config, data_path, mode='train')
                    self.val_dataset.image_files = val_files
                    self.val_dataset.mode = 'val'  # Set mode to val for proper behavior
                except Exception:
                    # If that fails, create without validation
                    print("! Could not create validation dataset, training without validation")
                    self.val_dataset = None
        
        if self.val_dataset:
            print(f"✓ Validation dataset loaded: {len(self.val_dataset)} images")
        else:
            print("! No validation dataset specified")
    
    def create_callbacks(self) -> list:
        """
        Create training callbacks.
        
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Model checkpointing
        checkpoint_path = self.checkpoint_dir / f"{self.config.project_name}_best.h5"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=self.config.training.monitor,
            mode=self.config.training.mode,
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if self.config.training.early_stopping:
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor=self.config.training.monitor,
                patience=self.config.training.patience,
                mode=self.config.training.mode,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping_callback)
        
        # Learning rate scheduling
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=self.config.training.monitor,
            factor=0.5,
            patience=self.config.training.patience // 2,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # TensorBoard logging
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=str(self.log_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
        
        # Custom callback for saving training history
        history_callback = TrainingHistoryCallback(self)
        callbacks.append(history_callback)
        
        return callbacks
    
    def train(self, data_path: str, val_data_path: Optional[str] = None, 
              resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the RDN model.
        
        Args:
            data_path: Path to training data directory
            val_data_path: Path to validation data directory (optional)
            resume_from: Path to checkpoint to resume from (optional)
            
        Returns:
            Training history dictionary
        """
        print("=" * 60)
        print(f"Starting RDN Training - {self.config.project_name}")
        print("=" * 60)
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Load datasets
        self.load_data(data_path, val_data_path)
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Prepare data generators
        print("Preparing data generators...")
        
        # Get training patches
        train_patches = self.train_dataset.get_patches(
            num_patches=self.config.training.batch_size * 100
        )
        
        if self.config.model.model_type == "rdn_denoising":
            train_x, train_y = train_patches  # (noisy, clean)
        else:
            train_x, train_y = train_patches  # (LR, HR)
        
        print(f"✓ Training patches prepared: {len(train_x)} patches")
        
        # Prepare validation data if available
        val_data = None
        if self.val_dataset:
            val_patches = self.val_dataset.get_patches(
                num_patches=self.config.training.batch_size * 20
            )
            if self.config.model.model_type == "rdn_denoising":
                val_x, val_y = val_patches  # (noisy, clean)
            else:
                val_x, val_y = val_patches  # (LR, HR)
            
            val_data = (val_x, val_y)
            print(f"✓ Validation patches prepared: {len(val_x)} patches")
        
        # Start training
        print("\nStarting training...")
        start_time = time.time()
        
        try:
            history = self.model.fit(
                x=train_x,
                y=train_y,
                batch_size=self.config.training.batch_size,
                epochs=self.config.training.epochs,
                validation_data=val_data,
                callbacks=callbacks,
                verbose=1,
                initial_epoch=self.current_epoch
            )
            
            training_time = time.time() - start_time
            
            print("\n" + "=" * 60)
            print("Training completed successfully!")
            print(f"Total training time: {training_time:.2f} seconds")
            print(f"Best validation PSNR: {self.best_val_psnr:.4f} dB")
            print("=" * 60)
            
            return history.history
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self.save_checkpoint(f"interrupted_epoch_{self.current_epoch}")
            return self.training_history
        
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            self.save_checkpoint(f"failed_epoch_{self.current_epoch}")
            raise e
    
    def evaluate(self, test_data_path: str) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.
        
        Args:
            test_data_path: Path to test data directory
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print("Evaluating model on test data...")
        
        # Load test dataset
        test_dataset = create_dataset(self.config, test_data_path, mode='test')
        
        # Get test patches
        test_patches = test_dataset.get_patches(num_patches=len(test_dataset) * 5)
        
        if self.config.model.model_type == "rdn_denoising":
            test_x, test_y = test_patches  # (noisy, clean)
        else:
            test_x, test_y = test_patches  # (LR, HR)
        
        # Predict
        predictions = self.model.predict(test_x, batch_size=self.config.training.batch_size)
        
        # Calculate metrics
        psnr_values = []
        ssim_values = []
        
        for i in range(len(predictions)):
            pred = predictions[i]
            target = test_y[i]
            
            psnr = calculate_psnr(target, pred, max_val=self.config.training.max_val)
            ssim = calculate_ssim(target, pred, max_val=self.config.training.max_val)
            
            psnr_values.append(psnr)
            ssim_values.append(ssim)
        
        metrics = {
            'mean_psnr': np.mean(psnr_values),
            'std_psnr': np.std(psnr_values),
            'mean_ssim': np.mean(ssim_values),
            'std_ssim': np.std(ssim_values),
            'num_samples': len(predictions)
        }
        
        print(f"Evaluation Results:")
        print(f"  Mean PSNR: {metrics['mean_psnr']:.4f} ± {metrics['std_psnr']:.4f} dB")
        print(f"  Mean SSIM: {metrics['mean_ssim']:.4f} ± {metrics['std_ssim']:.4f}")
        print(f"  Number of samples: {metrics['num_samples']}")
        
        return metrics
    
    def save_checkpoint(self, checkpoint_name: str):
        """
        Save model checkpoint.
        
        Args:
            checkpoint_name: Name for the checkpoint file
        """
        if self.model is None:
            print("No model to save")
            return
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.h5"
        self.model.save(str(checkpoint_path))
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.model = tf.keras.models.load_model(checkpoint_path)
        print(f"Model loaded from checkpoint: {checkpoint_path}")
    
    def predict_image(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Predict on a single image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            
        Returns:
            Processed image as numpy array
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        from utils.image_utils import load_image, save_image, normalize_image, denormalize_image
        
        # Load and preprocess image
        image = load_image(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Handle channel conversion for denoising
        if self.config.model.model_type == "rdn_denoising":
            channels = self.config.model.input_shape[-1]
            if channels == 1 and len(image.shape) == 3:
                image = np.mean(image, axis=2, keepdims=True)
            elif channels == 3 and len(image.shape) == 2:
                image = np.stack([image] * 3, axis=2)
        
        # Normalize
        image = normalize_image(image, method="0_1")
        
        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)
        
        # Predict
        prediction = self.model.predict(image_batch, verbose=0)
        
        # Remove batch dimension
        result = prediction[0]
        
        # Denormalize
        result = denormalize_image(result, method="0_1")
        
        # Save if output path provided
        if output_path:
            save_image(result, output_path)
            print(f"Result saved to: {output_path}")
        
        return result


class TrainingHistoryCallback(tf.keras.callbacks.Callback):
    """Custom callback to track training history."""
    
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
    
    def on_epoch_end(self, epoch, logs=None):
        """Update training history at the end of each epoch."""
        logs = logs or {}
        
        # Update current epoch
        self.trainer.current_epoch = epoch + 1
        
        # Update best validation PSNR
        val_psnr = logs.get('val_psnr', 0)
        if val_psnr > self.trainer.best_val_psnr:
            self.trainer.best_val_psnr = val_psnr
        
        # Update training history
        for key, value in logs.items():
            if key in self.trainer.training_history:
                self.trainer.training_history[key].append(value)
