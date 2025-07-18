"""
Data Loading and Dataset Classes for RDN Training

This module provides dataset classes and data loading utilities for both
super-resolution and denoising tasks.
"""

import os
import glob
import random
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Union
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_utils import (
    load_image, normalize_image, extract_patches, apply_degradation,
    resize_image, add_noise
)
from utils.config import Config


class BaseDataset:
    """Base dataset class with common functionality."""
    
    def __init__(self, config: Config, data_path: str, mode: str = 'train'):
        """
        Initialize base dataset.
        
        Args:
            config: Configuration object
            data_path: Path to dataset directory
            mode: Dataset mode ('train', 'val', 'test')
        """
        self.config = config
        self.data_path = Path(data_path)
        self.mode = mode
        self.patch_size = config.training.patch_size
        self.batch_size = config.training.batch_size
        
        # Image formats to search for
        self.image_formats = config.data.image_formats
        
        # Data augmentation settings
        self.horizontal_flip = config.training.horizontal_flip and mode == 'train'
        self.vertical_flip = config.training.vertical_flip and mode == 'train'
        self.rotation = config.training.rotation and mode == 'train'
        
        # Find image files
        self.image_files = self._find_image_files()
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {data_path}")
        
        print(f"Found {len(self.image_files)} images for {mode} mode")
    
    def _find_image_files(self) -> List[Path]:
        """Find all image files in the dataset directory."""
        image_files = []
        
        # Look for images in HR directory first, then root directory
        search_dirs = [
            self.data_path / self.mode / 'HR',
            self.data_path / self.mode,
            self.data_path / 'HR',
            self.data_path
        ]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                for ext in self.image_formats:
                    pattern = str(search_dir / f"*{ext}")
                    files = glob.glob(pattern, recursive=False)
                    image_files.extend([Path(f) for f in files])
                
                if image_files:  # If we found images, use this directory
                    break
        
        return sorted(list(set(image_files)))  # Remove duplicates and sort
    
    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation to image."""
        if not (self.horizontal_flip or self.vertical_flip or self.rotation):
            return image
        
        # Horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            image = np.fliplr(image)
        
        # Vertical flip
        if self.vertical_flip and random.random() > 0.5:
            image = np.flipud(image)
        
        # Rotation (90, 180, 270 degrees)
        if self.rotation and random.random() > 0.5:
            k = random.randint(1, 3)  # 1, 2, or 3 for 90, 180, 270 degrees
            image = np.rot90(image, k)
        
        return image
    
    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.image_files)


class SuperResolutionDataset(BaseDataset):
    """Dataset class for super-resolution training."""
    
    def __init__(self, config: Config, data_path: str, mode: str = 'train'):
        """
        Initialize super-resolution dataset.
        
        Args:
            config: Configuration object
            data_path: Path to dataset directory
            mode: Dataset mode ('train', 'val', 'test')
        """
        super().__init__(config, data_path, mode)
        
        self.scale_factor = config.model.scale_factor
        self.degradation_type = config.data.degradation_type
        self.blur_kernel_size = config.data.blur_kernel_size
        self.blur_sigma = config.data.blur_sigma
        self.noise_level = config.data.noise_level
        
        # Check if LR images exist, otherwise we'll generate them
        self.lr_path = self.data_path / self.mode / 'LR'
        self.use_existing_lr = self.lr_path.exists() and len(list(self.lr_path.glob('*'))) > 0
        
        if self.use_existing_lr:
            print(f"Using existing LR images from {self.lr_path}")
        else:
            print(f"Will generate LR images on-the-fly using {self.degradation_type}")
    
    def _load_image_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load HR and LR image pair."""
        hr_file = self.image_files[idx]
        
        # Load HR image
        hr_image = load_image(str(hr_file))
        if hr_image is None:
            raise ValueError(f"Could not load image: {hr_file}")
        
        # Normalize HR image
        hr_image = normalize_image(hr_image, method="0_1")
        
        # Load or generate LR image
        if self.use_existing_lr:
            # Try to find corresponding LR image
            lr_file = self.lr_path / hr_file.name
            if lr_file.exists():
                lr_image = load_image(str(lr_file))
                lr_image = normalize_image(lr_image, method="0_1")
            else:
                # Generate LR if corresponding file not found
                lr_image = apply_degradation(
                    hr_image, self.degradation_type, 
                    scale_factor=self.scale_factor,
                    blur_kernel_size=self.blur_kernel_size,
                    blur_sigma=self.blur_sigma,
                    noise_level=self.noise_level
                )
        else:
            # Generate LR image on-the-fly
            lr_image = apply_degradation(
                hr_image, self.degradation_type,
                scale_factor=self.scale_factor,
                blur_kernel_size=self.blur_kernel_size,
                blur_sigma=self.blur_sigma,
                noise_level=self.noise_level
            )
        
        return hr_image, lr_image
    
    def get_patches(self, num_patches: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get training patches from the dataset.
        
        Args:
            num_patches: Number of patches to extract (None for all)
            
        Returns:
            Tuple of (LR patches, HR patches)
        """
        if num_patches is None:
            num_patches = len(self.image_files) * 10  # 10 patches per image by default
        
        lr_patches = []
        hr_patches = []
        
        patches_per_image = max(1, num_patches // len(self.image_files))
        
        for idx in range(len(self.image_files)):
            try:
                hr_image, lr_image = self._load_image_pair(idx)
                
                # Apply augmentation
                if self.mode == 'train':
                    hr_image = self._augment_image(hr_image)
                    lr_image = self._augment_image(lr_image)
                
                # Extract patches from HR image
                hr_patches_img, _ = extract_patches(
                    hr_image, 
                    patch_size=self.patch_size,
                    stride=self.patch_size // 2
                )
                
                # Extract corresponding patches from LR image
                lr_patch_size = self.patch_size // self.scale_factor
                lr_patches_img, _ = extract_patches(
                    lr_image,
                    patch_size=lr_patch_size,
                    stride=lr_patch_size // 2
                )
                
                # Randomly select patches
                num_available = min(len(hr_patches_img), len(lr_patches_img))
                if num_available > patches_per_image:
                    indices = random.sample(range(num_available), patches_per_image)
                    hr_patches_img = hr_patches_img[indices]
                    lr_patches_img = lr_patches_img[indices]
                
                lr_patches.extend(lr_patches_img)
                hr_patches.extend(hr_patches_img)
                
            except Exception as e:
                print(f"Error processing image {self.image_files[idx]}: {e}")
                continue
        
        return np.array(lr_patches), np.array(hr_patches)
    
    def create_tf_dataset(self) -> tf.data.Dataset:
        """Create TensorFlow dataset for training."""
        def generator():
            while True:
                lr_patches, hr_patches = self.get_patches(self.batch_size * 10)
                
                # Shuffle patches
                indices = np.random.permutation(len(lr_patches))
                lr_patches = lr_patches[indices]
                hr_patches = hr_patches[indices]
                
                # Yield batches
                for i in range(0, len(lr_patches), self.batch_size):
                    batch_lr = lr_patches[i:i+self.batch_size]
                    batch_hr = hr_patches[i:i+self.batch_size]
                    
                    if len(batch_lr) == self.batch_size:
                        yield batch_lr, batch_hr
        
        # Define output signature
        lr_shape = (self.patch_size // self.scale_factor, 
                   self.patch_size // self.scale_factor, 3)
        hr_shape = (self.patch_size, self.patch_size, 3)
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(self.batch_size,) + lr_shape, dtype=tf.float32),
                tf.TensorSpec(shape=(self.batch_size,) + hr_shape, dtype=tf.float32)
            )
        )
        
        return dataset.prefetch(tf.data.AUTOTUNE)


class DenoisingDataset(BaseDataset):
    """Dataset class for denoising training."""
    
    def __init__(self, config: Config, data_path: str, mode: str = 'train'):
        """
        Initialize denoising dataset.
        
        Args:
            config: Configuration object
            data_path: Path to dataset directory
            mode: Dataset mode ('train', 'val', 'test')
        """
        super().__init__(config, data_path, mode)
        
        self.noise_level = config.data.noise_level
        self.channels = config.model.input_shape[-1] if config.model.input_shape[-1] else 1
        
        print(f"Denoising dataset initialized with noise level: {self.noise_level}")
        print(f"Using {self.channels} channel(s)")
    
    def _load_image_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load clean and noisy image pair."""
        image_file = self.image_files[idx]
        
        # Load clean image
        clean_image = load_image(str(image_file))
        if clean_image is None:
            raise ValueError(f"Could not load image: {image_file}")
        
        # Convert to grayscale if needed
        if self.channels == 1 and len(clean_image.shape) == 3:
            clean_image = np.mean(clean_image, axis=2, keepdims=True)
        elif self.channels == 3 and len(clean_image.shape) == 2:
            clean_image = np.stack([clean_image] * 3, axis=2)
        
        # Normalize clean image
        clean_image = normalize_image(clean_image, method="0_1")
        
        # Generate noisy image
        noisy_image = add_noise(clean_image, noise_type="gaussian", noise_level=self.noise_level)
        
        return clean_image, noisy_image
    
    def get_patches(self, num_patches: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get training patches from the dataset.
        
        Args:
            num_patches: Number of patches to extract (None for all)
            
        Returns:
            Tuple of (noisy patches, clean patches)
        """
        if num_patches is None:
            num_patches = len(self.image_files) * 10  # 10 patches per image by default
        
        noisy_patches = []
        clean_patches = []
        
        patches_per_image = max(1, num_patches // len(self.image_files))
        
        for idx in range(len(self.image_files)):
            try:
                clean_image, noisy_image = self._load_image_pair(idx)
                
                # Apply augmentation
                if self.mode == 'train':
                    clean_image = self._augment_image(clean_image)
                    noisy_image = self._augment_image(noisy_image)
                
                # Extract patches
                clean_patches_img, _ = extract_patches(
                    clean_image,
                    patch_size=self.patch_size,
                    stride=self.patch_size // 2
                )
                
                noisy_patches_img, _ = extract_patches(
                    noisy_image,
                    patch_size=self.patch_size,
                    stride=self.patch_size // 2
                )
                
                # Randomly select patches
                num_available = min(len(clean_patches_img), len(noisy_patches_img))
                if num_available > patches_per_image:
                    indices = random.sample(range(num_available), patches_per_image)
                    clean_patches_img = clean_patches_img[indices]
                    noisy_patches_img = noisy_patches_img[indices]
                
                noisy_patches.extend(noisy_patches_img)
                clean_patches.extend(clean_patches_img)
                
            except Exception as e:
                print(f"Error processing image {self.image_files[idx]}: {e}")
                continue
        
        return np.array(noisy_patches), np.array(clean_patches)
    
    def create_tf_dataset(self) -> tf.data.Dataset:
        """Create TensorFlow dataset for training."""
        def generator():
            while True:
                noisy_patches, clean_patches = self.get_patches(self.batch_size * 10)
                
                # Shuffle patches
                indices = np.random.permutation(len(noisy_patches))
                noisy_patches = noisy_patches[indices]
                clean_patches = clean_patches[indices]
                
                # Yield batches
                for i in range(0, len(noisy_patches), self.batch_size):
                    batch_noisy = noisy_patches[i:i+self.batch_size]
                    batch_clean = clean_patches[i:i+self.batch_size]
                    
                    if len(batch_noisy) == self.batch_size:
                        yield batch_noisy, batch_clean
        
        # Define output signature
        patch_shape = (self.patch_size, self.patch_size, self.channels)
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(self.batch_size,) + patch_shape, dtype=tf.float32),
                tf.TensorSpec(shape=(self.batch_size,) + patch_shape, dtype=tf.float32)
            )
        )
        
        return dataset.prefetch(tf.data.AUTOTUNE)


def create_dataset(config: Config, data_path: str, mode: str = 'train'):
    """
    Factory function to create appropriate dataset based on task type.
    
    Args:
        config: Configuration object
        data_path: Path to dataset directory
        mode: Dataset mode ('train', 'val', 'test')
        
    Returns:
        Dataset instance (SuperResolutionDataset or DenoisingDataset)
    """
    if config.model.model_type == "rdn_denoising":
        return DenoisingDataset(config, data_path, mode)
    else:
        return SuperResolutionDataset(config, data_path, mode)
