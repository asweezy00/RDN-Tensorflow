"""
Full Image Data Loading for RDN Training

This module provides dataset classes for full image training without patching,
specifically optimized for medical image denoising at 250x250 resolution.
"""

import os
import glob
import random
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Union
from pathlib import Path
import cv2

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_utils import load_image, normalize_image, add_noise
from utils.config import Config


class FullImageDenoisingDataset:
    """Dataset class for full image denoising training (no patching)."""
    
    def __init__(self, config: Config, data_path: str, mode: str = 'train'):
        """
        Initialize full image denoising dataset.
        
        Args:
            config: Configuration object
            data_path: Path to dataset directory
            mode: Dataset mode ('train', 'val', 'test')
        """
        self.config = config
        self.data_path = Path(data_path)
        self.mode = mode
        self.batch_size = config.training.batch_size
        
        # Image settings
        self.target_size = config.data.target_size  # [250, 250]
        self.channels = config.model.input_shape[-1] if config.model.input_shape[-1] else 1
        self.noise_level = config.data.noise_level
        
        # Image formats to search for
        self.image_formats = config.data.image_formats
        
        # Data augmentation settings (conservative for medical images)
        self.horizontal_flip = config.training.horizontal_flip and mode == 'train'
        self.vertical_flip = config.training.vertical_flip and mode == 'train'
        self.rotation = config.training.rotation and mode == 'train'
        
        # Find image files
        self.image_files = self._find_image_files()
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {data_path}")
        
        print(f"Found {len(self.image_files)} images for {mode} mode")
        print(f"Target size: {self.target_size}, Channels: {self.channels}")
    
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
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size while maintaining aspect ratio."""
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        # Calculate scaling factor to fit image in target size
        scale = min(target_w / w, target_h / h)
        
        # Resize image
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        if len(image.shape) == 3:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if len(image.shape) == 2:
                resized = np.expand_dims(resized, axis=2)
        
        # Create target size image with padding
        if len(resized.shape) == 3:
            padded = np.zeros((target_h, target_w, resized.shape[2]), dtype=resized.dtype)
        else:
            padded = np.zeros((target_h, target_w, 1), dtype=resized.dtype)
            resized = np.expand_dims(resized, axis=2)
        
        # Center the resized image
        start_y = (target_h - new_h) // 2
        start_x = (target_w - new_w) // 2
        padded[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        
        return padded
    
    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation to image (conservative for medical images)."""
        if not (self.horizontal_flip or self.vertical_flip or self.rotation):
            return image
        
        # Horizontal flip (safe for most X-rays)
        if self.horizontal_flip and random.random() > 0.5:
            image = np.fliplr(image)
        
        # Vertical flip (be careful with X-rays - they have orientation meaning)
        if self.vertical_flip and random.random() > 0.7:  # Lower probability
            image = np.flipud(image)
        
        # Rotation (avoid for medical images as orientation matters)
        # if self.rotation and random.random() > 0.8:  # Very low probability
        #     k = random.randint(1, 3)
        #     image = np.rot90(image, k)
        
        return image
    
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
        
        # Resize to target size
        clean_image = self._resize_image(clean_image)
        
        # Normalize clean image
        clean_image = normalize_image(clean_image, method="0_1")
        
        # Apply augmentation
        if self.mode == 'train':
            clean_image = self._augment_image(clean_image)
        
        # Generate noisy image
        noisy_image = add_noise(clean_image, noise_type="gaussian", noise_level=self.noise_level)
        
        return clean_image, noisy_image
    
    def get_batch(self, batch_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a batch of full images.
        
        Args:
            batch_size: Size of batch to return
            
        Returns:
            Tuple of (noisy images, clean images)
        """
        if batch_size is None:
