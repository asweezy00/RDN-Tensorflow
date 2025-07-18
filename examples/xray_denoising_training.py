"""
X-ray Image Denoising Training Script

This script provides a complete pipeline for training RDN models on chest X-ray images
for denoising tasks using the Coronahack dataset.

Usage:
    python xray_denoising_training.py [options]

Example:
    python xray_denoising_training.py --max_images 100 --epochs 20 --batch_size 4
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import random
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RDN modules
from utils.config import load_config, save_config
from utils.image_utils import load_image, save_image, normalize_image, add_noise
from training.trainer import RDNTrainer


class XrayDenoisingPipeline:
    """Complete pipeline for X-ray denoising training"""
    
    def __init__(self, args):
        self.args = args
        self.dataset_root = Path(args.dataset_path)
        self.output_dir = Path(args.output_dir)
        self.config = None
        self.trainer = None
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        print("🏥 X-ray Denoising Training Pipeline Initialized")
        print(f"📁 Dataset path: {self.dataset_root}")
        print(f"📁 Output directory: {self.output_dir}")
    
    def explore_dataset(self):
        """Explore and validate the X-ray dataset"""
        print("\n" + "="*60)
        print("📊 DATASET EXPLORATION")
        print("="*60)
        
        train_path = self.dataset_root / "train"
        test_path = self.dataset_root / "test"
        
        if not self.dataset_root.exists():
            print(f"❌ Dataset not found at {self.dataset_root}")
            print("Please ensure the Coronahack dataset is extracted in the correct location")
            return False
        
        # Count images
        train_images = list(train_path.glob("*")) if train_path.exists() else []
        test_images = list(test_path.glob("*")) if test_path.exists() else []
        
        print(f"✓ Dataset found at {self.dataset_root}")
        print(f"📈 Training images: {len(train_images)}")
        print(f"📈 Test images: {len(test_images)}")
        
        if len(train_images) == 0:
            print("❌ No training images found!")
            return False
        
        # Analyze sample images
        print("\n📋 Sample Image Analysis:")
        valid_count = 0
        total_size = 0
        formats = {}
        
        sample_images = random.sample(train_images, min(10, len(train_images)))
        
        for img_path in sample_images:
            try:
                img = load_image(str(img_path))
                if img is not None:
                    valid_count += 1
                    total_size += img.size
                    ext = img_path.suffix.lower()
                    formats[ext] = formats.get(ext, 0) + 1
                    
                    if valid_count <= 3:  # Show details for first 3 images
                        print(f"  📷 {img_path.name}: {img.shape} - {img.dtype}")
            except Exception as e:
                print(f"  ❌ Error loading {img_path.name}: {e}")
        
        print(f"\n✓ Valid images in sample: {valid_count}/{len(sample_images)}")
        print(f"📊 Image formats found: {formats}")
        
        return len(train_images) > 0
    
    def prepare_dataset(self):
        """Prepare and organize X-ray dataset for RDN training"""
        print("\n" + "="*60)
        print("🔧 DATA PREPARATION")
        print("="*60)
        
        # Define paths
        source_train_path = self.dataset_root / "train"
        target_data_path = self.output_dir / "prepared_data"
        
        # Create target directories
        train_dir = target_data_path / "train" / "HR"
        val_dir = target_data_path / "val" / "HR"
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Preparing data in: {target_data_path}")
        
        # Get source images
        source_images = list(source_train_path.glob("*"))
        print(f"📊 Found {len(source_images)} source images")
        
        # Filter and validate images
        print("🔍 Validating images...")
        valid_images = []
        
        for img_path in tqdm(source_images, desc="Validating"):
            try:
                img = load_image(str(img_path))
                if img is not None:
                    # Check minimum size requirements
                    min_size = 128  # Minimum size for training patches
                    if img.shape[0] >= min_size and img.shape[1] >= min_size:
                        valid_images.append(img_path)
                        
                        # Stop if we have enough images
                        if len(valid_images) >= self.args.max_images:
                            break
            except Exception as e:
                continue
        
        print(f"✓ Found {len(valid_images)} valid images")
        
        if len(valid_images) == 0:
            print("❌ No valid images found!")
            return None
        
