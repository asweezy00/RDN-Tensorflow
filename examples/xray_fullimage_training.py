"""
X-ray Full Image Denoising Training Script

This script trains RDN on full 250x250 X-ray images without patching.
Optimized for medical imaging with proper resizing and grayscale handling.

Usage:
    python xray_fullimage_training.py --dataset_path "Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset" --max_images 200 --epochs 20
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import shutil
import random
from tqdm import tqdm
import cv2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.image_utils import load_image, save_image
from training.trainer import RDNTrainer


def resize_image_to_250(image):
    """Resize image to exactly 250x250 while preserving aspect ratio"""
    if image is None:
        return None
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to 250x250
    resized = cv2.resize(image, (250, 250), interpolation=cv2.INTER_AREA)
    
    # Add channel dimension for consistency
    if len(resized.shape) == 2:
        resized = np.expand_dims(resized, axis=-1)
    
    return resized


def main():
    parser = argparse.ArgumentParser(description='X-ray Full Image Denoising Training')
    parser.add_argument('--dataset_path', default='Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset', 
                       help='Path to dataset')
    parser.add_argument('--max_images', type=int, default=200, help='Maximum images to use')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (small for full images)')
    parser.add_argument('--output_dir', default='xray_fullimage_output', help='Output directory')
    
    args = parser.parse_args()
    
    print("🏥 X-ray Full Image (250x250) Denoising Training")
    print("="*60)
    print(f"📊 Target image size: 250x250 pixels")
    print(f"🎯 No patching - training on full images")
    print(f"💾 Batch size: {args.batch_size} (optimized for full images)")
    
    # Setup paths
    dataset_root = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Explore dataset
    print("\n📊 Dataset Exploration:")
    train_path = dataset_root / "train"
    
    if not train_path.exists():
        print(f"❌ Dataset not found at {train_path}")
        return
    
    train_images = list(train_path.glob("*"))
    print(f"✓ Found {len(train_images)} training images")
    
    # 2. Prepare data with 250x250 resizing
    print("\n🔧 Preparing Data (Resizing to 250x250):")
    target_data_path = output_dir / "prepared_data"
    train_dir = target_data_path / "train" / "HR"
    val_dir = target_data_path / "val" / "HR"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter and process images
    valid_images = []
    print("🔍 Processing and validating images...")
    
    for img_path in tqdm(train_images[:args.max_images * 3], desc="Processing"):
        try:
            # Load original image
            img = load_image(str(img_path))
            if img is not None:
                # Resize to 250x250
                resized_img = resize_image_to_250(img)
                if resized_img is not None:
                    valid_images.append((img_path, resized_img))
                    
                    # Stop when we have enough
                    if len(valid_images) >= args.max_images:
                        break
        except Exception as e:
            continue
    
    print(f"✓ Successfully processed {len(valid_images)} images to 250x250")
    
    if len(valid_images) == 0:
        print("❌ No valid images found!")
        return
    
    # Split and save images
    random.shuffle(valid_images)
    split_idx = int(0.85 * len(valid_images))
    train_imgs = valid_images[:split_idx]
    val_imgs = valid_images[split_idx:]
    
    print(f"📊 Train/Val split: {len(train_imgs)}/{len(val_imgs)}")
    
    # Save training images
    print("💾 Saving training images (250x250)...")
    for i, (original_path, processed_img) in enumerate(tqdm(train_imgs, desc="Saving train")):
        try:
            save_path = train_dir / f"train_{i:04d}.png"
            save_image(processed_img, str(save_path))
        except Exception as e:
            print(f"Error saving {original_path}: {e}")
    
    # Save validation images
    print("💾 Saving validation images (250x250)...")
    for i, (original_path, processed_img) in enumerate(tqdm(val_imgs, desc="Saving val")):
        try:
            save_path = val_dir / f"val_{i:04d}.png"
            save_image(processed_img, str(save_path))
        except Exception as e:
            print(f"Error saving {original_path}: {e}")
    
    # 3. Load configuration
    print("\n⚙️ Loading Full Image Configuration:")
    config_path = "../configs/xray_fullimage_config.yaml"
    
    try:
        config = load_config(config_path)
        
        # Override with command line arguments
        if args.epochs:
            config.training.epochs = args.epochs
        if args.batch_size:
            config.training.batch_size = args.batch_size
            
        # Update output directories
        config.output_dir = str(output_dir / "training_output")
        config.checkpoint_dir = str(output_dir / "checkpoints")
        config.log_dir = str(output_dir / "logs")
        
