"""
Complete X-ray Image Denoising Training Script

This script provides a complete pipeline for training RDN models on chest X-ray images.

Usage:
    python xray_denoising_complete.py --dataset_path "Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset" --max_images 50 --epochs 10
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.image_utils import load_image, save_image, resize_image
from training.trainer import RDNTrainer


def main():
    parser = argparse.ArgumentParser(description='X-ray Denoising Training')
    parser.add_argument('--dataset_path', default='Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset', 
                       help='Path to dataset')
    parser.add_argument('--max_images', type=int, default=50, help='Maximum images to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--output_dir', default='xray_training_output', help='Output directory')
    
    args = parser.parse_args()
    
    print("🏥 X-ray Denoising Training Pipeline")
    print("="*50)
    
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
    
    # 2. Prepare data
    print("\n🔧 Preparing Data:")
    target_data_path = output_dir / "prepared_data"
    train_dir = target_data_path / "train" / "HR"
    val_dir = target_data_path / "val" / "HR"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter valid images
    valid_images = []
    for img_path in tqdm(train_images[:args.max_images * 2], desc="Validating"):
        try:
            img = load_image(str(img_path))
            if img is not None and img.shape[0] >= 128 and img.shape[1] >= 128:
                valid_images.append(img_path)
                if len(valid_images) >= args.max_images:
                    break
        except:
            continue
    
    print(f"✓ Found {len(valid_images)} valid images")
    
    # Split and copy images
    random.shuffle(valid_images)
    split_idx = int(0.8 * len(valid_images))
    train_imgs = valid_images[:split_idx]
    val_imgs = valid_images[split_idx:]
    
    print(f"📊 Train/Val split: {len(train_imgs)}/{len(val_imgs)}")
    print("🔄 Resizing all images to 250x250 pixels (full image training - no patching)")
    
    # Copy training images with 250x250 resizing
    for i, img_path in enumerate(tqdm(train_imgs, desc="Copying train")):
        try:
            img = load_image(str(img_path))
            if img is not None:
                if len(img.shape) == 3:
                    img = np.mean(img, axis=2, keepdims=True)
                # Resize to 250x250
                img = resize_image(img, (250, 250), method="bicubic")
                save_image(img, str(train_dir / f"train_{i:04d}.png"))
        except Exception as e:
            print(f"Error: {e}")
    
    # Copy validation images with 250x250 resizing
    for i, img_path in enumerate(tqdm(val_imgs, desc="Copying val")):
        try:
            img = load_image(str(img_path))
            if img is not None:
                if len(img.shape) == 3:
                    img = np.mean(img, axis=2, keepdims=True)
                # Resize to 250x250
                img = resize_image(img, (250, 250), method="bicubic")
                save_image(img, str(val_dir / f"val_{i:04d}.png"))
        except Exception as e:
            print(f"Error: {e}")
    
    # 3. Load configuration
    print("\n⚙️ Loading Configuration:")
    config_path = "../configs/xray_fullimage_config.yaml"
    
    try:
        config = load_config(config_path)
        config.training.epochs = args.epochs
        config.training.batch_size = args.batch_size
        config.output_dir = str(output_dir / "training_output")
        config.checkpoint_dir = str(output_dir / "checkpoints")
        config.log_dir = str(output_dir / "logs")
        
        print(f"✓ Configuration loaded")
        print(f"  - Epochs: {config.training.epochs}")
        print(f"  - Batch size: {config.training.batch_size}")
        
    except Exception as e:
        print(f"❌ Config error: {e}")
        return
    
    # 4. Train model
    print("\n🚀 Starting Training:")
    
    try:
        trainer = RDNTrainer(config)
        print("✓ Trainer initialized")
        
        history = trainer.train(data_path=str(target_data_path.resolve()))
        
        print("🎉 Training completed!")
        
        # 5. Save results
        if history:
            results_file = output_dir / "training_results.txt"
            with open(results_file, 'w') as f:
                f.write("X-ray Denoising Training Results (Full Image)\n")
                f.write("="*40 + "\n")
                f.write(f"Dataset: {dataset_root}\n")
                f.write(f"Images used: {len(valid_images)}\n")
                f.write(f"Image size: 250x250 pixels (full image - no patching)\n")
                f.write(f"Configuration: xray_fullimage_config.yaml\n")
                f.write(f"Epochs: {args.epochs}\n")
                f.write(f"Batch size: {args.batch_size}\n")
                f.write(f"Final loss: {history.get('loss', ['N/A'])[-1] if history.get('loss') else 'N/A'}\n")
            
            print(f"📊 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
