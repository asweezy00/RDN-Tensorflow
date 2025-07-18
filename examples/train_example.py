"""
Training Example for RDN Models

This example demonstrates how to train RDN models for both super-resolution
and denoising tasks using the training pipeline.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, create_default_config, save_config
from training.trainer import RDNTrainer
from training.prepare_data import DatasetPreparer


def train_super_resolution_example():
    """Example of training a super-resolution model."""
    print("=" * 60)
    print("Super-Resolution Training Example")
    print("=" * 60)
    
    # Create or load configuration
    config_path = "configs/super_resolution_config.yaml"
    
    try:
        config = load_config(config_path)
        print(f"✓ Loaded configuration from {config_path}")
    except FileNotFoundError:
        print("Creating default super-resolution configuration...")
        config = create_default_config("super_resolution")
        
        # Customize for example
        config.training.epochs = 5  # Short training for example
        config.training.batch_size = 4
        config.training.patch_size = 64
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        save_config(config, config_path)
        print(f"✓ Configuration saved to {config_path}")
    
    # Create sample data directory structure
    data_path = "data/example_sr"
    sample_data_dir = Path(data_path)
    
    # Create directories
    train_hr_dir = sample_data_dir / "train" / "HR"
    val_hr_dir = sample_data_dir / "val" / "HR"
    
    train_hr_dir.mkdir(parents=True, exist_ok=True)
    val_hr_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if sample images exist
    if not list(train_hr_dir.glob("*.png")) and not list(train_hr_dir.glob("*.jpg")):
        print("\n⚠️  No training images found!")
        print(f"Please add some high-resolution images to: {train_hr_dir}")
        print("You can:")
        print("1. Copy some images manually to the directory")
        print("2. Use the data preparation script to download DIV2K dataset")
        print("3. Run: python training/prepare_data.py --dataset div2k --scale_factor 2")
        return False
    
    # Initialize trainer
    trainer = RDNTrainer(config)
    
    print(f"\nStarting super-resolution training...")
    print(f"Data path: {data_path}")
    print(f"Scale factor: {config.model.scale_factor}x")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    
    try:
        # Train the model
        history = trainer.train(data_path=sample_data_dir.resolve())
        
        print("\n🎉 Super-resolution training completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False


def train_denoising_example():
    """Example of training a denoising model."""
    print("=" * 60)
    print("Denoising Training Example")
    print("=" * 60)
    
    # Create or load configuration
    config_path = "configs/denoising_config.yaml"
    
    try:
        config = load_config(config_path)
        print(f"✓ Loaded configuration from {config_path}")
    except FileNotFoundError:
        print("Creating default denoising configuration...")
        config = create_default_config("denoising")
        
        # Customize for example
        config.training.epochs = 5  # Short training for example
        config.training.batch_size = 4
        config.training.patch_size = 64
        config.data.noise_level = 25  # Moderate noise level
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        save_config(config, config_path)
        print(f"✓ Configuration saved to {config_path}")
    
    # Create sample data directory structure
    data_path = "data/example_denoising"
    sample_data_dir = Path(data_path)
    
    # Create directories
    train_dir = sample_data_dir / "train" / "HR"  # Clean images
    val_dir = sample_data_dir / "val" / "HR"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if sample images exist
    if not list(train_dir.glob("*.png")) and not list(train_dir.glob("*.jpg")):
        print("\n⚠️  No training images found!")
        print(f"Please add some clean images to: {train_dir}")
        print("The training pipeline will automatically add noise to create noisy/clean pairs")
        return False
    
    # Initialize trainer
    trainer = RDNTrainer(config)
    
    print(f"\nStarting denoising training...")
    print(f"Data path: {data_path}")
    print(f"Noise level: {config.data.noise_level}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    
    try:
        # Train the model
        history = trainer.train(data_path=sample_data_dir.resolve())
        
        print("\n🎉 Denoising training completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False


def prepare_sample_data():
    """Prepare sample data for training examples."""
    print("=" * 60)
    print("Sample Data Preparation")
    print("=" * 60)
    
    # Create synthetic sample images for demonstration
    import numpy as np
    from utils.image_utils import save_image
    
    # Create sample directories
    sr_dir = Path("data/example_sr/train/HR")
    denoising_dir = Path("data/example_denoising/train/HR")
    
    sr_dir.mkdir(parents=True, exist_ok=True)
    denoising_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic images
    print("Creating synthetic sample images...")
    
    for i in range(10):  # Create 10 sample images
        # Create a synthetic high-resolution image (256x256)
        image = np.random.rand(256, 256, 3) * 255
        
        # Add some structure to make it more realistic
        x, y = np.meshgrid(np.linspace(0, 4*np.pi, 256), np.linspace(0, 4*np.pi, 256))
        pattern = np.sin(x) * np.cos(y) * 0.3 + 0.7
        
        for c in range(3):
            image[:, :, c] = image[:, :, c] * pattern
        
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Save for super-resolution
        sr_filename = sr_dir / f"sample_{i:03d}.png"
        save_image(image, str(sr_filename))
        
        # Save for denoising (same images)
        denoising_filename = denoising_dir / f"sample_{i:03d}.png"
        save_image(image, str(denoising_filename))
    
    print(f"✓ Created {10} sample images for super-resolution training")
    print(f"✓ Created {10} sample images for denoising training")
    
    return True


def main():
    """Main function to run training examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RDN Training Examples")
    parser.add_argument('--task', choices=['sr', 'denoising', 'both', 'prepare'], 
                       default='both', help='Training task to run')
    
    args = parser.parse_args()
    
    if args.task == 'prepare':
        # Prepare sample data
        success = prepare_sample_data()
        if success:
            print("\n✅ Sample data prepared successfully!")
            print("You can now run training examples with:")
            print("  python examples/train_example.py --task sr")
            print("  python examples/train_example.py --task denoising")
        return
    
    elif args.task == 'sr':
        # Run super-resolution training example
        success = train_super_resolution_example()
        if success:
            print("\n✅ Super-resolution training example completed!")
        
    elif args.task == 'denoising':
        # Run denoising training example
        success = train_denoising_example()
        if success:
            print("\n✅ Denoising training example completed!")
        
    elif args.task == 'both':
        # Run both examples
        print("Running both training examples...\n")
        
        sr_success = train_super_resolution_example()
        print("\n" + "="*60 + "\n")
        denoising_success = train_denoising_example()
        
        if sr_success and denoising_success:
            print("\n✅ All training examples completed successfully!")
        elif sr_success:
            print("\n⚠️  Super-resolution completed, but denoising failed")
        elif denoising_success:
            print("\n⚠️  Denoising completed, but super-resolution failed")
        else:
            print("\n❌ Both training examples failed")


if __name__ == "__main__":
    main()
