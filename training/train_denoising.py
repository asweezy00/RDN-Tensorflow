"""
Denoising Training Script

This script provides a command-line interface for training RDN models
on denoising tasks.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, create_default_config, save_config
from training.trainer import RDNTrainer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RDN model for denoising",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/denoising_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data_path', '-d',
        type=str,
        required=True,
        help='Path to training data directory'
    )
    
    parser.add_argument(
        '--val_data_path', '-v',
        type=str,
        default=None,
        help='Path to validation data directory (optional)'
    )
    
    parser.add_argument(
        '--resume', '-r',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--noise_level', '-n',
        type=float,
        default=None,
        help='Override noise level from config'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=None,
        help='Override number of epochs from config'
    )
    
    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=None,
        help='Override batch size from config'
    )
    
    parser.add_argument(
        '--learning_rate', '-lr',
        type=float,
        default=None,
        help='Override learning rate from config'
    )
    
    parser.add_argument(
        '--patch_size', '-p',
        type=int,
        default=None,
        help='Override patch size from config'
    )
    
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='Override output directory from config'
    )
    
    parser.add_argument(
        '--create_config',
        action='store_true',
        help='Create default config file and exit'
    )
    
    return parser.parse_args()


def train_denoising(config_path: str, data_path: str, val_data_path: str = None,
                   resume_from: str = None, **overrides):
    """
    Train denoising model.
    
    Args:
        config_path: Path to configuration file
        data_path: Path to training data
        val_data_path: Path to validation data (optional)
        resume_from: Path to checkpoint to resume from (optional)
        **overrides: Configuration overrides
    """
    
    # Load or create configuration
    try:
        config = load_config(config_path)
        print(f"✓ Loaded configuration from {config_path}")
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        print("Creating default denoising configuration...")
        config = create_default_config("denoising")
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        save_config(config, config_path)
        print(f"✓ Default configuration saved to {config_path}")
    
    # Apply command-line overrides
    if overrides.get('noise_level'):
        config.data.noise_level = overrides['noise_level']
        print(f"Override: noise_level = {config.data.noise_level}")
    
    if overrides.get('epochs'):
        config.training.epochs = overrides['epochs']
        print(f"Override: epochs = {config.training.epochs}")
    
    if overrides.get('batch_size'):
        config.training.batch_size = overrides['batch_size']
        print(f"Override: batch_size = {config.training.batch_size}")
    
    if overrides.get('learning_rate'):
        config.training.learning_rate = overrides['learning_rate']
        print(f"Override: learning_rate = {config.training.learning_rate}")
    
    if overrides.get('patch_size'):
        config.training.patch_size = overrides['patch_size']
        print(f"Override: patch_size = {config.training.patch_size}")
    
    if overrides.get('output_dir'):
        config.output_dir = overrides['output_dir']
        config.checkpoint_dir = os.path.join(overrides['output_dir'], 'checkpoints')
        config.log_dir = os.path.join(overrides['output_dir'], 'logs')
        print(f"Override: output_dir = {config.output_dir}")
    
    # Validate data path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data path not found: {data_path}")
    
    if val_data_path and not os.path.exists(val_data_path):
        raise FileNotFoundError(f"Validation data path not found: {val_data_path}")
    
    # Initialize trainer
    trainer = RDNTrainer(config)
    
    # Start training
    channels = config.model.input_shape[-1] if config.model.input_shape[-1] else 1
    channel_type = "Grayscale" if channels == 1 else "RGB"
    
    print(f"\nTraining Configuration:")
    print(f"  Task: Denoising ({channel_type})")
    print(f"  Data path: {data_path}")
    print(f"  Validation path: {val_data_path or 'Auto-split from training data'}")
    print(f"  Noise level: {config.data.noise_level}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Patch size: {config.training.patch_size}")
    print(f"  Output directory: {config.output_dir}")
    
    try:
        history = trainer.train(
            data_path=data_path,
            val_data_path=val_data_path,
            resume_from=resume_from
        )
        
        print("\n🎉 Training completed successfully!")
        return history
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return None
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise e


def main():
    """Main function."""
    args = parse_arguments()
    
    # Create default config if requested
    if args.create_config:
        config = create_default_config("denoising")
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        save_config(config, args.config)
        print(f"Default denoising config created: {args.config}")
        return
    
    # Prepare overrides
    overrides = {}
    if args.noise_level:
        overrides['noise_level'] = args.noise_level
    if args.epochs:
        overrides['epochs'] = args.epochs
    if args.batch_size:
        overrides['batch_size'] = args.batch_size
    if args.learning_rate:
        overrides['learning_rate'] = args.learning_rate
    if args.patch_size:
        overrides['patch_size'] = args.patch_size
    if args.output_dir:
        overrides['output_dir'] = args.output_dir
    
    # Start training
    train_denoising(
        config_path=args.config,
        data_path=args.data_path,
        val_data_path=args.val_data_path,
        resume_from=args.resume,
        **overrides
    )


if __name__ == "__main__":
    main()
