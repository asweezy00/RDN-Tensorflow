"""
Basic Usage Example for RDN Implementation

This script demonstrates how to use the refactored RDN codebase for
both super-resolution and denoising tasks.
"""

import os
import sys
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_rdn, build_rdn_denoising, compile_model
from utils.config import load_config, create_default_config, save_config
from utils.image_utils import (
    load_image, save_image, normalize_image, denormalize_image,
    extract_patches, reconstruct_from_patches, apply_degradation,
    calculate_psnr, calculate_ssim
)


def demo_super_resolution():
    """Demonstrate super-resolution model creation and basic usage."""
    
    print("=== Super-Resolution Demo ===")
    
    # Load configuration
    try:
        config = load_config('configs/super_resolution_config.yaml')
        print("✓ Loaded configuration from file")
    except FileNotFoundError:
        print("! Configuration file not found, creating default...")
        config = create_default_config("super_resolution")
        save_config(config, 'configs/super_resolution_config.yaml')
        print("✓ Created default super-resolution configuration")
    
    # Build model
    print(f"Building RDN model with scale factor {config.model.scale_factor}...")
    model = build_rdn(
        scale_factor=config.model.scale_factor,
        input_shape=config.model.input_shape,
        num_features=config.model.num_features,
        growth_rate=config.model.growth_rate,
        num_blocks=config.model.num_blocks,
        num_layers=config.model.num_layers
    )
    
    # Compile model
    model = compile_model(
        model, 
        learning_rate=config.training.learning_rate,
        loss=config.training.loss,
        max_val=config.training.max_val
    )
    
    print(f"✓ Model created successfully!")
    print(f"  - Input shape: {model.input_shape}")
    print(f"  - Output shape: {model.output_shape}")
    print(f"  - Total parameters: {model.count_params():,}")
    
    # Demonstrate with dummy data
    print("\nTesting with dummy data...")
    dummy_lr = np.random.rand(1, 32, 32, 3).astype(np.float32)
    dummy_sr = model.predict(dummy_lr, verbose=0)
    
    expected_size = 32 * config.model.scale_factor
    print(f"✓ Input: {dummy_lr.shape} -> Output: {dummy_sr.shape}")
    print(f"✓ Expected output size: {expected_size}x{expected_size}")
    
    return model, config


def demo_denoising():
    """Demonstrate denoising model creation and basic usage."""
    
    print("\n=== Denoising Demo ===")
    
    # Load or create configuration
    try:
        config = load_config('configs/denoising_config.yaml')
        print("✓ Loaded denoising configuration from file")
    except FileNotFoundError:
        print("! Configuration file not found, creating default...")
        config = create_default_config("denoising")
        save_config(config, 'configs/denoising_config.yaml')
        print("✓ Created default denoising configuration")
    
    # Build specialized denoising model
    print("Building specialized RDN denoising model...")
    model = build_rdn_denoising(
        input_shape=config.model.input_shape,
        num_features=config.model.num_features,
        growth_rate=config.model.growth_rate,
        num_blocks=config.model.num_blocks,
        num_layers=config.model.num_layers
    )
    
    # Compile model
    model = compile_model(
        model,
        learning_rate=config.training.learning_rate,
        loss=config.training.loss,
        max_val=config.training.max_val
    )
    
    print(f"✓ Denoising model created successfully!")
    print(f"  - Input shape: {model.input_shape}")
    print(f"  - Output shape: {model.output_shape}")
    print(f"  - Total parameters: {model.count_params():,}")
    
    # Test with dummy noisy data
    print("\nTesting with dummy noisy data...")
    # Use the same number of channels as specified in the config
    channels = config.model.input_shape[-1] if config.model.input_shape[-1] is not None else 1
    dummy_noisy = np.random.rand(1, 64, 64, channels).astype(np.float32)
    dummy_clean = model.predict(dummy_noisy, verbose=0)
    
    print(f"✓ Input: {dummy_noisy.shape} -> Output: {dummy_clean.shape}")
    print("✓ Same size output (denoising preserves dimensions)")
    
    return model, config


def demo_image_processing():
    """Demonstrate image processing utilities."""
    
    print("\n=== Image Processing Demo ===")
    
    # Create a synthetic test image
    print("Creating synthetic test image...")
    test_image = np.random.randint(0, 256, (128, 128, 3)).astype(np.uint8)
    
    # Save test image
    os.makedirs('examples/temp', exist_ok=True)
    save_image(test_image, 'examples/temp/test_image.png', denormalize=False)
    print("✓ Saved synthetic test image")
    
    # Load and process image
    loaded_image = load_image('examples/temp/test_image.png')
    print(f"✓ Loaded image with shape: {loaded_image.shape}")
    
    # Normalize image
    normalized = normalize_image(loaded_image, method="0_1")
    print(f"✓ Normalized image (range: {normalized.min():.3f} - {normalized.max():.3f})")
    
    # Extract patches
    patches, original_shape = extract_patches(normalized, patch_size=32, stride=16)
    print(f"✓ Extracted {patches.shape[0]} patches of size {patches.shape[1:3]}")
    
    # Reconstruct from patches
    reconstructed = reconstruct_from_patches(
        patches, original_shape, patch_size=32, stride=16, overlap_method="average"
    )
    print(f"✓ Reconstructed image with shape: {reconstructed.shape}")
    
    # Apply degradation
    degraded = apply_degradation(loaded_image, "bicubic", scale_factor=2)
    print(f"✓ Applied bicubic degradation: {loaded_image.shape} -> {degraded.shape}")
    
    # Calculate metrics
    # Resize degraded back to original size for comparison
    from utils.image_utils import resize_image
    degraded_upscaled = resize_image(degraded, (loaded_image.shape[1], loaded_image.shape[0]))
    
    psnr = calculate_psnr(loaded_image, degraded_upscaled)
    ssim = calculate_ssim(loaded_image, degraded_upscaled)
    
    print(f"✓ Quality metrics - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    
    # Clean up
    os.remove('examples/temp/test_image.png')
    os.rmdir('examples/temp')
    print("✓ Cleaned up temporary files")


def main():
    """Main function to run all demos."""
    
    print("RDN Implementation Demo")
    print("=" * 50)
    
    try:
        # Demo super-resolution
        sr_model, sr_config = demo_super_resolution()
        
        # Demo denoising
        denoise_model, denoise_config = demo_denoising()
        
        # Demo image processing utilities
        demo_image_processing()
        
        print("\n" + "=" * 50)
        print("✅ All demos completed successfully!")
        print("\nNext steps:")
        print("1. Prepare your training data in the data/ directory")
        print("2. Modify configurations in configs/ as needed")
        print("3. Use the training scripts (to be implemented)")
        print("4. Run inference on your images")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
