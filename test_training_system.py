"""
Comprehensive Test for RDN Training System

This script tests the complete training pipeline including:
- Configuration loading
- Model building
- Data loading
- Training initialization
- Basic functionality verification
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        # Test model imports
        from models import build_rdn, build_rdn_denoising, compile_model
        print("✓ Model imports successful")
        
        # Test utility imports
        from utils.config import load_config, create_default_config, save_config
        from utils.image_utils import load_image, save_image, calculate_psnr, calculate_ssim
        print("✓ Utility imports successful")
        
        # Test training imports
        from training.trainer import RDNTrainer
        from training.data_loader import create_dataset, RDNDataset
        print("✓ Training imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_configuration_system():
    """Test configuration creation and loading."""
    print("\nTesting configuration system...")
    
    try:
        from utils.config import create_default_config, save_config, load_config
        
        # Test super-resolution config
        sr_config = create_default_config("super_resolution")
        assert sr_config.model.model_type == "rdn"
        assert sr_config.model.scale_factor == 2
        print("✓ Super-resolution config creation successful")
        
        # Test denoising config
        denoising_config = create_default_config("denoising")
        assert denoising_config.model.model_type == "rdn_denoising"
        assert hasattr(denoising_config.data, 'noise_level')
        print("✓ Denoising config creation successful")
        
        # Test config save/load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_config_path = f.name
        
        try:
            save_config(sr_config, temp_config_path)
            loaded_config = load_config(temp_config_path)
            assert loaded_config.model.scale_factor == sr_config.model.scale_factor
            print("✓ Config save/load successful")
        finally:
            os.unlink(temp_config_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_model_building():
    """Test model building functionality."""
    print("\nTesting model building...")
    
    try:
        from models import build_rdn, build_rdn_denoising, compile_model
        
        # Test super-resolution model
        sr_model = build_rdn(
            scale_factor=2,
            input_shape=(None, None, 3),
            num_features=32,  # Smaller for testing
            growth_rate=16,
            num_blocks=4,
            num_layers=4
        )
        
        assert sr_model is not None
        assert len(sr_model.input_shape) == 4  # (batch, height, width, channels)
        print("✓ Super-resolution model building successful")
        
        # Test denoising model
        denoising_model = build_rdn_denoising(
            input_shape=(None, None, 3),
            num_features=32,
            growth_rate=16,
            num_blocks=4,
            num_layers=4
        )
        
        assert denoising_model is not None
        assert denoising_model.input_shape == denoising_model.output_shape
        print("✓ Denoising model building successful")
        
        # Test model compilation
        compiled_model = compile_model(sr_model, learning_rate=0.001)
        assert compiled_model.optimizer is not None
        print("✓ Model compilation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Model building test failed: {e}")
        return False


def test_data_loading():
    """Test data loading functionality."""
    print("\nTesting data loading...")
    
    try:
        from training.data_loader import RDNDataset
        from utils.config import create_default_config
        from utils.image_utils import save_image
        
        # Create temporary test data
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir) / "test_data" / "train" / "HR"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create synthetic test images
            for i in range(3):
                # Create a simple test image
                test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                image_path = data_dir / f"test_{i:03d}.png"
                save_image(test_image, str(image_path))
            
            # Test dataset creation
            config = create_default_config("super_resolution")
            config.training.patch_size = 32  # Small patches for testing
            
            dataset = RDNDataset(config, str(data_dir.parent.parent), mode='train')
            assert len(dataset.image_files) == 3
            print("✓ Dataset creation successful")
            
            # Test patch extraction
            patches = dataset.get_patches(num_patches=5)
            lr_patches, hr_patches = patches
            
            assert len(lr_patches) == len(hr_patches)
            assert lr_patches[0].shape[-1] == 3  # RGB channels
            print("✓ Patch extraction successful")
            
            return True
            
        finally:
            shutil.rmtree(temp_dir)
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False


def test_trainer_initialization():
    """Test trainer initialization."""
    print("\nTesting trainer initialization...")
    
    try:
        from training.trainer import RDNTrainer
        from utils.config import create_default_config
        
        # Test super-resolution trainer
        sr_config = create_default_config("super_resolution")
        sr_config.model.num_features = 32  # Smaller for testing
        sr_config.model.num_blocks = 4
        sr_config.training.epochs = 1
        sr_config.training.batch_size = 2
        
        sr_trainer = RDNTrainer(sr_config)
        assert sr_trainer.config == sr_config
        assert sr_trainer.model is None  # Not built yet
        print("✓ Super-resolution trainer initialization successful")
        
        # Test denoising trainer
        denoising_config = create_default_config("denoising")
        denoising_config.model.num_features = 32  # Smaller for testing
        denoising_config.model.num_blocks = 4
        denoising_config.training.epochs = 1
        denoising_config.training.batch_size = 2
        
        denoising_trainer = RDNTrainer(denoising_config)
        assert denoising_trainer.config == denoising_config
        print("✓ Denoising trainer initialization successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Trainer initialization test failed: {e}")
        return False


def test_model_building_integration():
    """Test model building through trainer."""
    print("\nTesting model building integration...")
    
    try:
        from training.trainer import RDNTrainer
        from utils.config import create_default_config
        
        # Test super-resolution model building
        sr_config = create_default_config("super_resolution")
        sr_config.model.num_features = 32  # Smaller for testing
        sr_config.model.num_blocks = 4
        
        sr_trainer = RDNTrainer(sr_config)
        sr_model = sr_trainer.build_model()
        
        assert sr_model is not None
        assert sr_trainer.model is not None
        print("✓ Super-resolution model building through trainer successful")
        
        # Test denoising model building
        denoising_config = create_default_config("denoising")
        denoising_config.model.num_features = 32
        denoising_config.model.num_blocks = 4
        
        denoising_trainer = RDNTrainer(denoising_config)
        denoising_model = denoising_trainer.build_model()
        
        assert denoising_model is not None
        assert denoising_trainer.model is not None
        print("✓ Denoising model building through trainer successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Model building integration test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("RDN Training System Comprehensive Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration System Test", test_configuration_system),
        ("Model Building Test", test_model_building),
        ("Data Loading Test", test_data_loading),
        ("Trainer Initialization Test", test_trainer_initialization),
        ("Model Building Integration Test", test_model_building_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The RDN training system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
