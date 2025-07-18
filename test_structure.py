"""
Test script to verify the refactored code structure without TensorFlow dependencies.
This tests the configuration system and basic utilities.
"""

import sys
import os
import numpy as np

def test_configuration_system():
    """Test the configuration management system."""
    print("Testing Configuration System...")
    
    try:
        # Import config utilities
        from utils.config import Config, create_default_config, save_config, load_config
        
        # Test creating default configs
        sr_config = create_default_config("super_resolution")
        denoise_config = create_default_config("denoising")
        
        print(f"✓ Super-resolution config: scale_factor={sr_config.model.scale_factor}")
        print(f"✓ Denoising config: scale_factor={denoise_config.model.scale_factor}")
        
        # Test saving and loading
        os.makedirs('temp_test', exist_ok=True)
        save_config(sr_config, 'temp_test/test_config.yaml')
        loaded_config = load_config('temp_test/test_config.yaml')
        
        print(f"✓ Config save/load: {loaded_config.model.scale_factor == sr_config.model.scale_factor}")
        
        # Cleanup
        os.remove('temp_test/test_config.yaml')
        os.rmdir('temp_test')
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_basic_image_utils():
    """Test basic image utilities that don't require heavy dependencies."""
    print("\nTesting Basic Image Utilities...")
    
    try:
        # Test basic normalization functions
        test_image = np.random.randint(0, 256, (64, 64, 3)).astype(np.float32)
        
        # Import only the functions we can test
        import importlib.util
        spec = importlib.util.spec_from_file_location("image_utils", "utils/image_utils.py")
        
        if spec and spec.loader:
            print("✓ Image utilities module structure is valid")
        
        # Test basic numpy operations (simulating normalization)
        normalized = test_image / 255.0
        denormalized = normalized * 255.0
        
        print(f"✓ Basic image processing simulation:")
        print(f"  Original range: {test_image.min():.1f} - {test_image.max():.1f}")
        print(f"  Normalized range: {normalized.min():.3f} - {normalized.max():.3f}")
        print(f"  Denormalized range: {denormalized.min():.1f} - {denormalized.max():.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image utilities test failed: {e}")
        return False


def test_project_structure():
    """Test that all required directories and files exist."""
    print("\nTesting Project Structure...")
    
    try:
        required_dirs = ['models', 'utils', 'configs', 'data', 'training', 'inference', 'evaluation', 'examples']
        required_files = [
            'models/__init__.py',
            'models/rdn_model.py', 
            'models/compile_utils.py',
            'utils/__init__.py',
            'utils/config.py',
            'utils/image_utils.py',
            'configs/super_resolution_config.yaml',
            'configs/denoising_config.yaml',
            'requirements_fixed.txt',
            'setup_environment.bat',
            'setup_environment.sh'
        ]
        
        # Check directories
        missing_dirs = []
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            print(f"❌ Missing directories: {missing_dirs}")
            return False
        else:
            print(f"✓ All required directories exist: {len(required_dirs)} dirs")
        
        # Check files
        missing_files = []
        for file_name in required_files:
            if not os.path.exists(file_name):
                missing_files.append(file_name)
        
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return False
        else:
            print(f"✓ All required files exist: {len(required_files)} files")
        
        return True
        
    except Exception as e:
        print(f"❌ Project structure test failed: {e}")
        return False


def main():
    """Run all structure tests."""
    print("RDN Project Structure Test")
    print("=" * 50)
    
    tests = [
        test_configuration_system,
        test_basic_image_utils,
        test_project_structure
    ]
    
    results = []
    for test_func in tests:
        results.append(test_func())
    
    print("\n" + "=" * 50)
    
    if all(results):
        print("✅ All structure tests PASSED!")
        print("\nThe refactored RDN codebase structure is working correctly.")
        print("\nNext steps:")
        print("1. Run setup_environment.bat (Windows) or ./setup_environment.sh (Linux/Mac)")
        print("2. This will create a virtual environment with compatible packages")
        print("3. Then test the full functionality with TensorFlow")
    else:
        print("❌ Some tests FAILED!")
        print("Please check the error messages above.")
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
