"""
Test only the configuration system without any TensorFlow dependencies.
"""

import sys
import os
import yaml

def test_config_standalone():
    """Test configuration system directly without imports that trigger TensorFlow."""
    print("Testing Configuration System (Standalone)...")
    
    try:
        # Test YAML file reading directly
        with open('configs/super_resolution_config.yaml', 'r') as f:
            sr_config = yaml.safe_load(f)
        
        with open('configs/denoising_config.yaml', 'r') as f:
            denoise_config = yaml.safe_load(f)
        
        print(f"✓ Super-resolution config loaded: scale_factor={sr_config['model']['scale_factor']}")
        print(f"✓ Denoising config loaded: scale_factor={denoise_config['model']['scale_factor']}")
        
        # Test config structure
        required_sr_keys = ['model', 'training', 'data', 'checkpoint_dir', 'log_dir']
        required_denoise_keys = ['model', 'training', 'data', 'checkpoint_dir', 'log_dir']
        
        sr_missing = [key for key in required_sr_keys if key not in sr_config]
        denoise_missing = [key for key in required_denoise_keys if key not in denoise_config]
        
        if sr_missing:
            print(f"❌ Super-resolution config missing keys: {sr_missing}")
            return False
        
        if denoise_missing:
            print(f"❌ Denoising config missing keys: {denoise_missing}")
            return False
        
        print("✓ All required configuration keys present")
        
        # Test model parameters
        sr_model = sr_config['model']
        denoise_model = denoise_config['model']
        
        print(f"✓ SR model params: {sr_model['num_features']} features, {sr_model['num_blocks']} blocks")
        print(f"✓ Denoising model params: {denoise_model['num_features']} features, {denoise_model['num_blocks']} blocks")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_requirements_files():
    """Test that requirements files exist and are valid."""
    print("\nTesting Requirements Files...")
    
    try:
        # Check requirements_fixed.txt
        with open('requirements_fixed.txt', 'r') as f:
            fixed_reqs = f.read()
        
        required_packages = ['tensorflow', 'numpy', 'opencv-python', 'PyYAML']
        
        missing_packages = []
        for package in required_packages:
            if package not in fixed_reqs:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages in requirements_fixed.txt: {missing_packages}")
            return False
        
        print("✓ All required packages found in requirements_fixed.txt")
        
        # Count total packages
        lines = [line.strip() for line in fixed_reqs.split('\n') if line.strip() and not line.startswith('#')]
        print(f"✓ Total packages specified: {len(lines)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Requirements test failed: {e}")
        return False


def test_setup_scripts():
    """Test that setup scripts exist."""
    print("\nTesting Setup Scripts...")
    
    try:
        # Check Windows script
        if os.path.exists('setup_environment.bat'):
            print("✓ Windows setup script exists")
        else:
            print("❌ Windows setup script missing")
            return False
        
        # Check Linux/Mac script
        if os.path.exists('setup_environment.sh'):
            print("✓ Linux/Mac setup script exists")
        else:
            print("❌ Linux/Mac setup script missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Setup scripts test failed: {e}")
        return False


def main():
    """Run all standalone tests."""
    print("RDN Configuration and Setup Test")
    print("=" * 50)
    
    tests = [
        test_config_standalone,
        test_requirements_files,
        test_setup_scripts
    ]
    
    results = []
    for test_func in tests:
        results.append(test_func())
    
    print("\n" + "=" * 50)
    
    if all(results):
        print("✅ All configuration tests PASSED!")
        print("\nThe refactored RDN project setup is ready!")
        print("\nTo set up the environment and test with TensorFlow:")
        print("1. Run: setup_environment.bat (Windows)")
        print("2. Or run: ./setup_environment.sh (Linux/Mac)")
        print("3. This will create a virtual environment with compatible packages")
    else:
        print("❌ Some tests FAILED!")
        print("Please check the error messages above.")
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
