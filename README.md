# RDN TensorFlow - Residual Dense Network

A comprehensive implementation of Residual Dense Networks (RDN) for image super-resolution and denoising tasks using TensorFlow 2.x.

## Overview

This project provides a complete training and inference pipeline for RDN models, supporting both super-resolution and denoising tasks. The implementation is based on the paper "Residual Dense Network for Image Super-Resolution" and has been extended to support denoising applications.

## Features

- **Dual Task Support**: Super-resolution and denoising
- **Flexible Architecture**: Configurable RDN model parameters
- **Complete Training Pipeline**: Data loading, training, validation, and checkpointing
- **Configuration System**: YAML-based configuration management
- **Data Preparation**: Automated dataset preparation and augmentation
- **Multiple Loss Functions**: MAE, MSE, and Huber loss support
- **Comprehensive Metrics**: PSNR and SSIM evaluation
- **Command-line Interface**: Easy-to-use training scripts

## Project Structure

```
rdnmedimage/
├── models/                     # Model definitions
│   ├── __init__.py
│   ├── rdn_model.py           # RDN architecture implementation
│   └── compile_utils.py       # Model compilation utilities
├── training/                   # Training pipeline
│   ├── __init__.py
│   ├── trainer.py             # Main training class
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── train_super_resolution.py  # Super-resolution training script
│   ├── train_denoising.py     # Denoising training script
│   └── prepare_data.py        # Data preparation utilities
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   └── image_utils.py         # Image processing utilities
├── configs/                    # Configuration files
│   ├── super_resolution_config.yaml
│   └── denoising_config.yaml
├── examples/                   # Usage examples
│   ├── basic_usage.py         # Basic model usage
│   └── train_example.py       # Training examples
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd rdnmedimage
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment (optional):**
   ```bash
   # Windows
   setup_environment.bat
   
   # Linux/Mac
   chmod +x setup_environment.sh
   ./setup_environment.sh
   ```

## Quick Start

### 1. Basic Model Usage

```python
from models import build_rdn, build_rdn_denoising
from utils.config import create_default_config

# Create configuration
config = create_default_config("super_resolution")

# Build super-resolution model
sr_model = build_rdn(
    scale_factor=2,
    input_shape=(None, None, 3),
    num_features=64,
    growth_rate=32,
    num_blocks=16,
    num_layers=8
)

# Build denoising model
denoising_model = build_rdn_denoising(
    input_shape=(None, None, 3),
    num_features=64,
    growth_rate=32,
    num_blocks=16,
    num_layers=8
)
```

### 2. Training Super-Resolution Model

```bash
# Using command-line script
python training/train_super_resolution.py \
    --data_path /path/to/training/data \
    --config configs/super_resolution_config.yaml \
    --scale_factor 2 \
    --epochs 100 \
    --batch_size 16

# Using Python API
python examples/train_example.py --task sr
```

### 3. Training Denoising Model

```bash
# Using command-line script
python training/train_denoising.py \
    --data_path /path/to/training/data \
    --config configs/denoising_config.yaml \
    --noise_level 25 \
    --epochs 100 \
    --batch_size 16

# Using Python API
python examples/train_example.py --task denoising
```

## Configuration

The system uses YAML configuration files to manage all training parameters:

### Super-Resolution Configuration (`configs/super_resolution_config.yaml`)

```yaml
model:
  model_type: "rdn"
  scale_factor: 2
  input_shape: [null, null, 3]
  num_features: 64
  growth_rate: 32
  num_blocks: 16
  num_layers: 8

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.0001
  patch_size: 48
  validation_split: 0.1
  loss: "mae"
```

### Denoising Configuration (`configs/denoising_config.yaml`)

```yaml
model:
  model_type: "rdn_denoising"
  input_shape: [null, null, 3]
  num_features: 64
  growth_rate: 32
  num_blocks: 16
  num_layers: 8

data:
  noise_type: "gaussian"
  noise_level: 25

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.0001
  patch_size: 64
```

## Data Preparation

### Automatic Dataset Preparation

```bash
# Prepare DIV2K dataset for super-resolution
python training/prepare_data.py \
    --dataset div2k \
    --scale_factor 2 \
    --data_root data/

# Organize custom dataset
python training/prepare_data.py \
    --organize /path/to/your/images \
    --data_root data/
```

### Expected Data Structure

```
data/
├── train/
│   └── HR/          # High-resolution images
│       ├── image1.png
│       ├── image2.png
│       └── ...
├── val/
│   └── HR/          # Validation images
│       ├── val1.png
│       ├── val2.png
│       └── ...
└── test/
    └── HR/          # Test images
        ├── test1.png
        ├── test2.png
        └── ...
```

## Training Pipeline

### RDNTrainer Class

The `RDNTrainer` class provides a complete training pipeline:

```python
from training.trainer import RDNTrainer
from utils.config import load_config

# Load configuration
config = load_config("configs/super_resolution_config.yaml")

# Initialize trainer
trainer = RDNTrainer(config)

# Train model
history = trainer.train(
    data_path="data/DIV2K_x2/organized",
    val_data_path=None,  # Use auto-split if None
    resume_from=None     # Resume from checkpoint if provided
)

# Evaluate model
metrics = trainer.evaluate("data/test")

# Predict on single image
result = trainer.predict_image("input.png", "output.png")
```

### Training Features

- **Automatic Data Loading**: Handles image loading and preprocessing
- **Data Augmentation**: Random flips, rotations, and cropping
- **Patch-based Training**: Efficient training on image patches
- **Validation Monitoring**: PSNR and SSIM tracking
- **Checkpointing**: Automatic model saving and resuming
- **Early Stopping**: Prevents overfitting
- **TensorBoard Logging**: Training visualization

## Model Architecture

### RDN (Residual Dense Network)

The RDN architecture consists of:

1. **Shallow Feature Extraction**: Initial convolution layers
2. **Residual Dense Blocks (RDB)**: Core building blocks with dense connections
3. **Dense Feature Fusion (DFF)**: Combines features from all RDBs
4. **Upsampling**: Pixel shuffle for super-resolution (SR only)
5. **Reconstruction**: Final convolution layer

### Key Parameters

- `num_features`: Number of feature channels (default: 64)
- `growth_rate`: Growth rate for dense connections (default: 32)
- `num_blocks`: Number of residual dense blocks (default: 16)
- `num_layers`: Number of layers per RDB (default: 8)
- `scale_factor`: Upsampling factor for SR (2, 3, or 4)

## Advanced Usage

### Custom Training Loop

```python
from training.trainer import RDNTrainer
from training.data_loader import create_dataset

# Create custom dataset
dataset = create_dataset(config, "data/custom", mode='train')

# Get training patches
patches = dataset.get_patches(num_patches=1000)
train_x, train_y = patches

# Initialize trainer
trainer = RDNTrainer(config)
trainer.build_model()

# Custom training
model = trainer.model
history = model.fit(
    train_x, train_y,
    batch_size=config.training.batch_size,
    epochs=config.training.epochs,
    validation_split=0.1,
    callbacks=trainer.create_callbacks()
)
```

### Model Inference

```python
import numpy as np
from utils.image_utils import load_image, save_image
from models import build_rdn
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("outputs/checkpoints/best_model.h5")

# Load and preprocess image
image = load_image("input.jpg")
image = np.expand_dims(image / 255.0, axis=0)  # Normalize and add batch dim

# Predict
result = model.predict(image)

# Post-process and save
result = np.clip(result[0] * 255.0, 0, 255).astype(np.uint8)
save_image(result, "output.jpg")
```

## Performance Optimization

### Mixed Precision Training

Enable mixed precision for faster training on modern GPUs:

```yaml
# In config file
use_mixed_precision: true
```

### Memory Optimization

For large images or limited GPU memory:

```yaml
training:
  batch_size: 8        # Reduce batch size
  patch_size: 32       # Use smaller patches
  
gpu_memory_growth: true  # Enable memory growth
```

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better
- **SSIM (Structural Similarity Index)**: Higher is better (0-1 range)
- **Training Loss**: MAE, MSE, or Huber loss
- **Validation Metrics**: Tracked during training

## Troubleshooting

### Common Issues

1. **Out of Memory Error**:
   - Reduce batch size or patch size
   - Enable GPU memory growth
   - Use mixed precision training

2. **Slow Training**:
   - Increase batch size if memory allows
   - Use GPU acceleration
   - Enable mixed precision

3. **Poor Results**:
   - Check data quality and preprocessing
   - Adjust learning rate
   - Increase training epochs
   - Verify model architecture parameters

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See the original RDN-Tensorflow repository for more details.

## Citation

If you use this code in your research, please cite the original RDN paper:

```bibtex
@inproceedings{zhang2018residual,
  title={Residual dense network for image super-resolution},
  author={Zhang, Yulun and Tian, Yapeng and Kong, Yu and Zhong, Bineng and Fu, Yun},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2472--2481},
  year={2018}
}
```

## Acknowledgments

- Original RDN implementation and research
- TensorFlow team for the deep learning framework
- DIV2K dataset providers
- Open source community contributions

## Contact

For questions, issues, or contributions, please:
1. Open an issue on GitHub
2. Submit a pull request
3. Check the documentation and examples

---

**Note**: This is a refactored and enhanced version of the original RDN implementation, providing a complete training pipeline with modern TensorFlow 2.x support and additional features for both super-resolution and denoising tasks.
