"""
Configuration Management

This module provides configuration management for the RDN pipeline,
supporting both YAML and JSON configuration files.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    scale_factor: int = 2
    input_shape: list = field(default_factory=lambda: [None, None, 3])
    num_features: int = 64
    growth_rate: int = 64
    num_blocks: int = 6
    num_layers: int = 6
    model_type: str = "rdn"  # "rdn" or "rdn_denoising"


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    epochs: int = 200
    learning_rate: float = 1e-4
    patch_size: int = 32
    validation_split: float = 0.1
    loss: str = "mae"
    max_val: float = 1.0
    
    # Data augmentation
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotation: bool = True
    
    # Callbacks
    early_stopping: bool = True
    patience: int = 10
    monitor: str = "val_psnr"
    mode: str = "max"


@dataclass
class DataConfig:
    """Data processing configuration."""
    train_data_path: str = ""
    val_data_path: str = ""
    test_data_path: str = ""
    
    # Image processing
    image_formats: list = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.tiff', '.bmp'])
    normalize: bool = True
    
    # Degradation models
    degradation_type: str = "bicubic"  # "bicubic", "blur_downsample", "downsample_noise"
    noise_level: float = 30.0
    blur_kernel_size: int = 7
    blur_sigma: float = 1.6


@dataclass
class InferenceConfig:
    """Inference configuration."""
    model_path: str = ""
    input_path: str = ""
    output_path: str = ""
    
    # Processing options
    patch_size: int = 64
    overlap: int = 8
    batch_size: int = 1
    
    # Output options
    save_comparison: bool = True
    calculate_metrics: bool = True


@dataclass
class Config:
    """Main configuration class containing all sub-configurations."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # General settings
    project_name: str = "RDN_Project"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    seed: int = 42


def load_config(config_path: str) -> Config:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        Config: Configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is not supported
    """
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    file_ext = os.path.splitext(config_path)[1].lower()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if file_ext in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        elif file_ext == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")
    
    return dict_to_config(config_dict)


def save_config(config: Config, config_path: str) -> None:
    """
    Save configuration to a YAML or JSON file.
    
    Args:
        config (Config): Configuration object to save
        config_path (str): Path where to save the configuration
        
    Raises:
        ValueError: If config file format is not supported
    """
    
    config_dict = asdict(config)
    file_ext = os.path.splitext(config_path)[1].lower()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if file_ext in ['.yaml', '.yml']:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif file_ext == '.json':
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")


def dict_to_config(config_dict: Dict[str, Any]) -> Config:
    """
    Convert a dictionary to a Config object.
    
    Args:
        config_dict (dict): Configuration dictionary
        
    Returns:
        Config: Configuration object
    """
    
    # Extract sub-configurations
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    data_config = DataConfig(**config_dict.get('data', {}))
    inference_config = InferenceConfig(**config_dict.get('inference', {}))
    
    # Extract general settings
    general_settings = {k: v for k, v in config_dict.items() 
                       if k not in ['model', 'training', 'data', 'inference']}
    
    return Config(
        model=model_config,
        training=training_config,
        data=data_config,
        inference=inference_config,
        **general_settings
    )


def create_default_config(config_type: str = "super_resolution") -> Config:
    """
    Create a default configuration for different tasks.
    
    Args:
        config_type (str): Type of configuration ("super_resolution", "denoising")
        
    Returns:
        Config: Default configuration object
    """
    
    config = Config()
    
    if config_type == "super_resolution":
        config.model.scale_factor = 2
        config.model.model_type = "rdn"
        config.training.patch_size = 32
        config.data.degradation_type = "bicubic"
        
    elif config_type == "denoising":
        config.model.scale_factor = 1
        config.model.input_shape = [None, None, 1]  # Grayscale for denoising
        config.model.model_type = "rdn_denoising"
        config.training.patch_size = 64
        config.data.degradation_type = "downsample_noise"
        
    return config


def validate_config(config: Config) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config (Config): Configuration to validate
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    
    # Validate model configuration
    if config.model.scale_factor not in [1, 2, 3, 4]:
        raise ValueError("scale_factor must be 1, 2, 3, or 4")
    
    if config.model.model_type not in ["rdn", "rdn_denoising"]:
        raise ValueError("model_type must be 'rdn' or 'rdn_denoising'")
    
    # Validate training configuration
    if config.training.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    if config.training.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    
    if not 0 < config.training.validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1")
    
    # Validate data configuration
    if config.data.degradation_type not in ["bicubic", "blur_downsample", "downsample_noise"]:
        raise ValueError("Invalid degradation_type")
    
    return True


def get_config_template() -> str:
    """
    Get a configuration template as a string.
    
    Returns:
        str: YAML configuration template
    """
    
    template = """
# RDN Configuration Template

# Model Architecture
model:
  scale_factor: 2          # 1 for denoising, 2/3/4 for super-resolution
  input_shape: [null, null, 3]  # [height, width, channels]
  num_features: 64         # Number of feature maps
  growth_rate: 64          # Growth rate for dense layers
  num_blocks: 6            # Number of residual dense blocks
  num_layers: 6            # Number of dense layers per block
  model_type: "rdn"        # "rdn" or "rdn_denoising"

# Training Configuration
training:
  batch_size: 16
  epochs: 200
  learning_rate: 0.0001
  patch_size: 32           # Training patch size
  validation_split: 0.1
  loss: "mae"              # "mae" or "mse"
  max_val: 1.0             # Maximum pixel value for metrics
  
  # Data Augmentation
  horizontal_flip: true
  vertical_flip: true
  rotation: true
  
  # Callbacks
  early_stopping: true
  patience: 10
  monitor: "val_psnr"
  mode: "max"

# Data Configuration
data:
  train_data_path: ""
  val_data_path: ""
  test_data_path: ""
  
  # Image Processing
  image_formats: [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
  normalize: true
  
  # Degradation Models
  degradation_type: "bicubic"  # "bicubic", "blur_downsample", "downsample_noise"
  noise_level: 30.0
  blur_kernel_size: 7
  blur_sigma: 1.6

# Inference Configuration
inference:
  model_path: ""
  input_path: ""
  output_path: ""
  
  # Processing Options
  patch_size: 64
  overlap: 8
  batch_size: 1
  
  # Output Options
  save_comparison: true
  calculate_metrics: true

# General Settings
project_name: "RDN_Project"
output_dir: "outputs"
checkpoint_dir: "checkpoints"
log_dir: "logs"
seed: 42
"""
    
    return template.strip()
