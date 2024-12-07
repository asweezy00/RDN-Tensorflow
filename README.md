# Residual Dense Network (RDN) for Image Denoising and Super-Resolution (Tensorflow Implementation)

TensorFlow Implementation of Residual Dense Network for Image Super-Resolution and Image Denoising (CVPR 2018)
Based on Pytorch implementation found in: https://github.com/yjn870/RDN-pytorch

### Citations
Y. Zhang, Y. Tian, Y. Kong, B. Zhong, and Y. Fu, "Residual Dense Network for Image Super-Resolution," *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Salt Lake City, UT, USA, Jun. 2018, pp. 2472–2481, doi: 10.48550/arXiv.1802.08797.

Y. Zhang, Y. Tian, Y. Kong, B. Zhong and Y. Fu, "Residual Dense Network for Image Restoration," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 7, pp. 2480-2495, 1 July 2021, doi: 10.1109/TPAMI.2020.2968521.


## Overview
This repository contains an implementation of the Residual Dense Network (RDN) architecture using TensorFlow/Keras. The implementation supports both image denoising and super-resolution tasks, featuring dense connectivity, local and global feature fusion, and residual learning.

## Architecture
The network consists of several key components:

### 1. Dense Layer
- Custom implementation of dense connected layers
- Features:
  - 3x3 convolutional operation
  - ReLU activation
  - Dense connection pattern (concatenation with input)

### 2. Residual Dense Block (RDB)
- Building block of the network containing:
  - Multiple dense layers
  - Local feature fusion
  - Local residual learning
- Parameters:
  - Input channels
  - Growth rate
  - Number of dense layers

### 3. Network Structure
- **Shallow Feature Extraction**:
  - Two 3x3 convolutional layers
- **Residual Dense Blocks**:
  - Multiple RDBs connected sequentially
  - Dense connections within each block
- **Global Feature Fusion**:
  - Concatenation of all RDB outputs
  - 1x1 and 3x3 convolutional layers
- **Global Residual Learning**:
  - Skip connections from shallow features

## Features
- Support for both denoising and super-resolution
- Configurable scale factors (1x, 2x, 3x, 4x)
- Custom metrics implementation (PSNR, SSIM)
- Flexible input dimensions
- Comprehensive model compilation setup

## Building Model 

# For Denoising
<img width="700" alt="image" src="https://github.com/user-attachments/assets/795b648c-0840-4e23-85fb-7ad1e8bb0c91">

```python
denoising_model = build_rdn_denoising(
    input_shape=(64, 64, 1),
    num_features=64,
    growth_rate=64,
    num_blocks=6,
    num_layers=6
)
```

# For Super-Resolution
<img width="700" alt="image" src="https://github.com/user-attachments/assets/f50d80a5-6b75-43e0-9065-d2daec503eb6">

```python
sr_model = build_rdn(
    scale_factor=2,  # 1 for denoising, 2/3/4 for super-resolution
    input_shape=(64, 64, 1),
    num_features=64,
    growth_rate=64,
    num_blocks=6,
    num_layers=6
)
```

## Custom Metrics

### PSNR Metric
- Measures peak signal-to-noise ratio
- Range: Higher is better
- Implementation includes automatic state management

### SSIM Metric
- Measures structural similarity
- Range: 0 to 1 (higher is better)
- Accounts for structural information in images

## Training Tips
1. Ensure input images are normalized to [0, 1]
2. Use appropriate batch size based on available GPU memory
3. Monitor PSNR and SSIM metrics for model performance
4. Consider using learning rate scheduling for better convergence

## Model Variations
1. **Denoising Model (`build_rdn_denoising`)**
   - Specialized for image denoising
   - No upsampling layers
   - Direct residual connections

2. **Generic RDN (`build_rdn`)**
   - Supports both denoising and super-resolution
   - Configurable scale factor
   - Adaptive upsampling based on scale factor

## Performance Considerations
- Memory usage scales with:
  - Number of RDBs
  - Growth rate
  - Number of layers per RDB
- Consider reducing these parameters for memory-constrained environments



  
## Requirements
```python
tensorflow>=2.0

