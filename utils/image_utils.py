"""
Image Utilities

This module provides image processing utilities for the RDN pipeline including:
- Image loading and saving with multiple format support
- Normalization and denormalization
- Patch extraction and reconstruction
- Image quality metrics
- Degradation model implementations
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
from typing import List, Tuple, Optional, Union
import cv2
from scipy import ndimage


def load_image(image_path: str, target_channels: Optional[int] = None) -> np.ndarray:
    """
    Load an image from file with automatic format detection.
    
    Args:
        image_path (str): Path to the image file
        target_channels (int, optional): Target number of channels (1 or 3)
        
    Returns:
        np.ndarray: Image array with shape (H, W, C)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Load image using PIL for better format support
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode in ['RGBA', 'P']:
                img = img.convert('RGB')
            elif img.mode == 'L' and target_channels == 3:
                img = img.convert('RGB')
            elif img.mode in ['RGB', 'RGBA'] and target_channels == 1:
                img = img.convert('L')
            
            # Convert to numpy array
            image = np.array(img)
            
            # Ensure 3D array (H, W, C)
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
            
            return image.astype(np.float32)
            
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {str(e)}")


def save_image(image: np.ndarray, output_path: str, denormalize: bool = True) -> None:
    """
    Save an image array to file.
    
    Args:
        image (np.ndarray): Image array with shape (H, W, C) or (H, W)
        output_path (str): Output file path
        denormalize (bool): Whether to denormalize from [0,1] to [0,255]
        
    Raises:
        ValueError: If image format is invalid
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Handle different input formats
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    
    # Ensure numpy array
    image = np.array(image)
    
    # Remove batch dimension if present
    if len(image.shape) == 4:
        image = image[0]
    
    # Denormalize if needed
    if denormalize:
        if image.max() <= 1.0:
            image = image * 255.0
    
    # Clip values and convert to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Handle grayscale images
    if len(image.shape) == 3 and image.shape[-1] == 1:
        image = image.squeeze(-1)
    
    # Save using PIL
    try:
        pil_image = Image.fromarray(image)
        pil_image.save(output_path)
    except Exception as e:
        raise ValueError(f"Failed to save image to {output_path}: {str(e)}")


def normalize_image(image: np.ndarray, method: str = "0_1") -> np.ndarray:
    """
    Normalize image to specified range.
    
    Args:
        image (np.ndarray): Input image
        method (str): Normalization method ("0_1", "-1_1", "standardize")
        
    Returns:
        np.ndarray: Normalized image
    """
    
    image = image.astype(np.float32)
    
    if method == "0_1":
        return image / 255.0
    elif method == "-1_1":
        return (image / 127.5) - 1.0
    elif method == "standardize":
        mean = np.mean(image)
        std = np.std(image)
        return (image - mean) / (std + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def denormalize_image(image: np.ndarray, method: str = "0_1") -> np.ndarray:
    """
    Denormalize image from specified range back to [0, 255].
    
    Args:
        image (np.ndarray): Normalized image
        method (str): Normalization method used ("0_1", "-1_1")
        
    Returns:
        np.ndarray: Denormalized image
    """
    
    if method == "0_1":
        return image * 255.0
    elif method == "-1_1":
        return (image + 1.0) * 127.5
    else:
        raise ValueError(f"Unknown denormalization method: {method}")


def extract_patches(image: np.ndarray, patch_size: int, stride: Optional[int] = None,
                   padding: bool = True) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Extract patches from an image for training or inference.
    
    Args:
        image (np.ndarray): Input image with shape (H, W, C)
        patch_size (int): Size of square patches
        stride (int, optional): Stride for patch extraction (default: patch_size)
        padding (bool): Whether to pad image to ensure full coverage
        
    Returns:
        tuple: (patches, original_shape) where patches has shape (N, patch_size, patch_size, C)
    """
    
    if stride is None:
        stride = patch_size
    
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    
    # Ensure 3D array
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    
    original_shape = image.shape
    
    # Pad image if needed
    if padding:
        pad_h = (patch_size - h % stride) % stride
        pad_w = (patch_size - w % stride) % stride
        
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            h, w = image.shape[:2]
    
    # Calculate number of patches
    n_patches_h = (h - patch_size) // stride + 1
    n_patches_w = (w - patch_size) // stride + 1
    
    # Extract patches
    patches = []
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            start_h = i * stride
            start_w = j * stride
            patch = image[start_h:start_h + patch_size, start_w:start_w + patch_size]
            patches.append(patch)
    
    patches = np.array(patches)
    return patches, original_shape


def reconstruct_from_patches(patches: np.ndarray, original_shape: Tuple[int, int, int],
                           patch_size: int, stride: Optional[int] = None,
                           overlap_method: str = "average") -> np.ndarray:
    """
    Reconstruct an image from patches.
    
    Args:
        patches (np.ndarray): Patches with shape (N, patch_size, patch_size, C)
        original_shape (tuple): Original image shape (H, W, C)
        patch_size (int): Size of square patches
        stride (int, optional): Stride used for patch extraction
        overlap_method (str): Method to handle overlapping regions ("average", "first")
        
    Returns:
        np.ndarray: Reconstructed image
    """
    
    if stride is None:
        stride = patch_size
    
    h, w, channels = original_shape
    
    # Calculate padded dimensions
    pad_h = (patch_size - h % stride) % stride
    pad_w = (patch_size - w % stride) % stride
    padded_h = h + pad_h
    padded_w = w + pad_w
    
    # Initialize reconstruction arrays
    reconstructed = np.zeros((padded_h, padded_w, channels), dtype=np.float32)
    weight_map = np.zeros((padded_h, padded_w), dtype=np.float32)
    
    # Calculate number of patches
    n_patches_h = (padded_h - patch_size) // stride + 1
    n_patches_w = (padded_w - patch_size) // stride + 1
    
    # Reconstruct image
    patch_idx = 0
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            start_h = i * stride
            start_w = j * stride
            
            if overlap_method == "average":
                reconstructed[start_h:start_h + patch_size, start_w:start_w + patch_size] += patches[patch_idx]
                weight_map[start_h:start_h + patch_size, start_w:start_w + patch_size] += 1
            elif overlap_method == "first":
                mask = weight_map[start_h:start_h + patch_size, start_w:start_w + patch_size] == 0
                reconstructed[start_h:start_h + patch_size, start_w:start_w + patch_size][mask] = patches[patch_idx][mask]
                weight_map[start_h:start_h + patch_size, start_w:start_w + patch_size][mask] = 1
            
            patch_idx += 1
    
    # Average overlapping regions
    if overlap_method == "average":
        weight_map = np.maximum(weight_map, 1)  # Avoid division by zero
        reconstructed = reconstructed / weight_map[..., np.newaxis]
    
    # Remove padding
    reconstructed = reconstructed[:h, :w]
    
    return reconstructed


def apply_degradation(image: np.ndarray, degradation_type: str, scale_factor: int = 2,
                     noise_level: float = 30.0, blur_kernel_size: int = 7,
                     blur_sigma: float = 1.6) -> np.ndarray:
    """
    Apply degradation to create low-resolution images.
    
    Args:
        image (np.ndarray): High-resolution image
        degradation_type (str): Type of degradation
        scale_factor (int): Downsampling factor
        noise_level (float): Noise level for Gaussian noise
        blur_kernel_size (int): Kernel size for Gaussian blur
        blur_sigma (float): Sigma for Gaussian blur
        
    Returns:
        np.ndarray: Degraded low-resolution image
    """
    
    if degradation_type == "bicubic":
        # Simple bicubic downsampling
        h, w = image.shape[:2]
        new_h, new_w = h // scale_factor, w // scale_factor
        
        if len(image.shape) == 3:
            lr_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            lr_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
    elif degradation_type == "blur_downsample":
        # Blur then downsample
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1  # Ensure odd kernel size
            
        blurred = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), blur_sigma)
        
        h, w = blurred.shape[:2]
        new_h, new_w = h // scale_factor, w // scale_factor
        lr_image = cv2.resize(blurred, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
    elif degradation_type == "downsample_noise":
        # Downsample then add noise
        h, w = image.shape[:2]
        new_h, new_w = h // scale_factor, w // scale_factor
        
        downsampled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, downsampled.shape).astype(np.float32)
        lr_image = downsampled + noise
        lr_image = np.clip(lr_image, 0, 255)
        
    else:
        raise ValueError(f"Unknown degradation type: {degradation_type}")
    
    return lr_image.astype(np.float32)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1 (np.ndarray): First image
        img2 (np.ndarray): Second image
        max_val (float): Maximum possible pixel value
        
    Returns:
        float: PSNR value in dB
    """
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0,
                  window_size: int = 11, sigma: float = 1.5) -> float:
    """
    Calculate Structural Similarity Index Measure (SSIM) between two images.
    
    Args:
        img1 (np.ndarray): First image
        img2 (np.ndarray): Second image
        max_val (float): Maximum possible pixel value
        window_size (int): Size of the sliding window
        sigma (float): Standard deviation for Gaussian window
        
    Returns:
        float: SSIM value between 0 and 1
    """
    
    # Convert to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Constants for SSIM calculation
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    # Create Gaussian window
    window = np.outer(
        cv2.getGaussianKernel(window_size, sigma),
        cv2.getGaussianKernel(window_size, sigma)
    )
    
    # Calculate means
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    # Calculate SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator
    
    return float(np.mean(ssim_map))


def get_image_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Get all image files in a directory.
    
    Args:
        directory (str): Directory path
        extensions (list): List of file extensions to include
        
    Returns:
        list: List of image file paths
    """
    
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']
    
    image_files = []
    
    if not os.path.exists(directory):
        return image_files
    
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in extensions):
            image_files.append(os.path.join(directory, filename))
    
    return sorted(image_files)


def create_image_pairs(hr_dir: str, lr_dir: str, extensions: List[str] = None) -> List[Tuple[str, str]]:
    """
    Create pairs of high-resolution and low-resolution images.
    
    Args:
        hr_dir (str): High-resolution images directory
        lr_dir (str): Low-resolution images directory
        extensions (list): List of file extensions to include
        
    Returns:
        list: List of (HR, LR) image path tuples
    """
    
    hr_files = get_image_files(hr_dir, extensions)
    lr_files = get_image_files(lr_dir, extensions)
    
    # Create mapping based on filenames
    hr_dict = {os.path.basename(f): f for f in hr_files}
    lr_dict = {os.path.basename(f): f for f in lr_files}
    
    pairs = []
    for filename in hr_dict.keys():
        if filename in lr_dict:
            pairs.append((hr_dict[filename], lr_dict[filename]))
    
    return pairs


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                method: str = "bicubic") -> np.ndarray:
    """
    Resize an image to target size.
    
    Args:
        image (np.ndarray): Input image
        target_size (tuple): Target size (width, height)
        method (str): Interpolation method
        
    Returns:
        np.ndarray: Resized image
    """
    
    interpolation_methods = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4
    }
    
    if method not in interpolation_methods:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    return cv2.resize(image, target_size, interpolation=interpolation_methods[method])


def augment_image(image: np.ndarray, horizontal_flip: bool = False,
                 vertical_flip: bool = False, rotation: int = 0) -> np.ndarray:
    """
    Apply data augmentation to an image.
    
    Args:
        image (np.ndarray): Input image
        horizontal_flip (bool): Whether to flip horizontally
        vertical_flip (bool): Whether to flip vertically
        rotation (int): Rotation angle (0, 90, 180, 270)
        
    Returns:
        np.ndarray: Augmented image
    """
    
    augmented = image.copy()
    
    if horizontal_flip:
        augmented = cv2.flip(augmented, 1)
    
    if vertical_flip:
        augmented = cv2.flip(augmented, 0)
    
    if rotation != 0:
        if rotation == 90:
            augmented = cv2.rotate(augmented, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            augmented = cv2.rotate(augmented, cv2.ROTATE_180)
        elif rotation == 270:
            augmented = cv2.rotate(augmented, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return augmented


def add_noise(image: np.ndarray, noise_type: str = "gaussian", 
              noise_level: float = 25.0, **kwargs) -> np.ndarray:
    """
    Add noise to an image for denoising training.
    
    Args:
        image (np.ndarray): Clean input image
        noise_type (str): Type of noise ("gaussian", "poisson", "salt_pepper")
        noise_level (float): Noise level/intensity
        **kwargs: Additional noise parameters
        
    Returns:
        np.ndarray: Noisy image
    """
    
    image = image.astype(np.float32)
    noisy_image = image.copy()
    
    if noise_type == "gaussian":
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        noisy_image = image + noise
        
    elif noise_type == "poisson":
        # Add Poisson noise
        # Scale image to appropriate range for Poisson
        scaled = image / 255.0 * noise_level
        noisy_scaled = np.random.poisson(scaled).astype(np.float32)
        noisy_image = noisy_scaled / noise_level * 255.0
        
    elif noise_type == "salt_pepper":
        # Add salt and pepper noise
        prob = noise_level / 100.0  # Convert to probability
        noise_mask = np.random.random(image.shape[:2])
        
        # Salt noise (white pixels)
        salt_mask = noise_mask < prob / 2
        noisy_image[salt_mask] = 255.0
        
        # Pepper noise (black pixels)
        pepper_mask = noise_mask > (1 - prob / 2)
        noisy_image[pepper_mask] = 0.0
        
    elif noise_type == "uniform":
        # Add uniform noise
        noise = np.random.uniform(-noise_level, noise_level, image.shape).astype(np.float32)
        noisy_image = image + noise
        
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Clip values to valid range
    noisy_image = np.clip(noisy_image, 0, 255)
    
    return noisy_image.astype(np.float32)


def create_comparison_image(original: np.ndarray, processed: np.ndarray,
                          labels: List[str] = None) -> np.ndarray:
    """
    Create a side-by-side comparison image.
    
    Args:
        original (np.ndarray): Original image
        processed (np.ndarray): Processed image
        labels (list): Labels for the images
        
    Returns:
        np.ndarray: Comparison image
    """
    
    # Ensure same dimensions
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    # Create side-by-side comparison
    comparison = np.hstack([original, processed])
    
    return comparison
