"""
Data Preparation Utilities

This module provides utilities for preparing datasets for RDN training,
including dataset downloading, organization, and preprocessing.
"""

import os
import sys
import shutil
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_utils import load_image, save_image, resize_image


class DatasetPreparer:
    """Class for preparing datasets for training."""
    
    def __init__(self, data_root: str = "data"):
        """
        Initialize dataset preparer.
        
        Args:
            data_root: Root directory for datasets
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(exist_ok=True)
        
        # Common dataset URLs
        self.dataset_urls = {
            'div2k_train_hr': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
            'div2k_valid_hr': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip',
            'div2k_train_lr_bicubic_x2': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip',
            'div2k_valid_lr_bicubic_x2': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip',
            'div2k_train_lr_bicubic_x3': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip',
            'div2k_valid_lr_bicubic_x3': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X3.zip',
            'div2k_train_lr_bicubic_x4': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip',
            'div2k_valid_lr_bicubic_x4': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip',
        }
    
    def download_file(self, url: str, destination: Path, chunk_size: int = 8192) -> bool:
        """
        Download a file from URL.
        
        Args:
            url: URL to download from
            destination: Destination file path
            chunk_size: Download chunk size
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Downloading {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rProgress: {progress:.1f}%", end='', flush=True)
            
            print(f"\n✓ Downloaded: {destination}")
            return True
            
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            if destination.exists():
                destination.unlink()
            return False
    
    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """
        Extract archive file.
        
        Args:
            archive_path: Path to archive file
            extract_to: Directory to extract to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Extracting {archive_path}...")
            extract_to.mkdir(parents=True, exist_ok=True)
            
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix.lower() in ['.tar', '.tar.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
            
            print(f"✓ Extracted to: {extract_to}")
            return True
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False
    
    def organize_dataset(self, source_dir: Path, target_dir: Path, 
                        train_split: float = 0.8) -> bool:
        """
        Organize dataset into train/val/test structure.
        
        Args:
            source_dir: Source directory containing images
            target_dir: Target directory for organized dataset
            train_split: Fraction of data for training
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Organizing dataset from {source_dir} to {target_dir}...")
            
            # Find all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(source_dir.glob(f"**/*{ext}"))
                image_files.extend(source_dir.glob(f"**/*{ext.upper()}"))
            
            if not image_files:
                print("❌ No image files found")
                return False
            
            print(f"Found {len(image_files)} images")
            
            # Create directory structure
            train_dir = target_dir / 'train' / 'HR'
            val_dir = target_dir / 'val' / 'HR'
            test_dir = target_dir / 'test' / 'HR'
            
            for directory in [train_dir, val_dir, test_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Split files
            import random
            random.shuffle(image_files)
            
            train_count = int(len(image_files) * train_split)
            val_count = int(len(image_files) * 0.1)  # 10% for validation
            
            train_files = image_files[:train_count]
            val_files = image_files[train_count:train_count + val_count]
            test_files = image_files[train_count + val_count:]
            
            # Copy files to respective directories
            for i, file_path in enumerate(train_files):
                dest_path = train_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                if (i + 1) % 100 == 0:
                    print(f"Copied {i + 1}/{len(train_files)} training images")
            
            for i, file_path in enumerate(val_files):
                dest_path = val_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                if (i + 1) % 50 == 0:
                    print(f"Copied {i + 1}/{len(val_files)} validation images")
            
            for i, file_path in enumerate(test_files):
                dest_path = test_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                if (i + 1) % 50 == 0:
                    print(f"Copied {i + 1}/{len(test_files)} test images")
            
            print(f"✓ Dataset organized:")
            print(f"  Training: {len(train_files)} images")
            print(f"  Validation: {len(val_files)} images")
            print(f"  Test: {len(test_files)} images")
            
            return True
            
        except Exception as e:
            print(f"❌ Organization failed: {e}")
            return False
    
    def prepare_div2k(self, scale_factor: int = 2, download_lr: bool = False) -> bool:
        """
        Prepare DIV2K dataset for super-resolution training.
        
        Args:
            scale_factor: Scale factor (2, 3, or 4)
            download_lr: Whether to download LR images (otherwise generated on-the-fly)
            
        Returns:
            True if successful, False otherwise
        """
        print(f"Preparing DIV2K dataset (scale factor: {scale_factor})")
        
        dataset_dir = self.data_root / f"DIV2K_x{scale_factor}"
        downloads_dir = self.data_root / "downloads"
        downloads_dir.mkdir(exist_ok=True)
        
        # Download HR images
        hr_files = ['div2k_train_hr', 'div2k_valid_hr']
        
        for file_key in hr_files:
            if file_key not in self.dataset_urls:
                print(f"❌ Unknown dataset: {file_key}")
                continue
            
            url = self.dataset_urls[file_key]
            filename = Path(urlparse(url).path).name
            archive_path = downloads_dir / filename
            
            # Download if not exists
            if not archive_path.exists():
                if not self.download_file(url, archive_path):
                    return False
            
            # Extract
            extract_dir = downloads_dir / filename.stem
            if not extract_dir.exists():
                if not self.extract_archive(archive_path, extract_dir):
                    return False
        
        # Download LR images if requested
        if download_lr:
            lr_files = [f'div2k_train_lr_bicubic_x{scale_factor}', 
                       f'div2k_valid_lr_bicubic_x{scale_factor}']
            
            for file_key in lr_files:
                if file_key not in self.dataset_urls:
                    print(f"❌ Unknown dataset: {file_key}")
                    continue
                
                url = self.dataset_urls[file_key]
                filename = Path(urlparse(url).path).name
                archive_path = downloads_dir / filename
                
                # Download if not exists
                if not archive_path.exists():
                    if not self.download_file(url, archive_path):
                        return False
                
                # Extract
                extract_dir = downloads_dir / filename.stem
                if not extract_dir.exists():
                    if not self.extract_archive(archive_path, extract_dir):
                        return False
        
        # Organize dataset
        train_hr_dir = downloads_dir / "DIV2K_train_HR" / "DIV2K_train_HR"
        valid_hr_dir = downloads_dir / "DIV2K_valid_HR" / "DIV2K_valid_HR"
        
        # Create organized structure
        organized_dir = dataset_dir / "organized"
        train_dir = organized_dir / "train" / "HR"
        val_dir = organized_dir / "val" / "HR"
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy training images
        if train_hr_dir.exists():
            for img_file in train_hr_dir.glob("*.png"):
                shutil.copy2(img_file, train_dir / img_file.name)
            print(f"✓ Copied {len(list(train_dir.glob('*.png')))} training images")
        
        # Copy validation images
        if valid_hr_dir.exists():
            for img_file in valid_hr_dir.glob("*.png"):
                shutil.copy2(img_file, val_dir / img_file.name)
            print(f"✓ Copied {len(list(val_dir.glob('*.png')))} validation images")
        
        # Handle LR images if downloaded
        if download_lr:
            train_lr_dir = downloads_dir / f"DIV2K_train_LR_bicubic_X{scale_factor}" / f"DIV2K_train_LR_bicubic" / f"X{scale_factor}"
            valid_lr_dir = downloads_dir / f"DIV2K_valid_LR_bicubic_X{scale_factor}" / f"DIV2K_valid_LR_bicubic" / f"X{scale_factor}"
            
            train_lr_dest = organized_dir / "train" / "LR"
            val_lr_dest = organized_dir / "val" / "LR"
            
            train_lr_dest.mkdir(parents=True, exist_ok=True)
            val_lr_dest.mkdir(parents=True, exist_ok=True)
            
            # Copy LR training images
            if train_lr_dir.exists():
                for img_file in train_lr_dir.glob("*.png"):
                    shutil.copy2(img_file, train_lr_dest / img_file.name)
                print(f"✓ Copied {len(list(train_lr_dest.glob('*.png')))} LR training images")
            
            # Copy LR validation images
            if valid_lr_dir.exists():
                for img_file in valid_lr_dir.glob("*.png"):
                    shutil.copy2(img_file, val_lr_dest / img_file.name)
                print(f"✓ Copied {len(list(val_lr_dest.glob('*.png')))} LR validation images")
        
        print(f"✓ DIV2K dataset prepared at: {organized_dir}")
        return True


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare datasets for RDN training")
    parser.add_argument('--data_root', default='data', help='Root directory for datasets')
    parser.add_argument('--dataset', choices=['div2k'], default='div2k', help='Dataset to prepare')
    parser.add_argument('--scale_factor', type=int, choices=[2, 3, 4], default=2, help='Scale factor for super-resolution')
    parser.add_argument('--download_lr', action='store_true', help='Download LR images (otherwise generated on-the-fly)')
    parser.add_argument('--organize', help='Organize custom dataset from directory')
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer(args.data_root)
    
    if args.organize:
        # Organize custom dataset
        source_dir = Path(args.organize)
        target_dir = preparer.data_root / "custom_dataset"
        
        if not source_dir.exists():
            print(f"❌ Source directory not found: {source_dir}")
            return
        
        success = preparer.organize_dataset(source_dir, target_dir)
        if success:
            print(f"✅ Custom dataset organized at: {target_dir}")
        else:
            print("❌ Failed to organize dataset")
    
    elif args.dataset == 'div2k':
        # Prepare DIV2K dataset
        success = preparer.prepare_div2k(args.scale_factor, args.download_lr)
        if success:
            print(f"✅ DIV2K dataset prepared for {args.scale_factor}x super-resolution")
        else:
            print("❌ Failed to prepare DIV2K dataset")


if __name__ == "__main__":
    main()
