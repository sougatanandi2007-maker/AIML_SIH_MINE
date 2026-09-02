"""
===============================================================================
Module: src/data/download_dataset.py
Project: Low-Cost Real-Time Mine Subsidence Monitoring & Early Warning System
===============================================================================

EDUCATIONAL OVERVIEW:
--------------------
In any machine learning and sensor pipeline, data ingestion is the very first step.
Rather than requiring users to manually visit a website, download a zip file, and extract it,
we use programmatic ingestion via `kagglehub`.

Key Concepts:
1. `kagglehub.dataset_download()`: Connects to the Kaggle API (publicly accessible for open datasets)
   and downloads the latest version of the dataset to a local cache directory.
2. File Discovery: We do NOT assume the filenames inside the dataset repository.
   We programmatically scan the downloaded folder, identify all CSV, JSON, and Parquet data files,
   and mirror or copy them into our project's local `data/raw/` directory.
3. Pathlib & Shutil: Standard Python utilities for cross-platform file system operations (Windows, Linux, macOS).

Inputs:
- dataset_handle (str): The Kaggle dataset identifier, e.g., 'ziya07/multimodal-sensor-fusion-dataset'
- target_dir (str): Destination folder inside our project repo (default: 'data/raw')

Outputs:
- List of Path objects pointing to the copied raw data files.
===============================================================================
"""

import os
import shutil
from pathlib import Path
import kagglehub
from dotenv import load_dotenv

# Load optional environment variables from .env file
load_dotenv()

# Default dataset handle for the multimodal sensor dataset
DEFAULT_DATASET_HANDLE = os.getenv(
    "KAGGLE_DATASET_HANDLE", "ziya07/multimodal-sensor-fusion-dataset"
)


def download_dataset(
    dataset_handle: str = DEFAULT_DATASET_HANDLE,
    target_dir: str = "data/raw"
) -> list[Path]:
    """
    Downloads a dataset from Kaggle using kagglehub and copies all data files
    to the project's raw data directory.

    Args:
        dataset_handle (str): Kaggle dataset slug in 'username/dataset-name' format.
        target_dir (str): Relative or absolute path to copy raw files into.

    Returns:
        list[Path]: List of paths to the downloaded data files in `target_dir`.
    """
    print(f"[INFO] Initiating download for Kaggle dataset: '{dataset_handle}'...")
    
    # 1. Download dataset using kagglehub
    # kagglehub caches datasets locally and returns the root path of the downloaded folder
    download_path = kagglehub.dataset_download(dataset_handle)
    source_path = Path(download_path)
    destination_path = Path(target_dir)
    
    # Ensure target directory exists
    destination_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Dataset successfully downloaded to cache: {source_path}")
    print(f"[INFO] Copying data files to project directory: {destination_path.resolve()}...")

    # 2. Discover all files recursively in the downloaded cache
    discovered_files = list(source_path.rglob("*"))
    copied_files: list[Path] = []

    for item in discovered_files:
        if item.is_file():
            # Relative path from cache root to preserve subfolder structures if any
            relative_item_path = item.relative_to(source_path)
            dest_file_path = destination_path / relative_item_path
            
            # Ensure parent directories exist
            dest_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(item, dest_file_path)
            copied_files.append(dest_file_path)
            print(f"  -> Discovered & Copied: {relative_item_path.name} ({item.stat().st_size / 1024:.2f} KB)")

    print(f"[SUCCESS] Download completed. Total data files stored: {len(copied_files)}")
    return copied_files


if __name__ == "__main__":
    files = download_dataset()
    print("\nFiles available in data/raw/:")
    for f in files:
        print(f" - {f.resolve()}")
