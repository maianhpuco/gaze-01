#!/usr/bin/env python3
"""
Download MIMIC-CXR DICOM files using HTTP requests
This script downloads specific DICOM files needed for EGD-CXR visualization
"""

import os
import requests
import pandas as pd
from pathlib import Path
import time

# Set up paths
RAW_DATA_PATH = "/project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/egd-cxr/1.0.0"
MIMIC_DATA_PATH = "/project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/mimic-cxr/2.0.0/files"

def get_egd_cxr_cases():
    """Get DICOM IDs from EGD-CXR dataset"""
    master_sheet = pd.read_csv(f"{RAW_DATA_PATH}/master_sheet.csv")
    return master_sheet[['dicom_id', 'path']].head(5)  # Get first 5 cases for testing

def construct_dicom_url(dicom_path):
    """Construct URL for downloading DICOM file"""
    # MIMIC-CXR files are typically available via PhysioNet
    base_url = "https://physionet.org/files/mimic-cxr/2.0.0/files/"
    return base_url + dicom_path

def download_dicom_file(dicom_id, dicom_path, output_dir):
    """Download a single DICOM file"""
    url = construct_dicom_url(dicom_path)
    output_path = os.path.join(output_dir, f"{dicom_id}.dcm")
    
    print(f"Downloading {dicom_id} from {url}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download file
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Downloaded: {output_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to download {dicom_id}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error downloading {dicom_id}: {e}")
        return False

def download_sample_dicom_files():
    """Download sample DICOM files for testing"""
    print("=" * 60)
    print("MIMIC-CXR DICOM File Downloader")
    print("=" * 60)
    
    # Get sample cases
    cases = get_egd_cxr_cases()
    print(f"Found {len(cases)} cases to download")
    
    # Create output directory
    output_dir = f"{MIMIC_DATA_PATH}/downloaded_dicom"
    os.makedirs(output_dir, exist_ok=True)
    
    successful_downloads = 0
    
    for _, case in cases.iterrows():
        dicom_id = case['dicom_id']
        dicom_path = case['path']
        
        if download_dicom_file(dicom_id, dicom_path, output_dir):
            successful_downloads += 1
        
        # Add delay to be respectful to the server
        time.sleep(1)
    
    print(f"\nDownload Summary:")
    print(f"✓ Successful downloads: {successful_downloads}")
    print(f"✗ Failed downloads: {len(cases) - successful_downloads}")
    print(f"Files saved to: {output_dir}")

def check_existing_dicom_files():
    """Check if any DICOM files already exist"""
    print("Checking for existing DICOM files...")
    
    # Check in the main directory structure
    dcm_files = []
    for root, dirs, files in os.walk(MIMIC_DATA_PATH):
        for file in files:
            if file.endswith(('.dcm', '.DCM')):
                dcm_files.append(os.path.join(root, file))
    
    if dcm_files:
        print(f"Found {len(dcm_files)} existing DICOM files:")
        for dcm_file in dcm_files[:5]:  # Show first 5
            print(f"  - {dcm_file}")
        if len(dcm_files) > 5:
            print(f"  ... and {len(dcm_files) - 5} more")
    else:
        print("No existing DICOM files found")
    
    return dcm_files

def main():
    """Main function"""
    # Check for existing files first
    existing_files = check_existing_dicom_files()
    
    if not existing_files:
        print("\nNo DICOM files found. Attempting to download sample files...")
        print("Note: This requires internet access and may require PhysioNet credentials")
        
        # Ask user if they want to proceed
        response = input("Do you want to attempt downloading DICOM files? (y/n): ")
        if response.lower() == 'y':
            download_sample_dicom_files()
        else:
            print("Skipping download. Will use anatomical region images for visualization.")
    else:
        print(f"\nFound {len(existing_files)} existing DICOM files. No download needed.")

if __name__ == "__main__":
    main()
