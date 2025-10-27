#!/usr/bin/env python3
"""
Download MIMIC-CXR DICOM files using wget with authentication.

Updates:
- Fix 404s by constructing URLs from the dataset root (avoid duplicate 'files/').
- Prompt ONCE for password and pass via --password to avoid per-file prompts.
"""

import os
import subprocess
import pandas as pd
import time
import getpass

def download_dicom_with_wget(dicom_id, dicom_path, output_dir, username, password):
    """Download a single DICOM file using wget"""
    # Build URL from dataset root: master_sheet paths already begin with 'files/...'
    # Correct URL is: https://physionet.org/files/mimic-cxr/2.0.0/ + dicom_path
    base_url_root = "https://physionet.org/files/mimic-cxr/2.0.0/"
    url = base_url_root + dicom_path.lstrip('/')
    output_path = os.path.join(output_dir, f"{dicom_id}.dcm")
    
    print(f"Downloading {dicom_id}...")
    print(f"URL: {url}")
    print(f"Output: {output_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct wget command
    cmd = [
        'wget',
        '--user', username,
        '--password', password,
        '--no-check-certificate',
        '--timeout=60',
        '--tries=3',
        '--continue',
        '--output-document', output_path,
        url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✓ Successfully downloaded: {output_path}")
                print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
                return True
            else:
                print(f"✗ Download failed: File not created or empty")
                return False
        else:
            print(f"✗ Download failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Download timeout")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def download_sample_dicom_files():
    """Download sample DICOM files for testing"""
    print("=" * 60)
    print("MIMIC-CXR DICOM File Downloader with wget")
    print("=" * 60)
    
    # Credentials (password will be requested once interactively)
    username = "hiirooo"
    password = os.environ.get('PHYSIONET_PASSWORD') or getpass.getpass("PhysioNet password: ")
    
    # Get ALL cases from EGD-CXR dataset
    master_sheet = pd.read_csv("/project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/egd-cxr/1.0.0/master_sheet.csv")
    total_cases = len(master_sheet)
    print(f"Downloading full dataset: {total_cases} DICOM files...")
    
    # Create output directory (save all to central dicom_raw)
    output_dir = "/project/hnguyen2/mvu9/datasets/gaze_data/egd-cxr/dicom_raw"
    os.makedirs(output_dir, exist_ok=True)
    
    successful_downloads = 0
    already_existing = 0
    start_time = time.time()
    
    for idx, (_, case) in enumerate(master_sheet.iterrows(), start=1):
        dicom_id = case['dicom_id']
        dicom_path = case['path']
        output_path = os.path.join(output_dir, f"{dicom_id}.dcm")
        progress_pct = (idx / total_cases) * 100.0
        elapsed = time.time() - start_time
        avg_per_item = elapsed / idx if idx > 0 else 0
        remaining = total_cases - idx
        eta_min = (remaining * avg_per_item) / 60.0 if avg_per_item > 0 else 0.0
        
        # Skip if file already exists
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            already_existing += 1
            print(f"[{idx:4d}/{total_cases}] {progress_pct:5.1f}% | ✓ EXISTS      | {dicom_id[:24]}... | ETA: {eta_min:5.1f}m")
            continue
        
        if download_dicom_with_wget(dicom_id, dicom_path, output_dir, username, password):
            successful_downloads += 1
            print(f"[{idx:4d}/{total_cases}] {progress_pct:5.1f}% | ✓ DOWNLOADED | {dicom_id[:24]}... | ETA: {eta_min:5.1f}m")
        else:
            print(f"[{idx:4d}/{total_cases}] {progress_pct:5.1f}% | ✗ FAILED     | {dicom_id[:24]}... | ETA: {eta_min:5.1f}m")
        
        # Add delay between downloads
        time.sleep(2)
    
    print(f"\nDownload Summary:")
    print(f"✓ Successful downloads: {successful_downloads}")
    print(f"📁 Already existing: {already_existing}")
    print(f"✗ Failed downloads: {total_cases - successful_downloads - already_existing}")
    print(f"Files saved to: {output_dir}/")
    
    return successful_downloads > 0

def test_dicom_loading():
    """Test loading downloaded DICOM files"""
    print("\n" + "=" * 60)
    print("Testing DICOM File Loading")
    print("=" * 60)
    
    sample_dir = "sample"
    if not os.path.exists(sample_dir):
        print("No sample directory found")
        return
    
    dcm_files = [f for f in os.listdir(sample_dir) if f.endswith('.dcm')]
    
    if not dcm_files:
        print("No DICOM files found in sample directory")
        return
    
    print(f"Found {len(dcm_files)} DICOM files to test")
    
    try:
        import pydicom
        print("✓ PyDICOM available")
        
        for dcm_file in dcm_files:
            dcm_path = os.path.join(sample_dir, dcm_file)
            print(f"\nTesting: {dcm_file}")
            
            try:
                dicom = pydicom.dcmread(dcm_path)
                print(f"  ✓ Successfully loaded")
                print(f"  - Patient ID: {getattr(dicom, 'PatientID', 'N/A')}")
                print(f"  - Study Date: {getattr(dicom, 'StudyDate', 'N/A')}")
                print(f"  - Image size: {dicom.pixel_array.shape}")
                print(f"  - Pixel spacing: {getattr(dicom, 'PixelSpacing', 'N/A')}")
                
            except Exception as e:
                print(f"  ✗ Error loading: {e}")
                
    except ImportError:
        print("⚠ PyDICOM not available - cannot test DICOM loading")

def main():
    """Main function"""
    # Download sample DICOM files
    success = download_sample_dicom_files()
    
    if success:
        # Test loading the files
        test_dicom_loading()
    
    print("\n" + "=" * 60)
    print("Download Process Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
