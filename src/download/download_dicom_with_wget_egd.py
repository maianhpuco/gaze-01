#!/usr/bin/env python3
"""
Download all MIMIC-CXR DICOM files for EGD-CXR dataset using wget with authentication
Downloads all DICOM files listed in the master_sheet.csv with progress tracking
"""

import os
import subprocess
import pandas as pd
import time
from pathlib import Path

# Configuration
RAW_DATA_PATH = "/project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/egd-cxr/1.0.0"
OUTPUT_DIR = "/project/hnguyen2/mvu9/datasets/gaze_data/egd-cxr/dicom_raw"
USERNAME = "hiirooo"
# Use interactive password entry via --ask-password per user request
# Note: Do NOT hardcode passwords in source code
BASE_URL_ROOT = "https://physionet.org/files/mimic-cxr/2.0.0/"

def download_dicom_with_wget(dicom_id, dicom_path, output_dir, username, timeout=300):
    """Download a single DICOM file using wget"""
    # master_sheet.csv paths usually start with 'files/...'. Build correct URL under BASE_URL_ROOT
    normalized_path = dicom_path.lstrip('/')
    if normalized_path.startswith('files/'):
        normalized_path = normalized_path[len('files/'):]
    url = BASE_URL_ROOT + normalized_path
    output_path = os.path.join(output_dir, f"{dicom_id}.dcm")
    
    # Skip if file already exists
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True, "already_exists"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct wget command
    cmd = [
        'wget',
        '--user', username,
        '--ask-password',
        '--no-check-certificate',
        '--timeout=60',
        '--tries=3',
        '--continue',
        '--output-document', output_path,
        url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        if result.returncode == 0:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size = os.path.getsize(output_path) / (1024*1024)
                return True, f"downloaded ({file_size:.2f} MB)"
            else:
                return False, "file_not_created"
        else:
            return False, f"wget_error: {result.stderr.strip()}"
            
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, f"error: {str(e)}"

def download_all_egd_cxr_dicom_files():
    """Download all DICOM files for EGD-CXR dataset"""
    print("=" * 80)
    print("EGD-CXR DICOM File Downloader - Complete Dataset")
    print("=" * 80)
    
    # Load master sheet
    master_sheet_path = os.path.join(RAW_DATA_PATH, "master_sheet.csv")
    if not os.path.exists(master_sheet_path):
        print(f"✗ Master sheet not found: {master_sheet_path}")
        return False
    
    master_sheet = pd.read_csv(master_sheet_path)
    total_cases = len(master_sheet)
    
    print(f"📊 Dataset Information:")
    print(f"  - Total cases: {total_cases}")
    print(f"  - Output directory: {OUTPUT_DIR}")
    print(f"  - Base URL root: {BASE_URL_ROOT}")
    print(f"  - Username: {USERNAME}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Statistics tracking
    successful_downloads = 0
    already_existing = 0
    failed_downloads = 0
    error_summary = {}
    
    print(f"\n🚀 Starting download process...")
    print(f"⏱️  Estimated time: {total_cases * 2 / 60:.1f} minutes (assuming 2s per file)")
    print("-" * 80)
    
    start_time = time.time()
    
    for idx, (_, case) in enumerate(master_sheet.iterrows(), 1):
        dicom_id = case['dicom_id']
        dicom_path = case['path']
        
        # Calculate progress
        progress = (idx / total_cases) * 100
        
        # Download the file
        success, message = download_dicom_with_wget(
            dicom_id, dicom_path, OUTPUT_DIR, USERNAME
        )
        
        # Update statistics
        if success:
            if message == "already_exists":
                already_existing += 1
                status = "✓ EXISTS"
            else:
                successful_downloads += 1
                status = "✓ DOWNLOADED"
        else:
            failed_downloads += 1
            status = "✗ FAILED"
            
            # Track error types
            error_type = message.split(':')[0] if ':' in message else message
            error_summary[error_type] = error_summary.get(error_type, 0) + 1
        
        # Print progress
        elapsed_time = time.time() - start_time
        avg_time_per_file = elapsed_time / idx
        remaining_files = total_cases - idx
        eta_seconds = remaining_files * avg_time_per_file
        eta_minutes = eta_seconds / 60
        
        print(f"[{idx:4d}/{total_cases}] {progress:5.1f}% | {status} | {dicom_id[:20]}... | "
              f"ETA: {eta_minutes:.1f}m | {message}")
        
        # Add delay to be respectful to the server
        if success and message != "already_exists":
            time.sleep(1)
    
    # Final statistics
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("📈 DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"✅ Successful downloads: {successful_downloads}")
    print(f"📁 Already existing: {already_existing}")
    print(f"❌ Failed downloads: {failed_downloads}")
    print(f"📊 Total processed: {total_cases}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📁 Files saved to: {OUTPUT_DIR}")
    
    if error_summary:
        print(f"\n❌ Error Summary:")
        for error_type, count in error_summary.items():
            print(f"  - {error_type}: {count} files")
    
    # Calculate success rate
    success_rate = ((successful_downloads + already_existing) / total_cases) * 100
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    return successful_downloads > 0 or already_existing > 0

def verify_downloaded_files():
    """Verify downloaded DICOM files"""
    print("\n" + "=" * 80)
    print("🔍 VERIFYING DOWNLOADED FILES")
    print("=" * 80)
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"✗ Output directory not found: {OUTPUT_DIR}")
        return
    
    # Count DICOM files
    dcm_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.dcm')]
    total_files = len(dcm_files)
    
    print(f"📁 Found {total_files} DICOM files in {OUTPUT_DIR}")
    
    if total_files == 0:
        print("⚠ No DICOM files found")
        return
    
    # Test loading a few files
    try:
        import pydicom
        print("✓ PyDICOM available for verification")
        
        # Test first 5 files
        test_files = dcm_files[:5]
        successful_loads = 0
        
        for dcm_file in test_files:
            dcm_path = os.path.join(OUTPUT_DIR, dcm_file)
            try:
                dicom = pydicom.dcmread(dcm_path)
                successful_loads += 1
                print(f"  ✓ {dcm_file}: {dicom.pixel_array.shape}")
            except Exception as e:
                print(f"  ✗ {dcm_file}: Error - {e}")
        
        print(f"\n📊 Verification Results:")
        print(f"  - Tested: {len(test_files)} files")
        print(f"  - Successfully loaded: {successful_loads}")
        print(f"  - Load success rate: {(successful_loads/len(test_files))*100:.1f}%")
        
    except ImportError:
        print("⚠ PyDICOM not available - cannot verify DICOM files")

def print_run_command():
    """Print the command to run this script"""
    print("\n" + "=" * 80)
    print("🚀 COMMAND TO RUN THIS SCRIPT")
    print("=" * 80)
    print("Using GPU job and conda environment:")
    print()
    print("srun --jobid=206025 bash -c \"cd /project/hnguyen2/mvu9/folder_04_ma/gaze-01 && /project/hnguyen2/mvu9/conda_envs/wsi-agent/bin/python src/download/download_dicom_with_wget_egd.py\"")
    print()
    print("Or run directly:")
    print()
    print("cd /project/hnguyen2/mvu9/folder_04_ma/gaze-01")
    print("python3 src/download/download_dicom_with_wget_egd.py")
    print("=" * 80)

def main():
    """Main function"""
    print_run_command()
    
    # Ask for confirmation
    response = input("\nDo you want to start downloading all EGD-CXR DICOM files? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    # Download all files
    success = download_all_egd_cxr_dicom_files()
    
    if success:
        # Verify downloaded files
        verify_downloaded_files()
    
    print("\n" + "=" * 80)
    print("🎉 DOWNLOAD PROCESS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
