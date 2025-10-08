#!/usr/bin/env python3
"""
Download a single MIMIC-CXR DICOM file for testing using urllib
"""

import os
import urllib.request
import urllib.error
import pandas as pd
from pathlib import Path

def download_single_dicom():
    """Download a single DICOM file for testing"""
    print("=" * 50)
    print("Downloading Single MIMIC-CXR DICOM File")
    print("=" * 50)
    
    # Get the first case from EGD-CXR dataset
    master_sheet = pd.read_csv("/project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/egd-cxr/1.0.0/master_sheet.csv")
    first_case = master_sheet.iloc[0]
    
    dicom_id = first_case['dicom_id']
    dicom_path = first_case['path']
    
    print(f"DICOM ID: {dicom_id}")
    print(f"Path: {dicom_path}")
    
    # Create sample directory
    sample_dir = "sample"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Construct URL
    base_url = "https://physionet.org/files/mimic-cxr/2.0.0/files/"
    url = base_url + dicom_path
    output_path = os.path.join(sample_dir, f"{dicom_id}.dcm")
    
    print(f"Download URL: {url}")
    print(f"Output path: {output_path}")
    
    try:
        print("Starting download...")
        
        # Create request with headers
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36')
        
        with urllib.request.urlopen(req, timeout=60) as response:
            # Get file size if available
            file_size = response.headers.get('Content-Length')
            if file_size:
                file_size = int(file_size)
                print(f"File size: {file_size / (1024*1024):.2f} MB")
            
            # Download file
            with open(output_path, 'wb') as f:
                downloaded = 0
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if file_size:
                        progress = (downloaded / file_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✓ Successfully downloaded: {output_path}")
        if os.path.exists(output_path):
            print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
        return output_path
        
    except urllib.error.URLError as e:
        print(f"\n✗ URL Error: {e}")
        return None
    except urllib.error.HTTPError as e:
        print(f"\n✗ HTTP Error: {e.code} - {e.reason}")
        return None
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return None

def test_dicom_loading(dicom_path):
    """Test loading the downloaded DICOM file"""
    if not dicom_path or not os.path.exists(dicom_path):
        print("No DICOM file to test")
        return
    
    print("\n" + "=" * 50)
    print("Testing DICOM File Loading")
    print("=" * 50)
    
    try:
        import pydicom
        print("✓ PyDICOM available")
        
        dicom = pydicom.dcmread(dicom_path)
        print(f"✓ Successfully loaded DICOM file")
        print(f"  - Patient ID: {getattr(dicom, 'PatientID', 'N/A')}")
        print(f"  - Study Date: {getattr(dicom, 'StudyDate', 'N/A')}")
        print(f"  - Image size: {dicom.pixel_array.shape}")
        print(f"  - Pixel spacing: {getattr(dicom, 'PixelSpacing', 'N/A')}")
        
        return dicom
        
    except ImportError:
        print("⚠ PyDICOM not available - cannot test DICOM loading")
        return None
    except Exception as e:
        print(f"✗ Error loading DICOM: {e}")
        return None

def main():
    """Main function"""
    # Download single DICOM file
    dicom_path = download_single_dicom()
    
    # Test loading the DICOM file
    if dicom_path:
        test_dicom_loading(dicom_path)
    
    print("\n" + "=" * 50)
    print("Download Complete")
    print("=" * 50)

if __name__ == "__main__":
    main()
