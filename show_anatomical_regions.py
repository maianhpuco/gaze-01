#!/usr/bin/env python3
"""
Show Individual Anatomical Region Images
This script displays the individual anatomical region images for a specific case
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path

# Set up paths
RAW_DATA_PATH = "/project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/egd-cxr/1.0.0"

def show_anatomical_regions(dicom_id):
    """Display individual anatomical region images"""
    audio_dir = f"{RAW_DATA_PATH}/audio_segmentation_transcripts/{dicom_id}"
    
    if not os.path.exists(audio_dir):
        print(f"Audio directory not found: {audio_dir}")
        return
    
    # List of anatomical regions
    regions = ['aortic_knob', 'left_lung', 'right_lung', 'mediastanum']
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, region in enumerate(regions):
        img_path = f"{audio_dir}/{region}.png"
        
        if os.path.exists(img_path):
            try:
                # Load image
                img = Image.open(img_path)
                img_array = np.array(img)
                
                # Display image
                axes[i].imshow(img_array)
                axes[i].set_title(f"{region.replace('_', ' ').title()}\nShape: {img_array.shape}", 
                                fontsize=12, fontweight='bold')
                axes[i].axis('off')
                
                print(f"✓ {region}: {img_array.shape}")
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error loading\n{region}", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"{region} (Error)", fontsize=12)
                axes[i].axis('off')
                print(f"✗ {region}: Error - {e}")
        else:
            axes[i].text(0.5, 0.5, f"Not found\n{region}", 
                       ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f"{region} (Not Found)", fontsize=12)
            axes[i].axis('off')
            print(f"✗ {region}: File not found")
    
    plt.suptitle(f'Anatomical Region Images for Case: {dicom_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    output_path = f"/project/hnguyen2/mvu9/folder_04_ma/gaze-01/anatomical_regions_{dicom_id[:8]}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Anatomical regions saved to: {output_path}")
    
    plt.show()

def main():
    """Main function"""
    # Use the same case from the visualization
    case_id = "24c7496c-d7635dfe-b8e0b87f-d818affc-78ff7cf4"
    
    print("=" * 60)
    print("EGD-CXR Anatomical Region Images")
    print("=" * 60)
    
    show_anatomical_regions(case_id)

if __name__ == "__main__":
    main()
