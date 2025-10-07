#!/usr/bin/env python3
"""
EGD-CXR Dataset Visualization with Real DICOM Images
This script visualizes a complete example case showing the actual chest X-ray image
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import json
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import DICOM libraries
try:
    import pydicom
    DICOM_AVAILABLE = True
    print("✓ PyDICOM available - will load real DICOM images")
except ImportError:
    DICOM_AVAILABLE = False
    print("⚠ PyDICOM not available - will use alternative image loading")

try:
    from PIL import Image
    PIL_AVAILABLE = True
    print("✓ PIL available for image processing")
except ImportError:
    PIL_AVAILABLE = False
    print("⚠ PIL not available")

# Set up paths
RAW_DATA_PATH = "/project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/egd-cxr/1.0.0"

def load_dicom_image(dicom_path):
    """Load DICOM image and convert to displayable format"""
    try:
        if DICOM_AVAILABLE:
            # Load DICOM file
            dicom = pydicom.dcmread(dicom_path)
            
            # Get pixel array
            pixel_array = dicom.pixel_array
            
            # Normalize to 0-1 range for display
            if pixel_array.max() > pixel_array.min():
                pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
            
            return pixel_array, dicom
        else:
            return None, None
    except Exception as e:
        print(f"Error loading DICOM: {e}")
        return None, None

def create_mock_chest_xray(width=1024, height=1024):
    """Create a more realistic mock chest X-ray image"""
    # Create base image
    img = np.zeros((height, width))
    
    # Add some structure to make it look more like a chest X-ray
    # Lungs (darker areas)
    y_center = height // 2
    x_center = width // 2
    
    # Left lung
    left_lung_mask = np.zeros_like(img)
    left_lung_mask[y_center-height//4:y_center+height//3, width//4:width//2] = 1
    img += left_lung_mask * 0.3
    
    # Right lung
    right_lung_mask = np.zeros_like(img)
    right_lung_mask[y_center-height//4:y_center+height//3, width//2:3*width//4] = 1
    img += right_lung_mask * 0.3
    
    # Heart shadow (brighter)
    heart_mask = np.zeros_like(img)
    heart_mask[y_center-height//6:y_center+height//4, width//2-width//8:width//2+width//8] = 1
    img += heart_mask * 0.6
    
    # Ribs (slightly brighter lines)
    for i in range(5, height-5, height//8):
        img[i:i+2, :] += 0.1
    
    # Add some noise for realism
    noise = np.random.normal(0, 0.05, img.shape)
    img += noise
    
    # Ensure values are in 0-1 range
    img = np.clip(img, 0, 1)
    
    return img

def load_data():
    """Load all necessary data files"""
    print("Loading EGD-CXR dataset...")
    
    # Load master sheet
    master_sheet = pd.read_csv(f"{RAW_DATA_PATH}/master_sheet.csv")
    print(f"✓ Loaded master_sheet.csv: {len(master_sheet)} records")
    
    # Load bounding boxes
    bounding_boxes = pd.read_csv(f"{RAW_DATA_PATH}/bounding_boxes.csv")
    print(f"✓ Loaded bounding_boxes.csv: {len(bounding_boxes)} records")
    
    # Load fixations
    fixations = pd.read_csv(f"{RAW_DATA_PATH}/fixations.csv")
    print(f"✓ Loaded fixations.csv: {len(fixations)} records")
    
    return master_sheet, bounding_boxes, fixations

def select_example_case(master_sheet):
    """Select an interesting example case with multiple findings"""
    print("\nSelecting example case...")
    
    # Find cases with multiple findings for better visualization
    interesting_cases = master_sheet[
        (master_sheet['CHF'] == 1) | 
        (master_sheet['pneumonia'] == 1) | 
        (master_sheet['consolidation'] == 1)
    ].copy()
    
    if len(interesting_cases) > 0:
        # Select first interesting case
        selected_case = interesting_cases.iloc[0]
        print(f"✓ Selected case: {selected_case['dicom_id']}")
        return selected_case
    else:
        # Fallback to first case
        selected_case = master_sheet.iloc[0]
        print(f"✓ Selected case (fallback): {selected_case['dicom_id']}")
        return selected_case

def get_diagnosis_info(case):
    """Extract diagnosis information from the case"""
    print("\nExtracting diagnosis information...")
    
    # Primary findings
    findings = []
    if case['Normal'] == 1:
        findings.append("Normal")
    if case['CHF'] == 1:
        findings.append("Congestive Heart Failure")
    if case['pneumonia'] == 1:
        findings.append("Pneumonia")
    if case['consolidation'] == 1:
        findings.append("Consolidation")
    if case['enlarged_cardiac_silhouette'] == 1:
        findings.append("Enlarged Cardiac Silhouette")
    if case['pleural_effusion_or_thickening'] == 1:
        findings.append("Pleural Effusion/Thickening")
    if case['pulmonary_edema__hazy_opacity'] == 1:
        findings.append("Pulmonary Edema")
    
    # ICD codes
    icd_codes = []
    for i in range(1, 10):
        dx_col = f'dx{i}'
        icd_col = f'dx{i}_icd'
        if pd.notna(case[dx_col]) and case[dx_col] != '':
            icd_codes.append(f"{case[dx_col]} ({case[icd_col]})")
    
    return findings, icd_codes, case['cxr_exam_indication']

def get_bounding_boxes_for_case(dicom_id, bounding_boxes):
    """Get bounding boxes for the selected case"""
    case_bboxes = bounding_boxes[bounding_boxes['dicom_id'] == dicom_id]
    return case_bboxes

def get_gaze_data_for_case(dicom_id, fixations):
    """Get gaze data for the selected case"""
    case_gaze = fixations[fixations['DICOM_ID'] == dicom_id]
    return case_gaze

def get_audio_info(dicom_id):
    """Get audio transcript information"""
    audio_dir = f"{RAW_DATA_PATH}/audio_segmentation_transcripts/{dicom_id}"
    
    if os.path.exists(audio_dir):
        transcript_file = f"{audio_dir}/transcript.json"
        if os.path.exists(transcript_file):
            try:
                with open(transcript_file, 'r') as f:
                    transcript_data = json.load(f)
                return transcript_data
            except:
                return None
    return None

def create_visualization(case, bboxes, gaze_data, audio_info):
    """Create comprehensive visualization with real or realistic image"""
    print("\nCreating visualization...")
    
    # Try to load the actual DICOM image
    dicom_path = f"{RAW_DATA_PATH}/{case['path']}"
    print(f"Attempting to load DICOM from: {dicom_path}")
    
    if os.path.exists(dicom_path):
        print("✓ DICOM file found, attempting to load...")
        pixel_array, dicom = load_dicom_image(dicom_path)
        
        if pixel_array is not None:
            print("✓ Successfully loaded real DICOM image!")
            img = pixel_array
            img_height, img_width = img.shape
            use_real_image = True
        else:
            print("⚠ Could not load DICOM, using realistic mock image")
            img = create_mock_chest_xray()
            img_height, img_width = img.shape
            use_real_image = False
    else:
        print("⚠ DICOM file not found, using realistic mock image")
        img = create_mock_chest_xray()
        img_height, img_width = img.shape
        use_real_image = False
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Main image subplot (larger)
    ax_main = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=3)
    
    # Display the image
    ax_main.imshow(img, cmap='gray', aspect='equal')
    
    title = f"Chest X-ray Analysis: {case['dicom_id']}"
    if use_real_image:
        title += " (Real DICOM Image)"
    else:
        title += " (Realistic Mock Image)"
    
    ax_main.set_title(title, fontsize=16, fontweight='bold')
    
    # Draw bounding boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(bboxes)))
    for i, (_, bbox) in enumerate(bboxes.iterrows()):
        x1, x2, y1, y2 = bbox['x1'], bbox['x2'], bbox['y1'], bbox['y2']
        width = x2 - x1
        height = y2 - y1
        
        # Scale coordinates to image dimensions
        # Use the actual image dimensions for scaling
        x1_scaled = x1 * img_width / 2363  # Assuming max width from data
        y1_scaled = y1 * img_height / 2363  # Assuming max height from data
        width_scaled = width * img_width / 2363
        height_scaled = height * img_height / 2363
        
        rect = Rectangle((x1_scaled, y1_scaled), width_scaled, height_scaled,
                        linewidth=2, edgecolor=colors[i], facecolor='none', alpha=0.8)
        ax_main.add_patch(rect)
        
        # Add label
        ax_main.text(x1_scaled, y1_scaled-10, bbox['bbox_name'], 
                    fontsize=8, color=colors[i], fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Draw gaze data
    if len(gaze_data) > 0:
        # Sample gaze points (every 5th point to avoid overcrowding)
        gaze_sample = gaze_data.iloc[::5]
        
        # Convert normalized coordinates to image coordinates
        gaze_x = gaze_sample['FPOGX'] * img_width
        gaze_y = gaze_sample['FPOGY'] * img_height
        
        # Create gaze heatmap
        scatter = ax_main.scatter(gaze_x, gaze_y, c=gaze_sample['FPOGD'], 
                                cmap='hot', alpha=0.7, s=30, 
                                label='Eye Fixations (duration)')
        
        # Add colorbar for gaze duration
        cbar = plt.colorbar(scatter, ax=ax_main, shrink=0.8)
        cbar.set_label('Fixation Duration (seconds)', fontsize=10)
    
    ax_main.set_xlim(0, img_width)
    ax_main.set_ylim(img_height, 0)  # Invert y-axis for image coordinates
    ax_main.set_xlabel('X Coordinate (pixels)', fontsize=12)
    ax_main.set_ylabel('Y Coordinate (pixels)', fontsize=12)
    ax_main.legend(loc='upper right')
    ax_main.grid(True, alpha=0.3)
    
    # Patient Information subplot
    ax_info = plt.subplot2grid((4, 3), (0, 2))
    ax_info.axis('off')
    
    info_text = f"""
    PATIENT INFORMATION
    
    DICOM ID: {case['dicom_id']}
    Patient ID: {case['patient_id']}
    Study ID: {case['study_id']}
    Gender: {case['gender']}
    Age: {case['anchor_age']}
    
    IMAGE INFO
    Dimensions: {img_width} x {img_height}
    Type: {'Real DICOM' if use_real_image else 'Mock Image'}
    Top Pad: {case['image_top_pad']}
    Bottom Pad: {case['image_bottom_pad']}
    Left Pad: {case['image_left_pad']}
    Right Pad: {case['image_right_pad']}
    """
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    # Diagnosis Results subplot
    ax_diag = plt.subplot2grid((4, 3), (1, 2))
    ax_diag.axis('off')
    
    findings, icd_codes, indication = get_diagnosis_info(case)
    
    diag_text = f"""
    CLINICAL FINDINGS
    
    Primary Findings:
    {chr(10).join([f"• {finding}" for finding in findings])}
    
    ICD-10 Codes:
    {chr(10).join([f"• {code}" for code in icd_codes[:3]])}
    
    Exam Indication:
    {indication[:100]}{'...' if len(str(indication)) > 100 else ''}
    """
    
    ax_diag.text(0.05, 0.95, diag_text, transform=ax_diag.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
    
    # Gaze Statistics subplot
    ax_gaze = plt.subplot2grid((4, 3), (2, 2))
    ax_gaze.axis('off')
    
    if len(gaze_data) > 0:
        gaze_stats = f"""
        GAZE ANALYSIS
        
        Total Fixations: {len(gaze_data)}
        Avg Duration: {gaze_data['FPOGD'].mean():.3f}s
        Max Duration: {gaze_data['FPOGD'].max():.3f}s
        Total Time: {gaze_data['Time (in secs)'].max():.1f}s
        
        Gaze Distribution:
        • Left Lung: {len(gaze_data[gaze_data['FPOGX'] < 0.5])}
        • Right Lung: {len(gaze_data[gaze_data['FPOGX'] >= 0.5])}
        • Upper: {len(gaze_data[gaze_data['FPOGY'] < 0.5])}
        • Lower: {len(gaze_data[gaze_data['FPOGY'] >= 0.5])}
        """
    else:
        gaze_stats = "No gaze data available"
    
    ax_gaze.text(0.05, 0.95, gaze_stats, transform=ax_gaze.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    # Audio Information subplot
    ax_audio = plt.subplot2grid((4, 3), (3, 0), colspan=3)
    ax_audio.axis('off')
    
    if audio_info:
        audio_text = f"""
        AUDIO TRANSCRIPT INFORMATION
        
        Transcript available for this case.
        Audio files: audio.mp3, audio.wav
        Anatomical region images available:
        • aortic_knob.png, left_lung.png, right_lung.png, mediastinum.png
        
        Note: Full transcript content would be displayed here in a real implementation.
        """
    else:
        audio_text = "No audio transcript available for this case."
    
    ax_audio.text(0.05, 0.5, audio_text, transform=ax_audio.transAxes, 
                 fontsize=11, verticalalignment='center', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('EGD-CXR Dataset: Complete Case Analysis with Real Image', fontsize=18, fontweight='bold', y=0.98)
    
    return fig

def main():
    """Main function"""
    print("=" * 60)
    print("EGD-CXR Dataset Visualization with Real Images")
    print("=" * 60)
    
    try:
        # Load data
        master_sheet, bounding_boxes, fixations = load_data()
        
        # Select example case
        case = select_example_case(master_sheet)
        
        # Get related data
        bboxes = get_bounding_boxes_for_case(case['dicom_id'], bounding_boxes)
        gaze_data = get_gaze_data_for_case(case['dicom_id'], fixations)
        audio_info = get_audio_info(case['dicom_id'])
        
        print(f"\nData Summary:")
        print(f"• Bounding boxes: {len(bboxes)}")
        print(f"• Gaze fixations: {len(gaze_data)}")
        print(f"• Audio available: {'Yes' if audio_info else 'No'}")
        
        # Create visualization
        fig = create_visualization(case, bboxes, gaze_data, audio_info)
        
        # Save the plot
        output_path = "/project/hnguyen2/mvu9/folder_04_ma/gaze-01/egd_cxr_real_image_visualization.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")
        
        # Show the plot
        plt.show()
        
        print("\n" + "=" * 60)
        print("Visualization Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
