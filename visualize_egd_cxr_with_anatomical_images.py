#!/usr/bin/env python3
"""
EGD-CXR Dataset Visualization with Real Anatomical Images
This script creates a composite chest X-ray visualization using the actual anatomical region images
from the audio_segmentation_transcripts directories
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

# Try to import image processing libraries
try:
    from PIL import Image
    PIL_AVAILABLE = True
    print("✓ PIL available for image processing")
except ImportError:
    PIL_AVAILABLE = False
    print("⚠ PIL not available")

# Set up paths
RAW_DATA_PATH = "/project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/egd-cxr/1.0.0"

def load_anatomical_images(dicom_id):
    """Load anatomical region images for a specific case"""
    audio_dir = f"{RAW_DATA_PATH}/audio_segmentation_transcripts/{dicom_id}"
    
    anatomical_images = {}
    if os.path.exists(audio_dir):
        # List of anatomical regions we expect
        regions = ['aortic_knob', 'left_lung', 'right_lung', 'mediastanum']
        
        for region in regions:
            img_path = f"{audio_dir}/{region}.png"
            if os.path.exists(img_path) and PIL_AVAILABLE:
                try:
                    img = Image.open(img_path)
                    # Convert to numpy array
                    img_array = np.array(img)
                    anatomical_images[region] = img_array
                    print(f"✓ Loaded {region} image: {img_array.shape}")
                except Exception as e:
                    print(f"⚠ Could not load {region} image: {e}")
            else:
                print(f"⚠ {region} image not found or PIL not available")
    
    return anatomical_images

def create_composite_chest_xray(anatomical_images, width=1024, height=1024):
    """Create a composite chest X-ray from anatomical region images"""
    # Create base image
    composite = np.ones((height, width, 3)) * 0.1  # Dark background
    
    if not anatomical_images:
        # If no anatomical images, create a simple mock
        return create_simple_mock_xray(width, height)
    
    # Define regions for placing anatomical images
    regions_layout = {
        'left_lung': (width//4, height//4, width//2, height//2),
        'right_lung': (width//2, height//4, width//2, height//2),
        'mediastanum': (width//2-width//8, height//4, width//4, height//2),
        'aortic_knob': (width//2-width//16, height//4, width//8, height//8)
    }
    
    for region, img_array in anatomical_images.items():
        if region in regions_layout:
            x, y, w, h = regions_layout[region]
            
            # Resize image to fit the region
            if len(img_array.shape) == 3:
                # Color image
                img_resized = Image.fromarray(img_array).resize((w, h))
                img_resized = np.array(img_resized)
                
                # Blend with composite
                if img_resized.shape[2] == 4:  # RGBA
                    alpha = img_resized[:, :, 3:4] / 255.0
                    rgb = img_resized[:, :, :3] / 255.0
                    composite[y:y+h, x:x+w] = (1 - alpha) * composite[y:y+h, x:x+w] + alpha * rgb
                else:  # RGB
                    composite[y:y+h, x:x+w] = img_resized / 255.0
            else:
                # Grayscale image
                img_resized = Image.fromarray(img_array).resize((w, h))
                img_resized = np.array(img_resized)
                gray = img_resized / 255.0
                composite[y:y+h, x:x+w] = np.stack([gray, gray, gray], axis=2)
    
    return composite

def create_simple_mock_xray(width=1024, height=1024):
    """Create a simple mock chest X-ray when no anatomical images are available"""
    img = np.ones((height, width, 3)) * 0.1
    
    # Add some structure
    y_center = height // 2
    x_center = width // 2
    
    # Lungs (darker areas)
    left_lung = np.zeros((height//2, width//4, 3))
    left_lung[:, :, :] = 0.2
    img[y_center-height//4:y_center+height//4, width//4:width//2] = left_lung
    
    right_lung = np.zeros((height//2, width//4, 3))
    right_lung[:, :, :] = 0.2
    img[y_center-height//4:y_center+height//4, width//2:3*width//4] = right_lung
    
    # Heart shadow (brighter)
    heart = np.ones((height//3, width//4, 3)) * 0.4
    img[y_center-height//6:y_center+height//6, width//2-width//8:width//2+width//8] = heart
    
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
    """Create comprehensive visualization with real anatomical images"""
    print("\nCreating visualization...")
    
    # Load anatomical images
    anatomical_images = load_anatomical_images(case['dicom_id'])
    
    # Create composite chest X-ray
    img = create_composite_chest_xray(anatomical_images)
    img_height, img_width = img.shape[:2]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Main image subplot (larger)
    ax_main = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=3)
    
    # Display the composite image
    ax_main.imshow(img, aspect='equal')
    
    title = f"Chest X-ray Analysis: {case['dicom_id']}"
    if anatomical_images:
        title += f" (Real Anatomical Images: {len(anatomical_images)} regions)"
    else:
        title += " (Mock Image)"
    
    ax_main.set_title(title, fontsize=16, fontweight='bold')
    
    # Draw bounding boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(bboxes)))
    for i, (_, bbox) in enumerate(bboxes.iterrows()):
        x1, x2, y1, y2 = bbox['x1'], bbox['x2'], bbox['y1'], bbox['y2']
        width = x2 - x1
        height = y2 - y1
        
        # Scale coordinates to image dimensions
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
                                cmap='hot', alpha=0.8, s=40, 
                                label='Eye Fixations (duration)', edgecolors='white', linewidth=0.5)
        
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
    Anatomical Images: {len(anatomical_images)}
    Available Regions: {', '.join(anatomical_images.keys()) if anatomical_images else 'None'}
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
        Anatomical region images: {', '.join(anatomical_images.keys()) if anatomical_images else 'None available'}
        
        Note: This visualization uses the actual anatomical region images from the audio transcript directory.
        """
    else:
        audio_text = "No audio transcript available for this case."
    
    ax_audio.text(0.05, 0.5, audio_text, transform=ax_audio.transAxes, 
                 fontsize=11, verticalalignment='center', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('EGD-CXR Dataset: Complete Case Analysis with Real Anatomical Images', fontsize=18, fontweight='bold', y=0.98)
    
    return fig

def main():
    """Main function"""
    print("=" * 60)
    print("EGD-CXR Dataset Visualization with Real Anatomical Images")
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
        output_path = "/project/hnguyen2/mvu9/folder_04_ma/gaze-01/egd_cxr_anatomical_images_visualization.png"
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
