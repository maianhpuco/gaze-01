#!/usr/bin/env python3
"""
EGD-CXR Dataset Visualization Example
This script visualizes a complete example case showing:
- Chest X-ray image
- Anatomical bounding boxes
- Eye gaze data
- Audio transcript information
- Clinical diagnosis results
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

# Set up paths
RAW_DATA_PATH = "/project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/egd-cxr/1.0.0"
PROCESSED_DATA_PATH = "/project/hnguyen2/mvu9/processing_datasets/processing_gaze/egd-cxr"

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
    """Create comprehensive visualization"""
    print("\nCreating visualization...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Main image subplot (larger)
    ax_main = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=3)
    
    # Create a mock chest X-ray background (since we don't have actual DICOM loading)
    # In a real implementation, you would load the actual DICOM image here
    img_width, img_height = 1024, 1024  # Typical CXR dimensions
    ax_main.imshow(np.random.rand(img_height, img_width) * 0.3, cmap='gray', alpha=0.7)
    ax_main.set_title(f"Chest X-ray Analysis: {case['dicom_id']}", fontsize=16, fontweight='bold')
    
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
        # Sample gaze points (every 10th point to avoid overcrowding)
        gaze_sample = gaze_data.iloc[::10]
        
        # Convert normalized coordinates to image coordinates
        gaze_x = gaze_sample['FPOGX'] * img_width
        gaze_y = gaze_sample['FPOGY'] * img_height
        
        # Create gaze heatmap
        ax_main.scatter(gaze_x, gaze_y, c=gaze_sample['FPOGD'], 
                       cmap='hot', alpha=0.6, s=20, 
                       label='Eye Fixations (duration)')
        
        # Add colorbar for gaze duration
        cbar = plt.colorbar(ax_main.collections[0], ax=ax_main, shrink=0.8)
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
    plt.suptitle('EGD-CXR Dataset: Complete Case Analysis', fontsize=18, fontweight='bold', y=0.98)
    
    return fig

def main():
    """Main function"""
    print("=" * 60)
    print("EGD-CXR Dataset Visualization Example")
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
        output_path = "/project/hnguyen2/mvu9/folder_04_ma/gaze-01/egd_cxr_example_visualization.png"
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
