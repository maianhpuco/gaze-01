#!/usr/bin/env python3
"""
Create comprehensive EGD-CXR visualization plots with real DICOM images
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
    print("⚠ PyDICOM not available - will use anatomical region images")

try:
    from PIL import Image
    PIL_AVAILABLE = True
    print("✓ PIL available for image processing")
except ImportError:
    PIL_AVAILABLE = False
    print("⚠ PIL not available")

# Set up paths
RAW_DATA_PATH = "/project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/egd-cxr/1.0.0"
PLOTS_DIR = "/project/hnguyen2/mvu9/folder_04_ma/gaze-01/plots"
SAMPLE_DIR = "/project/hnguyen2/mvu9/folder_04_ma/gaze-01/sample"

def create_plots_directory():
    """Create plots directory if it doesn't exist"""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"✓ Plots directory: {PLOTS_DIR}")

def load_dicom_image(dicom_path):
    """Load DICOM image and convert to displayable format"""
    try:
        if DICOM_AVAILABLE and os.path.exists(dicom_path):
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

def load_anatomical_images(dicom_id):
    """Load anatomical region images for a specific case"""
    audio_dir = f"{RAW_DATA_PATH}/audio_segmentation_transcripts/{dicom_id}"
    
    anatomical_images = {}
    if os.path.exists(audio_dir) and PIL_AVAILABLE:
        # List of anatomical regions we expect
        regions = ['aortic_knob', 'left_lung', 'right_lung', 'mediastanum']
        
        for region in regions:
            img_path = f"{audio_dir}/{region}.png"
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    # Convert to numpy array
                    img_array = np.array(img)
                    anatomical_images[region] = img_array
                    print(f"✓ Loaded {region} image: {img_array.shape}")
                except Exception as e:
                    print(f"⚠ Could not load {region} image: {e}")
    
    return anatomical_images

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
    """Select the first case (which we have DICOM for)"""
    print("\nSelecting example case...")
    
    # Use the first case which we have DICOM for
    selected_case = master_sheet.iloc[0]
    print(f"✓ Selected case: {selected_case['dicom_id']}")
    return selected_case

def get_diagnosis_info(case):
    """Extract diagnosis information from the case"""
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

def plot_1_anatomical_regions(case, anatomical_images, dicom_image=None):
    """Plot 1: Anatomical region overlay on DICOM image"""
    print("\nCreating Plot 1: Anatomical Regions Overlay...")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Use DICOM image if available
    if dicom_image is not None:
        img = dicom_image
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)  # Convert to grayscale
        ax.imshow(img, cmap='gray')
        title_suffix = " (Real DICOM Image)"
    else:
        # Create composite from anatomical images
        img = np.ones((1024, 1024, 3)) * 0.1
        ax.imshow(img)
        title_suffix = f" (Anatomical Regions: {len(anatomical_images)})"
    
    # Add anatomical region labels
    if anatomical_images:
        regions_layout = {
            'left_lung': (0.25, 0.25, 0.5, 0.5),
            'right_lung': (0.5, 0.25, 0.5, 0.5),
            'mediastanum': (0.375, 0.25, 0.25, 0.5),
            'aortic_knob': (0.4375, 0.25, 0.125, 0.125)
        }
        
        for region, (x, y, w, h) in regions_layout.items():
            if region in anatomical_images:
                # Convert to image coordinates
                img_h, img_w = img.shape[:2]
                x_pix = x * img_w
                y_pix = y * img_h
                w_pix = w * img_w
                h_pix = h * img_h
                
                # Draw rectangle
                rect = Rectangle((x_pix, y_pix), w_pix, h_pix,
                               linewidth=3, edgecolor='red', facecolor='none', alpha=0.8)
                ax.add_patch(rect)
                
                # Add label
                ax.text(x_pix, y_pix-20, region.replace('_', ' ').title(), 
                       fontsize=12, color='red', fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_title(f'Anatomical Regions Overlay - {case["dicom_id"]}{title_suffix}', 
                fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Save plot
    plot_path = f"{PLOTS_DIR}/{case['dicom_id']}_anatomical_regions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()

def plot_2_bounding_boxes(case, bboxes, dicom_image=None):
    """Plot 2: Bounding boxes overlay on DICOM image"""
    print("\nCreating Plot 2: Bounding Boxes Overlay...")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Use DICOM image if available
    if dicom_image is not None:
        img = dicom_image
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)  # Convert to grayscale
        ax.imshow(img, cmap='gray')
        title_suffix = " (Real DICOM Image)"
    else:
        # Create mock image
        img = np.ones((1024, 1024, 3)) * 0.1
        ax.imshow(img)
        title_suffix = " (Mock Image)"
    
    # Draw bounding boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(bboxes)))
    for i, (_, bbox) in enumerate(bboxes.iterrows()):
        x1, x2, y1, y2 = bbox['x1'], bbox['x2'], bbox['y1'], bbox['y2']
        width = x2 - x1
        height = y2 - y1
        
        # Scale coordinates to image dimensions
        img_h, img_w = img.shape[:2]
        x1_scaled = x1 * img_w / 2363  # Assuming max width from data
        y1_scaled = y1 * img_h / 2363  # Assuming max height from data
        width_scaled = width * img_w / 2363
        height_scaled = height * img_h / 2363
        
        rect = Rectangle((x1_scaled, y1_scaled), width_scaled, height_scaled,
                        linewidth=2, edgecolor=colors[i], facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        
        # Add label
        ax.text(x1_scaled, y1_scaled-10, bbox['bbox_name'], 
                fontsize=8, color=colors[i], fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_title(f'Bounding Boxes Overlay - {case["dicom_id"]}{title_suffix}', 
                fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Save plot
    plot_path = f"{PLOTS_DIR}/{case['dicom_id']}_bounding_boxes.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()

def plot_3_fixation_analysis(case, gaze_data, dicom_image=None):
    """Plot 3: Fixation points with duration and transition lines"""
    print("\nCreating Plot 3: Fixation Analysis...")
    
    if len(gaze_data) == 0:
        print("⚠ No gaze data available for this case")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Use DICOM image if available
    if dicom_image is not None:
        img = dicom_image
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)  # Convert to grayscale
        ax.imshow(img, cmap='gray')
        title_suffix = " (Real DICOM Image)"
    else:
        # Create mock image
        img = np.ones((1024, 1024, 3)) * 0.1
        ax.imshow(img)
        title_suffix = " (Mock Image)"
    
    # Get image dimensions
    img_h, img_w = img.shape[:2]
    
    # Convert normalized coordinates to image coordinates
    gaze_x = gaze_data['FPOGX'] * img_w
    gaze_y = gaze_data['FPOGY'] * img_h
    
    # Draw transition lines between consecutive fixations
    for i in range(len(gaze_data) - 1):
        x1, y1 = gaze_x.iloc[i], gaze_y.iloc[i]
        x2, y2 = gaze_x.iloc[i + 1], gaze_y.iloc[i + 1]
        ax.plot([x1, x2], [y1, y2], 'b-', alpha=0.3, linewidth=1)
    
    # Draw fixation points with size based on duration
    scatter = ax.scatter(gaze_x, gaze_y, c=gaze_data['FPOGD'], 
                        cmap='hot', s=gaze_data['FPOGD'] * 1000, 
                        alpha=0.8, edgecolors='white', linewidth=1)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Fixation Duration (seconds)', fontsize=12)
    
    # Add start and end markers
    if len(gaze_data) > 0:
        ax.scatter(gaze_x.iloc[0], gaze_y.iloc[0], c='green', s=200, 
                  marker='o', label='Start', edgecolors='white', linewidth=2)
        ax.scatter(gaze_x.iloc[-1], gaze_y.iloc[-1], c='red', s=200, 
                  marker='s', label='End', edgecolors='white', linewidth=2)
    
    ax.set_title(f'Fixation Analysis - {case["dicom_id"]}{title_suffix}\n'
                f'Total Fixations: {len(gaze_data)}, Duration: {gaze_data["Time (in secs)"].max():.1f}s', 
                fontsize=16, fontweight='bold')
    ax.legend()
    ax.axis('off')
    
    # Save plot
    plot_path = f"{PLOTS_DIR}/{case['dicom_id']}_fixation_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()

def plot_4_comprehensive_info(case, bboxes, gaze_data, audio_info, dicom_image=None):
    """Plot 4: Comprehensive information panel"""
    print("\nCreating Plot 4: Comprehensive Information Panel...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Main image subplot (larger)
    ax_main = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=3)
    
    # Use DICOM image if available
    if dicom_image is not None:
        img = dicom_image
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)  # Convert to grayscale
        ax_main.imshow(img, cmap='gray')
        title_suffix = " (Real DICOM Image)"
    else:
        anatomical_images = load_anatomical_images(case['dicom_id'])
        img = np.ones((1024, 1024, 3)) * 0.1
        ax_main.imshow(img)
        title_suffix = f" (Anatomical Regions: {len(anatomical_images)})"
    
    # Draw bounding boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(bboxes)))
    for i, (_, bbox) in enumerate(bboxes.iterrows()):
        x1, x2, y1, y2 = bbox['x1'], bbox['x2'], bbox['y1'], bbox['y2']
        width = x2 - x1
        height = y2 - y1
        
        # Scale coordinates to image dimensions
        img_h, img_w = img.shape[:2]
        x1_scaled = x1 * img_w / 2363
        y1_scaled = y1 * img_h / 2363
        width_scaled = width * img_w / 2363
        height_scaled = height * img_h / 2363
        
        rect = Rectangle((x1_scaled, y1_scaled), width_scaled, height_scaled,
                        linewidth=2, edgecolor=colors[i], facecolor='none', alpha=0.8)
        ax_main.add_patch(rect)
    
    # Draw gaze data
    if len(gaze_data) > 0:
        gaze_x = gaze_data['FPOGX'] * img_w
        gaze_y = gaze_data['FPOGY'] * img_h
        
        scatter = ax_main.scatter(gaze_x, gaze_y, c=gaze_data['FPOGD'], 
                                cmap='hot', alpha=0.7, s=30, 
                                label='Eye Fixations (duration)', edgecolors='white', linewidth=0.5)
        
        cbar = plt.colorbar(scatter, ax=ax_main, shrink=0.8)
        cbar.set_label('Fixation Duration (seconds)', fontsize=10)
    
    ax_main.set_title(f'Comprehensive Analysis - {case["dicom_id"]}{title_suffix}', 
                     fontsize=16, fontweight='bold')
    ax_main.legend(loc='upper right')
    ax_main.axis('off')
    
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
    Dimensions: {img.shape[1]} x {img.shape[0]}
    Type: {'Real DICOM' if dicom_image is not None else 'Composite'}
    Top Pad: {case['image_top_pad']}
    Bottom Pad: {case['image_bottom_pad']}
    Left Pad: {case['image_left_pad']}
    Right Pad: {case['image_right_pad']}
    """
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    # Clinical Findings subplot
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
    
    # Gaze Analysis subplot
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
        Anatomical region images: left_lung.png, right_lung.png, mediastinum.png, aortic_knob.png
        
        Transcript structure: {list(audio_info.keys()) if isinstance(audio_info, dict) else 'Available'}
        
        Note: This visualization combines all available data modalities for comprehensive analysis.
        """
    else:
        audio_text = "No audio transcript available for this case."
    
    ax_audio.text(0.05, 0.5, audio_text, transform=ax_audio.transAxes, 
                 fontsize=11, verticalalignment='center', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('EGD-CXR Comprehensive Analysis', fontsize=18, fontweight='bold', y=0.98)
    
    # Save plot
    plot_path = f"{PLOTS_DIR}/{case['dicom_id']}_comprehensive_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()

def main():
    """Main function"""
    print("=" * 60)
    print("Comprehensive EGD-CXR Visualization with Real DICOM")
    print("=" * 60)
    
    try:
        # Create plots directory
        create_plots_directory()
        
        # Load data
        master_sheet, bounding_boxes, fixations = load_data()
        
        # Select example case
        case = select_example_case(master_sheet)
        
        # Get related data
        bboxes = get_bounding_boxes_for_case(case['dicom_id'], bounding_boxes)
        gaze_data = get_gaze_data_for_case(case['dicom_id'], fixations)
        audio_info = get_audio_info(case['dicom_id'])
        anatomical_images = load_anatomical_images(case['dicom_id'])
        
        print(f"\nData Summary:")
        print(f"• Bounding boxes: {len(bboxes)}")
        print(f"• Gaze fixations: {len(gaze_data)}")
        print(f"• Audio available: {'Yes' if audio_info else 'No'}")
        print(f"• Anatomical images: {len(anatomical_images)}")
        
        # Try to load DICOM image from sample directory
        dicom_path = f"{SAMPLE_DIR}/{case['dicom_id']}.dcm"
        dicom_image, dicom = load_dicom_image(dicom_path)
        
        if dicom_image is not None:
            print(f"✓ Successfully loaded DICOM image: {dicom_path}")
            if dicom:
                print(f"  - Patient ID: {getattr(dicom, 'PatientID', 'N/A')}")
                print(f"  - Study Date: {getattr(dicom, 'StudyDate', 'N/A')}")
                print(f"  - Image size: {dicom.pixel_array.shape}")
        else:
            print(f"⚠ DICOM image not found: {dicom_path}")
            print("Using anatomical region images for visualization")
        
        # Create all plots
        plot_1_anatomical_regions(case, anatomical_images, dicom_image)
        plot_2_bounding_boxes(case, bboxes, dicom_image)
        plot_3_fixation_analysis(case, gaze_data, dicom_image)
        plot_4_comprehensive_info(case, bboxes, gaze_data, audio_info, dicom_image)
        
        print("\n" + "=" * 60)
        print("All visualizations completed successfully!")
        print(f"Plots saved in: {PLOTS_DIR}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
