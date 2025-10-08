#!/usr/bin/env python3
"""
Create EGD-CXR visualization plots with real DICOM images and anatomical overlays
Fixed version with proper DICOM display and anatomical mask overlays
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

# Import required libraries
try:
    import pydicom
    DICOM_AVAILABLE = True
    print("✓ PyDICOM available")
except ImportError:
    DICOM_AVAILABLE = False
    print("⚠ PyDICOM not available")

try:
    from PIL import Image
    PIL_AVAILABLE = True
    print("✓ PIL available")
except ImportError:
    PIL_AVAILABLE = False
    print("⚠ PIL not available")

# Set up paths
RAW_DATA_PATH = "/project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/egd-cxr/1.0.0"
BASE_PLOTS_DIR = "/project/hnguyen2/mvu9/folder_04_ma/gaze-01/plots"
SAMPLE_DIR = "/project/hnguyen2/mvu9/folder_04_ma/gaze-01/sample"

def create_case_directory(dicom_id):
    """Create a directory for the specific case"""
    case_dir = os.path.join(BASE_PLOTS_DIR, dicom_id)
    os.makedirs(case_dir, exist_ok=True)
    print(f"✓ Created case directory: {case_dir}")
    return case_dir

def load_dicom_image(dicom_path):
    """Load DICOM image and convert to displayable format with proper normalization"""
    try:
        if DICOM_AVAILABLE and os.path.exists(dicom_path):
            # Load DICOM file
            dicom = pydicom.dcmread(dicom_path)
            
            # Get pixel array
            pixel_array = dicom.pixel_array
            
            # Apply proper DICOM windowing for chest X-rays
            # Use default window center and width for chest X-rays
            window_center = getattr(dicom, 'WindowCenter', 40)
            window_width = getattr(dicom, 'WindowWidth', 400)
            
            # If window center/width are lists, take the first value
            if isinstance(window_center, list):
                window_center = window_center[0]
            if isinstance(window_width, list):
                window_width = window_width[0]
            
            # Apply windowing
            min_val = window_center - window_width // 2
            max_val = window_center + window_width // 2
            
            # Clip values
            pixel_array = np.clip(pixel_array, min_val, max_val)
            
            # Normalize to 0-1 range
            if max_val > min_val:
                pixel_array = (pixel_array - min_val) / (max_val - min_val)
            else:
                pixel_array = pixel_array / pixel_array.max()
            
            # Invert for chest X-ray display (darker = more dense)
            pixel_array = 1.0 - pixel_array
            
            print(f"✓ Loaded DICOM: {pixel_array.shape}, range: {pixel_array.min():.3f}-{pixel_array.max():.3f}")
            return pixel_array, dicom
        else:
            return None, None
    except Exception as e:
        print(f"Error loading DICOM: {e}")
        return None, None

def load_anatomical_masks(dicom_id):
    """Load anatomical region mask images for a specific case"""
    audio_dir = f"{RAW_DATA_PATH}/audio_segmentation_transcripts/{dicom_id}"
    
    anatomical_masks = {}
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
                    anatomical_masks[region] = img_array
                    print(f"✓ Loaded {region} mask: {img_array.shape}")
                except Exception as e:
                    print(f"⚠ Could not load {region} mask: {e}")
    
    return anatomical_masks

def overlay_anatomical_masks(dicom_image, anatomical_masks):
    """Overlay anatomical masks on the DICOM image"""
    if not anatomical_masks:
        return dicom_image
    
    # Create a copy of the DICOM image
    overlay_image = dicom_image.copy()
    
    # Define colors for different anatomical regions
    colors = {
        'left_lung': [1.0, 0.0, 0.0, 0.3],    # Red with transparency
        'right_lung': [0.0, 1.0, 0.0, 0.3],   # Green with transparency
        'mediastanum': [0.0, 0.0, 1.0, 0.3],  # Blue with transparency
        'aortic_knob': [1.0, 1.0, 0.0, 0.3]   # Yellow with transparency
    }
    
    # Convert grayscale DICOM to RGB for overlay
    if len(overlay_image.shape) == 2:
        overlay_image = np.stack([overlay_image, overlay_image, overlay_image], axis=2)
    
    # Overlay each anatomical mask
    for region, mask in anatomical_masks.items():
        if region in colors:
            color = colors[region]
            
            # Resize mask to match DICOM image dimensions
            mask_resized = Image.fromarray(mask).resize((overlay_image.shape[1], overlay_image.shape[0]))
            mask_array = np.array(mask_resized)
            
            # Convert mask to binary (assuming white areas are the region)
            if len(mask_array.shape) == 3:
                mask_binary = np.mean(mask_array, axis=2) > 128
            else:
                mask_binary = mask_array > 128
            
            # Apply color overlay
            for i in range(3):  # RGB channels
                overlay_image[:, :, i] = np.where(mask_binary, 
                                                overlay_image[:, :, i] * (1 - color[3]) + color[i] * color[3],
                                                overlay_image[:, :, i])
    
    return overlay_image

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

def get_diagnosis_info(case):
    """Extract diagnosis information from the case"""
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
    
    return findings, case['cxr_exam_indication']

def plot_1_anatomical_regions(case, dicom_image, anatomical_masks, case_dir):
    """Plot 1: Anatomical region overlay on DICOM image"""
    print("\nCreating Plot 1: Anatomical Regions Overlay...")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Overlay anatomical masks on DICOM image
    if dicom_image is not None and anatomical_masks:
        overlay_image = overlay_anatomical_masks(dicom_image, anatomical_masks)
        ax.imshow(overlay_image)
        title_suffix = f" (Real DICOM + {len(anatomical_masks)} Anatomical Masks)"
    elif dicom_image is not None:
        ax.imshow(dicom_image, cmap='gray')
        title_suffix = " (Real DICOM Image)"
    else:
        ax.text(0.5, 0.5, 'No DICOM image available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16)
        title_suffix = " (No Image)"
    
    # Add legend for anatomical regions
    if anatomical_masks:
        legend_elements = []
        colors = {
            'left_lung': 'red',
            'right_lung': 'green', 
            'mediastanum': 'blue',
            'aortic_knob': 'yellow'
        }
        
        for region in anatomical_masks.keys():
            if region in colors:
                legend_elements.append(plt.Line2D([0], [0], color=colors[region], 
                                                lw=4, label=region.replace('_', ' ').title()))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title(f'Anatomical Regions Overlay{title_suffix}', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Save plot
    plot_path = os.path.join(case_dir, "anatomical_regions.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()

def plot_2_bounding_boxes(case, bboxes, dicom_image, case_dir):
    """Plot 2: Bounding boxes overlay on DICOM image"""
    print("\nCreating Plot 2: Bounding Boxes Overlay...")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Display DICOM image
    if dicom_image is not None:
        ax.imshow(dicom_image, cmap='gray')
        title_suffix = " (Real DICOM Image)"
    else:
        ax.text(0.5, 0.5, 'No DICOM image available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16)
        title_suffix = " (No Image)"
    
    # Draw bounding boxes
    if len(bboxes) > 0 and dicom_image is not None:
        colors = plt.cm.Set3(np.linspace(0, 1, len(bboxes)))
        for i, (_, bbox) in enumerate(bboxes.iterrows()):
            x1, x2, y1, y2 = bbox['x1'], bbox['x2'], bbox['y1'], bbox['y2']
            width = x2 - x1
            height = y2 - y1
            
            # Scale coordinates to image dimensions
            img_h, img_w = dicom_image.shape[:2]
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
    
    ax.set_title(f'Bounding Boxes Overlay{title_suffix}', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Save plot
    plot_path = os.path.join(case_dir, "bounding_boxes.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()

def plot_3_fixation_analysis(case, gaze_data, dicom_image, case_dir):
    """Plot 3: Fixation points with duration and transition lines"""
    print("\nCreating Plot 3: Fixation Analysis...")
    
    if len(gaze_data) == 0:
        print("⚠ No gaze data available for this case")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Display DICOM image
    if dicom_image is not None:
        ax.imshow(dicom_image, cmap='gray')
        title_suffix = " (Real DICOM Image)"
    else:
        ax.text(0.5, 0.5, 'No DICOM image available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16)
        title_suffix = " (No Image)"
    
    # Draw gaze data
    if dicom_image is not None:
        img_h, img_w = dicom_image.shape[:2]
        
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
        
        ax.legend()
    
    ax.set_title(f'Fixation Analysis{title_suffix}\n'
                f'Total Fixations: {len(gaze_data)}, Duration: {gaze_data["Time (in secs)"].max():.1f}s', 
                fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Save plot
    plot_path = os.path.join(case_dir, "fixation_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()

def plot_4_comprehensive_info(case, bboxes, gaze_data, dicom_image, anatomical_masks, case_dir):
    """Plot 4: Comprehensive information panel"""
    print("\nCreating Plot 4: Comprehensive Information Panel...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Main image subplot (larger)
    ax_main = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=3)
    
    # Display DICOM image with overlays
    if dicom_image is not None:
        # Overlay anatomical masks if available
        if anatomical_masks:
            display_image = overlay_anatomical_masks(dicom_image, anatomical_masks)
        else:
            display_image = dicom_image
            if len(display_image.shape) == 2:
                display_image = np.stack([display_image, display_image, display_image], axis=2)
        
        ax_main.imshow(display_image)
        title_suffix = f" (Real DICOM + {len(anatomical_masks)} Masks)"
    else:
        ax_main.text(0.5, 0.5, 'No DICOM image available', ha='center', va='center', 
                    transform=ax_main.transAxes, fontsize=16)
        title_suffix = " (No Image)"
    
    # Draw bounding boxes
    if len(bboxes) > 0 and dicom_image is not None:
        colors = plt.cm.Set3(np.linspace(0, 1, len(bboxes)))
        for i, (_, bbox) in enumerate(bboxes.iterrows()):
            x1, x2, y1, y2 = bbox['x1'], bbox['x2'], bbox['y1'], bbox['y2']
            width = x2 - x1
            height = y2 - y1
            
            # Scale coordinates to image dimensions
            img_h, img_w = dicom_image.shape[:2]
            x1_scaled = x1 * img_w / 2363
            y1_scaled = y1 * img_h / 2363
            width_scaled = width * img_w / 2363
            height_scaled = height * img_h / 2363
            
            rect = Rectangle((x1_scaled, y1_scaled), width_scaled, height_scaled,
                            linewidth=2, edgecolor=colors[i], facecolor='none', alpha=0.8)
            ax_main.add_patch(rect)
    
    # Draw gaze data
    if len(gaze_data) > 0 and dicom_image is not None:
        gaze_x = gaze_data['FPOGX'] * dicom_image.shape[1]
        gaze_y = gaze_data['FPOGY'] * dicom_image.shape[0]
        
        scatter = ax_main.scatter(gaze_x, gaze_y, c=gaze_data['FPOGD'], 
                                cmap='hot', alpha=0.7, s=30, 
                                label='Eye Fixations (duration)', edgecolors='white', linewidth=0.5)
        
        cbar = plt.colorbar(scatter, ax=ax_main, shrink=0.8)
        cbar.set_label('Fixation Duration (seconds)', fontsize=10)
    
    ax_main.set_title(f'Comprehensive Analysis{title_suffix}', fontsize=16, fontweight='bold')
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
    Dimensions: {dicom_image.shape[1] if dicom_image is not None else 'N/A'} x {dicom_image.shape[0] if dicom_image is not None else 'N/A'}
    Type: {'Real DICOM' if dicom_image is not None else 'Not Available'}
    Anatomical Masks: {len(anatomical_masks)}
    """
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    # Clinical Findings subplot
    ax_diag = plt.subplot2grid((4, 3), (1, 2))
    ax_diag.axis('off')
    
    findings, indication = get_diagnosis_info(case)
    
    diag_text = f"""
    CLINICAL FINDINGS
    
    Primary Findings:
    {chr(10).join([f"• {finding}" for finding in findings])}
    
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
    
    # Summary subplot
    ax_summary = plt.subplot2grid((4, 3), (3, 0), colspan=3)
    ax_summary.axis('off')
    
    summary_text = f"""
    EGD-CXR COMPREHENSIVE ANALYSIS SUMMARY
    
    This visualization combines multiple data modalities:
    • Real DICOM chest X-ray image with proper windowing
    • Anatomical region masks overlaid with transparency
    • Bounding box annotations for anatomical structures
    • Eye gaze fixation data with duration-based sizing
    • Clinical findings and patient information
    
    Data Quality: {'High' if dicom_image is not None and len(anatomical_masks) > 0 else 'Partial'}
    """
    
    ax_summary.text(0.05, 0.5, summary_text, transform=ax_summary.transAxes, 
                   fontsize=11, verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('EGD-CXR Comprehensive Analysis', fontsize=18, fontweight='bold', y=0.98)
    
    # Save plot
    plot_path = os.path.join(case_dir, "comprehensive_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()

def main():
    """Main function"""
    print("=" * 60)
    print("EGD-CXR Visualization with Real DICOM and Anatomical Overlays")
    print("=" * 60)
    
    try:
        # Create base plots directory
        os.makedirs(BASE_PLOTS_DIR, exist_ok=True)
        
        # Load data
        master_sheet, bounding_boxes, fixations = load_data()
        
        # Select the first case (which we have DICOM for)
        case = master_sheet.iloc[0]
        dicom_id = case['dicom_id']
        
        print(f"\nProcessing case: {dicom_id}")
        
        # Create case-specific directory
        case_dir = create_case_directory(dicom_id)
        
        # Get related data
        bboxes = bounding_boxes[bounding_boxes['dicom_id'] == dicom_id]
        gaze_data = fixations[fixations['DICOM_ID'] == dicom_id]
        anatomical_masks = load_anatomical_masks(dicom_id)
        
        print(f"\nData Summary:")
        print(f"• Bounding boxes: {len(bboxes)}")
        print(f"• Gaze fixations: {len(gaze_data)}")
        print(f"• Anatomical masks: {len(anatomical_masks)}")
        
        # Load DICOM image
        dicom_path = f"{SAMPLE_DIR}/{dicom_id}.dcm"
        dicom_image, dicom = load_dicom_image(dicom_path)
        
        if dicom_image is not None:
            print(f"✓ Successfully loaded DICOM image: {dicom_path}")
            if dicom:
                print(f"  - Patient ID: {getattr(dicom, 'PatientID', 'N/A')}")
                print(f"  - Study Date: {getattr(dicom, 'StudyDate', 'N/A')}")
                print(f"  - Image size: {dicom.pixel_array.shape}")
        else:
            print(f"⚠ DICOM image not found: {dicom_path}")
        
        # Create all plots
        plot_1_anatomical_regions(case, dicom_image, anatomical_masks, case_dir)
        plot_2_bounding_boxes(case, bboxes, dicom_image, case_dir)
        plot_3_fixation_analysis(case, gaze_data, dicom_image, case_dir)
        plot_4_comprehensive_info(case, bboxes, gaze_data, dicom_image, anatomical_masks, case_dir)
        
        print("\n" + "=" * 60)
        print("All visualizations completed successfully!")
        print(f"Plots saved in: {case_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
