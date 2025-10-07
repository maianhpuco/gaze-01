#!/usr/bin/env python3
"""
Detailed EGD-CXR Case Analysis
Provides comprehensive analysis of a specific case with all available information
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# Set up paths
RAW_DATA_PATH = "/project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/egd-cxr/1.0.0"

def analyze_case(dicom_id):
    """Perform detailed analysis of a specific case"""
    print(f"Detailed Analysis for Case: {dicom_id}")
    print("=" * 80)
    
    # Load data
    master_sheet = pd.read_csv(f"{RAW_DATA_PATH}/master_sheet.csv")
    bounding_boxes = pd.read_csv(f"{RAW_DATA_PATH}/bounding_boxes.csv")
    fixations = pd.read_csv(f"{RAW_DATA_PATH}/fixations.csv")
    
    # Get case data
    case = master_sheet[master_sheet['dicom_id'] == dicom_id].iloc[0]
    case_bboxes = bounding_boxes[bounding_boxes['dicom_id'] == dicom_id]
    case_gaze = fixations[fixations['DICOM_ID'] == dicom_id]
    
    print(f"\n1. PATIENT & STUDY INFORMATION")
    print("-" * 40)
    print(f"DICOM ID: {case['dicom_id']}")
    print(f"Patient ID: {case['patient_id']}")
    print(f"Study ID: {case['study_id']}")
    print(f"Stay ID: {case['stay_id']}")
    print(f"Gender: {case['gender']}")
    print(f"Age: {case['anchor_age']}")
    print(f"Image Path: {case['path']}")
    
    print(f"\n2. IMAGE PROCESSING INFORMATION")
    print("-" * 40)
    print(f"Top Padding: {case['image_top_pad']} pixels")
    print(f"Bottom Padding: {case['image_bottom_pad']} pixels")
    print(f"Left Padding: {case['image_left_pad']} pixels")
    print(f"Right Padding: {case['image_right_pad']} pixels")
    
    print(f"\n3. CLINICAL DIAGNOSES")
    print("-" * 40)
    # Primary diagnoses
    diagnoses = []
    for i in range(1, 10):
        dx_col = f'dx{i}'
        icd_col = f'dx{i}_icd'
        if pd.notna(case[dx_col]) and case[dx_col] != '':
            diagnoses.append(f"{case[dx_col]} (ICD-10: {case[icd_col]})")
    
    for i, dx in enumerate(diagnoses, 1):
        print(f"Diagnosis {i}: {dx}")
    
    print(f"\n4. CLINICAL FINDINGS (Binary Labels)")
    print("-" * 40)
    findings = {
        'Normal': case['Normal'],
        'CHF (Congestive Heart Failure)': case['CHF'],
        'Pneumonia': case['pneumonia'],
        'Consolidation': case['consolidation'],
        'Enlarged Cardiac Silhouette': case['enlarged_cardiac_silhouette'],
        'Linear/Patchy Atelectasis': case['linear__patchy_atelectasis'],
        'Lobar/Segmental Collapse': case['lobar__segmental_collapse'],
        'Pleural Effusion/Thickening': case['pleural_effusion_or_thickening'],
        'Pulmonary Edema': case['pulmonary_edema__hazy_opacity'],
        'Normal Anatomically': case['normal_anatomically'],
        'Elevated Hemidiaphragm': case['elevated_hemidiaphragm'],
        'Hyperaeration': case['hyperaeration'],
        'Vascular Redistribution': case['vascular_redistribution']
    }
    
    positive_findings = [finding for finding, value in findings.items() if value == 1]
    negative_findings = [finding for finding, value in findings.items() if value == 0]
    
    print("Positive Findings:")
    for finding in positive_findings:
        print(f"  ✓ {finding}")
    
    print("\nNegative Findings:")
    for finding in negative_findings:
        print(f"  ✗ {finding}")
    
    print(f"\n5. CHEXPERT LABELS")
    print("-" * 40)
    chexpert_labels = {
        'Atelectasis': case['atelectasis__chx'],
        'Cardiomegaly': case['cardiomegaly__chx'],
        'Consolidation': case['consolidation__chx'],
        'Edema': case['edema__chx'],
        'Enlarged Cardiomediastinum': case['enlarged_cardiomediastinum__chx'],
        'Fracture': case['fracture__chx'],
        'Lung Lesion': case['lung_lesion__chx'],
        'Lung Opacity': case['lung_opacity__chx'],
        'No Finding': case['no_finding__chx'],
        'Pleural Effusion': case['pleural_effusion__chx'],
        'Pleural Other': case['pleural_other__chx'],
        'Pneumonia': case['pneumonia__chx'],
        'Pneumothorax': case['pneumothorax__chx'],
        'Support Devices': case['support_devices__chx']
    }
    
    for label, value in chexpert_labels.items():
        if pd.notna(value):
            status = {-1: "Uncertain", 0: "Negative", 1: "Positive"}.get(value, "Unknown")
            print(f"  {label}: {status}")
    
    print(f"\n6. EXAMINATION INDICATION")
    print("-" * 40)
    print(f"Clinical Indication: {case['cxr_exam_indication']}")
    
    print(f"\n7. ANATOMICAL BOUNDING BOXES")
    print("-" * 40)
    print(f"Total anatomical regions: {len(case_bboxes)}")
    print("\nAnatomical Regions:")
    for _, bbox in case_bboxes.iterrows():
        print(f"  • {bbox['bbox_name']}: ({bbox['x1']:.0f}, {bbox['y1']:.0f}) to ({bbox['x2']:.0f}, {bbox['y2']:.0f})")
    
    print(f"\n8. EYE GAZE ANALYSIS")
    print("-" * 40)
    if len(case_gaze) > 0:
        print(f"Total fixations: {len(case_gaze)}")
        print(f"Session duration: {case_gaze['Time (in secs)'].max():.2f} seconds")
        print(f"Average fixation duration: {case_gaze['FPOGD'].mean():.3f} seconds")
        print(f"Maximum fixation duration: {case_gaze['FPOGD'].max():.3f} seconds")
        print(f"Minimum fixation duration: {case_gaze['FPOGD'].min():.3f} seconds")
        
        # Gaze distribution
        left_gaze = len(case_gaze[case_gaze['FPOGX'] < 0.5])
        right_gaze = len(case_gaze[case_gaze['FPOGX'] >= 0.5])
        upper_gaze = len(case_gaze[case_gaze['FPOGY'] < 0.5])
        lower_gaze = len(case_gaze[case_gaze['FPOGY'] >= 0.5])
        
        print(f"\nGaze Distribution:")
        print(f"  Left side: {left_gaze} fixations ({left_gaze/len(case_gaze)*100:.1f}%)")
        print(f"  Right side: {right_gaze} fixations ({right_gaze/len(case_gaze)*100:.1f}%)")
        print(f"  Upper half: {upper_gaze} fixations ({upper_gaze/len(case_gaze)*100:.1f}%)")
        print(f"  Lower half: {lower_gaze} fixations ({lower_gaze/len(case_gaze)*100:.1f}%)")
        
        # Top fixation areas
        print(f"\nTop 5 Longest Fixations:")
        top_fixations = case_gaze.nlargest(5, 'FPOGD')[['FPOGX', 'FPOGY', 'FPOGD', 'Time (in secs)']]
        for i, (_, fixation) in enumerate(top_fixations.iterrows(), 1):
            print(f"  {i}. Duration: {fixation['FPOGD']:.3f}s, Position: ({fixation['FPOGX']:.3f}, {fixation['FPOGY']:.3f}), Time: {fixation['Time (in secs)']:.2f}s")
    else:
        print("No gaze data available for this case.")
    
    print(f"\n9. AUDIO TRANSCRIPT INFORMATION")
    print("-" * 40)
    audio_dir = f"{RAW_DATA_PATH}/audio_segmentation_transcripts/{dicom_id}"
    
    if os.path.exists(audio_dir):
        print(f"Audio directory exists: {audio_dir}")
        
        # List available files
        audio_files = list(Path(audio_dir).glob("*"))
        print(f"Available files ({len(audio_files)}):")
        for file in audio_files:
            file_type = file.suffix
            if file_type == '.mp3':
                print(f"  • Audio file: {file.name}")
            elif file_type == '.wav':
                print(f"  • Audio file: {file.name}")
            elif file_type == '.json':
                print(f"  • Transcript: {file.name}")
            elif file_type == '.png':
                print(f"  • Anatomical image: {file.name}")
            elif file_type == '.html':
                print(f"  • Study info: {file.name}")
            else:
                print(f"  • {file.name}")
        
        # Try to load transcript
        transcript_file = f"{audio_dir}/transcript.json"
        if os.path.exists(transcript_file):
            try:
                with open(transcript_file, 'r') as f:
                    transcript_data = json.load(f)
                print(f"\nTranscript structure: {list(transcript_data.keys()) if isinstance(transcript_data, dict) else 'List with items'}")
            except Exception as e:
                print(f"Could not load transcript: {e}")
    else:
        print("No audio transcript directory found for this case.")
    
    print(f"\n10. SUMMARY")
    print("-" * 40)
    print(f"This case represents a {'normal' if case['Normal'] == 1 else 'abnormal'} chest X-ray")
    if positive_findings:
        print(f"Key findings: {', '.join(positive_findings)}")
    print(f"Radiologist spent {case_gaze['Time (in secs)'].max():.1f} seconds analyzing the image")
    print(f"Focus areas: {len(case_bboxes)} anatomical regions identified")
    print(f"Audio recording: {'Available' if os.path.exists(audio_dir) else 'Not available'}")

def main():
    """Main function"""
    # Use the same case from the visualization
    case_id = "24c7496c-d7635dfe-b8e0b87f-d818affc-78ff7cf4"
    analyze_case(case_id)

if __name__ == "__main__":
    main()
