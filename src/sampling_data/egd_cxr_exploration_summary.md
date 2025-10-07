# EGD-CXR Dataset Exploration Summary

## Dataset Overview
The EGD-CXR (Eye Gaze Dataset for Chest X-Ray) dataset is a comprehensive collection of eye tracking data from radiologists examining chest X-ray images.

**Dataset Path:** `/project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/egd-cxr/1.0.0`

## Dataset Structure

### Main Files
1. **`master_sheet.csv`** (392KB) - 1,083 records, 59 columns
   - Contains metadata for each chest X-ray study
   - Key columns: `dicom_id`, `path`, `study_id`, `patient_id`, `gender`, `anchor_age`
   - Clinical labels: `Normal`, `CHF`, `pneumonia`, `consolidation`, etc.
   - Anatomical findings: `enlarged_cardiac_silhouette`, `pleural_effusion_or_thickening`, etc.

2. **`eye_gaze.csv`** (406MB) - Large file with eye tracking data
   - 37 columns including gaze coordinates, pupil data, and timestamps
   - Key columns: `SESSION_ID`, `MEDIA_ID`, `DICOM_ID`, `Time (in secs)`
   - Gaze coordinates: `FPOGX`, `FPOGY` (fixation point), `BPOGX`, `BPOGY` (binocular point)
   - Pupil data: `LPCX`, `LPCY`, `RPCX`, `RPCY` (left/right pupil centers)
   - Validity flags: `FPOGV`, `BPOGV`, `LPV`, `RPV`

3. **`fixations.csv`** (15MB) - Processed fixation data
   - Same structure as eye_gaze.csv but contains only fixation events
   - Includes fixation duration (`FPOGD`) and start time (`FPOGS`)

4. **`bounding_boxes.csv`** (1.7MB) - Anatomical region annotations
   - 6 columns: `dicom_id`, `bbox_name`, `x1`, `x2`, `y1`, `y2`
   - Contains bounding boxes for anatomical structures like:
     - Cardiac silhouette
     - Left/right clavicle
     - Costophrenic angles
     - Lungs

### Subdirectories

1. **`audio_segmentation_transcripts/`** (1,084 subdirectories)
   - Each subdirectory contains:
     - `index.html` - Study information
     - `audio.mp3` & `audio.wav` - Audio recordings
     - `transcript.json` - Text transcripts
     - Anatomical region images: `aortic_knob.png`, `left_lung.png`, `right_lung.png`, `mediastanum.png`

2. **`inclusion_exclusion_criteria_outputs/`**
   - `CHF.csv` (493 records) - Congestive Heart Failure cases
   - `normals.csv` (7MB) - Normal cases
   - `pneumonia.csv` (995KB) - Pneumonia cases

## Data Characteristics

### Clinical Labels
- **Normal cases:** 1,000+ records
- **CHF (Congestive Heart Failure):** 493 records
- **Pneumonia:** ~1,000 records
- **Other conditions:** Various cardiac and pulmonary conditions

### Eye Tracking Data
- **Sampling rate:** High-frequency eye tracking (appears to be ~60Hz based on timestamps)
- **Coordinate system:** Normalized coordinates (0-1 range)
- **Data quality:** Includes validity flags for data quality assessment
- **Temporal resolution:** Sub-second precision in timestamps

### Anatomical Annotations
- Bounding boxes for key anatomical structures
- Multiple anatomical regions per image
- Coordinate system matches the eye tracking data

## Key Insights

1. **Rich multimodal data:** Combines eye tracking, audio transcripts, and clinical annotations
2. **Large scale:** Over 1,000 chest X-ray studies with complete eye tracking data
3. **Clinical relevance:** Focused on chest X-ray interpretation with specific clinical conditions
4. **High temporal resolution:** Detailed eye movement patterns during image interpretation
5. **Structured annotations:** Well-organized anatomical region bounding boxes

## Potential Use Cases

1. **Radiology education:** Understanding expert gaze patterns
2. **AI model training:** Using expert attention for model development
3. **Clinical decision support:** Analyzing diagnostic patterns
4. **Human-computer interaction:** Improving radiology workstation design
5. **Medical image analysis:** Attention-based deep learning models

## File Sizes and Scale
- **Total dataset size:** ~3.69 GB
- **Total files:** 8,665 files
- **Main data files:** 4 large CSV files with comprehensive tracking data
- **Audio/visual data:** 1,083+ audio recordings with transcripts
- **Anatomical images:** 4,332 PNG files showing anatomical regions

This dataset provides a unique opportunity to study expert radiologist behavior during chest X-ray interpretation, combining high-resolution eye tracking data with clinical annotations and audio transcripts.
