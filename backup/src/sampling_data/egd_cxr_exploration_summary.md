# EGD-CXR Dataset Exploration Summary

## Abstract
We created a rich multimodal dataset for the Chest X-Ray (CXR) domain. The data was collected using an eye tracking system while a radiologist interpreted and read 1,083 public CXR images. The dataset contains the following aligned modalities: image, transcribed report text, dictation audio and eye gaze data. We hope this dataset can contribute to various fields of research with applications in machine learning such as deep learning explainability, multi-modal fusion, disease classification, and automated radiology report generation to name a few. The images were selected from the MIMIC-CXR Database and were associated with studies from 1,038 subjects (female: 495, male: 543) who had age range 20 - 80 years old.

## Background
CXR is the most common imaging modality in the United States. It makes up to 74% of all imaging modalities ordered by physicians. In recent years with the proliferation of deep learning techniques and publicly available CXR datasets, numerous machine learning approaches have been proposed and deployed in radiology settings for disease detection. 

Eye tracking in radiology has been extensively studied for the purposes of education, perception understanding, fatigue measurement. More recently, efforts have shown use of eye gaze data to improve segmentation and disease classification in Computed Tomography (CT) radiography by combining them in deep learning techniques.

Currently, there is a lack of public datasets that capture eye gaze data in CXR space and given their promising utilization in machine learning, we are releasing the first of its kind dataset to the research community to explore and implement novel applications.

## Dataset Overview
The EGD-CXR (Eye Gaze Data for Chest X-Ray) dataset is a comprehensive collection of eye-tracking data from radiologists examining chest X-ray images, along with corresponding diagnostic labels and metadata. This is the first public dataset of its kind that captures eye gaze data in the CXR domain.

## Methods
The dataset was collected using an eye tracking system (GP3 HD Eye Tracker, Gazepoint). A radiologist, American Board of Radiology (ABR) certified, with 5 years of attending experience performed interpretation/reading on 1,083 CXR images. The analysis software (Gazepoint Analysis UX Edition) allowed for recording and exporting of eye gaze data and dictation audio.

To identify the images for this study we used MIMIC-CXR Database, which is a large public dataset containing CXR in conjunction with the MIMIC-IV Clinical Database that contains clinical outcomes. Inclusion and exclusion criteria were applied on the Emergency Department clinical notes from MIMIC-IV Clinical Database resulting in a subset of 1,083 cases covering equally 3 prominent clinical conditions (i.e. Normal, Pneumonia and Congestive Heart Failure (CHF)). The corresponding CXR images of these cases were extracted from the MIMIC-CXR database.

The radiologist performed radiology reading on these CXR images using Gazepoint's GP3 Eye Tracker, Gazepoint Analysis UX Edition software, a headset microphone, a PC computer and a monitor (Dell S2719DGF) set at 1920x1080 resolution. Radiology reading took place in multiple sessions (i.e. 30 cases per session) over a period of 2 months (i.e. March - May 2020). The Gazepoint Analysis UX Edition exported raw and processed eye gaze fixations (.csv format) and voice dictation (audio) of radiologist's reading. The audio files were further processed with speech-to-text software (i.e. Google Speech-to-Text API) to extract text transcripts along with dictation word time-related information (.json format). Furthermore, these transcripts were manually corrected. The final dataset contained the eye gaze signal information (.csv), audio files (.wav, .mp3) and transcript files (.json).

## Dataset Structure

### Directory Tree
```
/project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/egd-cxr/1.0.0/
├── audio_segmentation_transcripts/          # 1,084 directories with audio transcripts
│   ├── 002da0d9-ce49c30d-4dfcc1f8-746d2401-d8044d48/
│   ├── 0066734a-35568fde-fd52ba23-ec66f3de-88d4aaf9/
│   ├── 00fe73b4-5215bb4f-94bbccc4-ac5f4f6f-52805cfb/
│   └── ... (1,081 more directories)
├── inclusion_exclusion_criteria_outputs/    # Inclusion/exclusion criteria files
├── bounding_boxes.csv                       # 18,405 rows - Anatomical region bounding boxes
├── eye_gaze.csv                            # 1,498,953 rows - Raw eye gaze data
├── fixations.csv                           # 48,959 rows - Eye fixation data
├── master_sheet.csv                        # 1,417 rows - Main metadata and labels
├── index.html                              # Dataset documentation
├── LICENSE.txt                             # Dataset license
├── SHA256SUMS.txt                          # File integrity checksums
└── table_descriptions.pdf                  # Detailed table descriptions
```

## Data Description
The dataset consists of the following data documents:

### 1. master_sheet.csv (1,417 rows)
**Description**: Master spreadsheet containing DICOM_IDs (i.e. original MIMIC-CXR Database IDs) along with disease labels. This file provides the following key information:

- **DICOM_ID column**: Maps each row to the original MIMIC CXR image as well as the rest of the documents in this dataset
- **Granular disease labels**: Given by the MIMIC CXR database (i.e. CheXpert NLP tool)
- **Reason for exam sentences**: Sectioned out from Indication section of the original MIMIC-CXR report

**Column Descriptions**:
- **dicom_id**: Unique identifier for the DICOM image file
- **path**: File path to the DICOM image within the dataset
- **study_id**: Unique identifier for the radiology study
- **patient_id**: Unique identifier for the patient
- **stay_id**: Unique identifier for the hospital stay
- **gender**: Patient gender (F/M)
- **anchor_age**: Patient age range (e.g., "20 - 30")
- **image_top_pad, image_bottom_pad, image_left_pad, image_right_pad**: Padding values for image preprocessing
- **dx1-dx9**: Primary to ninth diagnosis codes
- **dx1_icd-dx9_icd**: Corresponding ICD-10 diagnosis codes
- **normal_reports**: Flag indicating if reports are normal
- **Normal**: Binary label for normal cases (0/1)
- **CHF**: Binary label for Congestive Heart Failure (0/1)
- **pneumonia**: Binary label for pneumonia (0/1)
- **consolidation**: Binary label for lung consolidation (0/1)
- **enlarged_cardiac_silhouette**: Binary label for enlarged cardiac silhouette (0/1)
- **linear__patchy_atelectasis**: Binary label for linear/patchy atelectasis (0/1)
- **lobar__segmental_collapse**: Binary label for lobar/segmental collapse (0/1)
- **not_otherwise_specified_opacity___pleural__parenchymal_opacity__**: Binary label for unspecified opacity (0/1)
- **pleural_effusion_or_thickening**: Binary label for pleural effusion/thickening (0/1)
- **pulmonary_edema__hazy_opacity**: Binary label for pulmonary edema (0/1)
- **normal_anatomically**: Binary label for anatomically normal (0/1)
- **elevated_hemidiaphragm**: Binary label for elevated hemidiaphragm (0/1)
- **hyperaeration**: Binary label for hyperaeration (0/1)
- **vascular_redistribution**: Binary label for vascular redistribution (0/1)
- **atelectasis__chx**: CheXpert label for atelectasis (-1/0/1)
- **cardiomegaly__chx**: CheXpert label for cardiomegaly (-1/0/1)
- **consolidation__chx**: CheXpert label for consolidation (-1/0/1)
- **edema__chx**: CheXpert label for edema (-1/0/1)
- **enlarged_cardiomediastinum__chx**: CheXpert label for enlarged cardiomediastinum (-1/0/1)
- **fracture__chx**: CheXpert label for fracture (-1/0/1)
- **lung_lesion__chx**: CheXpert label for lung lesion (-1/0/1)
- **lung_opacity__chx**: CheXpert label for lung opacity (-1/0/1)
- **no_finding__chx**: CheXpert label for no finding (-1/0/1)
- **pleural_effusion__chx**: CheXpert label for pleural effusion (-1/0/1)
- **pleural_other__chx**: CheXpert label for other pleural findings (-1/0/1)
- **pneumonia__chx**: CheXpert label for pneumonia (-1/0/1)
- **pneumothorax__chx**: CheXpert label for pneumothorax (-1/0/1)
- **support_devices__chx**: CheXpert label for support devices (-1/0/1)
- **cxr_exam_indication**: Clinical indication for the chest X-ray examination

**First 3 rows sample**:
```csv
dicom_id,path,study_id,patient_id,stay_id,gender,anchor_age,image_top_pad,image_bottom_pad,image_left_pad,image_right_pad,dx1,dx1_icd,dx2,dx2_icd,dx3,dx3_icd,dx4,dx4_icd,dx5,dx5_icd,dx6,dx6_icd,dx7,dx7_icd,dx8,dx8_icd,dx9,dx9_icd,normal_reports,Normal,CHF,pneumonia,consolidation,enlarged_cardiac_silhouette,linear__patchy_atelectasis,lobar__segmental_collapse,not_otherwise_specified_opacity___pleural__parenchymal_opacity__,pleural_effusion_or_thickening,pulmonary_edema__hazy_opacity,normal_anatomically,elevated_hemidiaphragm,hyperaeration,vascular_redistribution,atelectasis__chx,cardiomegaly__chx,consolidation__chx,edema__chx,enlarged_cardiomediastinum__chx,fracture__chx,lung_lesion__chx,lung_opacity__chx,no_finding__chx,pleural_effusion__chx,pleural_other__chx,pneumonia__chx,pneumothorax__chx,support_devices__chx,cxr_exam_indication
24c7496c-d7635dfe-b8e0b87f-d818affc-78ff7cf4,files/p15/p15628804/s58573295/24c7496c-d7635dfe-b8e0b87f-d818affc-78ff7cf4.dcm,58573295,15628804,33811834,F,20 - 30,86,86,448,448,"Heart failure, unspecified",I50.9,,,,,,,,,,,,,,,,,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,,1,,-1,,,,,,,,,,,___F with CHF and  shortness of breath// ?Pulmonary edema
78711a04-264d5305-d5feec9b-ebef1cec-fdc6db9c,files/p19/p19462352/s51900589/78711a04-264d5305-d5feec9b-ebef1cec-fdc6db9c.dcm,51900589,19462352,32954494,F,20 - 30,0,0,534,534,HYPERTENSION NOS,401.9,"CONGESTIVE HEART FAILURE, UNSPEC",428,,,,,,,,,,,,,,,0,0,1,0,0,0,0,0,0,1,1,0,0,0,1,,,0,1,,,,,,1,,1,,,"___F with hypertension, tachycardia"
a770d8d6-7b6a62ff-815ab876-c81709a8-9a654a54,files/p11/p11255143/s50941783/a770d8d6-7b6a62ff-815ab876-c81709a8-9a654a54.dcm,50941783,11255143,34005408,F,20 - 30,0,0,534,534,"CONGESTIVE HEART FAILURE, UNSPEC",428,,,,,,,,,,,,,,,,,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,,,,,,,,,1,,,,,1,History of myocardial infarction.  Shortness of breath.
```

### 2. fixations.csv (48,959 rows)
**Description**: Spreadsheet containing fixation eye gaze data as exported by Gazepoint Analysis UX Edition software containing DICOM_IDs. This file is a subset of eye_gaze.csv containing a single data entry per fixation. Fixation is defined as the maintaining of the eye gaze on a single location (i.e. eye gaze cluster). The Gazepoint Analysis UX Edition software generates this file by post-processing (i.e. 'sweeping') the eye_gaze.csv file and storing the last entry for each fixation.

**Key Columns**:
- **DICOM_ID**: Maps rows to the original MIMIC image name
- **TIME (in secs)**: Presents the time elapsed in seconds since the last system initialization or calibration (i.e. when a new CXR image was presented to the radiologist)
- **FPOGX**: The X coordinates of the fixation POG, as a fraction of the screen size. (0, 0) is top left, (0.5, 0.5) is the screen center, and (1.0, 1.0) is bottom right
- **FPOGY**: The Y coordinates of the fixation POG, as a fraction of the screen size. (0, 0) is top left, (0.5, 0.5) is the screen center, and (1.0, 1.0) is bottom right
- **X_ORIGINAL**: The X coordinate of the fixation in original MIMIC DICOM image coordinates
- **Y_ORIGINAL**: The Y coordinate of the fixation in original MIMIC DICOM image coordinates

**Column Descriptions**:
- **SESSION_ID**: Unique identifier for the eye-tracking session
- **MEDIA_ID**: Identifier for the media being viewed (typically 0)
- **DICOM_ID**: Unique identifier linking to the DICOM image
- **CNT**: Counter/index for the data point
- **Time (in secs)**: Timestamp in seconds from session start
- **TIMETICK(f=10000000)**: High-resolution timestamp (10MHz frequency)
- **FPOGX, FPOGY**: Fixation point of gaze coordinates (normalized 0-1)
- **FPOGS**: Fixation start time in seconds
- **FPOGD**: Fixation duration in seconds
- **FPOGID**: Fixation point ID
- **FPOGV**: Fixation point validity flag (0/1)
- **BPOGX, BPOGY**: Binocular point of gaze coordinates (normalized 0-1)
- **BPOGV**: Binocular point validity flag (0/1)
- **LPCX, LPCY**: Left pupil center coordinates (normalized 0-1)
- **LPD**: Left pupil diameter in millimeters
- **LPS**: Left pupil size
- **LPV**: Left pupil validity flag (0/1)
- **RPCX, RPCY**: Right pupil center coordinates (normalized 0-1)
- **RPD**: Right pupil diameter in millimeters
- **RPS**: Right pupil size
- **RPV**: Right pupil validity flag (0/1)
- **BKID**: Blink ID
- **BKDUR**: Blink duration
- **BKPMIN**: Blink pupil minimum
- **LPMM**: Left pupil measurement
- **LPMMV**: Left pupil measurement validity
- **RPMM**: Right pupil measurement
- **RPMMV**: Right pupil measurement validity
- **SACCADE_MAG**: Saccade magnitude
- **SACCADE_DIR**: Saccade direction
- **VID_FRAME**: Video frame number
- **X_ORIGINAL, Y_ORIGINAL**: Original pixel coordinates

**First 3 rows sample**:
```csv
SESSION_ID,MEDIA_ID,DICOM_ID,CNT,Time (in secs),TIMETICK(f=10000000),FPOGX,FPOGY,FPOGS,FPOGD,FPOGID,FPOGV,BPOGX,BPOGY,BPOGV,LPCX,LPCY,LPD,LPS,LPV,RPCX,RPCY,RPD,RPS,RPV,BKID,BKDUR,BKPMIN,LPMM,LPMMV,RPMM,RPMMV,SACCADE_MAG,SACCADE_DIR,VID_FRAME,X_ORIGINAL,Y_ORIGINAL
1,0,1a3f39ce-ebe90275-9a66145a-af03360e-ee3b163b,45,0.72363,696216388878.0,0.45224,0.33879000000000004,0.01794,0.7056899999999999,2,1,0.46134,0.26619,1,0.37859,0.5943,18.33718,1.0527600000000001,1,0.6379,0.5979800000000001,18.19434,1.0606799999999998,1,0,0.0,20,3.5064800000000003,1,3.56681,1,0.0,0.0,0.0,998,1035
1,0,1a3f39ce-ebe90275-9a66145a-af03360e-ee3b163b,68,1.09607,696220113269.0,0.44419,0.28933000000000003,0.73987,0.3562,3,1,0.37864000000000003,0.21775,1,0.37749,0.594,17.92645,1.0606799999999998,1,0.6367,0.59809,18.35567,1.06859,1,0,0.0,19,3.7337599999999997,1,3.7751099999999997,1,55.60791,106.13765,0.0,952,884
1,0,1a3f39ce-ebe90275-9a66145a-af03360e-ee3b163b,86,1.3869,696223021482.0,0.39018,0.28168000000000004,1.11212,0.27478,4,1,0.38154,0.43718,1,0.37654,0.59767,18.143629999999998,1.0087899999999999,1,0.63485,0.60175,19.20525,1.0087899999999999,1,0,0.0,19,3.65403,1,3.8783199999999995,1,104.02783000000001,175.4447,0.0,642,860
```

### 3. bounding_boxes.csv (18,405 rows)
**Description**: Spreadsheet containing bounding boxes coordinates for the anatomical structures containing DICOM_IDs. This file is provided as a supplemental source to help researchers for useful in-depth and correlation analysis (e.g. eye gaze vs. anatomical structures) and/or anatomical structure segmentation purposes.

**Column Descriptions**:
- **dicom_id**: Unique identifier linking to the DICOM image
- **bbox_name**: Name of the anatomical region/landmark
  - Common regions include: cardiac silhouette, left/right clavicle, left/right costophrenic angle, left/right hilar structures, left/right lower/mid/upper lung zone, left/right lung, mediastinum, etc.
- **x1**: Left boundary coordinate of the bounding box (pixels)
- **x2**: Right boundary coordinate of the bounding box (pixels)
- **y1**: Top boundary coordinate of the bounding box (pixels)
- **y2**: Bottom boundary coordinate of the bounding box (pixels)

**First 3 rows sample**:
```csv
dicom_id,bbox_name,x1,x2,y1,y2
002da0d9-ce49c30d-4dfcc1f8-746d2401-d8044d48,cardiac silhouette,1007.0,1743.0,1278.0,2040.0
002da0d9-ce49c30d-4dfcc1f8-746d2401-d8044d48,left clavicle,1369.0,2363.0,309.0,671.0
002da0d9-ce49c30d-4dfcc1f8-746d2401-d8044d48,left costophrenic angle,2105.0,2363.0,2156.0,2415.0
```

### 4. eye_gaze.csv (1,498,953 rows)
**Description**: Spreadsheet containing raw eye gaze data as exported by Gazepoint Analysis UX Edition software containing DICOM_IDs. This file contains one (1) row for every data sample collected from the eye tracker. Both fixations.csv and eye_gaze.csv spreadsheets contain the same columns, but eye_gaze.csv contains the raw sampled eye gaze signal while fixations.csv contains the post-processed fixation data.

**Key Columns** (same as fixations.csv):
- **DICOM_ID**: Maps rows to the original MIMIC image name
- **TIME (in secs)**: Presents the time elapsed in seconds since the last system initialization or calibration
- **FPOGX**: The X coordinates of the fixation POG, as a fraction of the screen size
- **FPOGY**: The Y coordinates of the fixation POG, as a fraction of the screen size
- **X_ORIGINAL**: The X coordinate of the fixation in original MIMIC DICOM image coordinates
- **Y_ORIGINAL**: The Y coordinate of the fixation in original MIMIC DICOM image coordinates

**Column Descriptions**:
- **SESSION_ID**: Unique identifier for the eye-tracking session
- **MEDIA_ID**: Identifier for the media being viewed (typically 0)
- **DICOM_ID**: Unique identifier linking to the DICOM image
- **CNT**: Counter/index for the data point
- **Time (in secs)**: Timestamp in seconds from session start
- **TIMETICK(f=10000000)**: High-resolution timestamp (10MHz frequency)
- **FPOGX, FPOGY**: Fixation point of gaze coordinates (normalized 0-1)
- **FPOGS**: Fixation start time in seconds
- **FPOGD**: Fixation duration in seconds
- **FPOGID**: Fixation point ID
- **FPOGV**: Fixation point validity flag (0/1)
- **BPOGX, BPOGY**: Binocular point of gaze coordinates (normalized 0-1)
- **BPOGV**: Binocular point validity flag (0/1)
- **LPCX, LPCY**: Left pupil center coordinates (normalized 0-1)
- **LPD**: Left pupil diameter in millimeters
- **LPS**: Left pupil size
- **LPV**: Left pupil validity flag (0/1)
- **RPCX, RPCY**: Right pupil center coordinates (normalized 0-1)
- **RPD**: Right pupil diameter in millimeters
- **RPS**: Right pupil size
- **RPV**: Right pupil validity flag (0/1)
- **BKID**: Blink ID
- **BKDUR**: Blink duration
- **BKPMIN**: Blink pupil minimum
- **LPMM**: Left pupil measurement
- **LPMMV**: Left pupil measurement validity
- **RPMM**: Right pupil measurement
- **RPMMV**: Right pupil measurement validity
- **SACCADE_MAG**: Saccade magnitude
- **SACCADE_DIR**: Saccade direction
- **VID_FRAME**: Video frame number
- **X_ORIGINAL, Y_ORIGINAL**: Original pixel coordinates

**Note**: This file contains the same structure as fixations.csv but includes all eye tracking data points (not just fixations), making it much larger and more comprehensive for detailed gaze analysis.

### 5. audio_segmentation_transcripts/ (1,084 subdirectories)
**Description**: Folder containing subfolders named using DICOM_IDs. Each subfolder contains the following files:

- **audio.wav**: The dictation audio in wav format
- **audio.mp3**: The dictation audio in mp3 format  
- **transcript.json**: The transcript of the dictation audio with timestamps for each spoken phrase. Specifically:
  - `phrase` tag contains phrase text
  - `begin_time` tag contains the starting time (in seconds) of dictation for phrase
  - `end_time` tag contains the end time (in seconds) of dictation for phrase
- **left_lung.png, right_lung.png, mediastinum.png and aortic_knob.png**: The manually segmentation images of four (4) key anatomies: left lung, right lung, mediastinum, aortic knob, respectively

### 6. inclusion_exclusion_criteria_outputs/
**Description**: Folder containing 3 spreadsheet files that were generated after applying inclusion/exclusion criteria. These 3 spreadsheet files can be used by the sampling script to generate the master_sheet.csv. This is optional and is shared for reproducible purposes.

- **CHF.csv**: Congestive Heart Failure cases
- **normals.csv**: Normal cases  
- **pneumonia.csv**: Pneumonia cases

## Clinical Diagnostic Labels

### Primary Clinical Conditions Count:
- **Normal**: 357 cases
- **CHF (Congestive Heart Failure)**: 92 cases  
- **Pneumonia**: 322 cases
- **Consolidation**: 359 cases
- **Enlarged Cardiac Silhouette**: 172 cases
- **Pleural Effusion or Thickening**: 382 cases
- **Pulmonary Edema (Hazy Opacity)**: 260 cases

### Additional Labels Available:
The dataset also includes numerous other diagnostic labels such as:
- Linear/Patchy Atelectasis
- Lobar/Segmental Collapse
- Not Otherwise Specified Opacity
- Normal Anatomically
- Elevated Hemidiaphragm
- Hyperaeration
- Vascular Redistribution
- Various CheXpert labels (atelectasis__chx, cardiomegaly__chx, etc.)

## Dataset Statistics Summary
- **Total Images**: 1,417 chest X-ray images
- **Total Eye Gaze Records**: 1,498,953 data points
- **Total Fixation Records**: 48,959 fixations
- **Total Bounding Box Records**: 18,405 anatomical regions
- **Audio Transcript Directories**: 1,084 sessions
- **Dataset Size**: ~420MB total

## File Locations
- **Raw Data**: `/project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/egd-cxr/1.0.0/`
- **Processed Data**: `/project/hnguyen2/mvu9/processing_datasets/processing_gaze/egd-cxr/`
- **Configuration**: `/project/hnguyen2/mvu9/folder_04_ma/gaze-01/config_maui/data_egd-cxr.yaml`

## Key Features
1. **Multi-modal Data**: Combines eye-tracking, audio transcripts, and diagnostic labels
2. **Rich Metadata**: Includes patient demographics, study information, and clinical history
3. **Anatomical Annotations**: Bounding boxes for key anatomical regions
4. **Temporal Data**: Eye gaze and fixation data with precise timing
5. **Clinical Labels**: Comprehensive diagnostic classifications for chest X-ray findings

This dataset is particularly valuable for studying radiologist behavior patterns, developing AI-assisted diagnostic tools, and understanding the relationship between visual attention and diagnostic accuracy in chest X-ray interpretation.

## Usage Notes

### Data Access Requirements
The dataset requires access to the CXR DICOM images found in the MIMIC-CXR database. Users need to obtain access to the MIMIC-CXR database separately to use the actual chest X-ray images.

### Recommended Data Usage
- **For most experiments**: Use the `fixations.csv` spreadsheet because it contains the eye gaze signal as post-processed by the Gazepoint Analysis UX Edition
- **For raw signal access**: Use `eye_gaze.csv` if you want access to the raw sampled eye gaze signal
- **Data linking**: Use the DICOM_ID tag found across all data documents to work with combinations of information from different data sources

### Research Applications
This dataset can contribute to various fields of research with applications in machine learning such as:
- **Deep learning explainability**: Understanding how AI models make decisions
- **Multi-modal fusion**: Combining eye gaze, audio, and image data
- **Disease classification**: Improving diagnostic accuracy using gaze patterns
- **Automated radiology report generation**: Using gaze data to improve report quality
- **Radiology education**: Understanding expert behavior patterns
- **Clinical decision support**: Analyzing diagnostic patterns
- **Human-computer interaction**: Improving radiology workstation design

### Data Collection Details
- **Eye Tracker**: GP3 HD Eye Tracker (Gazepoint)
- **Monitor**: Dell S2719DGF at 1920x1080 resolution
- **Radiologist**: ABR certified with 5 years attending experience
- **Collection Period**: March - May 2020 (2 months)
- **Session Structure**: 30 cases per session
- **Audio Processing**: Google Speech-to-Text API with manual correction

### Coordinate Systems
- **Screen Coordinates**: FPOGX, FPOGY are normalized (0-1) with (0,0) at top-left, (0.5,0.5) at center, (1,1) at bottom-right
- **DICOM Coordinates**: X_ORIGINAL, Y_ORIGINAL are in original MIMIC DICOM image coordinates
- **Bounding Boxes**: x1, y1, x2, y2 are in original MIMIC DICOM image coordinates

### Example Usage
Examples of data usage can be found at: https://github.com/cxr-eye-gaze/eye-gaze-dataset

### Dataset Statistics Summary
- **Total Images**: 1,083 chest X-ray images
- **Total Subjects**: 1,038 (Female: 495, Male: 543)
- **Age Range**: 20-80 years old
- **Clinical Conditions**: Normal, Pneumonia, Congestive Heart Failure (CHF)
- **Total Eye Gaze Records**: 1,498,953 data points
- **Total Fixation Records**: 48,959 fixations
- **Total Bounding Box Records**: 18,405 anatomical regions
- **Audio Transcript Directories**: 1,084 sessions
- **Dataset Size**: ~420MB total