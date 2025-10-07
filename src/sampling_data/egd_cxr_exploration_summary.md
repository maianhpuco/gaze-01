# EGD-CXR Dataset Exploration Summary

## Dataset Overview
The EGD-CXR (Eye Gaze Data for Chest X-Ray) dataset is a comprehensive collection of eye-tracking data from radiologists examining chest X-ray images, along with corresponding diagnostic labels and metadata.

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

## Tabular Data Files

### 1. master_sheet.csv (1,417 rows)
**Description**: Main metadata file containing patient information, diagnostic labels, and study details.

**First 3 rows sample**:
```csv
dicom_id,path,study_id,patient_id,stay_id,gender,anchor_age,image_top_pad,image_bottom_pad,image_left_pad,image_right_pad,dx1,dx1_icd,dx2,dx2_icd,dx3,dx3_icd,dx4,dx4_icd,dx5,dx5_icd,dx6,dx6_icd,dx7,dx7_icd,dx8,dx8_icd,dx9,dx9_icd,normal_reports,Normal,CHF,pneumonia,consolidation,enlarged_cardiac_silhouette,linear__patchy_atelectasis,lobar__segmental_collapse,not_otherwise_specified_opacity___pleural__parenchymal_opacity__,pleural_effusion_or_thickening,pulmonary_edema__hazy_opacity,normal_anatomically,elevated_hemidiaphragm,hyperaeration,vascular_redistribution,atelectasis__chx,cardiomegaly__chx,consolidation__chx,edema__chx,enlarged_cardiomediastinum__chx,fracture__chx,lung_lesion__chx,lung_opacity__chx,no_finding__chx,pleural_effusion__chx,pleural_other__chx,pneumonia__chx,pneumothorax__chx,support_devices__chx,cxr_exam_indication
24c7496c-d7635dfe-b8e0b87f-d818affc-78ff7cf4,files/p15/p15628804/s58573295/24c7496c-d7635dfe-b8e0b87f-d818affc-78ff7cf4.dcm,58573295,15628804,33811834,F,20 - 30,86,86,448,448,"Heart failure, unspecified",I50.9,,,,,,,,,,,,,,,,,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,,1,,-1,,,,,,,,,,,___F with CHF and  shortness of breath// ?Pulmonary edema
78711a04-264d5305-d5feec9b-ebef1cec-fdc6db9c,files/p19/p19462352/s51900589/78711a04-264d5305-d5feec9b-ebef1cec-fdc6db9c.dcm,51900589,19462352,32954494,F,20 - 30,0,0,534,534,HYPERTENSION NOS,401.9,"CONGESTIVE HEART FAILURE, UNSPEC",428,,,,,,,,,,,,,,,0,0,1,0,0,0,0,0,0,1,1,0,0,0,1,,,0,1,,,,,,1,,1,,,"___F with hypertension, tachycardia"
a770d8d6-7b6a62ff-815ab876-c81709a8-9a654a54,files/p11/p11255143/s50941783/a770d8d6-7b6a62ff-815ab876-c81709a8-9a654a54.dcm,50941783,11255143,34005408,F,20 - 30,0,0,534,534,"CONGESTIVE HEART FAILURE, UNSPEC",428,,,,,,,,,,,,,,,,,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,,,,,,,,,1,,,,,1,History of myocardial infarction.  Shortness of breath.
```

### 2. fixations.csv (48,959 rows)
**Description**: Eye fixation data with coordinates, duration, and timing information.

**First 3 rows sample**:
```csv
SESSION_ID,MEDIA_ID,DICOM_ID,CNT,Time (in secs),TIMETICK(f=10000000),FPOGX,FPOGY,FPOGS,FPOGD,FPOGID,FPOGV,BPOGX,BPOGY,BPOGV,LPCX,LPCY,LPD,LPS,LPV,RPCX,RPCY,RPD,RPS,RPV,BKID,BKDUR,BKPMIN,LPMM,LPMMV,RPMM,RPMMV,SACCADE_MAG,SACCADE_DIR,VID_FRAME,X_ORIGINAL,Y_ORIGINAL
1,0,1a3f39ce-ebe90275-9a66145a-af03360e-ee3b163b,45,0.72363,696216388878.0,0.45224,0.33879000000000004,0.01794,0.7056899999999999,2,1,0.46134,0.26619,1,0.37859,0.5943,18.33718,1.0527600000000001,1,0.6379,0.5979800000000001,18.19434,1.0606799999999998,1,0,0.0,20,3.5064800000000003,1,3.56681,1,0.0,0.0,0.0,998,1035
1,0,1a3f39ce-ebe90275-9a66145a-af03360e-ee3b163b,68,1.09607,696220113269.0,0.44419,0.28933000000000003,0.73987,0.3562,3,1,0.37864000000000003,0.21775,1,0.37749,0.594,17.92645,1.0606799999999998,1,0.6367,0.59809,18.35567,1.06859,1,0,0.0,19,3.7337599999999997,1,3.7751099999999997,1,55.60791,106.13765,0.0,952,884
1,0,1a3f39ce-ebe90275-9a66145a-af03360e-ee3b163b,86,1.3869,696223021482.0,0.39018,0.28168000000000004,1.11212,0.27478,4,1,0.38154,0.43718,1,0.37654,0.59767,18.143629999999998,1.0087899999999999,1,0.63485,0.60175,19.20525,1.0087899999999999,1,0,0.0,19,3.65403,1,3.8783199999999995,1,104.02783000000001,175.4447,0.0,642,860
```

### 3. bounding_boxes.csv (18,405 rows)
**Description**: Anatomical region bounding boxes for chest X-ray images.

**First 3 rows sample**:
```csv
dicom_id,bbox_name,x1,x2,y1,y2
002da0d9-ce49c30d-4dfcc1f8-746d2401-d8044d48,cardiac silhouette,1007.0,1743.0,1278.0,2040.0
002da0d9-ce49c30d-4dfcc1f8-746d2401-d8044d48,left clavicle,1369.0,2363.0,309.0,671.0
002da0d9-ce49c30d-4dfcc1f8-746d2401-d8044d48,left costophrenic angle,2105.0,2363.0,2156.0,2415.0
```

### 4. eye_gaze.csv (1,498,953 rows)
**Description**: Raw eye gaze tracking data (file too large for sample display - 406MB).

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