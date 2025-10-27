# wsi-agent
# gaze-01


============================================================
SPLIT STATISTICS
============================================================

TRAIN SPLIT:
  Total cases: 756
    CHF       :  254 cases ( 33.6%)
    pneumonia :  251 cases ( 33.2%)
    Normal    :  251 cases ( 33.2%)

VAL SPLIT:
  Total cases: 108
    CHF       :   36 cases ( 33.3%)
    pneumonia :   36 cases ( 33.3%)
    Normal    :   36 cases ( 33.3%)

TEST SPLIT:
  Total cases: 219
    CHF       :   73 cases ( 33.3%)
    pneumonia :   73 cases ( 33.3%)
    Normal    :   73 cases ( 33.3%)

OVERALL TOTALS:
  Total cases: 1083
    train:  756 cases ( 69.8%)
    val  :  108 cases ( 10.0%)
    test :  219 cases ( 20.2%) 

    
```
egd-cxr/
├── master_sheet.csv (1,083 records, 59 columns)
├── eye_gaze.csv (406MB - raw eye tracking data)
├── fixations.csv (15MB - processed fixations)
├── bounding_boxes.csv (anatomical annotations)
├── audio_segmentation_transcripts/ (1,084 subdirectories)
│   ├── [study_id]/
│   │   ├── audio.mp3/wav
│   │   ├── transcript.json
│   │   └── anatomical region images
└── inclusion_exclusion_criteria_outputs/
    ├── CHF.csv (493 records)
    ├── normals.csv (7MB)
    └── pneumonia.csv 
``` 
download mimic dcom in the egd list : 
    python src/download/download_dicom_with_wget_egd.py 
