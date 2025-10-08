# wsi-agent
# gaze-01

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
