# EGD-CXR Project Tree Structure

## Overview
This project contains comprehensive visualization tools for the EGD-CXR (Eye Gaze Dataset for Chest X-Ray) dataset, including real DICOM image processing, anatomical mask overlays, and multi-modal data analysis.

## Directory Structure

```
/project/hnguyen2/mvu9/folder_04_ma/gaze-01/
├── config_maui/                          # Configuration files
│   ├── data_egd-cxr.yaml                # EGD-CXR dataset configuration
│   └── data_reflacx.yaml                # REFLACX dataset configuration
├── plots/                                # Generated visualization plots
│   └── 24c7496c-d7635dfe-b8e0b87f-d818affc-78ff7cf4/  # Case-specific plots
│       ├── anatomical_regions.png       # DICOM + anatomical mask overlays
│       ├── bounding_boxes.png           # DICOM + bounding box annotations
│       ├── fixation_analysis.png        # DICOM + eye gaze fixations
│       └── comprehensive_analysis.png   # All data combined
├── sample/                               # Sample DICOM files
│   ├── 24c7496c-d7635dfe-b8e0b87f-d818affc-78ff7cf4.dcm
│   ├── 78711a04-264d5305-d5feec9b-ebef1cec-fdc6db9c.dcm
│   └── a770d8d6-7b6a62ff-815ab876-c81709a8-9a654a54.dcm
├── src/sampling_data/                    # Data processing and exploration
│   ├── egd_cxr_exploration_summary.md   # Comprehensive dataset documentation
│   ├── explore_egd_cxr.py               # Dataset exploration script
│   ├── sampling_egd_cxr.py              # Basic sampling script
│   ├── sampling_egd_cxr_enhanced.py     # Enhanced sampling script
│   ├── sampling_egd_cxr_final.py        # Final sampling script
│   └── sampling.py                      # General sampling utilities
├── create_egd_cxr_plots.py              # Main visualization script (NEW)
├── download_dicom_with_wget.py          # DICOM download utility
├── download_mimic_dicom.py              # Alternative download script
├── download_single_dicom.py             # Single file download
├── download_single_dicom_urllib.py      # Single file download (urllib)
├── README.md                            # Project documentation
└── slurm-*.out                          # SLURM job output files
```

## Key Files Description

### Visualization Scripts
- **`create_egd_cxr_plots.py`** (NEW): Main comprehensive visualization script
  - Loads real DICOM images with proper windowing
  - Overlays anatomical region masks with transparency
  - Creates 4 different visualization types
  - Organizes output by DICOM ID in separate folders

### Data Processing Scripts
- **`src/sampling_data/explore_egd_cxr.py`**: Dataset exploration and analysis
- **`src/sampling_data/sampling_egd_cxr_*.py`**: Various sampling strategies

### Download Scripts
- **`download_dicom_with_wget.py`**: Downloads MIMIC-CXR DICOM files using wget
- **`download_single_dicom.py`**: Downloads individual DICOM files for testing

### Configuration
- **`config_maui/data_egd-cxr.yaml`**: Dataset path configuration

### Documentation
- **`src/sampling_data/egd_cxr_exploration_summary.md`**: Comprehensive dataset documentation
- **`README.md`**: Project overview and usage instructions

## Generated Visualizations

### 1. Anatomical Regions Overlay (`anatomical_regions.png`)
- Real DICOM chest X-ray image
- Overlaid anatomical region masks with color coding:
  - Red: Left Lung
  - Green: Right Lung  
  - Blue: Mediastinum
  - Yellow: Aortic Knob
- Transparency for mask visibility

### 2. Bounding Boxes Overlay (`bounding_boxes.png`)
- Real DICOM chest X-ray image
- Annotated bounding boxes for anatomical structures
- Color-coded regions with labels

### 3. Fixation Analysis (`fixation_analysis.png`)
- Real DICOM chest X-ray image
- Eye gaze fixation points with duration-based sizing
- Transition lines between consecutive fixations
- Start/end markers and color-coded duration

### 4. Comprehensive Analysis (`comprehensive_analysis.png`)
- Multi-panel visualization combining all data
- Main image with all overlays
- Patient information panel
- Clinical findings panel
- Gaze analysis statistics
- Summary information

## Data Sources

### EGD-CXR Dataset
- **Location**: `/project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/egd-cxr/1.0.0/`
- **Components**:
  - `master_sheet.csv`: Patient and clinical information
  - `fixations.csv`: Eye gaze fixation data
  - `bounding_boxes.csv`: Anatomical structure annotations
  - `audio_segmentation_transcripts/`: Audio files and anatomical masks

### MIMIC-CXR Dataset
- **Location**: `/project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/mimic-cxr/2.0.0/`
- **Components**: DICOM chest X-ray images
- **Access**: Requires PhysioNet credentials

## Technical Features

### DICOM Processing
- Proper windowing for chest X-ray display
- Automatic normalization and inversion
- Support for various DICOM formats

### Anatomical Mask Overlay
- Automatic mask resizing to match DICOM dimensions
- Color-coded regions with transparency
- Binary mask processing

### Eye Gaze Visualization
- Normalized coordinate conversion
- Duration-based point sizing
- Transition line visualization
- Statistical analysis

### File Organization
- Case-specific directories
- Descriptive file naming
- High-resolution output (300 DPI)

## Usage

### Running Visualizations
```bash
# Using GPU job and conda environment
srun --jobid=206025 bash -c "cd /project/hnguyen2/mvu9/folder_04_ma/gaze-01 && /project/hnguyen2/mvu9/conda_envs/wsi-agent/bin/python create_egd_cxr_plots.py"
```

### Downloading DICOM Files
```bash
# Download sample DICOM files
python3 download_dicom_with_wget.py
```

## Dependencies
- Python 3.11
- PyDICOM 3.0.1
- Matplotlib 3.10.6
- Pillow 11.3.0
- Pandas
- NumPy

## Recent Updates
- ✅ Fixed DICOM image display (proper windowing)
- ✅ Added anatomical mask overlays with transparency
- ✅ Organized output files by DICOM ID
- ✅ Created comprehensive multi-panel visualizations
- ✅ Cleaned up old plotting code and results
- ✅ Updated project structure documentation
