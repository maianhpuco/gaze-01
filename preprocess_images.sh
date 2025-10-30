#!/bin/bash
# Preprocess DICOM images to PNG for faster training

set -euo pipefail

echo "🚀 Starting DICOM to PNG preprocessing..."
echo "📁 DICOM source: /project/hnguyen2/mvu9/datasets/gaze_data/egd-cxr/dicom_raw"
echo "📁 PNG output: /project/hnguyen2/mvu9/datasets/gaze_data/egd-cxr/png_raw"
echo "📋 Master sheet: /project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/egd-cxr/1.0.0/master_sheet.csv"

# Create output directory
mkdir -p /project/hnguyen2/mvu9/datasets/gaze_data/egd-cxr/png_raw

# Run preprocessing
python preprocess_dicom_to_png.py \
  --dicom-dir /project/hnguyen2/mvu9/datasets/gaze_data/egd-cxr/dicom_raw \
  --output-dir /project/hnguyen2/mvu9/datasets/gaze_data/egd-cxr/png_raw \
  --master-sheet /project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/egd-cxr/1.0.0/master_sheet.csv \
  --batch-size 32 \
  --num-workers 8 \
  --target-size 224 224

echo "✅ Preprocessing complete!"
echo "🎯 You can now use the fast training script:"
echo "   python main_resnet_img_fast.py --config configs/data_egd_cxr_single_label.yaml"
