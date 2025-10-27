# SBatch Files for ST01 Training Scripts

This directory contains SLURM sbatch scripts to facilitate training runs for the ST01 RNN-based model with various modality combinations.

## Naming Convention

Files follow the format: `train_st01_[modalities].sbatch`

- **prtr**: Pretrained ResNet-18 image encoder (always used for chest X-ray images)
- **img**: Image features (chest X-ray)
- **bs**: Bounding box + Segmentation (always used together)
- **gaze**: Gaze/fixation information
- **text**: Transcript/transcription data

## Available Scripts:

- **`train_st01_prtr_img_bs_gaze_text.sbatch`**:
  - **Purpose**: Full multimodal training with pretrained ResNet-18, bounding boxes, segmentation, gaze, and transcript
  - **Command**: `python main_st01_prtr_img_bs_gaze_text.py --config configs/st01_prtr_img_bs_gaze_text.yaml --pretrained-image`

- **`train_st01_img_bs_gaze_text_no_pretrain.sbatch`**:
  - **Purpose**: Full multimodal training without pretrained image encoder (CNN init only)
  - **Command**: `python main_st01_prtr_img_bs_gaze_text.py --config configs/st01_prtr_img_bs_gaze_text.yaml --no-pretrained-image`

- **`train_st01_prtr_img_bs_gaze_text_rerun.sbatch`**:
  - **Purpose**: Full multimodal training using wrapper script
  - **Command**: `python main_st01_img_bs_gaze_text.py --config configs/st01_prtr_img_bs_gaze_text.yaml --pretrained-image`

- **`train_st01_prtr_img_gaze.sbatch`**:
  - **Purpose**: Pretrained image + gaze only
  - **Command**: `python main_st01_prtr_img_bs_gaze_text.py --config configs/st01_prtr_img_bs_gaze_text.yaml --pretrained-image --no-bbox --no-seg --no-text`

- **`train_st01_prtr_img_bs.sbatch`**:
  - **Purpose**: Pretrained image + bounding boxes + segmentation
  - **Command**: `python main_st01_prtr_img_bs_gaze_text.py --config configs/st01_prtr_img_bs_gaze_text.yaml --pretrained-image --no-gaze --no-text`

- **`train_st01_prtr_img.sbatch`**:
  - **Purpose**: Pretrained image only
  - **Command**: `python main_st01_prtr_img_bs_gaze_text.py --config configs/st01_prtr_img_bs_gaze_text.yaml --pretrained-image --no-bbox --no-seg --no-gaze --no-text`

- **`train_st01_gaze.sbatch`**:
  - **Purpose**: Gaze only
  - **Command**: `python main_st01_prtr_img_bs_gaze_text.py --config configs/st01_prtr_img_bs_gaze_text.yaml --no-image --no-bbox --no-seg --no-text`

- **`train_st01_gaze_text.sbatch`**:
  - **Purpose**: Gaze + transcript
  - **Command**: `python main_st01_prtr_img_bs_gaze_text.py --config configs/st01_prtr_img_bs_gaze_text.yaml --no-image --no-bbox --no-seg`

- **`train_st01_prtr_img_text.sbatch`**:
  - **Purpose**: Pretrained image + transcript
  - **Command**: `python main_st01_prtr_img_bs_gaze_text.py --config configs/st01_prtr_img_bs_gaze_text.yaml --pretrained-image --no-bbox --no-seg --no-gaze`

- **`train_st01_prtr_img_bs_text.sbatch`**:
  - **Purpose**: Pretrained image + bounding boxes + segmentation + transcript
  - **Command**: `python main_st01_prtr_img_bs_gaze_text.py --config configs/st01_prtr_img_bs_gaze_text.yaml --pretrained-image --no-gaze`

- **`train_st01_bs_text.sbatch`**:
  - **Purpose**: Bounding boxes + segmentation + transcript (BEST performing combination)
  - **Command**: `python main_st01_prtr_img_bs_gaze_text.py --config configs/st01_prtr_img_bs_gaze_text.yaml --no-image --no-gaze`

- **`train_st01_bs_text_spoken.sbatch`**:
  - **Purpose**: Bounding boxes + segmentation + transcript (spoken only)
  - **Command**: `python main_st01_prtr_img_bs_gaze_text.py --config configs/st01_prtr_img_bs_gaze_text.yaml --no-image --no-gaze`

- **`train_st01_bs_gaze_text.sbatch`**:
  - **Purpose**: Bounding boxes + segmentation + gaze + transcript
  - **Command**: `python main_st01_prtr_img_bs_gaze_text.py --config configs/st01_prtr_img_bs_gaze_text.yaml --no-image`

- **`train_st01_bs_gaze.sbatch`**:
  - **Purpose**: Bounding boxes + segmentation + gaze
  - **Command**: `python main_st01_prtr_img_bs_gaze_text.py --config configs/st01_prtr_img_bs_gaze_text.yaml --no-image --no-text`

## Usage:

To submit a job, use the `sbatch` command followed by the script name:

```bash
sbatch sbatch_files/train_st01_prtr_img_bs_gaze_text.sbatch
```

To submit all experiments at once:

```bash
bash sbatch_files/submit_all_experiments.sh
```

Logs for each job will be stored in the `logs_st01/` directory.

## Key Features:

- **Pretrained ResNet-18**: All image-based experiments use pretrained ResNet-18 for chest X-ray images
- **Consistent Resource Allocation**: 24 hours, 1 GPU, 32GB RAM, 8 CPUs
- **Proper Logging**: All logs saved to `logs_st01/` with descriptive names
- **Error Handling**: Robust error handling with `set -euo pipefail`
- **Modular Design**: Each experiment is isolated and can be run independently