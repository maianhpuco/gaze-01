#!/bin/bash
#SBATCH --job-name=st_rm_bbox_seg_img
#SBATCH --output=logs/st_rm_bbox_seg_img_%j.out
#SBATCH --error=logs/st_rm_bbox_seg_img_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -e

mkdir -p logs

echo "Starting silence-thought (gaze only) training at: $(date)"

python main_train_st_rm_bbox_seg_img.py --config config_maui/st_edg_cxr_rm_bbox_seg_img.yaml

echo "Gaze-only training completed at: $(date)"
echo "Done"
