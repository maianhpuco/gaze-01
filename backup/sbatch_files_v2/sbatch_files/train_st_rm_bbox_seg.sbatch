#!/bin/bash
#SBATCH --job-name=st_rm_bbox_seg
#SBATCH --output=logs/st_rm_bbox_seg_%j.out
#SBATCH --error=logs/st_rm_bbox_seg_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -e

mkdir -p logs

echo "Starting silence-thought (no bbox/seg) training at: $(date)"

python main_train_st_rm_bbox_seg.py --config config_maui/st_edg_cxr_rm_bbox_seg.yaml

echo "No-bbox/seg training completed at: $(date)"
echo "Done"
