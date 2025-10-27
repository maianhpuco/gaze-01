#!/bin/bash
#SBATCH --job-name=st4_img_bs_txt
#SBATCH --output=logs_st4/st4_img_bs_txt_%j.out
#SBATCH --error=logs_st4/st4_img_bs_txt_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

mkdir -p logs_st4

echo "Starting ST4 (image+bbox+seg+text+gaze with imbalance fixes) at: $(date)"

python st4/main_train_st4_image_bbox_seg_text.py --config config_v4/st4_image_bbox_seg_text.yaml

echo "Finished ST4 (image+bbox+seg+text+gaze with imbalance fixes) at: $(date)"
echo "Done"
