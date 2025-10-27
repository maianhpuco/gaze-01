#!/bin/bash
#SBATCH --job-name=st3_img_bs_txt
#SBATCH --output=logs_st3/st3_img_bs_txt_%j.out
#SBATCH --error=logs_st3/st3_img_bs_txt_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

mkdir -p logs_st3

echo "Starting ST3 (image+bbox+seg+text+gaze, pos-weight BCE) at: $(date)"

python st3/main_train_st3_image_bbox_seg_text.py --config config_v3/st3_image_bbox_seg_text.yaml

echo "Finished ST3 (image+bbox+seg+text+gaze, pos-weight BCE) at: $(date)"
echo "Done"
