#!/bin/bash
#SBATCH --job-name=st2_img_bs_txt_gz
#SBATCH --output=logs_st2/st2_img_bs_txt_gz_%j.out
#SBATCH --error=logs_st2/st2_img_bs_txt_gz_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

mkdir -p logs_st2

echo "Starting ST2 (image+bbox+seg+text+gaze) at: $(date)"

python st2/main_train_st2.py --config config_v2/st2_image_bbox_seg_text.yaml

echo "Finished ST2 (image+bbox+seg+text+gaze) at: $(date)"
echo "Done"
