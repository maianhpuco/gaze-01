#!/bin/bash
#SBATCH --job-name=st01_img_bs_txt_gz
#SBATCH --output=logs/st01_img_bs_txt_gz_%j.out
#SBATCH --error=logs/st01_img_bs_txt_gz_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -e

mkdir -p logs

echo "Starting ST1 pretrained (image+bbox+seg+text+gaze) at: $(date)"

python main_train_st01_pretrained_image_bbox_seg_text.py --config config/st01_pretrained_image_bbox_seg_text.yaml

echo "Finished ST1 pretrained (image+bbox+seg+text+gaze) at: $(date)"
echo "Done"
