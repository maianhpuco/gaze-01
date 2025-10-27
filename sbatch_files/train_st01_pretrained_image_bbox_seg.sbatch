#!/bin/bash
#SBATCH --job-name=st01_img_bs
#SBATCH --output=logs/st01_img_bs_%j.out
#SBATCH --error=logs/st01_img_bs_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -e

mkdir -p logs

echo "Starting ST1 pretrained (image+bbox+seg) at: $(date)"

python main_train_st01_pretrained_image_bbox_seg.py --config config/st01_pretrained_image_bbox_seg.yaml

echo "Finished ST1 pretrained (image+bbox+seg) at: $(date)"
echo "Done"

