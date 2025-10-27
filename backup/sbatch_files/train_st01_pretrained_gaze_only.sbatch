#!/bin/bash
#SBATCH --job-name=st01_gaze
#SBATCH --output=logs/st01_gaze_%j.out
#SBATCH --error=logs/st01_gaze_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -e

mkdir -p logs

echo "Starting ST1 pretrained (gaze only) at: $(date)"

python main_train_st01_pretrained_gaze_only.py --config config/st01_pretrained_gaze_only.yaml

echo "Finished ST1 pretrained (gaze only) at: $(date)"
echo "Done"

