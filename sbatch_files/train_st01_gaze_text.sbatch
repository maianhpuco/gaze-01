#!/bin/bash
#SBATCH --job-name=st01_transcript_gaze_no_image
#SBATCH --output=logs_st01/st01_transcript_gaze_no_image_%j.out
#SBATCH --error=logs_st01/st01_transcript_gaze_no_image_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

mkdir -p logs_st01

echo "Starting ST01 (transcript + gaze, no image) at: $(date)"

python main_st01_prtr_img_bs_gaze_text.py --config configs/st01_prtr_img_bs_gaze_text.yaml --no-image --no-bbox --no-seg

echo "Finished ST01 (transcript + gaze, no image) at: $(date)"
echo "Done"
