#!/bin/bash
#SBATCH --job-name=st01_bs_text
#SBATCH --output=logs_st01/st01_bs_text_%j.out
#SBATCH --error=logs_st01/st01_bs_text_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

mkdir -p logs_st01

echo "Starting ST01 (bs+text, BEST) at: $(date)"

python main_st01_prtr_img_bs_gaze_text.py --config configs/st01_prtr_img_bs_gaze_text.yaml --no-image --no-gaze

echo "Finished ST01 (bs+text, BEST) at: $(date)"
echo "Done"
