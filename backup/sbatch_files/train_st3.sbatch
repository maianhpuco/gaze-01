#!/bin/bash
#SBATCH --job-name=st_v3
#SBATCH --output=logs/st3_%j.out
#SBATCH --error=logs/st3_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -e

mkdir -p logs

echo "Starting Silence-Thought v3 (OneCycleLR) training at: $(date)"

python main_train_st3.py --config config/st3_egd_cxr.yaml

echo "Finished Silence-Thought v3 training at: $(date)"
echo "Done"

