#!/bin/bash
#SBATCH --job-name=st01_pre
#SBATCH --output=logs/st01_pre_%j.out
#SBATCH --error=logs/st01_pre_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -e

mkdir -p logs

echo "Starting Silence-Thought pretrained run at: $(date)"

python main_train_st01_pretrained.py --config config/st01_pretrained.yaml

echo "Finished Silence-Thought pretrained run at: $(date)"
echo "Done"

