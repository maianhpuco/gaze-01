#!/bin/bash
#SBATCH --job-name=st01_no_text
#SBATCH --output=logs/st01_no_text_%j.out
#SBATCH --error=logs/st01_no_text_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -e

mkdir -p logs

echo "Starting ST1 pretrained (no text) at: $(date)"

python main_train_st01_pretrained_no_text.py --config config/st01_pretrained_no_text.yaml

echo "Finished ST1 pretrained (no text) at: $(date)"
echo "Done"

