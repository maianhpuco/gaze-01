#!/bin/bash
#SBATCH --job-name=st4_pretrained
#SBATCH --output=logs_st4/st4_pretrained_%j.out
#SBATCH --error=logs_st4/st4_pretrained_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

mkdir -p logs_st4

echo "Starting ST4 pretrained run (pos-weight + thresholds + sampler) at: $(date)"

python st4/main_train_st4.py --config config_v4/st4_pretrained.yaml

echo "Finished ST4 pretrained run at: $(date)"
echo "Done"
