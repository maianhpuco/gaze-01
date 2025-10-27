#!/bin/bash
#SBATCH --job-name=st3_pretrained
#SBATCH --output=logs_st3/st3_pretrained_%j.out
#SBATCH --error=logs_st3/st3_pretrained_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

mkdir -p logs_st3

echo "Starting ST3 pretrained run (pos-weight BCE) at: $(date)"

python st3/main_train_st3.py --config config_v3/st3_pretrained.yaml

echo "Finished ST3 pretrained run (pos-weight BCE) at: $(date)"
echo "Done"
