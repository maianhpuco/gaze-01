#!/bin/bash
#SBATCH --job-name=st2_pretrained
#SBATCH --output=logs_st2/st2_pretrained_%j.out
#SBATCH --error=logs_st2/st2_pretrained_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

mkdir -p logs_st2

echo "Starting ST2 pretrained (all modalities) at: $(date)"

python st2/main_train_st2.py --config config_v2/st2_pretrained.yaml

echo "Finished ST2 pretrained (all modalities) at: $(date)"
echo "Done"
