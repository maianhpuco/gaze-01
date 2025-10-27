#!/bin/bash
#SBATCH --job-name=st_v2
#SBATCH --output=logs/st2_%j.out
#SBATCH --error=logs/st2_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -e

mkdir -p logs

echo "Starting Silence-Thought v2 training at: $(date)"

python main_train_st2.py --config config/st2_egd_cxr.yaml

echo "Finished Silence-Thought v2 training at: $(date)"
echo "Done"

