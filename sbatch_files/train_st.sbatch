#!/bin/bash
#SBATCH --job-name=st_full
#SBATCH --output=logs/st_full_%j.out
#SBATCH --error=logs/st_full_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -e

# Create logs directory
mkdir -p logs

echo "Starting base Silence-Thought training at: $(date)"

python main_train_st.py --config config_maui/st_edg_cxr.yaml

echo "Base training completed at: $(date)"
echo "Done"
