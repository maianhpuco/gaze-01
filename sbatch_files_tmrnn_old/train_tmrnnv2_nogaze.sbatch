#!/bin/bash
#SBATCH --job-name=tmrnnv2_nogaze
#SBATCH --output=logs_tmrnn_v2/tmrnnv2_nogaze_%j.out
#SBATCH --error=logs_tmrnn_v2/tmrnnv2_nogaze_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
set -euo pipefail
mkdir -p logs_tmrnn_v2
export HF_HOME=/project/hnguyen2/mvu9/.cache/huggingface
export TRANSFORMERS_VERBOSITY=error
unset HF_TOKEN HUGGINGFACE_TOKEN HUGGINGFACE_HUB_TOKEN
cd /project/hnguyen2/mvu9/folder_04_ma/gaze-01

echo "Starting training at: $(date)"
srun python -u main_train_v2.py --config configs_tmrnn_v2/st_tmrnn_nogaze.yaml
