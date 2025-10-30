#!/bin/bash
#SBATCH --job-name=tmrnn_full
#SBATCH --output=logs_tmrnn/tmrnn_full_%j.out
#SBATCH --error=logs_tmrnn/tmrnn_full_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

set -euo pipefail
mkdir -p logs_tmrnn_v2

export HF_HOME=/project/hnguyen2/mvu9/.cache/huggingface
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_VERBOSITY=error
unset HF_TOKEN
unset HUGGINGFACE_TOKEN
unset HUGGINGFACE_HUB_TOKEN

cd /project/hnguyen2/mvu9/folder_04_ma/gaze-01
echo "Starting training at: $(date)"
srun python -u main_train.py --config configs/st_tmrnn.yaml
echo "Finished at: $(date)"