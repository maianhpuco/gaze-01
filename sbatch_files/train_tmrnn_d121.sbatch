#!/bin/bash
#SBATCH --job-name=tmrnn_d121
#SBATCH --output=logs_tmrnn/tmrnn_d121_%j.out
#SBATCH --error=logs_tmrnn/tmrnn_d121_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
mkdir -p logs_tmrnn

export HF_HOME=/project/hnguyen2/mvu9/.cache/huggingface
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_VERBOSITY=error
unset HF_TOKEN
unset HUGGINGFACE_TOKEN
unset HUGGINGFACE_HUB_TOKEN

cd /project/hnguyen2/mvu9/folder_04_ma/gaze-01
echo "Starting training at: $(date)"
srun python -u main_train_tmrnn.py --config configs_tmrnn_v2/st_tmrnn_d121_all.yaml
echo "Finished at: $(date)" 

