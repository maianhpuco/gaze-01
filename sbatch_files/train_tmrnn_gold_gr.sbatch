#!/bin/bash
#SBATCH --job-name=gold_gr
#SBATCH --output=logs_tmrnn/gold_gr_%j.out
#SBATCH --error=logs_tmrnn/gold_gr_%j.err
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
srun python -u main_train_tmrnn_gold_gr.py --config configs_tmrnn_v2/st_tmrnn_gold_gr.yaml
echo "Finished at: $(date)" 

