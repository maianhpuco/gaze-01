#!/bin/bash
#SBATCH --job-name=silence_thought_quick
#SBATCH --output=logs/train_quick_%j.out
#SBATCH --error=logs/train_quick_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Quick training run for testing/debugging
echo "Quick training run - Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"

# Create logs directory
mkdir -p logs

# Set environment
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

# Change to project directory
cd /project/hnguyen2/mvu9/folder_04_ma/gaze-01

# Create checkpoint directory
mkdir -p runs/checkpoints

# Quick training parameters (shorter run for testing)
python main_train_silence_thought.py \
    --config config/data_egd-cxr.yaml \
    --batch-size 2 \
    --epochs 5 \
    --lr 2e-4 \
    --weight-decay 1e-4 \
    --max-fixations 10 \
    --txt-dim 128 \
    --enc-dim 128 \
    --max-decode-len 32 \
    --num-workers 2 \
    --seed 42 \
    --checkpoint-dir runs/checkpoints

echo "Quick training completed at $(date)"
