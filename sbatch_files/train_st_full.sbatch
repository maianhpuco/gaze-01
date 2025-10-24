#!/bin/bash
#SBATCH --job-name=silence_thought_full
#SBATCH --output=logs/train_full_%j.out
#SBATCH --error=logs/train_full_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Full training run with high performance settings
echo "Full training run - Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"

# Create logs directory
mkdir -p logs

# Set environment for multi-GPU
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

# Change to project directory
cd /project/hnguyen2/mvu9/folder_04_ma/gaze-01

# Create checkpoint directory
mkdir -p runs/checkpoints

# Full training parameters (production run)
python main_train_silence_thought.py \
    --config config/data_egd-cxr.yaml \
    --batch-size 8 \
    --epochs 100 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --max-fixations 50 \
    --txt-dim 512 \
    --enc-dim 512 \
    --max-decode-len 128 \
    --num-workers 8 \
    --seed 42 \
    --checkpoint-dir runs/checkpoints

echo "Full training completed at $(date)"
