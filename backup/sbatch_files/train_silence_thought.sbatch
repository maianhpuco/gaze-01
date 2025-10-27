#!/bin/bash
#SBATCH --job-name=silence_thought_train
#SBATCH --output=logs/train_silence_thought_%j.out
#SBATCH --error=logs/train_silence_thought_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up environment
echo "Setting up environment..."

# Load required modules (adjust based on your cluster's available modules)
# module load python/3.11
# module load cuda/11.8
# module load gcc/9.3.0

# Activate conda environment (adjust path as needed)
# source /project/hnguyen2/mvu9/conda_envs/wsi-agent/bin/activate
# conda activate wsi-agent

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

# Print environment information
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"

# Change to project directory
cd /project/hnguyen2/mvu9/folder_04_ma/gaze-01

# Create checkpoint directory
mkdir -p runs/checkpoints

# Training parameters (adjust as needed)
CONFIG_FILE="config_maui/st_edg_cxr.yaml"
BATCH_SIZE=64
EPOCHS=20
LEARNING_RATE=2e-4
WEIGHT_DECAY=1e-4
MAX_FIXATIONS=20
TXT_DIM=256
ENC_DIM=256
MAX_DECODE_LEN=64
NUM_WORKERS=4
SEED=42

echo "Starting training with parameters:"
echo "  Config: $CONFIG_FILE"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Max fixations: $MAX_FIXATIONS"
echo "  Text dim: $TXT_DIM"
echo "  Encoder dim: $ENC_DIM"
echo "  Max decode length: $MAX_DECODE_LEN"
echo "  Num workers: $NUM_WORKERS"
echo "  Seed: $SEED"

# Run training
echo "Starting training at $(date)"
python main_train_silence_thought.py \
    --config "$CONFIG_FILE" \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --weight-decay $WEIGHT_DECAY \
    --max-fixations $MAX_FIXATIONS \
    --txt-dim $TXT_DIM \
    --enc-dim $ENC_DIM \
    --max-decode-len $MAX_DECODE_LEN \
    --num-workers $NUM_WORKERS \
    --seed $SEED \
    --checkpoint-dir runs/checkpoints

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully at $(date)"
else
    echo "Training failed with exit code $? at $(date)"
    exit 1
fi

# Print final job information
echo "Job completed at $(date)"
echo "Total runtime: $SECONDS seconds"
echo "Checkpoint directory: runs/checkpoints"
echo "Log files: logs/train_silence_thought_${SLURM_JOB_ID}.out"
echo "Error files: logs/train_silence_thought_${SLURM_JOB_ID}.err"

# List generated checkpoints
echo "Generated checkpoints:"
ls -la runs/checkpoints/ 2>/dev/null || echo "No checkpoints found"

echo "Job finished successfully!"
