# Training Guide for Silence-Thought Model

This guide explains how to train the silence-thought model using SLURM job scheduler.

## Available Training Scripts

### 1. Standard Training (`train_silence_thought.sbatch`)
- **Duration**: 24 hours
- **Resources**: 1 GPU, 8 CPUs, 32GB RAM
- **Purpose**: Standard training run with balanced parameters
- **Parameters**:
  - Batch size: 4
  - Epochs: 50
  - Learning rate: 2e-4
  - Max fixations: 20
  - Text/Encoder dim: 256

### 2. Quick Training (`train_silence_thought_quick.sbatch`)
- **Duration**: 4 hours
- **Resources**: 1 GPU, 4 CPUs, 16GB RAM
- **Purpose**: Quick testing and debugging
- **Parameters**:
  - Batch size: 2
  - Epochs: 5
  - Learning rate: 2e-4
  - Max fixations: 10
  - Text/Encoder dim: 128

### 3. Full Training (`train_silence_thought_full.sbatch`)
- **Duration**: 48 hours
- **Resources**: 2 GPUs, 16 CPUs, 64GB RAM
- **Purpose**: Production training with maximum resources
- **Parameters**:
  - Batch size: 8
  - Epochs: 100
  - Learning rate: 1e-4
  - Max fixations: 50
  - Text/Encoder dim: 512

## How to Submit Jobs

### Submit Standard Training
```bash
sbatch train_silence_thought.sbatch
```

### Submit Quick Training (for testing)
```bash
sbatch train_silence_thought_quick.sbatch
```

### Submit Full Training (production)
```bash
sbatch train_silence_thought_full.sbatch
```

## Monitoring Jobs

### Check Job Status
```bash
squeue -u $USER
```

### View Job Output
```bash
# View standard output
tail -f logs/train_silence_thought_<JOB_ID>.out

# View error output
tail -f logs/train_silence_thought_<JOB_ID>.err
```

### Cancel Job
```bash
scancel <JOB_ID>
```

## Output Files

### Checkpoints
- **Location**: `runs/checkpoints/`
- **Format**: `{timestamp}_best.pt` (best validation loss)
- **Format**: `{timestamp}_last.pt` (final epoch)

### Logs
- **Location**: `logs/`
- **Format**: `train_silence_thought_<JOB_ID>.out`
- **Format**: `train_silence_thought_<JOB_ID>.err`

## Customizing Training Parameters

To modify training parameters, edit the sbatch file:

```bash
# Example: Change batch size and epochs
python main_train_silence_thought.py \
    --config config/data_egd-cxr.yaml \
    --batch-size 6 \
    --epochs 30 \
    --lr 3e-4 \
    # ... other parameters
```

## Environment Setup

The sbatch files assume:
1. Python environment is activated
2. Required modules are loaded
3. CUDA is available
4. Project directory is accessible

### Manual Environment Setup
```bash
# Activate conda environment
source /project/hnguyen2/mvu9/conda_envs/wsi-agent/bin/activate

# Or use conda activate
conda activate wsi-agent

# Load modules (if needed)
module load python/3.11
module load cuda/11.8
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or max fixations
2. **CUDA Errors**: Check GPU availability and CUDA installation
3. **Data Loading Errors**: Verify data paths in config file
4. **Permission Errors**: Check file permissions and directory access

### Debug Mode
For debugging, use the quick training script:
```bash
sbatch train_silence_thought_quick.sbatch
```

### Check System Resources
```bash
# Check available GPUs
nvidia-smi

# Check available memory
free -h

# Check disk space
df -h
```

## Best Practices

1. **Start with Quick Training**: Test your setup with the quick script first
2. **Monitor Resources**: Use `htop` or `nvidia-smi` to monitor usage
3. **Save Checkpoints**: The training script automatically saves best and final checkpoints
4. **Check Logs**: Regularly monitor output and error logs
5. **Backup Results**: Copy important checkpoints to a safe location

## Example Workflow

```bash
# 1. Test with quick training
sbatch train_silence_thought_quick.sbatch

# 2. Monitor the job
squeue -u $USER
tail -f logs/train_quick_<JOB_ID>.out

# 3. If successful, run standard training
sbatch train_silence_thought.sbatch

# 4. For production, use full training
sbatch train_silence_thought_full.sbatch
```

## Configuration Files

Make sure your configuration file (`config/data_egd-cxr.yaml`) contains the correct paths:

```yaml
dataset_name: "egd-cxr"
input_path:
  gaze_raw: /path/to/gaze/data
  dicom_raw: /path/to/dicom/data
  segmentation_dir: /path/to/segmentation/data
  transcripts_dir: /path/to/transcript/data
output_path:
  base_plots_dir: plots
split_files:
  dir: config/splits
```
