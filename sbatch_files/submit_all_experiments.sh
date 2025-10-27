#!/bin/bash
# Quick submission script for all ST01 experiments

echo "Submitting all ST01 experiments..."

# Core experiments
echo "Submitting core experiments..."
sbatch sbatch_files/train_st01_prtr_img_bs_gaze_text.sbatch

# Image-based experiments
echo "Submitting image-based experiments..."
sbatch sbatch_files/train_st01_prtr_img_gaze.sbatch
sbatch sbatch_files/train_st01_prtr_img_bs.sbatch
sbatch sbatch_files/train_st01_prtr_img.sbatch
sbatch sbatch_files/train_st01_prtr_img_text.sbatch
sbatch sbatch_files/train_st01_prtr_img_bs_text.sbatch

# Gaze-based experiments
echo "Submitting gaze-based experiments..."
sbatch sbatch_files/train_st01_gaze.sbatch
sbatch sbatch_files/train_st01_gaze_text.sbatch

# Best performing combination
echo "Submitting best performing combinations..."
sbatch sbatch_files/train_st01_bs_text.sbatch

# Additional combinations
echo "Submitting additional combinations..."
sbatch sbatch_files/train_st01_bs_gaze_text.sbatch
sbatch sbatch_files/train_st01_bs_gaze.sbatch

echo "All experiments submitted!"
echo "Use 'squeue -u \$USER' to monitor job status"
