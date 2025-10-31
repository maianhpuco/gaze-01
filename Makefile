PYTHON ?= python

# .PHONY: prepare_transcripts test_dataset train_model eval_model

# prepare_transcripts:
# 	$(PYTHON) prepare_transcripts.py --config-path config/data_egd-cxr.yaml

# test_dataset:
# 	$(PYTHON) sanity_check_dataset.py --config-path config/data_egd-cxr.yaml --split train --batch-size 1 --num-batches 1 --max-fixations 5 --show-json --print-raw --output-dir sanity_check/dataset

# train_model:
# 	$(PYTHON) main_train_silence_thought.py --config config_maui/st_edg_cxr.yaml

# eval_model:
# 	@if [ -z "$(CHECKPOINT)" ]; then \
# 		echo "Missing CHECKPOINT variable. Usage: make eval_model CHECKPOINT=/path/to/checkpoint.pt"; \
# 		exit 1; \
# 	fi
# 	$(PYTHON) main_test_silence_thought.py --config config_maui/st_edg_cxr.yaml --checkpoint $(CHECKPOINT)


split:
	create_splits.py --config-path configs/data_egd_cxr_single_label.yaml --output-dir configs/splits --train 0.7 --val 0.1 --test 0.2 --seed 42 --stratify 

# Train with all modalities enabled
train_all: 
	python main_st01_prtr_img_bs_gaze_text.py --config configs/st01_prtr_img_bs_gaze_text.yaml

# Train with custom parameters

	# python main_st01_prtr_img_bs_gaze_text.py --config configs/st01_prtr_img_bs_gaze_text.yaml --epochs 30 --batch-size 8

# Experiment with disabled components
	# python main_st01_prtr_img_bs_gaze_text.py --config configs/st01_prtr_img_bs_gaze_text.yaml --no-text --no-gaze


	# Submit full multimodal training
# sbatch sbatch_files/train_st01_img_bs_gaze_text.sbatch

# # Submit gaze-only training for baseline
# sbatch sbatch_files/train_st01_gaze_only.sbatch

# # Submit without images (since DICOM files aren't available)
# sbatch sbatch_files/train_st01_bbox_seg_gaze_text.sbatch 

# Quick test (1 epoch) - 2 hours

seq:
	sbatch sbatch_files/sbatch_sigma_seq_001_eps.sbatch
	sbatch sbatch_files/sbatch_sigma_seq_010_eps.sbatch
	sbatch sbatch_files/sbatch_sigma_seq_020_eps.sbatch
	sbatch sbatch_files/sbatch_sigma_seq_100_eps.sbatch 


dicom: 
	sbatch sbatch_files/sbatch_st01_img_dicom_001_eps.sbatch
	sbatch sbatch_files/sbatch_st01_img_dicom_010_eps.sbatch
	sbatch sbatch_files/sbatch_st01_img_dicom_020_eps.sbatch
	sbatch sbatch_files/sbatch_st01_img_dicom_100_eps.sbatch 
	sbatch sbatch_files/sbatch_st01_img_dicom_200_eps.sbatch 

check_data:
	python main_dataset.py --config configs/data_egd_cxr_single_label.yaml --save-json 



# export TORCHXRV_CACHE=/project/hnguyen2/mvu9/.cache/torchxrayvision
# export HF_HOME=/project/hnguyen2/mvu9/.cache/huggingface


train_full_adv:
	python main_train.py --config configs/st_tmrnn.yaml 

all_cnn:
	sbatch sbatch_files/train_imgcnn_r18.sbatch
	sbatch sbatch_files/train_imgcnn_r50.sbatch
	sbatch sbatch_files/train_imgcnn_d121.sbatch
	sbatch sbatch_files/train_imgcnn_txrvd121.sbatch 

tmrnn: 
	sbatch sbatch_files/train_tmrnn_r18.sbatch
	sbatch sbatch_files/train_tmrnn_r50.sbatch
	sbatch sbatch_files/train_tmrnn_d121.sbatch
	sbatch sbatch_files/train_tmrnn_txrvd121.sbatch 