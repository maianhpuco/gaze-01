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