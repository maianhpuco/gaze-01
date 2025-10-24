PYTHON ?= python

.PHONY: prepare_transcripts test_dataset train_model eval_model

prepare_transcripts:
	$(PYTHON) prepare_transcripts.py --config-path config/data_egd-cxr.yaml

test_dataset:
	$(PYTHON) sanity_check_dataset.py --config-path config/data_egd-cxr.yaml --split train --batch-size 1 --num-batches 1 --max-fixations 5 --show-json --print-raw --output-dir sanity_check/dataset

train_model:
	$(PYTHON) main_train_silence_thought.py --config config_maui/st_edg_cxr.yaml

eval_model:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Missing CHECKPOINT variable. Usage: make eval_model CHECKPOINT=/path/to/checkpoint.pt"; \
		exit 1; \
	fi
	$(PYTHON) main_test_silence_thought.py --config config_maui/st_edg_cxr.yaml --checkpoint $(CHECKPOINT)
