PYTHON ?= python

.PHONY: prepare_transcripts test_dataset train_model

prepare_transcripts:
	$(PYTHON) prepare_transcripts.py --config-path config/data_egd-cxr.yaml

test_dataset:
	$(PYTHON) main_testdata.py --config-path config/data_egd-cxr.yaml --split train --batch-size 1 --num-batches 1 --max-fixations 5 --show-json --print-raw

train_model:
	$(PYTHON) main_train_silence_thought.py --config config/data_egd-cxr.yaml --epochs 3 --batch-size 2 --max-fixations 8
 
