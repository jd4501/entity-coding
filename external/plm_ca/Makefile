.PHONY: clean data setup mimiciv mimiciii mdace download_roberta download_models prepare_everything

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = explainable_medical_coding
PYTHON_INTERPRETER ?= python
MODULE_PREFIX := explainable_medical_coding.data
DATA_RAW ?= data/raw
DATA_PROCESSED ?= data/processed

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

setup:
	pip install poetry
	poetry config virtualenvs.in-project true
	poetry install
	poetry run pre-commit install

## Make Dataset
mimiciv:
	poetry run $(PYTHON_INTERPRETER) -m $(MODULE_PREFIX).prepare_mimiciv $(DATA_RAW) $(DATA_PROCESSED)
	poetry run $(PYTHON_INTERPRETER) -m $(MODULE_PREFIX).make_mimiciv_icd10
	poetry run $(PYTHON_INTERPRETER) -m $(MODULE_PREFIX).make_mimiciv_icd9

mimiciii:
	poetry run $(PYTHON_INTERPRETER) -m $(MODULE_PREFIX).prepare_mimiciii $(DATA_RAW) $(DATA_PROCESSED)
	poetry run $(PYTHON_INTERPRETER) -m $(MODULE_PREFIX).make_mimiciii_clean
	poetry run $(PYTHON_INTERPRETER) -m $(MODULE_PREFIX).make_mimiciii_full
	poetry run $(PYTHON_INTERPRETER) -m $(MODULE_PREFIX).make_mimiciii_50

mdace:
	poetry run $(PYTHON_INTERPRETER) -m $(MODULE_PREFIX).prepare_mdace $(DATA_RAW) $(DATA_PROCESSED)
	poetry run $(PYTHON_INTERPRETER) -m $(MODULE_PREFIX).make_mdace_icd9_inpatient
	poetry run $(PYTHON_INTERPRETER) -m $(MODULE_PREFIX).make_mdace_icd9_inpatient_code
	poetry run $(PYTHON_INTERPRETER) -m $(MODULE_PREFIX).make_mdace_icd10_inpatient

download_roberta:
	mkdir -p models
	wget https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-hf.tar.gz -P models
	tar -xvzf models/RoBERTa-base-PM-M3-Voc-hf.tar.gz -C models
	rm models/RoBERTa-base-PM-M3-Voc-hf.tar.gz
	mv models/RoBERTa-base-PM-M3-Voc/RoBERTa-base-PM-M3-Voc-hf models/roberta-base-pm-m3-voc-hf
	rm -r models/RoBERTa-base-PM-M3-Voc

download_models:
	mkdir -p models
	poetry run gdown --id 1Gna1tEQqtSrBQC_2QX_1HGpZBYAFQCix -O models/temp.tar.gz
	tar -xvzf models/temp.tar.gz
	rm models/temp.tar.gz

prepare_everything:
	$(MAKE) setup
	$(MAKE) mimiciv
	$(MAKE) mimiciii
	$(MAKE) mdace
	$(MAKE) download_roberta
	$(MAKE) download_models
