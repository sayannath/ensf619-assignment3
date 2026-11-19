#!/bin/bash
# Runs every measurement. Needs a CUDA GPU.
set -e

python run_experiment.py
python run_experiment.py --configs optimized baseline --out results/check_reversed.csv \
    --index-out results/check_reversed_index.csv
python k_sweep.py
python check_facts.py
python retrieval_energy.py
python make_grading_sheet.py
python summarize.py
