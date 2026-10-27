#!/bin/bash
set -e
cd "$(dirname "$0")/.."
python3 scripts/12_prepare_real_pairs.py
python3 scripts/03_train.py --model meghgan --real-finetune checkpoints/meghgan.pt
python3 scripts/03_train.py --model diffusion --real-finetune checkpoints/diffusion.pt
python3 scripts/04_evaluate.py > results/evaluate_log.txt 2>&1
python3 scripts/11_eval_real_pairs.py > results/real_pairs_log.txt 2>&1
python3 scripts/07_make_figures.py
echo REAL-CLOUD PIPELINE DONE
