#!/bin/bash
set -e
cd "$(dirname "$0")/.."
python3 scripts/03_train.py --model meghgan --no-adv --tag meghgan_noadv
python3 scripts/03_train.py --model meghgan --charbonnier --tag meghgan_charb
echo E4 TRAINING DONE
