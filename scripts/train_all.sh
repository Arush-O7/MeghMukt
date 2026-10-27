#!/bin/bash
set -e
cd "$(dirname "$0")/.."
run() { tag=$1; shift; [ -f "checkpoints/$tag.pt" ] && [ -f "results/train_logs/$tag.json" ] && { echo "skip $tag"; return; }; python3 scripts/03_train.py "$@"; }
run unet               --model unet
run meghgan            --model meghgan
run meghgan_nospectral --model meghgan --no-spectral --tag meghgan_nospectral
run diffusion          --model diffusion
run meghgan_liss4ft    --model meghgan --finetune checkpoints/meghgan.pt
run unet_liss4ft       --model unet --finetune checkpoints/unet.pt
echo ALL TRAINING DONE
