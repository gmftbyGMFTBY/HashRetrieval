#!/bin/bash

# ========== HOW TO USE ========== #
# ./prepare_corpus.sh <dataset> <mode> <gpu_ids>
dataset=$1
mode=$2     # es/faiss
gpus=$3

if [ $mode = 'es' ]; then
    # init elasticsearch
    python -m utils.searcher --dataset $dataset --mode $mode
elif [ $mode = 'faiss' ]; then
    # init faiss
    ./run.sh inference $dataset dual-bert $gpus
    python -m utils.searcher --dataset $dataset --mode $mode --model dual-bert --dim 768
else
    echo "[!] wrong mode: $mode (es or faiss)"
fi