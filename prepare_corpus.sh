#!/bin/bash

# ========== HOW TO USE ========== #
# ./prepare_corpus.sh <dataset> <mode> <model> <gpu_ids>
dataset=$1
mode=$2     # es/faiss
model=$3    # es/dual-bert/hash-bert
gpus=$4

if [ $mode = 'es' ]; then
    # init elasticsearch
    python -m utils.searcher --dataset $dataset --mode $mode
elif [ $mode = 'faiss' ]; then
    # init faiss
    ./run.sh inference $dataset $model $gpus
    if [ $model = 'dual-bert' ]; then
        dim=768
    else
        dim=128
    fi
    python -m utils.searcher --dataset $dataset --mode $mode --model $model --dim $dim
else
    echo "[!] wrong mode: $mode (es or faiss)"
fi