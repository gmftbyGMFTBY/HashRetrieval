#!/bin/bash

# ========== HOW TO USE ========== #
# ./chat.sh <dataset> <coarse_mode> <topk> <gpu_id> <faiss_gpu_id>
# coarse_mode: es, dense, hash

dataset=$1
coarse_mode=$2
topk=$3
cuda=$4
faiss_cuda=$5

CUDA_VISIBLE_DEVICES=$cuda python agent.py \
    --dataset $dataset \
    --coarse $coarse_mode \
    --batch_size 32 \
    --seed 50 \
    --max_len 256 \
    --topk $topk \
    --test_mode coarse \
    --gpu $faiss_cuda