#!/bin/bash

# ========== HOW TO USE ========== #
# ./chat.sh <dataset> <coarse_mode> <gpu_id>
# coarse_mode: es, dense, hash

dataset=$1
coarse_mode=$2
cuda=$3

CUDA_VISIBLE_DEVICES=$cuda python agent.py \
    --dataset $dataset \
    --coarse $coarse_mode \
    --batch_size 32 \
    --seed 50 \
    --max_len 256 \
    --topk 100 \
    --test_mode coarse