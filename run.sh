#!/bin/bash

# ========== How to run this script ========== #
# ./run.sh <train/test> <dataset_name> <model_name> <cuda_ids>
# for example: ./run/sh ecommerce train dual-bert 0,1,2,3
mode=$1
dataset=$2
model=$3
cuda=$4

if [ $mode = 'init' ]; then
    models=(dual-bert cross-bert)
    datasets=(ecommerce douban LCCC)
    mkdir bak ckpt rest
    for m in ${models[@]}
    do
        for d in ${datasets[@]}
        do
            mkdir -p ckpt/$d/$m
            mkdir -p rest/$d/$m
            mkdir -p bak/$d/$m
        done
    done
elif [ $mode = 'backup' ]; then
    cp ckpt/$dataset/$model/* bak/$dataset/$model/
elif [ $mode = 'train' ]; then
    ./run.sh backup $dataset $model
    rm ckpt/$dataset/$model/*
    rm rest/$dataset/$model/events*    # clear the tensorboard cache
    
    # batch for cross-bert is 32, for dual-bert is 16
    gpu_ids=(${cuda//,/ })
    CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 29400 main.py \
        --dataset $dataset \
        --model $model \
        --mode train \
        --batch_size 16 \
        --epoch 5 \
        --seed 50 \
        --max_len 256 \
        --multi_gpu $cuda
elif [ $mode = 'test' ]; then
    one_batch_model=(dual-bert)
    if [[ ${one_batch_model[@]} =~ $model ]]; then
        batch_size=1
    else
        batch_size=32
    fi

    CUDA_VISIBLE_DEVICES=$cuda python main.py \
        --dataset $dataset \
        --model $model \
        --mode test \
        --batch_size $batch_size \
        --max_len 256 \
        --seed 50 \
        --multi_gpu $cuda
elif [ $mode = 'inference' ]; then
    # inference and generate the real-vector for the utterances (faiss uses it)
    gpu_ids=(${cuda//,/ })
    CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 29400 main.py \
        --mode inference \
        --dataset $dataset \
        --model $model \
        --multi_gpu $cuda \
        --max_len 256 \
        --seed 50 \
        --batch_size 32
    # reconstruct the results
    python -m utils.reconstruct --model $model --dataset $dataset --num_nodes ${#gpu_ids[@]}
else
    echo "[!] mode needs to be init/backup/train/test/inference, but got $mode"
fi

