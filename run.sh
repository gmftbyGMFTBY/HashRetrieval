#!/bin/bash

# ========== How to run this script ========== #
# ./run.sh <train/test> <dataset_name> <model_name> <cuda_ids>
# for example: ./run/sh ecommerce train dual-bert 0,1,2,3
mode=$1
dataset=$2
model=$3
cuda=$4

if [ $mode = 'init' ]; then
    models=(dual-bert cross-bert hash-bert bert-ruber bert-ruber-ft)
    datasets=(ecommerce douban zh50w lccc)
    mkdir bak ckpt rest generated
    for m in ${models[@]}
    do
        for d in ${datasets[@]}
        do
            mkdir -p ckpt/$d/$m
            mkdir -p rest/$d/$m
            mkdir -p bak/$d/$m
        done
    done
    for d in ${datasets[@]}
    do
        mkdir -p generated/$d/es 
        mkdir -p generated/$d/dense
        mkdir -p generated/$d/hash
    done
elif [ $mode = 'statistic' ]; then
    python -m utils.statistic --dataset ecommerce
    python -m utils.statistic --dataset douban
    python -m utils.statistic --dataset zh50w
elif [ $mode = 'backup' ]; then
    cp ckpt/$dataset/$model/* bak/$dataset/$model/
    cp rest/$dataset/$model/event* bak/$dataset/$model/
elif [ $mode = 'train' ]; then
    ./run.sh backup $dataset $model
    rm ckpt/$dataset/$model/*
    rm rest/$dataset/$model/events*    # clear the tensorboard cache
    
    # cross-bert: 32; dual-bert, bert-ruber, bert-ruber-ft: 16; Hash-bert: 16~128.
    if [ $model = 'cross-bert' ]; then
        batch_size=32
    elif [ $model = 'hash-bert' ]; then
        batch_size=16
    else
        batch_size=16
    fi
    
    gpu_ids=(${cuda//,/ })
    CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 29500 main.py \
        --dataset $dataset \
        --model $model \
        --mode train \
        --batch_size $batch_size \
        --epoch 5 \
        --seed 50 \
        --max_len 256 \
        --multi_gpu $cuda \
        --hash_code_size 512 \
        --neg_samples $batch_size
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
    CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 29300 main.py \
        --mode inference \
        --dataset $dataset \
        --model $model \
        --multi_gpu $cuda \
        --max_len 256 \
        --seed 50 \
        --batch_size 64 \
        --hash_code_size 512 \
        --neg_samples 16
        
    # reconstruct the results
    python -m utils.reconstruct --model $model --dataset $dataset --num_nodes ${#gpu_ids[@]}
elif [ $model = 'ruber-score' ]; then
    echo ""
else
    echo "[!] mode needs to be init/backup/train/test/inference, but got $mode"
fi