#!/bin/bash

# ./run.sh <mode> <dataset> <cuda>
mode=$1    # train, test
dataset=$2    # E-Commerce, Douban, Zh50w (or LCCC)
cuda=$3

mkdir ckpt
mkdir rest
mkdir bak
