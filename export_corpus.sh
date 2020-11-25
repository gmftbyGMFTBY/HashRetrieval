#!/bin/bash

# export the samples under the generated/<dataset_name>/ folder

datasets=(ecommerce douban zh50w lccc)
for d in ${datasets[@]}
do
    python -m utils.annotate --dataset $d --seed 50 --samples 200 --max_length 100
done