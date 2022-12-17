#!/bin/bash


device=0
folds=0
# folds=1,2,3,4
process=plain
config_file_list=(
    ./configs/vtnet/exp1.yaml
)

for config_file in ${config_file_list[@]}; do

    CUDA_VISIBLE_DEVICES=$device \
    python train.py              \
        --data_root      ./data/source          \
        --exp_root       ./experiments/$process \
        --config_file    $config_file           \
        --folds          $folds                 \
        --process_method $process

done;