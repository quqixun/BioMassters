#!/bin/bash


device=0
folds=0
# folds=1,2,3,4
process=plain
config_file_list=(
    # ./configs/swin_unetr/exp4-2.yaml
    # ./configs/swin_unetr/exp6.yaml
    ./configs/swin_unetr/exp7.yaml
    # ./configs/swin_unetr/exp8.yaml
    # ./configs/swin_unetr/exp9.yaml
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