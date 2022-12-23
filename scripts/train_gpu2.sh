#!/bin/bash


device=2
folds=0
# folds=1,2
process=plain
config_file_list=(
    # ./configs/swin_unetr/exp2.yaml
    # ./configs/swin_unetr/exp3.yaml
    # ./configs/swin_unetr/exp4.yaml
    # ./configs/swin_unetr/exp4-2.yaml
    # ./configs/swin_unetr/exp6.yaml
    # ./configs/swin_unetr/exp7.yaml
    # ./configs/swin_unetr/exp10.yaml
    ./configs/swin_unetr/exp11.yaml
    ./configs/swin_unetr/exp12.yaml
)

for config_file in ${config_file_list[@]}; do

    CUDA_VISIBLE_DEVICES=$device \
    python train.py              \
        --data_root      ./data/source          \
        --exp_root       ./experiments/$process \
        --config_file    $config_file           \
        --process_method $process               \
        --folds          $folds

done;