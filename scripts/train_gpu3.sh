#!/bin/bash


device=3
folds=0
# folds=3,4
process=plain
config_file_list=(
    # ./configs/swin_unetr/exp4.yaml
    # ./configs/swin_unetr/exp5.yaml
    # ./configs/swin_unetr/exp8.yaml
    # ./configs/swin_unetr/exp9.yaml
    # ./configs/swin_unetr/exp10.yaml
    ./configs/swin_unetr/exp12.yaml
    # ./configs/vt2unet/exp1.yaml
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