#!/bin/bash


device=1
folds=0
# folds=1,2,3,4
process=plain
config_file_list=(
    ./configs/swin_unetr/exp30.yaml
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
