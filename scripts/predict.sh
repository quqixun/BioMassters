#!/bin/bash


device=0
data_root=./data/source
config_file=./configs/swin_unetr/exp1.yaml


CUDA_VISIBLE_DEVICES=$device \
python predict.py            \
    --data_root      $data_root          \
    --exp_root       ./experiments/log2  \
    --output_root    ./predictions/log2  \
    --config_file    $config_file        \
    --process_method log2                \
    --folds          0,1,2,3,4


CUDA_VISIBLE_DEVICES=$device \
python predict.py            \
    --data_root      $data_root          \
    --exp_root       ./experiments/plain \
    --output_root    ./predictions/plain \
    --config_file    $config_file        \
    --process_method plain               \
    --folds          0,1,2,3,4
