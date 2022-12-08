#!/bin/bash


device=2
config_file=./configs/exp1.yaml


CUDA_VISIBLE_DEVICES=$device \
python train.py              \
    --data_root      ./data/source       \
    --exp_root       ./experiments/log2  \
    --config_file    $config_file        \
    --folds          0,1,2               \
    --process_method log2


CUDA_VISIBLE_DEVICES=$device \
python train.py              \
    --data_root      ./data/source       \
    --exp_root       ./experiments/plain \
    --config_file    $config_file        \
    --folds          3,4                 \
    --process_method plain
