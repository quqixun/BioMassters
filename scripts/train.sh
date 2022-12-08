#!/bin/bash


device=0
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


# CUDA_VISIBLE_DEVICES=$device \
# python train.py              \
#     --data_root      ./data/process_log2 \
#     --exp_root       ./experiments/log2  \
#     --config_file    ./configs/exp1.yaml \
#     --folds          0,1,2,3,4           \
#     --process_method log2                \
#     --processed


# CUDA_VISIBLE_DEVICES=$device \
# python train.py              \
#     --data_root      ./data/process_plain \
#     --exp_root       ./experiments/plain  \
#     --config_file    ./configs/exp1.yaml  \
#     --folds          0,1,2,3,4            \
#     --process_method plain                \
#     --processed
