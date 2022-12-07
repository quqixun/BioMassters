#!/bin/bash


device=0
data_source=./data/source
data_plain=./data/process_plain


# CUDA_VISIBLE_DEVICES=$device \
# python train.py              \
#     --data_root      $data_source        \
#     --exp_root       ./experiments       \
#     --config_file    ./configs/exp1.yaml \
#     --folds          0,1,2,3,4           \
#     --process_method plain


CUDA_VISIBLE_DEVICES=$device \
python train.py              \
    --data_root      $data_plain         \
    --exp_root       ./experiments       \
    --config_file    ./configs/exp1.yaml \
    --folds          0,1,2,3,4           \
    --processed                          \
    --process_method plain
