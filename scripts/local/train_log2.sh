#!/bin/bash


device=0
data_source=./data/source
data_log2=./data/process_log2


# CUDA_VISIBLE_DEVICES=$device \
# python train.py              \
#     --data_root      $data_source        \
#     --exp_root       ./experiments       \
#     --config_file    ./configs/exp1.yaml \
#     --folds          0,1,2,3,4           \
#     --process_method log2


CUDA_VISIBLE_DEVICES=$device \
python train.py              \
    --data_root      $data_log2          \
    --exp_root       ./experiments       \
    --config_file    ./configs/exp1.yaml \
    --folds          0,1,2,3,4           \
    --processed                          \
    --process_method log2
