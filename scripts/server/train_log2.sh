#!/bin/bash


device=2
data_source=/mnt/dataset/quqixun/Github/BioMassters/data/source
data_log2=/mnt/dataset/quqixun/Github/BioMassters/data/process_log2


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
