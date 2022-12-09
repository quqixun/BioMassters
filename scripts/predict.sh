#!/bin/bash


device=0
data_root=./data/source


CUDA_VISIBLE_DEVICES=$device \
python predict.py            \
    --data_root      $data_root          \
    --exp_root       ./experiments/log2  \
    --output_root    ./predictions/log2  \
    --config_file    ./configs/exp1.yaml \
    --process_method log2                \
    --folds          0,3


# CUDA_VISIBLE_DEVICES=$device \
# python predict.py            \
#     --data_root      $data_root          \
#     --exp_root       ./experiments/plain \
#     --output_root    ./predictions/plain \
#     --config_file    ./configs/exp1.yaml \
#     --process_method plain               \
#     --folds          0,1,2,3,4
