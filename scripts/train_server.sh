#!/bin/bash


CUDA_VISIBLE_DEVICES=0                \
python train.py                       \
    --data_root   ./data/source       \
    --exp_root    ./experiments       \
    --config_file ./configs/exp1.yaml \
    --folds       0,1,2,3,4
