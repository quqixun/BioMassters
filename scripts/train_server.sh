#!/bin/bash


CUDA_VISIBLE_DEVICES=2,3              \
python train.py                       \
    --data_root   /mnt/dataset/quqixun/Github/BioMassters/data/source \
    --exp_root    ./experiments       \
    --config_file ./configs/exp1.yaml \
    --folds       0,1,2,3,4
