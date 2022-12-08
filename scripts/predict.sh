#!/bin/bash


device=0


CUDA_VISIBLE_DEVICES=$device \
python predict.py            \
    --data_root   ./data/source       \
    --exp_root    ./experiments       \
    --output_root ./predictions       \
    --config_file ./configs/exp2.yaml
