#!/bin/bash


source_root=./data/source
split_seed=42
split_folds=5

# --------------------------------------

python process.py \
    --source_root    $source_root \
    --process_method log2

python process.py \
    --source_root    $source_root \
    --process_method plain

python split.py \
    --data_root   $source_root \
    --split_seed  $split_seed  \
    --split_folds $split_folds

# --------------------------------------

# process_root=./data/process_log2

# python process.py \
#     --source_root    $source_root  \
#     --process_method log2          \
#     --process_root   $process_root \
#     --apply_preprocess

# python split.py \
#     --data_root   $process_root \
#     --split_seed  $split_seed   \
#     --split_folds $split_folds

# --------------------------------------

# process_root=./data/process_plain

#     --source_root    $source_root  \
#     --process_method plain         \
#     --process_root   $process_root \
#     --apply_preprocess

# python split.py \
#     --data_root   $process_root \
#     --split_seed  $split_seed   \
#     --split_folds $split_folds
