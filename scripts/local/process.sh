#!/bin/bash


source_root=./data/source
split_seed=42
split_folds=5

# --------------------------------------

python split.py \
    --data_root   $source_root \
    --split_seed  $split_seed  \
    --split_folds $split_folds

# --------------------------------------

process_root=./data/process_log2
process_method=log2

python process.py \
    --source_root    $source_root    \
    --process_root   $process_root   \
    --process_method $process_method \
    --apply_preprocess

python split.py \
    --data_root   $process_root \
    --split_seed  $split_seed   \
    --split_folds $split_folds

# --------------------------------------

process_root=./data/process_plain
process_method=plain

python process.py \
    --source_root    $source_root    \
    --process_root   $process_root   \
    --process_method $process_method \
    --apply_preprocess

python split.py \
    --data_root   $process_root \
    --split_seed  $split_seed   \
    --split_folds $split_folds
