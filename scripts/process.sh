#!/bin/bash


source_root=./data/source
# source_root=/mnt/dataset/quqixun/Github/BioMassters/data/source


process_root=./data/process_log2
# process_root=/mnt/dataset/quqixun/Github/BioMassters/data/process_log2
process_method=log2

python process.py \
    --source_root    $source_root  \
    --process_root   $process_root \
    --process_method $process_method


process_root=./data/process_plain
# process_root=/mnt/dataset/quqixun/Github/BioMassters/data/process_plain
process_method=plain

python process.py \
    --source_root    $source_root  \
    --process_root   $process_root \
    --process_method $process_method
