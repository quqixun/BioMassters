import os
import pickle
import random

from copy import deepcopy


if __name__ == '__main__':

    seed = 42
    k_folds = 5

    # source_dir = './data/source'
    source_dir = '/mnt/dataset/quqixun/Github/BioMassters/data/source'

    data_dir = os.path.join(source_dir, 'train')
    subjects = os.listdir(data_dir)
    random.seed(seed)
    random.shuffle(subjects)

    per_fold = round(len(subjects) / k_folds)

    splits = {}
    for k in range(k_folds):
        start = k * per_fold
        end = (k + 1) * per_fold if k < (k_folds - 1) else len(subjects)

        val = [os.path.join(data_dir, i) for i in deepcopy(subjects[start:end])]
        train = [os.path.join(data_dir, i) for i in subjects if i not in val]
        splits[k] = {'train': train, 'val': val}

    splits_path = os.path.join(source_dir, 'splits.pkl')
    with open(splits_path, 'wb') as f:
        pickle.dump(splits, f)
