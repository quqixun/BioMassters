import os
import gc
import pickle
import rasterio
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from os.path import join as opj


matplotlib.use('Agg')


PROCESS_INFO = {
    'label': {'2pow': -6, 'min': -3.0, 'max': 14.0},
    'S1': {
        0: {'2pow': None, 'min': -25.0, 'max': 30.0},
        1: {'2pow': None, 'min': -61.0, 'max': 30.0},
        2: {'2pow': None, 'min': -25.0, 'max': 30.0},
        3: {'2pow': None, 'min': -70.0, 'max': 20.0},
    },
    'S2': {
        0:  {'2pow': -1,   'min': 0.0, 'max': 15.0 },
        1:  {'2pow': -1,   'min': 0.0, 'max': 15.0 },
        2:  {'2pow': -1,   'min': 0.0, 'max': 15.0 },
        3:  {'2pow': -1,   'min': 0.0, 'max': 15.0 },
        4:  {'2pow': -1,   'min': 0.0, 'max': 15.0 },
        5:  {'2pow': -1,   'min': 0.0, 'max': 15.0 },
        6:  {'2pow': -1,   'min': 0.0, 'max': 15.0 },
        7:  {'2pow': -1,   'min': 0.0, 'max': 15.0 },
        8:  {'2pow': -1,   'min': 0.0, 'max': 15.0 },
        9:  {'2pow': -1,   'min': 0.0, 'max': 15.0 },
        10: {'2pow': None, 'min': 0.0, 'max': 105.0},
    }
}


def read_raster(data_path):

    if os.path.isfile(data_path):
        raster = rasterio.open(data_path)
        data = raster.read()
    else:
        data = None

    return data


def process_data(data, data_name, data_index=None):

    if data_name == 'label':
        process_dict = PROCESS_INFO['label']
    else:
        process_dict = PROCESS_INFO[data_name][data_index]
    
    if process_dict['2pow'] is not None:
        min_thresh = 2 ** process_dict['2pow']
        data = np.where(data < min_thresh, min_thresh, data)
        data = np.log2(data)
    
    min_ = process_dict['min']
    max_ = process_dict['max']
    range_ = max_ - min_

    if (data_name == 'S2') and (data_index == 10):
        data = np.where(data < min_, min_, data)
        data = np.where(data > max_, min_, data)
    else:
        data = np.where(data < min_, min_, data)
        data = np.where(data > max_, max_, data)
    data = (data - min_) / range_

    return data


def calc_stats(data, data_name, exclude_zero=False, p=None, hist=True):

    data_ = data.copy()

    if exclude_zero:
        data = data_[np.where(data > 0)]

    if p is not None:
        assert 0 <= p <= 100
        data_min = np.percentile(data, 100 - p)
        data_max = np.percentile(data, p)
        data = np.where(data < data_min, data_min, data)
        data = np.where(data > data_max, data_max, data)
    else:
        data_min = np.min(data)
        data_max = np.max(data)

    data_avg = np.mean(data)
    data_std = np.std(data)

    print(f'Statistics of {data_name} with percentile {p}:')
    print(f'- min: {data_min:.3f}')
    print(f'- max: {data_max:.3f}')
    print(f'- avg: {data_avg:.3f}')
    print(f'- std: {data_std:.3f}')

    if hist:
        plot_dir = f'./data/source/plot/'
        os.makedirs(plot_dir, exist_ok=True)
        plot_file = f'stats_{data_name}_p{p}.png'
        plot_path = opj(plot_dir, plot_file)

        plt.figure()
        plt.title(f'{data_name} - P:{p}')
        plt.hist(data.reshape(-1), bins=100, log=False)
        plt.tight_layout()
        # plt.show()
        plt.savefig(plot_path)
        plt.close()

    return {
        'min': data_min,
        'max': data_max,
        'avg': data_avg,
        'std': data_std,
    }


if __name__ == '__main__':

    data_dir = './data/source/train'
    subjects = os.listdir(data_dir)
    subjects.sort()

    stats = {}

    # --------------------------------------------------------------------------
    # statistics of label

    label_list = []
    for subject in tqdm(subjects, ncols=88):
        subject_dir = opj(data_dir, subject)

        # load label data
        label_path = opj(subject_dir, f'{subject}_agbm.tif')
        label = read_raster(label_path)
        if label is not None:
            label_list.append(label)

    label = np.array(label_list)
    label = process_data(label, 'label')
    stats['label'] = calc_stats(label, 'label', exclude_zero=False, p=None)
    del label_list, label
    gc.collect()

    # --------------------------------------------------------------------------
    # statistics of S1 and S2

    feat_dict = {'S1': 4, 'S2': 11}
    for fname, fnum in feat_dict.items():
        stats[fname] = {}

        for index in range(fnum):
            ith_feat_list = []

            print(f'Feature: {fname} - index: {index}')
            for subject in tqdm(subjects[:1000], ncols=88):
                subject_dir = opj(data_dir, subject)

                for month in range(12):
                    feat_file = f'{subject}_{fname}_{month:02d}.tif'
                    feat_path = opj(subject_dir, fname, feat_file)
                    feat = read_raster(feat_path)
                    if feat is not None:
                        assert feat.shape[0] == fnum
                        ith_feat = feat[index]
                        ith_feat = process_data(ith_feat, fname, index)
                        ith_feat_list.append(ith_feat)

            ith_feat = np.array(ith_feat_list)
            ith_fname = f'{fname}-{index}'
            stats[fname][index] = calc_stats(ith_feat, ith_fname, exclude_zero=True, p=None)
            del ith_feat_list, ith_feat
            gc.collect()

    # --------------------------------------------------------------------------
    # save statistics

    stats_path = './data/source/stats.pkl'
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    print(stats)
    
