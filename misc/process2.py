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


def read_raster(data_path):

    if os.path.isfile(data_path):
        raster = rasterio.open(data_path)
        data = raster.read()
    else:
        data = None

    return data


def calc_stats(data, data_name, p=None, hist=True):

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
        plot_dir = './data/source/plot'
        os.makedirs(plot_dir, exist_ok=True)
        plot_file = f'stats_{data_name}_p{p}.png'
        plot_path = opj(plot_dir, plot_file)

        plt.figure()
        plt.title(f'{data_name} - P:{p}')
        plt.hist(data.reshape(-1), bins=100, log=True)
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
    stats['label'] = {}
    for subject in tqdm(subjects, ncols=88):
        subject_dir = opj(data_dir, subject)

        # load label data
        label_path = opj(subject_dir, f'{subject}_agbm.tif')
        label = read_raster(label_path)
        if label is not None:
            label_list.append(label)

    label = np.array(label_list)
    stats['label'][None] = calc_stats(label, 'label', p=None)
    stats['label'][99.5] = calc_stats(label, 'label', p=99.5)
    stats['label'][99.9] = calc_stats(label, 'label', p=99.9)

    label = np.where(label < 2 ** (-6), 2 ** (-6), label)
    label_log2 = np.log2(label)
    stats['label']['log2'] = calc_stats(label_log2, 'label-log2', p=None)

    del label_list, label, label_log2
    gc.collect()

    # --------------------------------------------------------------------------
    # statistics of S1 and S2

    feat_dict = {'S1': 4, 'S2': 11}
    for fname, fnum in feat_dict.items():
        stats[fname] = {}

        for i in range(fnum):
            ith_feat_list = []
            stats[fname][i] = {}

            for subject in tqdm(subjects, ncols=88):
                subject_dir = opj(data_dir, subject)

                for j in range(12):
                    feat_file = f'{subject}_{fname}_{j:02d}.tif'
                    feat_path = opj(subject_dir, fname, feat_file)
                    feat = read_raster(feat_path)
                    if feat is not None:
                        assert feat.shape[0] == fnum
                        ith_feat = feat[i]
                        ith_feat_list.append(ith_feat)

            ith_feat = np.array(ith_feat_list)
            ith_fname = f'{fname}-{i}'
            stats[fname][i][None] = calc_stats(ith_feat, ith_fname, p=None)
            stats[fname][i][99.5] = calc_stats(ith_feat, ith_fname, p=99.5)
            stats[fname][i][99.9] = calc_stats(ith_feat, ith_fname, p=99.9)
            del ith_feat_list, ith_feat
            gc.collect()

    # --------------------------------------------------------------------------
    # save statistics

    stats_path = './data/source/stats.pkl'
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
