import os
import gc
import pickle
import rasterio
import warnings
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from os.path import join as opj


matplotlib.use('Agg')
warnings.filterwarnings('ignore')


GT_SHAPE = (1,  256, 256)
S1_SHAPE = (4,  256, 256)
S2_SHAPE = (11, 256, 256)


def read_raster(data_path, return_zero=False, data_shape=None):

    if os.path.isfile(data_path):
        raster = rasterio.open(data_path)
        data = raster.read()
    else:
        if return_zero:
            assert data_shape is not None
            data = np.zeros(data_shape).astype(np.float32)
        else:
            data = None

    return data


PROCESS2_INFO = {
    'S1': {
        0: {'min': -25.0, 'max': 30.0},
        1: {'min': -63.0, 'max': 29.0},
        2: {'min': -25.0, 'max': 32.0},
        3: {'min': -70.0, 'max': 23.0},
    },
    'S2': {
        10: {'min': 0.0, 'max': 100.0},
    }
}


def process2_data(
    data, data_name, data_index=None,
    norm_stats=None, norm_method='minmax'
):

    if data_name == 'label':
        data = np.where(data < 0.0, 0.0, data)
    elif data_name == 'S1':
        process_dict = PROCESS2_INFO['S1'][data_index]
        min_ = process_dict['min']
        max_ = process_dict['max']
        data = np.where(data < min_, min_, data)
        data = np.where(data > max_, max_, data)
    elif data_name == 'S2':
        if data_index == 10:
            process_dict = PROCESS2_INFO['S2'][10]
            min_ = process_dict['min']
            max_ = process_dict['max']
            data = np.where(data < min_, min_, data)
            data = np.where(data > max_, min_, data)

    if norm_stats is not None:
        if norm_method == 'minmax':
            min_ = norm_stats['min']
            max_ = norm_stats['max']
            range_ = max_ - min_
            data = (data - min_) / range_
            data = np.clip(data, 0.0, 1.0)
        elif norm_method == 'zscore':
            avg = norm_stats['avg']
            std = norm_stats['std']
            data = (data - avg) / std

    return data


def calc_stats(
    data, data_name, exclude_zero=False,
    p=None, hist=False, plot_dir=None
):

    if exclude_zero:
        data = data[np.where(data > 0)]

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

    source_dir = './data/source'
    # source_dir = '/mnt/dataset/quqixun/Github/BioMassters/data/source'
    source_data_dir = os.path.join(source_dir, 'train')
    subjects = os.listdir(source_data_dir)
    subjects.sort()

    process_dir = './data/process2'
    # process_dir = '/mnt/dataset/quqixun/Github/BioMassters/data/process1'
    process2_data_dir = os.path.join(process_dir, 'train')
    os.makedirs(process2_data_dir, exist_ok=True)
    plot_dir = os.path.join(process_dir, 'plot')
    stats_path = os.path.join(process_dir, 'stats.pkl')
    stats = {}

    # --------------------------------------------------------------------------
    # statistics of label

    label_list = []
    stats['label'] = {}
    for subject in tqdm(subjects, ncols=88):
        subject_dir = opj(source_data_dir, subject)

        # load label data
        label_path = opj(subject_dir, f'{subject}_agbm.tif')
        label = read_raster(label_path)
        label = process2_data(label, 'label')
        if label is not None:
            label_list.append(label)

    label = np.array(label_list)
    stats['label'] = calc_stats(
        label, 'label', exclude_zero=False,
        p=99.9, hist=True, plot_dir=plot_dir
    )

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
            for subject in tqdm(subjects, ncols=88):
                subject_dir = opj(source_data_dir, subject)

                for month in range(12):
                    feat_file = f'{subject}_{fname}_{month:02d}.tif'
                    feat_path = opj(subject_dir, fname, feat_file)
                    feat = read_raster(feat_path)
                    if feat is not None:
                        assert feat.shape[0] == fnum
                        ith_feat = feat[index]
                        ith_feat = process2_data(ith_feat, fname, index)
                        ith_feat_list.append(ith_feat)

            ith_feat = np.array(ith_feat_list)
            ith_fname = f'{fname}-{index}'
            stats[fname][index] = calc_stats(
                ith_feat, ith_fname, exclude_zero=False,
                p=99.9, hist=True, plot_dir=plot_dir
            )
            del ith_feat_list, ith_feat
            gc.collect()

    # --------------------------------------------------------------------------
    # save statistics

    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    print(stats)

    # --------------------------------------------------------------------------
    # processes dataset

    print('Preprocessing ...')
    for subject in tqdm(subjects, ncols=88):
        subject_dir = os.path.join(source_data_dir, subject)

        # loads label data
        label_path = opj(subject_dir, f'{subject}_agbm.tif')
        assert os.path.isfile(label_path), f'label {label_path} is not exist'
        label_src = read_raster(label_path, return_zero=True, data_shape=GT_SHAPE)
        label = process2_data(label_src, 'label', norm_stats=stats['label'], norm_method='minmax')
        label_src = np.expand_dims(label_src, axis=-1)
        label = np.expand_dims(label, axis=-1)
        # label_src, label: (1, 256, 256, 1)

        # loads S1 and S2 features
        feature_list = []
        for month in range(12):
            s1_path = opj(subject_dir, 'S1', f'{subject}_S1_{month:02d}.tif')
            s2_path = opj(subject_dir, 'S2', f'{subject}_S2_{month:02d}.tif')
            s1 = read_raster(s1_path, return_zero=True, data_shape=S1_SHAPE)
            s2 = read_raster(s2_path, return_zero=True, data_shape=S2_SHAPE)
            s1 = [process2_data(s1[i], 'S1', i, stats['S1'][i], 'zscore') for i in range(4)]
            s2 = [process2_data(s2[i], 'S2', i, stats['S2'][i], 'zscore') for i in range(11)]
            feature = np.expand_dims(np.stack(s1 + s2, axis=-1), axis=0)
            feature_list.append(feature)
        feature = np.concatenate(feature_list, axis=0)
        # feature: (12, 256, 256, 15)

        subject_dict = {
            'label_src': label_src,
            'label': label,
            'feature': feature
        }

        # for m in range(feature.shape[0]):
        #     plt.figure(f'{subject} - {m:02d}', figsize=(15, 15))
        #     plt.subplot(4, 4, 1)
        #     plt.title('GT')
        #     plt.imshow(label[0, :, :, 0], cmap='coolwarm')
        #     plt.axis('off')
        #     for f in range(feature.shape[-1]):
        #         plt.subplot(4, 4, f + 2)
        #         plt.title(f'M{m}-F{f + 1}')
        #         plt.imshow(feature[m, :, :, f], cmap='coolwarm')
        #         plt.axis('off')
        #     plt.tight_layout()
        #     plt.show()

        subject_path = os.path.join(process2_data_dir, f'{subject}.pkl')
        with open(subject_path, 'wb') as f:
            pickle.dump(subject_dict, f)
