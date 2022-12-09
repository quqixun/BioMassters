import os
import gc
import sys
import pickle
import argparse
import warnings
import numpy as np

from tqdm import tqdm
from libs.process import *
from os.path import join as opj


warnings.filterwarnings('ignore')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BioMassters Preprocessing')
    parser.add_argument('--source_root',      type=str, help='dir path of source dataset')
    parser.add_argument('--process_method',   type=str, help='method for processing, log2 or plain')
    parser.add_argument('--process_root',     type=str, help='dir path of processed dataset')
    parser.add_argument('--apply_preprocess', action='store_true',
                        help='if save preprocessed dataset, it will take huge space on dick')
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # input arguments

    source_root      = args.source_root
    process_root     = args.process_root
    process_method   = args.process_method
    apply_preprocess = args.apply_preprocess
    assert process_method in ['log2', 'plain']

    # --------------------------------------------------------------------------
    # creats path for output files and directories

    if apply_preprocess:
        assert process_root is not None
        process_data_dir = os.path.join(process_root, 'train')
        os.makedirs(process_data_dir, exist_ok=True)
        plot_dir = os.path.join(process_root, 'plot')
        os.makedirs(plot_dir, exist_ok=True)
        stats_path_in_process = os.path.join(process_root, 'stats.pkl')
        stats_path_in_source  = os.path.join(source_root, f'stats_{process_method}.pkl')
    else:
        plot_dir = os.path.join(source_root, 'plot', process_method)
        os.makedirs(plot_dir, exist_ok=True)
        stats_path_in_process = None
        stats_path_in_source  = os.path.join(source_root, f'stats_{process_method}.pkl')

    # --------------------------------------------------------------------------
    # gets list of all subjects for training

    source_data_dir = os.path.join(source_root, 'train')
    subjects = os.listdir(source_data_dir)
    subjects.sort()

    # --------------------------------------------------------------------------
    # gets data and function according to processing method

    if process_method == 'log2':
        percentile           = None
        exclude_mins4label   = False
        exclude_mins4feature = False
        remove_outliers_func = remove_outliers_by_log2

    elif process_method == 'plain':
        percentile           = 99.9
        exclude_mins4label   = False
        exclude_mins4feature = False
        remove_outliers_func = remove_outliers_by_plain

    # --------------------------------------------------------------------------
    # computes statistics of agbm labels

    print('Label')
    stats = {}
    label_list = []
    for subject in tqdm(subjects, ncols=88):
        subject_dir = opj(source_data_dir, subject)
        label_path = opj(subject_dir, f'{subject}_agbm.tif')

        label = read_raster(label_path)
        label = remove_outliers_func(label, 'label')

        if label is not None:
            label_list.append(label)

    label = np.array(label_list)
    stats['label'] = calculate_statistics(
        label, 'label', exclude_mins=exclude_mins4label,
        p=percentile, hist=True, plot_dir=plot_dir
    )
    del label_list, label
    gc.collect()

    # --------------------------------------------------------------------------
    # computes statistics of S1 and S2 features

    feat_dict = {'S1': 4, 'S2': 11}
    for fname, fnum in feat_dict.items():
        stats[fname] = {}

        for index in range(fnum):
            print(f'Feature: {fname} - index: {index}')
            ith_feat_list = []
            for subject in tqdm(subjects, ncols=88):
                subject_dir = opj(source_data_dir, subject)

                for month in range(12):
                    feat_file = f'{subject}_{fname}_{month:02d}.tif'
                    feat_path = opj(subject_dir, fname, feat_file)
                    feat = read_raster(feat_path)
                    if feat is not None:
                        assert feat.shape[0] == fnum
                        ith_feat = feat[index]
                        ith_feat = remove_outliers_func(ith_feat, fname, index)
                        ith_feat_list.append(ith_feat)

            ith_feat = np.array(ith_feat_list)
            ith_fname = f'{fname}-{index}'
            stats[fname][index] = calculate_statistics(
                ith_feat, ith_fname, exclude_mins=exclude_mins4feature,
                p=percentile, hist=True, plot_dir=plot_dir
            )
            del ith_feat_list, ith_feat
            gc.collect()

    # --------------------------------------------------------------------------
    # save statistics

    if stats_path_in_process is not None:
        with open(stats_path_in_process, 'wb') as f:
            pickle.dump(stats, f)
    
    with open(stats_path_in_source, 'wb') as f:
        pickle.dump(stats, f)

    print(stats)

    # --------------------------------------------------------------------------
    # processes dataset

    if not apply_preprocess:
        sys.exit(0)

    print('Preprocessing ...')
    for subject in tqdm(subjects, ncols=88):
        subject_dir = os.path.join(source_data_dir, subject)

        # loads label data
        label_path = opj(subject_dir, f'{subject}_agbm.tif')
        assert os.path.isfile(label_path), f'label {label_path} is not exist'
        label_src = read_raster(label_path, True, GT_SHAPE)
        label = remove_outliers_func(label_src, 'label')
        label = normalize(label, stats['label'], 'minmax')

        label_src = np.expand_dims(label_src, axis=-1)
        label = np.expand_dims(label, axis=-1)
        # label_src, label: (1, 256, 256, 1)

        # loads S1 and S2 features
        feature_list = []
        for month in range(12):
            s1_path = opj(subject_dir, 'S1', f'{subject}_S1_{month:02d}.tif')
            s2_path = opj(subject_dir, 'S2', f'{subject}_S2_{month:02d}.tif')
            s1 = read_raster(s1_path, True, S1_SHAPE)
            s2 = read_raster(s2_path, True, S2_SHAPE)

            s1_list = []
            for index in range(4):
                s1i = remove_outliers_func(s1[index], 'S1', index)
                s1i = normalize(s1i, stats['S1'][index], 'zscore')
                s1_list.append(s1i)

            s2_list = []
            for index in range(11):
                s2i = remove_outliers_func(s2[index], 'S2', index)
                s2i = normalize(s2i, stats['S2'][index], 'zscore')
                s2_list.append(s2i)

            feature = np.stack(s1_list + s2_list, axis=-1)
            feature = np.expand_dims(feature, axis=0)
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

        subject_path = os.path.join(process_data_dir, f'{subject}.pkl')
        with open(subject_path, 'wb') as f:
            pickle.dump(subject_dict, f)
