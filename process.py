import os
import gc
import argparse

from tqdm import tqdm
from libs.process import *
from os.path import join as opj



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BioMassters Training')
    parser.add_argument('--source_root',    type=str, help='dir path of source data')
    parser.add_argument('--process_root',   type=str, help='dir path of processed data')
    parser.add_argument('--process_method', type=str, help='method for processing, log2 or plain')
    args = parser.parse_args()

    # main(args)

    # --------------------------------------------------------------------------
    # input arguments

    source_root    = args.source_root
    process_root   = args.process_root
    process_method = args.process_method
    assert process_method in ['log2', 'plain']

    # source_root = './data/source'
    # source_root = '/mnt/dataset/quqixun/Github/BioMassters/data/source'

    # process_root = './data/process_log2'
    # process_root = '/mnt/dataset/quqixun/Github/BioMassters/data/process_log2'
    # process_method = 'log2'

    # process_root = './data/process_plain'
    # process_root = '/mnt/dataset/quqixun/Github/BioMassters/data/process_plain'
    # process_method = 'plain'

    # --------------------------------------------------------------------------
    # creats path for output files and directories

    process_data_dir = os.path.join(process_root, 'train')
    os.makedirs(process_data_dir, exist_ok=True)

    plot_dir = os.path.join(process_root, 'plot')
    os.makedirs(plot_dir, exist_ok=True)
    
    stats_path = os.path.join(process_root, 'stats.pkl')
    stats = {}

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
