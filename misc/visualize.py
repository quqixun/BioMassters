import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from os.path import join as opj


GT_SHAPE = (1,  256, 256)
S1_SHAPE = (4,  256, 256)
S2_SHAPE = (11, 256, 256)

PROCESS_INFO = {
    'label': {'2pow': -6, 'min': -3.0, 'max': 14.0},
    'S1': {
        0: {'2pow': None, 'min': -25.0, 'max': 30.0},
        1: {'2pow': None, 'min': -63.0, 'max': 29.0},
        2: {'2pow': None, 'min': -25.0, 'max': 32.0},
        3: {'2pow': None, 'min': -70.0, 'max': 23.0},
    },
    'S2': {
        0:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        1:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        2:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        3:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        4:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        5:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        6:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        7:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        8:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        9:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        10: {'2pow': None, 'min': 0.0, 'max': 100.0},
    }
}


def read_raster(data_path, data_shape=None):

    if os.path.isfile(data_path):
        raster = rasterio.open(data_path)
        data = raster.read()
    else:
        assert data_shape is not None
        data = np.zeros(data_shape).astype(np.float32)

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


if __name__ == '__main__':

    data_dir = './data/source/train'
    subjects = os.listdir(data_dir)
    subjects.sort()

    plot_kwargs = {'vmin': 0.0, 'vmax': 1.0, 'cmap': 'coolwarm'}

    for subject in tqdm(subjects, ncols=88):
        subject_dir = opj(data_dir, subject)

        # load label data
        label_path = opj(subject_dir, f'{subject}_agbm.tif')
        label = read_raster(label_path, GT_SHAPE)
        label = process_data(label, 'label')

        # load features
        for i in range(12):
            s1_path = opj(subject_dir, 'S1', f'{subject}_S1_{i:02d}.tif')
            s2_path = opj(subject_dir, 'S2', f'{subject}_S2_{i:02d}.tif')
            s1 = read_raster(s1_path, S1_SHAPE)
            s2 = read_raster(s2_path, S2_SHAPE)

            plt.figure(f'{subject} - {i:02d}', figsize=(15, 15))
            plt.subplot(4, 4, 1)
            plt.title('GT')
            plt.imshow(label[0], **plot_kwargs)
            plt.axis('off')
            for s1_index in range(s1.shape[0]):
                s1_index_data = s1[s1_index]
                s1_index_data = process_data(s1_index_data, 'S1', s1_index)
                plt.subplot(4, 4, s1_index + 2)
                plt.title(f'S1-{s1_index + 1}')
                plt.imshow(s1_index_data, **plot_kwargs)
                plt.axis('off')
            for s2_index in range(s2.shape[0]):
                s2_index_data = s2[s2_index]
                s2_index_data = process_data(s2_index_data, 'S2', s2_index)
                plt.subplot(4, 4, s2_index + 2 + s1.shape[0])
                plt.title(f'S2-{s2_index + 1}')
                plt.imshow(s2_index_data, **plot_kwargs)
                plt.axis('off')
            plt.tight_layout()
            plt.show()
