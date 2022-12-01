import os
import torch
import random
import rasterio
import numpy as np


# ------------------------------------------------------------------------------
# training initialization


def check_train_args(args):

    if not os.path.isdir(args.data_root):
        raise IOError(f'data_root {args.data_root} is not exist')

    if not os.path.isfile(args.config_file):
        raise IOError(f'config_file {args.config_file} is not exist')

    return


def init_environment(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return


# ------------------------------------------------------------------------------
# reads, preprocesses, recovers | labels and features


GT_SHAPE = (1,  256, 256)
S1_SHAPE = (4,  256, 256)
S2_SHAPE = (11, 256, 256)


def read_raster(data_path, data_shape=None):

    if os.path.isfile(data_path):
        raster = rasterio.open(data_path)
        data = raster.read()
    else:
        assert data_shape is not None
        data = np.zeros(data_shape).astype(np.float32)

    return data


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


def process_data(data, data_name, data_index=None, norm_stats=None):

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

    if (data_name != 'label') and (norm_stats is not None):
        data = (data - norm_stats['avg']) / norm_stats['std']

    return data


def recover_label(data):

    process_dict = PROCESS_INFO['label']
    min_ = process_dict['min']
    max_ = process_dict['max']
    range_ = max_ - min_

    data = data * range_ + min_
    data = 2 ** data
    min_thresh = 2 ** process_dict['2pow']
    data = np.where(data < min_thresh, 0, data)

    return data
