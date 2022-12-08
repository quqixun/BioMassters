__all__ = ['get_dataloader']


import os
import pickle
import numpy as np
import volumentations as V
import matplotlib.pyplot as plt

from ..utils import *
from ..process import *
from os.path import join as opj
from torch.utils.data import Dataset, DataLoader


class BMDataset(Dataset):

    def __init__(
        self, mode, data_list, augment=False, norm_stats=None,
        processed=True, process_method=None
    ):
        super(BMDataset, self).__init__()

        self.mode           = mode
        self.augment        = augment
        self.processed      = processed
        self.transform      = None
        self.data_list      = data_list
        self.norm_stats     = norm_stats
        self.process_method = process_method

        if self.augment:
            self.transform = V.Compose([
                V.Flip(1, p=0.2),
                V.Flip(2, p=0.2),
                V.RandomRotate90((1, 2), p=0.2)
            ], p=1.0)

        if self.processed:
            self.load_data_func = self._load_processed_data
        else:
            assert self.norm_stats is not None
            assert self.process_method in ['log2', 'plain']

            self.load_data_func = self._load_original_data
            if process_method == 'log2':
                self.remove_outliers_func = remove_outliers_by_log2
            elif process_method == 'plain':
                self.remove_outliers_func = remove_outliers_by_plain

    def __len__(self):
        return len(self.data_list)

    def _load_processed_data(self, subject_path):

        with open(subject_path, 'rb') as f:
            data = pickle.load(f)

        if self.mode == 'train':
            label = data['label']
        elif self.mode == 'val':
            label = data['label_src']
        feature = data['feature']

        return label, feature

    def _load_original_data(self, subject_path):

        subject = os.path.basename(subject_path)

        # loads label data
        label_path = opj(subject_path, f'{subject}_agbm.tif')
        assert os.path.isfile(label_path), f'label {label_path} is not exist'
        label = read_raster(label_path, True, GT_SHAPE)
        if self.mode == 'train':
            label = self.remove_outliers_func(label, 'label')
            label = normalize(label, self.norm_stats['label'], 'minmax')
        label = np.expand_dims(label, axis=-1)

        # loads S1 and S2 features
        feature_list = []
        for month in range(12):
            s1_path = opj(subject_path, 'S1', f'{subject}_S1_{month:02d}.tif')
            s2_path = opj(subject_path, 'S2', f'{subject}_S2_{month:02d}.tif')
            s1 = read_raster(s1_path, True, S1_SHAPE)
            s2 = read_raster(s2_path, True, S2_SHAPE)

            s1_list = []
            for index in range(4):
                s1i = self.remove_outliers_func(s1[index], 'S1', index)
                s1i = normalize(s1i, self.norm_stats['S1'][index], 'zscore')
                s1_list.append(s1i)

            s2_list = []
            for index in range(11):
                s2i = self.remove_outliers_func(s2[index], 'S2', index)
                s2i = normalize(s2i, self.norm_stats['S2'][index], 'zscore')
                s2_list.append(s2i)

            feature = np.stack(s1_list + s2_list, axis=-1)
            feature = np.expand_dims(feature, axis=0)
            feature_list.append(feature)
        feature = np.concatenate(feature_list, axis=0)

        return label, feature

    def __getitem__(self, index):

        subject_path = self.data_list[index]
        label, feature = self.load_data_func(subject_path)
        # label:   ( 1, 256, 256,  1)
        # feature: (12, 256, 256, 15)

        if self.augment:
            data = {'image': feature, 'mask': label}
            aug_data = self.transform(**data)
            feature, label = aug_data['image'], aug_data['mask']
            if label.shape[0] > 1:
                label = label[:1]

        # for m in range(feature.shape[0]):
        #     subject = os.path.basename(subject_path)
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

        feature = feature.transpose(3, 0, 1, 2).astype(np.float32)
        # feature: (15, 12, 256, 256)
        label = label[0].transpose(2, 0, 1).astype(np.float32)
        # label: (1, 256, 256)
        return feature, label


def get_dataloader(
    mode, data_list, configs, norm_stats=None,
    processed=True, process_method=None
):
    assert mode in ['train', 'val']

    if mode == 'train':
        batch_size = configs.train_batch
        drop_last  = True
        shuffle    = True
        augment    = configs.apply_augment
    else:  # mode == 'val'
        batch_size = configs.val_batch
        drop_last  = False
        shuffle    = False
        augment    = False

    dataset = BMDataset(
        mode           = mode,
        data_list      = data_list,
        augment        = augment,
        norm_stats     = norm_stats,
        processed      = processed,
        process_method = process_method
    )

    dataloader = DataLoader(
        dataset,
        batch_size  = batch_size,
        num_workers = configs.num_workers,
        pin_memory  = configs.pin_memory,
        drop_last   = drop_last,
        shuffle     = shuffle,
    )

    return dataloader
