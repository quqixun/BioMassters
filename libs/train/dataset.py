__all__ = ['get_dataloader']


import os
import warnings
import numpy as np
import volumentations as V
# import matplotlib.pyplot as plt

from ..utils import *
from os.path import join as opj
from torch.utils.data import Dataset, DataLoader


warnings.filterwarnings('ignore')


class BMDataset(Dataset):

    def __init__(
        self, mode, data_list, norm_stats,
        augment=False, image_size=[12, 256, 256]
    ):
        super(BMDataset, self).__init__()

        self.mode       = mode
        self.augment    = augment
        self.transform  = None
        self.data_list  = data_list
        self.norm_stats = norm_stats

        if self.augment:
            self.transform = V.Compose([
                V.RandomResizedCrop(
                    image_size, scale_limit=(0.8, 1.2),
                    interpolation=1, resize_type=0, p=1.0
                ),
                V.Flip(1, p=1.0),
                V.Flip(2, p=1.0),
                V.RandomRotate90((1, 2), p=1.0)
            ], p=1.0)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        subject_dir = self.data_list[index]
        subject = os.path.basename(subject_dir)

        # loads label data
        label_path = opj(subject_dir, f'{subject}_agbm.tif')
        assert os.path.isfile(label_path), f'label {label_path} is not exist'
        label = read_raster(label_path)
        label = process_data(label, 'label')
        label = np.expand_dims(label, axis=-1)
        # label: (1, 256, 256, 1)

        # loads S1 and S2 features
        feature_list = []
        for month in range(12):
            s1_path = opj(subject_dir, 'S1', f'{subject}_S1_{month:02d}.tif')
            s2_path = opj(subject_dir, 'S2', f'{subject}_S2_{month:02d}.tif')
            s1 = read_raster(s1_path, S1_SHAPE)
            s2 = read_raster(s2_path, S2_SHAPE)
            s1 = [process_data(s1[i], 'S1', i, self.norm_stats['S1'][i]) for i in range(4)]
            s2 = [process_data(s2[i], 'S2', i, self.norm_stats['S2'][i]) for i in range(11)]
            feature = np.expand_dims(np.stack(s1 + s2, axis=-1), axis=0)
            feature_list.append(feature)
        feature = np.concatenate(feature_list, axis=0)
        # feature: (12, 256, 256, 15)

        if self.augment:
            data = {'image': feature, 'mask': label}
            aug_data = self.transform(**data)
            feature, label = aug_data['image'], aug_data['mask']
            if label.shape[0] > 1:
                label = label[:1]

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

        feature = feature.transpose(3, 0, 1, 2)
        # feature: (15, 12, 256, 256)
        label = label[0].transpose(2, 0, 1)
        # label: (1, 256, 256)
        return feature, label


def get_dataloader(mode, data_list, configs, norm_stats):
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
        mode       = mode,
        data_list  = data_list,
        norm_stats = norm_stats,
        augment    = augment,
        image_size = configs.image_size
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
