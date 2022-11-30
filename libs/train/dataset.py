import os
import warnings
import numpy as np
import volumentations as V
import matplotlib.pyplot as plt

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
                    interpolation=1, resize_type=0, p=0.2
                ),
                V.Flip(1, p=0.5),
                V.Flip(2, p=0.5)
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
        # label: (256, 256, 1)
        print(label.shape)

        # loads S1 and S2 features
        S_list = []
        for month in range(12):
            s1_path = opj(subject_dir, 'S1', f'{subject}_S1_{month:02d}.tif')
            s2_path = opj(subject_dir, 'S2', f'{subject}_S2_{month:02d}.tif')
            s1 = read_raster(s1_path, S1_SHAPE)
            s2 = read_raster(s2_path, S2_SHAPE)
            s1 = [process_data(s1[i], 'S1', i, self.norm_stats['S1'][i]) for i in range(4)]
            s2 = [process_data(s2[i], 'S2', i, self.norm_stats['S2'][i]) for i in range(11)]
            s = np.expand_dims(np.stack(s1 + s2, axis=-1), axis=0)
            S_list.append(s)

        S = np.concatenate(S_list, axis=0)
        # S: (12, 256, 256, 15)
        print(S.shape)

        return S, label

        # if self.augment:
        #     # normalize imgae to [0, 1]
        #     image[..., 0] = normalize(image[..., 0], 't2w', **self.minmax_norm_params)
        #     image[..., 1] = normalize(image[..., 1], 'adc', **self.minmax_norm_params)
        #     image[..., 2] = normalize(image[..., 2], 'hbv', **self.minmax_norm_params)
        #     # image: (D, H, W, 3)

        #     # augmentation
        #     data = {'image': image, 'mask': slabel}
        #     aug_data = self.transform(**data)
        #     image, slabel = aug_data['image'], aug_data['mask']

        # # normalization
        # image[..., 0] = normalize(image[..., 0], 't2w', **self.norm_params)
        # image[..., 1] = normalize(image[..., 1], 'adc', **self.norm_params)
        # image[..., 2] = normalize(image[..., 2], 'hbv', **self.norm_params)
        # # image: (D, H, W, 3)

        # if self.apply_diff:
        #     hbv_adc_diff = image[..., 2] - image[..., 1]
        #     hbv_adc_diff = normalize(hbv_adc_diff, None, **self.norm_params)
        #     image = np.concatenate([image, np.expand_dims(hbv_adc_diff, -1)], axis=-1)
        #     # image: (D, H, W, 4)

        # # num = image.shape[0]
        # # plt.figure(figsize=(35, 8))
        # # for i in range(num):
        # #     plt.subplot(5, num, i + 1)
        # #     plt.title(f't2w-{i}')
        # #     plt.imshow(image[i, :, :, 0], cmap='gray')
        # #     plt.axis('off')
        # #     plt.subplot(5, num, i + num + 1)
        # #     plt.title(f'adc-{i}')
        # #     plt.imshow(image[i, :, :, 1], cmap='gray')
        # #     plt.axis('off')
        # #     plt.subplot(5, num, i + num * 2 + 1)
        # #     plt.title(f'hbv-{i}')
        # #     plt.imshow(image[i, :, :, 2], cmap='gray')
        # #     plt.axis('off')
        # #     final = 3
        # #     if self.apply_diff:
        # #         final = 4
        # #         plt.subplot(5, num, i + num * 3 + 1)
        # #         plt.title(f'diff-{i}')
        # #         plt.imshow(image[i, :, :, 3], cmap='gray')
        # #         plt.axis('off')
        # #     plt.subplot(5, num, i + num * final + 1)
        # #     plt.title(f'slabel-{i}')
        # #     plt.imshow(slabel[i, :, :, 0], cmap='gray')
        # #     plt.axis('off')
        # # plt.tight_layout()
        # # plt.show()

        # # image: [3 or 4, D, H, W] | slable: [D, H, w]
        # image  = image.transpose(3, 0, 1, 2).astype(np.float32)
        # slabel = slabel.transpose(3, 0, 1, 2).astype(np.float32)

        # return image, slabel, clabel


def get_dataloader(mode, configs, data_list, norm_stats):
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
