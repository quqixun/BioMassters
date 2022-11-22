import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from os.path import join as opj


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


if __name__ == '__main__':

    data_dir = './data/source/train'
    subjects = os.listdir(data_dir)
    subjects.sort()

    for subject in tqdm(subjects, ncols=88):
        subject_dir = opj(data_dir, subject)

        # load label data
        label_path = opj(subject_dir, f'{subject}_agbm.tif')
        label = read_raster(label_path, GT_SHAPE)

        # load features
        for i in range(12):
            s1_path = opj(subject_dir, 'S1', f'{subject}_S1_{i:02d}.tif')
            s2_path = opj(subject_dir, 'S2', f'{subject}_S2_{i:02d}.tif')
            s1 = read_raster(s1_path, S1_SHAPE)
            s2 = read_raster(s2_path, S2_SHAPE)

            plt.figure(f'{subject} - {i:02d}', figsize=(15, 15))
            plt.subplot(4, 4, 1)
            plt.title('GT')
            plt.imshow(label[0])
            plt.axis('off')
            for is1 in range(s1.shape[0]):
                plt.subplot(4, 4, is1 + 2)
                plt.title(f'S1-{is1 + 1}')
                plt.imshow(s1[is1])
                plt.axis('off')
            for is2 in range(s2.shape[0]):
                plt.subplot(4, 4, is2 + 2 + s1.shape[0])
                plt.title(f'S2-{is2 + 1}')
                plt.imshow(s2[is2])
                plt.axis('off')
            plt.tight_layout()
            plt.show()
