import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from ..utils import *
from PIL import Image
from tqdm import tqdm
from os.path import join as opj
from ..models import define_model


class BMPredictor(object):

    def __init__(self, configs, exp_dir, norm_stats):

        self.norm_stats = norm_stats
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # loads models
        self.models = []
        for i in range(configs.cv):
            model_path = opj(exp_dir, f'fold{i}', 'model.pth')
            model_dict = torch.load(model_path, map_location='cpu')
            model = define_model(configs.model)
            model.load_state_dict(model_dict)
            model = model.to(self.device)
            model.eval()
            self.models.append(model)

    @torch.no_grad()
    def forward(self, data_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        subjects = os.listdir(data_dir)
        for subject in tqdm(subjects, ncols=88):
            subject_dir = opj(data_dir, subject)
            output_path = opj(output_dir, f'{subject}_agbm.tif')

            feature = self._load_data(subject_dir)
            preds = []
            for model in self.models:
                pred = model(feature)
                pred = recover_data(pred.cpu().numpy()[0, 0])
                preds.append(pred)

            pred = np.mean(preds, axis=0)

            plt.figure()
            plt.imshow(pred)
            plt.tight_layout()
            plt.show()

            pred = Image.fromarray(pred)
            pred.save(output_path, format='TIFF', save_all=True)

    def _load_data(self, subject_dir):
        subject = os.path.basename(subject_dir)

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
        feature = feature.transpose(3, 0, 1, 2).astype(np.float32)
        feature = np.expand_dims(feature, axis=0)
        # feature: (1, 15, 12, 256, 256)

        feature = torch.from_numpy(feature)
        feature = feature.to(self.device)
        return feature