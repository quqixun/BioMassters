import os
import yaml
import pickle
import argparse
import warnings

from libs.utils import *
from libs.predict import *
from omegaconf import OmegaConf
from os.path import join as opj


warnings.filterwarnings('ignore')


def main(args):

    # --------------------------------------------------------------------------
    # loads configs

    with open(args.config_file, 'r') as f:
        configs = yaml.load(f, Loader=yaml.Loader)
    configs = OmegaConf.create(configs)

    # --------------------------------------------------------------------------
    # loads data stats

    stats_path = opj(args.data_root, f'stats_{args.process_method}.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # --------------------------------------------------------------------------
    # gets folds

    if args.folds is not None:
        folds = list(map(int, args.folds.split(',')))
        folds = [f for f in folds if f < configs.cv]
        if len(folds) == 0:
            raise ValueError(f'folds {args.folds} are not availabel')
    else:
        folds = list(range(configs.cv))

    # --------------------------------------------------------------------------
    # predicting test data

    exp_dir = opj(args.exp_root, configs.exp)
    data_dir = opj(args.data_root, 'test')
    folds_str = 'folds_' + '-'.join([str(f) for f in folds])
    output_dir = opj(args.output_root, configs.exp, folds_str)

    # prints information
    print('-' * 100)
    print('BioMassters Predicting ...\n')
    print(f'- Data Dir : {data_dir}')
    print(f'- Exp Dir  : {exp_dir}')
    print(f'- Out Dir  : {output_dir}')
    print(f'- Configs  : {args.config_file}')
    print(f'- Models   : {folds}\n')

    model_paths = [opj(exp_dir, f'fold{f}', 'model.pth') for f in folds]
    predictor = BMPredictor(model_paths, configs, stats, args.process_method)
    predictor.forward(data_dir, output_dir)

    print('-' * 100, '\n')
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BioMassters Predicting')
    parser.add_argument('--data_root',      type=str, help='dir path of training data')
    parser.add_argument('--exp_root',       type=str, help='root dir of experiments')
    parser.add_argument('--output_root',    type=str, help='root dir of outputs')
    parser.add_argument('--config_file',    type=str, help='yaml path of configs')
    parser.add_argument('--process_method', type=str, help='method for processing, log2 or plain')
    parser.add_argument('--folds',          type=str, help='list of folds, separated by ,')
    args = parser.parse_args()

    check_predict_args(args)
    main(args)


