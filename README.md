# BioMassters

Competition Page: https://www.drivendata.org/competitions/99/biomass-estimation/page/536/

## 1. Environment

```bash
conda create --name biomassters python=3.9
conda activate biomassters

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## 2. Dataset Preparation

- Download Metadata from [DATA DOWNLOAD page](https://www.drivendata.org/competitions/99/biomass-estimation/data/), and put all files in [./data/information](./data/information) in following structure:

```bash
./data/information
├── biomassters-download-instructions.txt  # Instructions to download satellite images and AGBM data
├── features_metadata.csv                  # Metadata for satellite images
└── train_agbm_metadata.csv                # Metadata for training set AGBM tifs
```

- Download Image data by running ```./scripts/download.sh```, data is saved in [./data/source](./data/source):

```bash
./data/source
├── test
│   ├── aa5e092e
│   │   ├── S1
│   │   │   ├── aa5e092e_S1_00.tif
│   │   │   ├── ...
│   │   │   └── aa5e092e_S1_11.tif
│   │   └── S2
│   │       ├── aa5e092e_S2_00.tif
│   │       ├── ...
│   │       └── aa5e092e_S2_11.tif
|   ├── ...
│   └── fff812c0
└── train
    ├── aa018d7b
    ├── ...
    └── fff05995
```

- Calculate statistics for normalization and split dataset into 5 folds by running ```./scripts/process.sh```:

```bash
./data/source
├── plot              # data distribution
├── splits.pkl        # 5 folds for cross validation
├── stats_log2.pkl    # statistics of log2 transformed dataset
├── stats_plain.pkl   # statistics of original dataset
├── test
└── train
```

## 3. Training

Train model with arguments:

- ```data_root```: root directory of training dataset
- ```exp_root```: root direcroty to save checkpoints, logs and models
- ```config_file```: file path of configurations
- ```process_method```: processing method to calculate statistics, ```log2``` or ```plain```
- ```folds```: list of folds, separated by ```,```

```bash
device=0
process=plain
folds=0,1,2,3,4
data_root=./data/source
config_file=./configs/swin_unetr/exp1.yaml

CUDA_VISIBLE_DEVICES=$device \
python train.py              \
    --data_root      $data_root             \
    --exp_root       ./experiments/$process \
    --config_file    $config_file           \
    --process_method $process               \
    --folds          $folds
```

## 4. Predicting

Make predictions with almost the same arguments as training:

- ```data_root```: root directory of training dataset
- ```exp_root```: root direcroty to save checkpoints, logs and models
- ```output_root```: root directory to save predictions
- ```config_file```: file path of configurations
- ```process_method```: processing method to calculate statistics, ```log2``` or ```plain```
- ```folds```: list of folds, separated by ```,```

```bash
device=0
process=plain
folds=0,1,2,3,4
data_root=./data/source
config_file=./configs/swin_unetr/exp1.yaml

CUDA_VISIBLE_DEVICES=$device \
python predict.py            \
    --data_root      $data_root             \
    --exp_root       ./experiments/$process \
    --output_root    ./predictions/$process \
    --config_file    $config_file           \
    --process_method $process               \
    --folds          $folds
```



