from pathlib import Path
from re import L

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedGroupKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from timm import create_model

from kuma_utils.torch.callbacks import (
    EarlyStopping, SaveSnapshot, SaveEveryEpoch, SaveAllSnapshots)
from kuma_utils.torch.hooks import TrainHook
from kuma_utils.metrics import AUC

from general import *
from datasets import *
from loss_functions import *
from metrics import *
from transforms import *
from architectures import *


class Baseline:
    name = 'baseline'
    seed = 2022
    train_path = DATA_DIR/'train.csv'
    image_dir = Path('input/rsna-breast-cancer-detection/image_resized_1024')
    cv = 5
    splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=seed)
    target_cols = ['cancer']
    group_col = 'patient_id'
    dataset = PatientLevelDataset
    dataset_params = dict(
        image_dir=image_dir
    )

    model = MultiViewModel
    model_params = dict(
        classification_model='tf_efficientnet_b0',
        pretrained=True,
    )
    weight_path = None
    num_epochs = 20
    batch_size = 16
    optimizer = optim.Adam
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)
    scheduler = CosineAnnealingWarmRestarts
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)
    scheduler_target = None
    batch_scheduler = False
    criterion = BCEWithLogitsLoss()
    eval_metric = Pfbeta()
    monitor_metrics = [AUC().torch]
    amp = True
    parallel = None
    deterministic = False
    clip_grad = None
    max_grad_norm = 100
    hook = TrainHook()
    callbacks = [
        EarlyStopping(patience=5, maximize=True, skip_epoch=0),
        SaveSnapshot()
    ]

    preprocess = dict(
        train=None,
        test=None,
    )

    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(rotate_limit=30), 
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
    )

    pseudo_labels = None
    debug = False

