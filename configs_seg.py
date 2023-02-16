from pathlib import Path
from re import L

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedGroupKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from timm import create_model
try:
    from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler
except:
    ExhaustiveWeightedRandomSampler = None

from kuma_utils.torch.callbacks import (
    EarlyStopping, SaveSnapshot, SaveEveryEpoch, SaveAllSnapshots)
from kuma_utils.torch.hooks import TrainHook
from kuma_utils.metrics import AUC
import segmentation_models_pytorch as smp

from general import *
from datasets import *
from loss_functions import *
from metrics import *
from transforms import *
from architectures import *
from training_extras import *
from smp_encoder import *


class Baseline:
    name = 'baseline'
    seed = 2022
    train_path = DATA_DIR/'train.csv'
    addon_train_path = None
    image_dir = Path('input/rsna-breast-cancer-detection/image_resized_1024')
    cv = 5
    splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=seed)
    target_cols = ['cancer']
    group_col = 'patient_id'
    dataset = PatientLevelDataset
    dataset_params = dict()
    sampler = None
    oversample_ntimes = 0

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
    eval_metric = Pfbeta(binarize=True)
    monitor_metrics = [AUC().torch, Pfbeta(binarize=False)]
    amp = True
    parallel = None
    deterministic = False
    clip_grad = None
    max_grad_norm = 100
    hook = TrainHook()
    callbacks = [
        EarlyStopping(patience=6, maximize=True, skip_epoch=0),
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


class Baseline4(Baseline): # Equivalent to Model05v3aug2, 4 fold cv
    name = 'baseline_4'
    cv = 4
    seed = 2022
    splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=seed)
    dataset_params = dict(
        sample_criteria='low_value_for_implant'
    )
    preprocess = dict(
        train=A.Compose([
            AutoFlip(sample_width=200), 
            RandomCropROI(threshold=(0.08, 0.12), buffer=(-20, 100)), A.Resize(1024, 512)]),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1024, 512)]),
    )
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(rotate_limit=30),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.OneOf([
                A.GridDistortion(),
                A.OpticalDistortion(),
            ], p=0.1),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), 
            A.CoarseDropout(max_holes=16, max_height=64, max_width=64, p=0.2),
            ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ])
    )
    image_dir = Path('input/rsna-breast-cancer-detection/image_resized_2048V')
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=True)
    optimizer = optim.AdamW
    optimizer_params = dict(lr=1e-5, weight_decay=1e-6)
    criterion = nn.BCEWithLogitsLoss()
    eval_metric = Pfbeta(average_both=True)
    monitor_metrics = [AUC().torch, Pfbeta(binarize=False),  Pfbeta(binarize=True),]
    parallel = 'ddp'
    callbacks = [
        EarlyStopping(patience=6, maximize=True, skip_epoch=5),
        SaveSnapshot()
    ]


class SegBaseline(Baseline4):
    name = 'seg_baseline'
    dataset = SegmentationDataset
    dataset_params = dict()
    train_path = Path('input/rsna-breast-cancer-detection/vindr_findings.csv')
    model = smp.Unet
    model_params = dict(
        # encoder_name="convnext_small",
        encoder_name="resnet200d",
        encoder_weights="imagenet",
        in_channels=1,
        classes=3,
    )
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(rotate_limit=30),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.OneOf([
                A.GridDistortion(),
                A.OpticalDistortion(),
            ], p=0.1),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), 
            # A.CoarseDropout(max_holes=16, max_height=64, max_width=64, p=0.2),
            ToTensorV2(transpose_mask=True)
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2(transpose_mask=True)
        ])
    )
    criterion = smp.losses.FocalLoss(mode='multilabel')
    eval_metric = IoU(threshold=0.5)
    monitor_metrics = []
    encoder_lr = 2e-4
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)
    hook = TrainHook(evaluate_in_batch=True)
