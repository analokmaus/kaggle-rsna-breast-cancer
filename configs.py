from pathlib import Path
from re import L

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedGroupKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from timm import create_model
from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler

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


class Aug00(Baseline):
    name = 'aug_00'

    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(rotate_limit=30), 
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.25),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
    )

    parallel = 'ddp'


class Aug01(Baseline):
    name = 'aug_01'

    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(rotate_limit=30), 
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.25),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
    )

    parallel = 'ddp'


class Aug02(Baseline):
    name = 'aug_02'

    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(rotate_limit=30), 
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.25),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), 
            A.CoarseDropout(max_holes=8, max_height=128, max_width=64, p=0.25),
            ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
    )

    parallel = 'ddp'
    

class Aug03(Baseline):
    name = 'aug_03'

    preprocess = dict(
        train=A.CLAHE(p=1.0),
        test=A.CLAHE(p=1.0)
    )

    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(rotate_limit=30), 
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.25),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), 
            # A.CoarseDropout(max_holes=8, max_height=128, max_width=64, p=0.25),
            ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
    )

    parallel = 'ddp'


class Dataset00(Aug00):
    name = 'dataset_00'
    sampler = ExhaustiveWeightedRandomSampler


class Dataset01(Aug00):
    name = 'dataset_01'
    image_dir = Path('input/rsna-breast-cancer-detection/image_resized_1024W')


class Dataset01loss0(Dataset01):
    name = 'dataset_01_loss0'
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.]))


class Dataset01v0(Dataset01):
    name = 'dataset_01_v0'
    oversample_ntimes = 3


class Loss00(Aug00):
    name = 'loss_00'
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.]))


class Loss01(Aug00):
    name = 'loss_01'
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.]))


class Model00(Aug00):
    name = 'model_00'
    model_params = dict(
        classification_model='tf_efficientnet_b2',
        pretrained=True,
    )


class Model01(Dataset01):
    name = 'model_01'
    model = MultiInstanceModel
    model_params = dict(
        classification_model='tf_efficientnet_b0',
        pretrained=True,
    )
    preprocess = dict(
        train=ImageToTile(tile_size=256, num_tiles=16, criterion='brightness', concat=False, dropout=0.1),
        test=ImageToTile(tile_size=256, num_tiles=16, criterion='brightness', concat=False, dropout=0),
    )


class Model02(Dataset01):
    name = 'model_02'
    model_params = dict(
        classification_model='tf_efficientnet_b0',
        pretrained=True,
        spatial_pool=True,
    )


class Model02v0(Model02):
    name = 'model_02_v0'
    model_params = dict(
        classification_model='tf_efficientnet_b0',
        pretrained=True,
        spatial_pool=True,
        dropout=0.2
    )


class Model02loss0(Model02):
    name = 'model_02_loss0'
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.]))
    callbacks = [
        EarlyStopping(patience=5, maximize=True, skip_epoch=5),
        SaveSnapshot()
    ]


class Model02ds0(Model02loss0):
    name = 'model_02_ds0'
    image_dir = Path('input/rsna-breast-cancer-detection/image_resized_1024')
    