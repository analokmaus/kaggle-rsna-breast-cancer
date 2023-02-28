from pathlib import Path
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedGroupKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from timm import create_model

from kuma_utils.torch.callbacks import (
    EarlyStopping, SaveSnapshot, SaveEveryEpoch, SaveAllSnapshots, SaveAverageSnapshot, CollectTopK)
from kuma_utils.torch.hooks import TrainHook
from kuma_utils.metrics import AUC

from general import *
from datasets import *
from loss_functions import *
from metrics import *
from transforms import *
from architectures import *
from training_extras import *
from global_objectives.losses import AUCPRLoss, PRLoss, TPRFPRLoss


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
    monitor_metrics = [AUC().torch, PRAUC().torch, Pfbeta(binarize=False)]
    amp = True
    parallel = None
    deterministic = False
    clip_grad = None
    max_grad_norm = 100
    grad_accumulations = 1
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
    monitor_metrics = [AUC().torch, PRAUC().torch, Pfbeta(binarize=False),  Pfbeta(binarize=True),]
    parallel = 'ddp'
    callbacks = [
        EarlyStopping(patience=6, maximize=True, skip_epoch=5),
        SaveSnapshot()
    ]


class Aug07(Baseline4):
    name = 'aug_07'
    dataset_params = dict(
        sample_criteria='low_value_for_implant',
        bbox_path='input/rsna-breast-cancer-detection/rsna-yolo-crop/001_baseline/det_result_001_baseline.csv',
    )
    preprocess = dict(
        train=A.Compose([
            RandomCropBBox(buffer=(-20, 100)), 
            AutoFlip(sample_width=100), A.Resize(1024, 512)], 
            bbox_params=A.BboxParams(format='pascal_voc')),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1024, 512)],
            bbox_params=A.BboxParams(format='pascal_voc')),
    )


class Aug07lr0(Aug07):
    name = 'aug_07_lr0'
    criterion = AUCPRLoss()
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(0.1, 0.2, 45, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
            A.OneOf([
                A.GaussianBlur(),
                A.MotionBlur(),
                A.MedianBlur(),
            ], p=0.25),
            A.CLAHE(p=0.1), 
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.25),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), 
            A.CoarseDropout(max_holes=20, max_height=64, max_width=64, p=0.2),
            ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ])
    )
    callbacks = [
        CollectTopK(3, maximize=True), 
        SaveAverageSnapshot(num_snapshot=3)
    ]
    eval_metric = PRAUC().torch
    monitor_metrics = [Pfbeta(binarize=False), Pfbeta(binarize=True)]
    dataset_params = dict(
        sample_criteria='valid_area',
        bbox_path='input/rsna-breast-cancer-detection/bbox_all.csv',
    )


class Model12(Aug07lr0):
    name = 'model_12'
    model = MultiLevelModel
    model_params = dict(
        global_model='convnext_tiny.fb_in22k_ft_in1k_384',
        local_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        crop_size=128,
        crop_num=8,
    )
    criterion = MultiLevelLoss(loss_type='aucpr')
    hook = MultiLevelTrain()
    num_epochs = 30


class Model12v1(Model12):
    name = 'model_12_v1'
    model_params = dict(
        global_model='convnext_tiny.fb_in22k_ft_in1k_384',
        local_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        crop_size=256,
        crop_num=4,
    )


class Model14(Model12v1):
    name = 'model_14'
    model = MultiLevelModel2
    model_params = dict(
        global_model='convnext_tiny.fb_in22k_ft_in1k_384',
        local_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        crop_size=256,
        crop_num=4,
    )


class Res02(Baseline4):
    name = 'res_02'
    dataset_params = dict(
        sample_criteria='low_value_for_implant'
    )
    preprocess = dict(
        train=A.Compose([
            AutoFlip(sample_width=200), 
            RandomCropROI(threshold=(0.08, 0.12), buffer=(-20, 100)), 
            A.Resize(1536, 768)]),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1536, 768)]),
    )
    model_params = dict(
        classification_model='convnext_tiny.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=True)
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(rotate_limit=30),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.OneOf([
                A.GridDistortion(),
                A.OpticalDistortion(),
            ], p=0.2),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), 
            A.CoarseDropout(max_holes=16, max_height=96, max_width=96, p=0.2),
            ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
    )


class Res02lr0(Res02):
    name = 'res_02_lr0'
    criterion = AUCPRLoss()
    preprocess = dict(
        train=A.Compose([
            RandomCropBBox(buffer=(-20, 100)), 
            AutoFlip(sample_width=100), A.Resize(1536, 768)], 
            bbox_params=A.BboxParams(format='pascal_voc')),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1536, 768)],
            bbox_params=A.BboxParams(format='pascal_voc')),
    )
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(0.1, 0.2, 45, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
            A.OneOf([
                A.GaussianBlur(),
                A.MotionBlur(),
                A.MedianBlur(),
            ], p=0.25),
            A.CLAHE(p=0.1), 
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.25),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), 
            A.CoarseDropout(max_holes=20, max_height=96, max_width=96, p=0.2),
            ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ])
    )
    callbacks = [
        CollectTopK(3, maximize=True), 
        SaveAverageSnapshot(num_snapshot=3)
    ]
    eval_metric = PRAUC().torch
    monitor_metrics = [Pfbeta(binarize=False), Pfbeta(binarize=True)]
    dataset_params = dict(
        sample_criteria='valid_area',
        bbox_path='input/rsna-breast-cancer-detection/bbox_all.csv',
    )


class Res02mod2(Res02lr0):
    name = 'res_02_mod2'
    model = MultiLevelModel2
    model_params = dict(
        global_model='convnext_nano.in12k_ft_in1k',
        local_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        crop_size=256,
        crop_num=4,
        # percent_t=0.1, 
    )
    criterion = MultiLevelLoss(loss_type='aucpr')
    hook = MultiLevelTrain()
    num_epochs = 30


class Res02mod3(Res02mod2):
    name = 'res_02_mod3'
    model_params = dict(
        global_model='convnext_nano.in12k_ft_in1k',
        local_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        crop_size=384,
        crop_num=2,
    )


class Res02mod4(Res02mod2):
    name = 'res_02_mod4'
    model_params = dict(
        global_model='convnext_pico.d1_in1k',
        local_model='convnext_base.fb_in22k_ft_in1k_384',
        pretrained=True,
        crop_size=384,
        crop_num=2,
    )


class Res02mod5(Res02mod2):
    name = 'res_02_mod5'
    model_params = dict(
        global_model='convnext_nano.in12k_ft_in1k',
        local_model='convnext_base.fb_in22k_ft_in1k_384',
        pretrained=True,
        crop_size=384,
        crop_num=2,
    )
