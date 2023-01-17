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
from training_extras import *


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
    eval_metric = Pfbeta(binarize=True)
    monitor_metrics = [AUC().torch, Pfbeta(binarize=False)]
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


class Model01v0(Model01):
    name = 'model_01_v0'
    preprocess = dict(
        train=ImageToTile(tile_size=256, num_tiles=8, criterion='brightness', concat=False, dropout=0),
        test=ImageToTile(tile_size=256, num_tiles=8, criterion='brightness', concat=False, dropout=0),
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.]))
    callbacks = [
        EarlyStopping(patience=5, maximize=True, skip_epoch=5),
        SaveSnapshot()
    ]


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


class Model02aug0(Model02loss0):
    name = 'model_02_aug0'
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
            A.CoarseDropout(max_holes=16, max_height=64, max_width=64, p=0.25),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), 
            ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
    )


class Model02aug1(Model02loss0):
    name = 'model_02_aug1'
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(rotate_limit=30), 
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.OneOf([
                A.ElasticTransform(),
                A.GridDistortion(),
                A.OpticalDistortion(),
            ], p=0.25),
            A.CoarseDropout(max_holes=16, max_height=64, max_width=64, p=0.25),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), 
            ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
    )


class Model02ds0(Model02loss0):
    name = 'model_02_ds0'
    image_dir = Path('input/rsna-breast-cancer-detection/image_resized_1024')
    

class Model02v1(Model02loss0):
    name = 'model_02_v1'
    model_params = dict(
        classification_model='tf_efficientnet_b0',
        pretrained=True,
        spatial_pool=True,
        custom_classifier='concat',
    )


class Model02v2(Model02loss0):
    name = 'model_02_v2'
    model_params = dict(
        classification_model='tf_efficientnet_b0',
        pretrained=True,
        spatial_pool=True,
        custom_classifier='gem',
    )


class Model02v3(Model02loss0):
    name = 'model_02_v3'
    model_params = dict(
        classification_model='tf_efficientnet_b0',
        pretrained=True,
        spatial_pool=True,
        custom_attention='triplet'
    )



class Baseline2(Baseline):
    name = 'baseline_2'
    dataset_params = dict(flip_lr=False)
    preprocess = dict(
        train=A.Compose([AutoFlip()]),
        test=A.Compose([AutoFlip()]),
    )
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(rotate_limit=20),
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
    image_dir = Path('input/rsna-breast-cancer-detection/image_resized_1024W')
    model_params = dict(
        classification_model='tf_efficientnet_b0',
        pretrained=True,
        spatial_pool=True,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.]))
    callbacks = [
        EarlyStopping(patience=5, maximize=True, skip_epoch=5),
        SaveSnapshot()
    ]


class Aug04(Baseline2):
    name = 'aug_04'
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(rotate_limit=20),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.OneOf([
                A.ElasticTransform(),
                A.GridDistortion(),
                A.OpticalDistortion(),
            ], p=0.2),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
    )


class Aug05(Baseline2):
    name = 'aug_05'
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
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
    )


class Dataset02(Baseline2):
    name = 'dataset_02'
    image_dir = Path('input/rsna-breast-cancer-detection/image_resized_2048W')
    preprocess = dict(
        train=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1024, 512)]),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1024, 512)]),
    )


class Dataset02v0(Dataset02):
    name = 'dataset_02_v0'
    preprocess = dict(
        train=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=160), A.Resize(1024, 512)]),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=160), A.Resize(1024, 512)]),
    )


class Dataset02mod0(Dataset02):
    name = 'dataset_02_mod0'
    model_params = dict(
        classification_model='tf_efficientnet_b4',
        pretrained=True,
        spatial_pool=True,
    )
    

class Dataset02aug0(Dataset02):
    name = 'dataset_02_aug0'
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(rotate_limit=20),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
    )


class Dataset02aug1(Dataset02):
    name = 'dataset_02_aug1'
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
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
    )


class Dataset02v1(Dataset02aug1):
    name = 'dataset_02_v1'
    image_dir = Path('input/rsna-breast-cancer-detection/image_resized_2048V')


class Dataset02v2(Dataset02aug1):
    name = 'dataset_02_v2'
    image_dir = Path('input/rsna-breast-cancer-detection/image_resized_2048V(0, 98)')


class Baseline3(Dataset02):
    name = 'baseline_3'
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
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
    )
    image_dir = Path('input/rsna-breast-cancer-detection/image_resized_2048V')


class Model03(Baseline3):
    name = 'model_03'
    model_params = dict(
        classification_model='tf_efficientnet_b4',
        pretrained=True,
        spatial_pool=True,
    )


class Model04(Baseline3):
    name = 'model_04'
    model = MultiLevelModel
    model_params = dict(
        global_model='tf_efficientnet_b0',
        local_model='tf_efficientnet_b0',
        pretrained=True,
        crop_size=128,
        crop_num=8,
    )
    criterion = MultiLevelLoss(pos_weight=torch.tensor([5.]))
    hook = MultiLevelTrain()


class Model04v0(Model04):
    name = 'model_04_v0'
    model_params = dict(
        global_model='tf_efficientnet_b0',
        local_model='tf_efficientnet_b4',
        pretrained=True,
        crop_size=128,
        crop_num=8,
    )


class Model04v1(Model04):
    name = 'model_04_v1'
    criterion = MultiLevelLoss(pos_weight=torch.tensor([5.]), weights=(2, 2, 1))


class Model04v2(Model04):
    name = 'model_04_v2'
    model_params = dict(
        global_model='tf_efficientnet_b0',
        local_model='tf_efficientnet_b0',
        pretrained=True,
        crop_size=64,
        crop_num=16,
    )


class Model04v3(Model04):
    name = 'model_04_v3'
    model_params = dict(
        global_model='tf_efficientnet_b0',
        local_model='tf_efficientnet_b0',
        pretrained=True,
        crop_size=128,
        crop_num=8,
        local_attention=True,
    )


class Model04v4(Model04):
    name = 'model_04_v4'
    model_params = dict(
        global_model='tf_efficientnet_b0',
        local_model='tf_efficientnet_b0',
        pretrained=True,
        crop_size=128,
        crop_num=8,
        percent_t=0.1
    )


class Model04v4si(Model04v4):
    name = 'model_04_v4si'
    model_params = dict(
        global_model='tf_efficientnet_b0',
        local_model='tf_efficientnet_b0',
        pretrained=True,
        crop_size=128,
        crop_num=8,
        percent_t=0.1,
        num_view=1,
    )
    dataset = ImageLevelDataset
    dataset_params = dict()
    hook = SingleImageAggregatedTrain(multilevel=True)


class Model04v5(Model04):
    name = 'model_04_v5'
    model_params = dict(
        global_model='convnext_nano.in12k_ft_in1k',
        local_model='convnext_nano.in12k_ft_in1k',
        pretrained=True,
        crop_size=128,
        crop_num=8,
    )
    optimizer_params = dict(lr=5e-6, weight_decay=1e-6)
    optimizer = optim.AdamW


class Model04v6(Model04):
    name = 'model_04_v6'
    model_params = dict(
        global_model='convnext_nano.in12k_ft_in1k',
        local_model='vit_tiny_patch16_224.augreg_in21k_ft_in1k',
        pretrained=True,
        crop_size=224,
        crop_num=4,
    )
    optimizer_params = dict(lr=1e-5, weight_decay=1e-6)


class Aug06(Baseline3):
    name = 'aug_06'
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
        ]), 
    )


class Aug06si(Aug06):
    name = 'aug_06_si'
    model = ClassificationModel
    model_params = dict(
        classification_model='tf_efficientnet_b0',
        pretrained=True)
    dataset = ImageLevelDataset
    dataset_params = dict()
    hook = SingleImageAggregatedTrain()
    

class Model05(Aug06):
    name = 'model_05'
    model_params = dict(
        classification_model='convnext_nano.in12k_ft_in1k',
        pretrained=True,
        spatial_pool=True,)
    optimizer_params = dict(lr=1e-5, weight_decay=1e-6)


class Model05rep(Model05):
    name = 'model_05_rep'


class Model05lr0(Model05):
    name = 'model_05_lr0'
    optimizer = optim.AdamW


class Model05v0(Model05):
    name = 'model_05_v0'
    model_params = dict(
        classification_model='tf_efficientnetv2_s.in21k_ft_in1k',
        pretrained=True,
        spatial_pool=True)
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)


class Model05v1(Model05):
    name = 'model_05_v1'
    model_params = dict(
        classification_model='vit_tiny_patch16_384.augreg_in21k_ft_in1k',
        pretrained=True,
        spatial_pool=True)
    optimizer_params = dict(lr=5e-4, weight_decay=1e-6)


class Model05v2(Model05):
    name = 'model_05_v2'
    model_params = dict(
        classification_model='convnext_tiny.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=True)
    optimizer = optim.AdamW
    