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
        EarlyStopping(patience=6, maximize=True, skip_epoch=5),
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
        EarlyStopping(patience=6, maximize=True, skip_epoch=5),
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
        EarlyStopping(patience=6, maximize=True, skip_epoch=5),
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
        global_model='convnext_small.fb_in22k_ft_in1k_384',
        local_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        crop_size=128,
        crop_num=8,
    )
    optimizer_params = dict(lr=1e-5, weight_decay=1e-6)
    optimizer = optim.AdamW
    eval_metric = Pfbeta(average_both=True)
    monitor_metrics = [AUC().torch, Pfbeta(binarize=False),  Pfbeta(binarize=True)]


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


class Model04res0(Model04):
    name = 'model_04_res0'
    preprocess = dict(
        train=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1536, 768)]),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1536, 768)]),
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
            A.CoarseDropout(max_holes=16, max_height=96, max_width=96, p=0.2),
            ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
    )
    model_params = dict(
        global_model='convnext_tiny.fb_in22k_ft_in1k_384',
        local_model='convnext_tiny.fb_in22k_ft_in1k_384',
        pretrained=True,
        crop_size=128,
        crop_num=8,
    )
    optimizer_params = dict(lr=1e-5, weight_decay=1e-6)
    optimizer = optim.AdamW
    eval_metric = Pfbeta(average_both=True)
    monitor_metrics = [AUC().torch, Pfbeta(binarize=False),  Pfbeta(binarize=True)]


class Model04res0a(Model04res0):
    name = 'model_04_res0a'
    dataset_params = dict(
        sample_criteria='low_value_for_implant'
    )
    weight_path = Path('results/model_04_res0')
    num_epochs = 10
    callbacks = [
        EarlyStopping(patience=5, maximize=True, skip_epoch=0),
        SaveSnapshot()
    ]


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


class Model05v2lr0(Model05v2):
    name = 'model_05_v2_lr0'
    optimizer = optim.Adam


class Model05v2si(Model05v2):
    name = 'model_05_v2_si'
    model = ClassificationModel
    model_params = dict(
        classification_model='convnext_tiny.fb_in22k_ft_in1k_384',
        pretrained=True)
    dataset = ImageLevelDataset
    dataset_params = dict()
    hook = SingleImageAggregatedTrain()


class Model05v3(Model05):
    name = 'model_05_v3'
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=True)
    optimizer = optim.AdamW
    eval_metric = Pfbeta(average_both=True)
    monitor_metrics = [AUC().torch, Pfbeta(binarize=False),  Pfbeta(binarize=True),]


class Model05v3val(Model05v3):
    name = 'model_05_v3_val'
    eval_metric = Pfbeta(binarize=True)
    monitor_metrics = [AUC().torch, Pfbeta(binarize=False)]


class Model05v3loss0(Model05v3):
    name = 'model_05_v3_loss0'
    criterion = nn.BCEWithLogitsLoss()


class Model05v3loss1(Model05v3):
    name = 'model_05_v3_loss1'
    criterion = FocalLoss()


class Model05v3aug0(Model05v3loss0):
    name = 'model_05_v3_aug0'
    preprocess = dict(
        train=A.Compose([
            AutoFlip(sample_width=200), 
            RandomCropROI(threshold=(0.08, 0.12), buffer=(-20, 100)), A.Resize(1024, 512)]),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1024, 512)]),
    )


class Model05v3prep0(Model05v3loss0):
    name = 'model_05_v3_prep0'
    preprocess = dict(
        train=A.Compose([
            AutoFlip(sample_width=200), 
            CropROI2(threshold=10, buffer=80), A.Resize(1024, 512)]),
        test=A.Compose([AutoFlip(sample_width=200), CropROI2(threshold=10, buffer=80), A.Resize(1024, 512)]),
    )


class Model05v3aug1(Model05v3loss0):
    name = 'model_05_v3_aug1'
    preprocess = dict(
        train=A.Compose([
            AutoFlip(sample_width=200), 
            RandomCropROI2(buffer=(0, 120)), A.Resize(1024, 512)]),
        test=A.Compose([AutoFlip(sample_width=200), CropROI2(threshold=10, buffer=80), A.Resize(1024, 512)]),
    )


class Model05v3prep1(Model05v3loss0):
    name = 'model_05_v3_prep1'
    preprocess = dict(
        train=A.Compose([
            AutoFlip(sample_width=200), CropROI(buffer=20), A.Resize(1024, 512)]),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=20), A.Resize(1024, 512)]),
    )


class Model05v3arch0(Model05v3loss0):
    name = 'model_05_v3_arch0'
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=False)


class Model05v3ds0(Model05v3loss0):
    name = 'model_05_v3_ds0'
    dataset_params = dict(
        sample_criteria='low_value_for_implant'
    )


class Model05v3aug2(Model05v3aug0):
    name = 'model_05_v3_aug2'
    dataset_params = dict(
        sample_criteria='low_value_for_implant'
    )


class Model05v3aug3(Model05v3ds0):
    name = 'model_05_v3_aug3'
    preprocess = dict(
        train=A.Compose([
            AutoFlip(sample_width=200), 
            RandomCropROI(threshold=(0.08, 0.12), buffer=(0, 120)), A.Resize(1024, 512)]),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1024, 512)]),
    )


class Model05v3aug4(Model05v3ds0):
    name = 'model_05_v3_aug4'
    preprocess = dict(
        train=A.Compose([
            AutoFlip(sample_width=200), 
            RandomCropROI(threshold=(0.08, 0.12), buffer=(-40, 80)), A.Resize(1024, 512)]),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=40), A.Resize(1024, 512)]),
    )


class Model05v3aug3arch0(Model05v3aug3):
    name = 'model_05_v3_aug3_arch0'
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=False, 
        dropout=0.05)


class Model05v3aug2arch0(Model05v3aug2):
    name = 'model_05_v3_aug2_arch0'
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=False, 
        dropout=0.05)


class Model05v3aug4arch0(Model05v3aug4):
    name = 'model_05_v3_aug4_arch0'
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=False,
        dropout=0.05)


class Model05v3aug5(Model05v3aug2arch0):
    name = 'model_05_v3_aug5'
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(rotate_limit=30),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.2),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), 
            A.CoarseDropout(max_holes=24, max_height=64, max_width=64, p=0.2),
            ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
    )
    

class Model05v4(Model05):
    name = 'model_05_v4'
    model_params = dict(
        classification_model='convnext_base.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=True)
    optimizer = optim.AdamW
    eval_metric = Pfbeta(average_both=True)
    monitor_metrics = [AUC().torch, Pfbeta(binarize=False),  Pfbeta(binarize=True)]
    criterion = nn.BCEWithLogitsLoss()


class Model05v4val(Model05v4):
    name = 'model_05_v4_val'
    eval_metric = Pfbeta(binarize=True)
    monitor_metrics = [AUC().torch, Pfbeta(binarize=False)]


class Res00(Model05v2):
    name = 'res_00'
    optimizer = optim.AdamW
    eval_metric = Pfbeta(average_both=True)
    monitor_metrics = [AUC().torch, Pfbeta(binarize=False),  Pfbeta(binarize=True),]
    criterion = nn.BCEWithLogitsLoss()
    preprocess = dict(
        train=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1536, 768)]),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1536, 768)]),
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
            A.CoarseDropout(max_holes=16, max_height=96, max_width=96, p=0.2),
            ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
    )


class Res00aug0(Res00):
    name = 'res_00_aug0'
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


class Res01(Res00):
    name = 'res_01'
    preprocess = dict(
        train=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(2048, 1024)]),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(2048, 1024)]),
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
            A.CoarseDropout(max_holes=16, max_height=128, max_width=128, p=0.2),
            ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
    )
    batch_size = 8
    

class Model06(Model05v3loss0):
    name = 'model_06'
    dataset = PatientLevelDataset
    dataset_params = dict(
        sample_num=1, 
        view_category=[['MLO', 'LMO', 'LM', 'ML'], ['CC', 'AT'], ['MLO', 'LMO', 'LM', 'ML', 'CC', 'AT']], 
        replace=False,
    )
    model_params = dict(
        classification_model='convnext_tiny.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=True,
        num_view=3,
    )


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


class Baseline4ddsm(Baseline4):
    name = 'pretrain_baseline4_ddsm'
    dataset = PatientLevelDatasetDDSM
    train_path = Path('input/rsna-breast-cancer-detection/ddsm/DDSM/ddsm_with_meta.csv')
    image_dir = Path('input/rsna-breast-cancer-detection/ddsm/DDSM/ddsm_image_resized_2048V')
    num_epochs = 10


class Baseline4vindr(Baseline4):
    name = 'pretrain_baseline4_vindr'
    train_path = Path('input/rsna-breast-cancer-detection/vindr_train.csv')
    image_dir = Path('input/rsna-breast-cancer-detection/vindr_mammo_resized_2048V')
    num_epochs = 10


class Baseline4pr0(Baseline4):
    name = 'baseline_4_pr0'
    weight_path = Path('results/pretrain_baseline4_ddsm/nocv.pt')


class Baseline4pr1(Baseline4):
    name = 'baseline_4_pr1'
    weight_path = Path('results/pretrain_baseline4_vindr/nocv.pt')


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


class Aug08(Aug07):
    name = 'aug_08'
    preprocess = dict(
        train=A.Compose([
            MixedCropBBox(buffer=(-20, 100), bbox_p=0.5), 
            AutoFlip(sample_width=100), A.Resize(1024, 512)], 
            bbox_params=A.BboxParams(format='pascal_voc')),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1024, 512)],
            bbox_params=A.BboxParams(format='pascal_voc')),
    )


class AuxLoss00(Baseline4):
    name = 'aux_00'
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=True,
        num_classes=2)
    target_cols = ['cancer']
    dataset_params = dict(
        aux_target_cols=['age']
    )
    criterion = AuxLoss(loss_types=('bce', 'mse'), weights=(1., 1.))
    hook = AuxLossTrain()


class AuxLoss01(AuxLoss00):
    name = 'aux_01'
    dataset_params = dict(
        aux_target_cols=['biopsy']
    )
    criterion = AuxLoss(loss_types=('bce', 'bce'), weights=(1., 1.))


class AuxLoss01v0(AuxLoss01):
    name = 'aux_01_v0'
    criterion = AuxLoss(loss_types=('bce', 'bce'), weights=(2., 1.))


class AuxLoss02(AuxLoss00):
    name = 'aux_02'
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=True,
        num_classes=3)
    target_cols = ['cancer']
    dataset_params = dict(
        aux_target_cols=['age', 'biopsy']
    )
    criterion = AuxLoss(loss_types=('bce', 'mse', 'bce'), weights=(2., 1., 1.))


class AuxLoss02pr0(AuxLoss02):
    name = 'aux_02_pr0'
    weight_path = Path('results/pretrain_baseline4_ddsm/nocv.pt')
    optimizer_params = dict(lr=8e-6, weight_decay=1e-6)


class AuxLoss02v0(AuxLoss02):
    name = 'aux_02_v0'
    criterion = AuxLoss(loss_types=('bce', 'bce', 'bce'), weights=(2., 1., 1.))


class AuxLoss03(AuxLoss00):
    name = 'aux_03'
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=True,
        num_classes=4)
    target_cols = ['cancer']
    dataset_params = dict(
        aux_target_cols=['age', 'biopsy', 'invasive']
    )
    criterion = AuxLoss(loss_types=('bce', 'mse', 'bce', 'bce'), weights=(3., 1., 1., 1.))


class AuxLoss04(AuxLoss00):
    name = 'aux_04'
    dataset_params = dict(
        aux_target_cols=['machine_id']
    )
    criterion = AuxLoss(loss_types=('bce', 'bce'), weights=(2., 1.))


class Dataset03(Baseline4):
    name = 'dataset_03'
    group_col = 'machine_id'


class Dataset04(Baseline4):
    name = 'dataset_04'
    train_path = DATA_DIR/'train_concat_vindr_birads05.csv'


class Dataset05(Baseline4):
    name = 'dataset_05'
    addon_train_path = DATA_DIR/'vindr_train_birads05.csv'
    

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
            A.CoarseDropout(max_holes=16, max_height=96, max_width=96, p=0.2),
            ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ]), 
    )
