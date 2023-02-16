from pathlib import Path
import cv2
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
    monitor_metrics = [AUC().torch, PRAUC().torch, Pfbeta(binarize=False), Pfbeta(binarize=True)]


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
    monitor_metrics = [AUC().torch, PRAUC().torch, Pfbeta(binarize=False),  Pfbeta(binarize=True),]
    parallel = 'ddp'
    callbacks = [
        EarlyStopping(patience=6, maximize=True, skip_epoch=5),
        SaveSnapshot()
    ]

class Baseline4mod0(Baseline4):
    name = 'baseline_4_mod0'
    model_params = dict(
        classification_model='convnext_tiny.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=True)


class Baseline4mod1(Baseline4):
    name = 'baseline_4_mod1'
    model_params = dict(
        classification_model='convnext_nano.in12k_ft_in1k',
        pretrained=True,
        spatial_pool=True)


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


class Baseline4vindr2(Baseline4):
    name = 'pretrain_baseline4_vindr2'
    train_path = Path('input/rsna-breast-cancer-detection/vindr_train.csv')
    # image_dir = Path('input/rsna-breast-cancer-detection/vindr_mammo_resized_2048V')
    image_dir = Path('input/rsna-breast-cancer-detection/image_resized_2048V')
    num_epochs = 15
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=True,
        dropout=0.05,
        num_classes=2)
    target_cols = ['BIRADS']
    dataset_params = dict(
        aux_target_cols=['density']
    )
    criterion = AuxLoss(loss_types=('mse', 'mse'), weights=(1., 1.))
    eval_metric = nn.L1Loss()
    monitor_metrics = []
    callbacks = [
        EarlyStopping(patience=5, maximize=False, skip_epoch=0, target='train_metric'),
        SaveSnapshot()
    ]
    hook = AuxLossTrain()


class Baseline4vindr2a(Baseline4vindr2):
    name = 'pretrain_baseline4_vindr2a'
    dataset_params = dict(
        sample_criteria='valid_area',
        bbox_path='input/rsna-breast-cancer-detection/bbox_all.csv',
        aux_target_cols=['density']
    )
    preprocess = dict(
        train=A.Compose([
            RandomCropBBox(buffer=(-20, 100)), 
            AutoFlip(sample_width=100), A.Resize(1024, 512)], 
            bbox_params=A.BboxParams(format='pascal_voc')),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1024, 512)],
            bbox_params=A.BboxParams(format='pascal_voc')),
    )
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(0.1, 0.2, 15),
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


class Baseline4vindr3(Baseline4vindr):
    name = 'pretrain_baseline4_vindr3'
    callbacks = [
        # EarlyStopping(patience=6, maximize=True, skip_epoch=0),
        CollectTopK(k=3, maximize=True, target='train_metric'), 
        SaveAverageSnapshot(num_snapshot=3)
    ]


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


class Aug07prep0(Aug07):
    name = 'aug_07_prep0'
    image_dir = Path('input/rsna-breast-cancer-detection/image_resized_2048VS')


class Aug07pr0(Aug07):
    name = 'aug_07_pr0'
    weight_path = Path('results/pretrain_baseline4_vindr2/nocv.pt')


class Aug07aug0(Aug07):
    name = 'aug_07_aug0'
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(0.1, 0.2, 15),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
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


class Aug07aug1(Aug07):
    name = 'aug_07_aug1'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            # A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
            A.CLAHE(clip_limit=(1,4), p=0.5),
            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3),
            ], p=0.2),
            A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                A.MedianBlur(),
            ], p=0.2),
            A.PiecewiseAffine(p=0.2),
            A.Sharpen(p=0.2),
            # A.Cutout(max_h_size=int(1024 * 0.2), max_w_size=int(512 * 0.2), num_holes=5, p=0.5),
            A.CoarseDropout(max_holes=20, max_height=64, max_width=64, p=0.2),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2(),
        ]),
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ])
    )


class Aug07v0(Aug07):
    name = 'aug_07_v0'
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=True,
        dropout=0.1)


class Aug07v1(Aug07):
    name = 'aug_07_v1'
    callbacks = [
        # EarlyStopping(patience=6, maximize=True, skip_epoch=0),
        SaveEveryEpoch(), 
        SaveAverageSnapshot(num_snapshot=3)
    ]


class Aug07pl0(Aug07v1):
    name = 'aug_07_pl0'
    addon_train_path = Path('input/rsna-breast-cancer-detection/vindr_train_pl_v1_soft.csv')
    monitor_metrics = [ContinuousAUC(98.).torch, PRAUC().torch, Pfbeta(binarize=False), Pfbeta(binarize=True)]


class Aug07pl0es0(Aug07pl0):
    name = 'aug_07_pl0_es0'
    callbacks = [
        EarlyStopping(patience=6, maximize=True, skip_epoch=0),
        # SaveEveryEpoch(), 
        SaveAverageSnapshot(num_snapshot=3) # Save average of 3 best model
    ]


class Aug07pl1(Aug07pl0):
    name = 'aug_07_pl1'
    dataset_params = dict(
        sample_criteria='low_value_for_implant',
        bbox_path='input/rsna-breast-cancer-detection/bbox_all.csv',
    )
    callbacks = [
        CollectTopK(3, maximize=True), 
        SaveAverageSnapshot(num_snapshot=3)
    ]


class Aug07pl1aug0(Aug07pl1):
    name = 'aug_07_pl1_aug0'
    dataset_params = dict(
        sample_criteria='valid_area',
        bbox_path='input/rsna-breast-cancer-detection/bbox_all.csv',
    )
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(0.1, 0.2, 15),
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


class Aug07pl2(Aug07pl1):
    name = 'aug_07_pl2'
    addon_train_path = Path('input/rsna-breast-cancer-detection/vindr_train_pl_v1_soft_2575.csv')


class Aug07pl2aug0(Aug07pl2):
    name = 'aug_07_pl2_aug0'
    eval_metric = PRAUC().torch
    monitor_metrics = [ContinuousAUC(98.).torch, Pfbeta(binarize=False), Pfbeta(binarize=True)]
    dataset_params = dict(
        sample_criteria='valid_area',
        bbox_path='input/rsna-breast-cancer-detection/bbox_all.csv',
    )
    preprocess = dict(
        train=A.Compose([
            RandomCropBBox(buffer=(-50, 200)), 
            AutoFlip(sample_width=100), A.Resize(1024, 512)], 
            bbox_params=A.BboxParams(format='pascal_voc')),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1024, 512)],
            bbox_params=A.BboxParams(format='pascal_voc')),
    )
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(0.1, 0.2, 15),
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


class Aug07pl2aug1(Aug07pl2aug0):
    name = 'aug_07_pl2_aug1'
    preprocess = dict(
        train=A.Compose([
            A.OneOf([
                RandomCropBBox2(buffer=(-50, 150), always_apply=False, p=0.75),
                RandomCropROI2(threshold=(0.08, 0.12), buffer=(-50, 150), always_apply=False, p=0.25)], 
                p=1.0
            ),
            AutoFlip(sample_width=100), A.Resize(1024, 512)], 
            bbox_params=A.BboxParams(format='pascal_voc')),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1024, 512)],
            bbox_params=A.BboxParams(format='pascal_voc')),
    )
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(0.1, 0.2, 15),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
            A.OneOf([
                A.GaussianBlur(),
                A.MotionBlur(),
                A.MedianBlur(),
            ], p=0.1),
            A.CLAHE(p=0.1), 
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.25),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), 
            A.CoarseDropout(max_holes=16, max_height=64, max_width=64, p=0.2),
            ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ])
    )


class Aug07pl2aug2(Aug07pl2aug0):
    name = 'aug_07_pl2_aug2'
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


class SiteAug07pl2aug2(Aug07pl2aug2):
    name = 'site_aug_07_pl2_aug2'
    num_epochs = 5
    optimizer_params = dict(lr=5e-6, weight_decay=1e-6)
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(0.1, 0.2, 15, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.VerticalFlip(p=0.1),
            A.HorizontalFlip(p=0.1),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.25),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), 
            ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ])
    )
    callbacks = [
        # CollectTopK(2, maximize=True), 
        SaveEveryEpoch(),
        SaveAverageSnapshot(num_snapshot=2)
    ]
    weight_path = Path('results/aug_07_pl2_aug2/')


class Aug07mod0(Aug07pl2aug0):
    name = 'aug_07_mod0'
    model_params = dict(
        classification_model='convnext_base.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=True)
    batch_size = 8
    optimizer_params = dict(lr=2e-5, weight_decay=1e-6)
    grad_accumulations = 2


class Aug07mod1(Aug07pl2aug0):
    name = 'aug_07_mod1'
    model = MultiViewSiameseModel
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True)
    

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
    monitor_metrics = [ContinuousAUC(98.).torch, Pfbeta(binarize=False), Pfbeta(binarize=True)]
    dataset_params = dict(
        sample_criteria='valid_area',
        bbox_path='input/rsna-breast-cancer-detection/bbox_all.csv',
    )


class Aug07lr1(Aug07lr0):
    name = 'aug_07_lr1'
    criterion = FocalLoss()


class Aug07lr2(Aug07lr0):
    name = 'aug_07_lr2'
    criterion = PRLoss(target_recall=0.5)


class Aug07lr3(Aug07lr0):
    name = 'aug_07_lr3'
    criterion = TPRFPRLoss(target_fpr=0.25)


class Aug08(Aug07):
    name = 'aug_08'
    preprocess = dict(
        train=A.Compose([
            A.OneOf([
                RandomCropBBox(buffer=(-20, 100), always_apply=False, p=0.5),
                RandomCropROI(threshold=(0.08, 0.12), buffer=(-20, 100), always_apply=False, p=0.5)], 
                p=1.0
            ),
            AutoFlip(sample_width=100), A.Resize(1024, 512)], 
            bbox_params=A.BboxParams(format='pascal_voc')),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1024, 512)],
            bbox_params=A.BboxParams(format='pascal_voc')),
    )


class Aug09(Aug07):
    name = 'aug_09'
    preprocess = dict(
        train=A.Compose([
            A.OneOf([
                RandomCropBBox(buffer=(-20, 100), always_apply=False, p=0.75),
                RandomCropROI(threshold=(0.08, 0.12), buffer=(-20, 100), always_apply=False, p=0.25)], 
                p=1.0
            ),
            AutoFlip(sample_width=100), A.Resize(1024, 512)], 
            bbox_params=A.BboxParams(format='pascal_voc')),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1024, 512)],
            bbox_params=A.BboxParams(format='pascal_voc')),
    )


class Aug10(Baseline4):
    name = 'aug_10'
    dataset_params = dict(
        sample_criteria='low_value_for_implant',
        bbox_path='input/rsna-breast-cancer-detection/rsna-yolo-crop/001_baseline/det_result_001_baseline.csv',
    )
    preprocess = dict(
        train=A.Compose([
            RandomCropBBox2(buffer=(-20, 100)), 
            AutoFlip(sample_width=100), A.Resize(1024, 512)], 
            bbox_params=A.BboxParams(format='pascal_voc')),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1024, 512)],
            bbox_params=A.BboxParams(format='pascal_voc')),
    )


class Aug10es0(Aug10):
    name = 'aug_10_es0'
    eval_metric = AUC().torch
    monitor_metrics = [PRAUC().torch, Pfbeta(binarize=True), Pfbeta(binarize=False)]


class Aug10es1(Aug10):
    name = 'aug_10_es1'
    callbacks = [
        SaveEveryEpoch(), 
        SaveAverageSnapshot(num_snapshot=3)
    ]


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


class AuxLoss02aug0(AuxLoss02v0):
    name = 'aux_02_aug0'
    dataset_params = dict(
        aux_target_cols=['age', 'biopsy'],
        sample_criteria='valid_area',
        bbox_path='input/rsna-breast-cancer-detection/bbox_all.csv',
    )
    preprocess = dict(
        train=A.Compose([
            RandomCropBBox(buffer=(-50, 200)), 
            AutoFlip(sample_width=100), A.Resize(1024, 512)], 
            bbox_params=A.BboxParams(format='pascal_voc')),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1024, 512)],
            bbox_params=A.BboxParams(format='pascal_voc')),
    )
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(0.1, 0.2, 15),
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


class AuxLoss05(AuxLoss00):
    name = 'aux_05'
    train_path = Path('input/rsna-breast-cancer-detection/train_meta2.csv')
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=True,
        num_classes=5)
    target_cols = ['cancer']
    dataset_params = dict(
        aux_target_cols=['age', 'biopsy', 'birads_pl', 'density_pl']
    )
    criterion = AuxLoss(loss_types=('bce', 'mse', 'bce', 'mse', 'mse'), weights=(4., 1., 1., 2., 1.))
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(0.1, 0.2, 15, border_mode=cv2.BORDER_CONSTANT),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
            A.OneOf([
                A.GaussianBlur(),
                A.MotionBlur(),
                A.MedianBlur(),
            ], p=0.1),
            A.CLAHE(p=0.1), 
            A.OneOf([
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.25),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), 
            A.CoarseDropout(max_holes=16, max_height=64, max_width=64, p=0.2),
            ToTensorV2()
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ])
    )
    eval_metric = PRAUC().torch
    monitor_metrics = [ContinuousAUC(98.).torch, Pfbeta(binarize=False), Pfbeta(binarize=True)]
    callbacks = [
        CollectTopK(3, maximize=True), 
        SaveAverageSnapshot(num_snapshot=3)
    ]


class AuxLoss06(AuxLoss00):
    name = 'aux_06'
    train_path = Path('input/rsna-breast-cancer-detection/train_meta_ishikei.csv')
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=True,
        num_classes=7)
    target_cols = ['cancer']
    dataset_params = dict(
        aux_target_cols=['biopsy', 'b0', 'b1', 'b2', 'b3', 'b4']
    )
    criterion = AuxLoss(loss_types=('bce', 'bce', 'bce', 'bce', 'bce', 'bce', 'bce'), weights=(6., 1., 1., 1., 1., 1., 1.))
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
    eval_metric = PRAUC().torch
    monitor_metrics = [ContinuousAUC(98.).torch, Pfbeta(binarize=False), Pfbeta(binarize=True)]
    callbacks = [
        CollectTopK(3, maximize=True), 
        SaveAverageSnapshot(num_snapshot=3)
    ]


class Dataset03(Baseline4):
    name = 'dataset_03'
    group_col = 'machine_id'


class Dataset04(Baseline4):
    name = 'dataset_04'
    train_path = DATA_DIR/'train_concat_vindr_birads05.csv'


class Dataset05(Baseline4):
    name = 'dataset_05'
    addon_train_path = DATA_DIR/'vindr_train_birads05.csv'


class Model07(Baseline4):
    name = 'model_07'
    model = MultiViewSiameseModel
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True)


class Model08(Baseline4):
    name = 'model_08'
    dataset = PatientLevelDatasetLR
    dataset_params = dict(
        sample_criteria='low_value_for_implant', img_size=2048,
        transform_imagewise=False, separate_channel=True,
    )
    model = MultiViewSiameseLRModel
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        in_chans=1,
        pretrained=True)
    hook = LRTrain()
    batch_size = 8


class Model08aug0(Model08):
    name = 'model_08_aug0'
    dataset_params = dict(
        sample_criteria='low_value_for_implant', img_size=2048,
        transform_imagewise=True, separate_channel=True,
    )


class Model08v0(Model08):
    name = 'model_08_v0'
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        in_chans=1,
        pool_view=True,
        pretrained=True)


class Model08v1(Model08):
    name = 'model_08_v1'
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        in_chans=1,
        dropout=0.1,
        pretrained=True)


class Model09(Model08):
    name = 'model_09'
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        in_chans=1,
        pool_view=True,
        dropout=0.1,
        pretrained=True)


class Model09aux0(Model09):
    name = 'model_09_aux0'
    model_params = dict(
        classification_model='convnext_small.fb_in22k_ft_in1k_384',
        in_chans=1,
        num_classes=3,
        pool_view=True,
        dropout=0.1,
        pretrained=True)
    target_cols = ['cancer']
    dataset_params = dict(
        aux_target_cols=['age', 'biopsy'], 
        sample_criteria='low_value_for_implant', img_size=2048,
        transform_imagewise=False, separate_channel=True,
    )
    criterion = AuxLoss(loss_types=('bce', 'bce', 'bce'), weights=(2., 1., 1.))


class Model10(Aug07aug0):
    name = 'model_10'
    model = MultiLevelModel
    model_params = dict(
        global_model='convnext_small.fb_in22k_ft_in1k_384',
        local_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        crop_size=128,
        crop_num=8,
    )
    criterion = MultiLevelLoss()
    hook = MultiLevelTrain()


class Model10v0(Model10):
    name = 'model_10_v0'
    model_params = dict(
        global_model='convnext_small.fb_in22k_ft_in1k_384',
        local_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        crop_size=128,
        crop_num=8,
        pool_view=True
    )

class Model11(Aug07):
    name = 'model_11'
    model = MultiLevelModel2
    model_params = dict(
        global_model='convnext_tiny.fb_in22k_ft_in1k_384',
        local_model='convnext_small.fb_in22k_ft_in1k_384',
        pretrained=True,
        crop_size=128,
        crop_num=8,
        pool_view=True
    )
    criterion = MultiLevelLoss2(weights=(3., 2., 1.))
    dataset = PatientLevelDatasetWithFindingMask
    dataset_params = dict(
        sample_criteria='valid_area',
        bbox_path='input/rsna-breast-cancer-detection/bbox_all.csv',
        mask_path='input/rsna-breast-cancer-detection/rsna-yolo-crop/vindr_001_baseline/det_result_vindr_001_baseline.csv',
        mask_score=0.05,
        mask_filter=True,
    )
    hook = MultiLevelTrain2()
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(0.1, 0.2, 15, border_mode=0),
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
            A.CoarseDropout(max_holes=16, max_height=64, max_width=64, p=0.2),
            ToTensorV2(transpose_mask=True)
        ]), 
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2(transpose_mask=True)
        ])
    )
    eval_metric = PRAUC().torch
    monitor_metrics = [ContinuousAUC(98.).torch, Pfbeta(binarize=False), Pfbeta(binarize=True)]


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


class Res02vindr(Res02):
    name = 'pretrain_res02_vindr'
    train_path = Path('input/rsna-breast-cancer-detection/vindr_train.csv')
    image_dir = Path('input/rsna-breast-cancer-detection/vindr_mammo_resized_2048V')
    num_epochs = 10


class Res02Aux0(Res02):
    name ='res_02_aux0'
    model_params = dict(
        classification_model='convnext_tiny.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=True,
        num_classes=3)
    target_cols = ['cancer']
    dataset_params = dict(
        aux_target_cols=['age', 'biopsy']
    )
    criterion = AuxLoss(loss_types=('bce', 'mse', 'bce'), weights=(2., 1., 1.))
    hook = AuxLossTrain()


class Res02pr0(Res02Aux0):
    name = 'res_02_pr0'
    weight_path = Path('results/pretrain_res02_vindr/nocv.pt')


class Res02aug0(Res02Aux0):
    name = 'res_02_aug0'
    dataset_params = dict(
        sample_criteria='low_value_for_implant',
        aux_target_cols=['age', 'biopsy'],
        bbox_path='input/rsna-breast-cancer-detection/rsna-yolo-crop/001_baseline/det_result_001_baseline.csv',
    )
    preprocess = dict(
        train=A.Compose([
            RandomCropBBox(buffer=(-20, 100)), 
            AutoFlip(sample_width=100), A.Resize(1536, 768)], 
            bbox_params=A.BboxParams(format='pascal_voc')),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1536, 768)],
            bbox_params=A.BboxParams(format='pascal_voc')),
    )


class Res02aug1(Res02Aux0):
    name = 'res_02_aug1'
    transforms = dict(
        train=A.Compose([
            A.ShiftScaleRotate(0.1, 0.2, 15),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
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
        SaveEveryEpoch(), 
        SaveAverageSnapshot(num_snapshot=3)
    ]


class Res02aug2(Res02Aux0):
    name = 'res_02_aug2'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            # A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
            A.CLAHE(clip_limit=(1,4), p=0.5),
            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3),
            ], p=0.2),
            A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                A.MedianBlur(),
            ], p=0.2),
            A.PiecewiseAffine(p=0.2),
            A.Sharpen(p=0.2),
            A.CoarseDropout(max_holes=16, max_height=96, max_width=96, p=0.2),
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2(),
        ]),
        test=A.Compose([
            A.Normalize(mean=0.485, std=0.229, always_apply=True), ToTensorV2()
        ])
    )
    callbacks = [
        CollectTopK(3, maximize=True), 
        SaveAverageSnapshot(num_snapshot=3)
    ]


class Res02pl0pr0(Res02):
    name = 'res_02_pl0_pr0'
    addon_train_path = Path('input/rsna-breast-cancer-detection/vindr_train_pl_v1_soft_2575.csv')
    weight_path = Path('results/pretrain_res02_vindr/nocv.pt')
    eval_metric = PRAUC().torch
    monitor_metrics = [ContinuousAUC(98.).torch, Pfbeta(binarize=False), Pfbeta(binarize=True)]
    dataset_params = dict(
        sample_criteria='valid_area',
        bbox_path='input/rsna-breast-cancer-detection/bbox_all.csv',
    )
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
            A.ShiftScaleRotate(0.1, 0.2, 15),
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
            A.CoarseDropout(max_holes=20, max_height=64, max_width=64, p=0.25),
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


class Res02pl0ds0(Res02pl0pr0):
    name = 'res_02_pl0_ds0'
    image_dir = Path('input/rsna-breast-cancer-detection/image_resized_3072V')
    dataset_params = dict(
        sample_criteria='valid_area')
    preprocess = dict(
        train=A.Compose([
            AutoFlip(sample_width=200), RandomCropROI(buffer=(-20, 100)), A.Resize(1536, 768)]),
        test=A.Compose([AutoFlip(sample_width=200), CropROI(buffer=80), A.Resize(1536, 768)]),
    )


class Res02mod0(Res02):
    name = 'res_02_mod0'
    dataset = PatientLevelDatasetLR
    dataset_params = dict(
        sample_criteria='low_value_for_implant', img_size=2048,
        transform_imagewise=False, separate_channel=True,
    )
    model = MultiViewSiameseLRModel
    model_params = dict(
        classification_model='convnext_tiny.fb_in22k_ft_in1k_384',
        in_chans=1,
        pretrained=True)
    hook = LRTrain()
    batch_size = 8


class Distillation00(Aug07pl2aug2):
    name = 'distil_00'
    teach_configs = [Aug07, Aug07pl2aug2, AuxLoss03]
    num_epochs = 10
    hook = Distillation()
    inference_hook = Aug07pl2aug2.hook
    criterion = nn.MSELoss()
    eval_metric = PRAUC().torch
    monitor_metrics = [ContinuousAUC(98.).torch, Pfbeta(binarize=False), Pfbeta(binarize=True)]
    callbacks = [
        CollectTopK(2, maximize=True), 
        SaveAverageSnapshot(num_snapshot=2)
    ]
    batch_size = 16


class Distillation00a(Distillation00):
    name = 'distil_00_a'
    num_epochs = 5
    weight_path = Path('results/distil_00/')
    

class Distillation01(Distillation00):
    name = 'distil_01'
    num_epochs = 15
    teach_configs = [Aug07, Aug07pl2aug2, AuxLoss03]
    model_params = dict(
        classification_model='convnext_tiny.fb_in22k_ft_in1k_384',
        pretrained=True,
        spatial_pool=True)