import argparse
from pathlib import Path
from pprint import pprint
import sys
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.utils.data as D
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
import pickle
import traceback
from torch.nn import SyncBatchNorm

from kuma_utils.torch import TorchTrainer, TorchLogger
from kuma_utils.torch.utils import get_time, seed_everything, fit_state_dict
from kuma_utils.torch.temperature_scaling import TemperatureScaler
from timm.layers import convert_sync_batchnorm

from configs_seg import *
from utils import print_config, notify_me
from metrics import Pfbeta


def oversample_data(df, n_times=0):
    def add_oversample_id(df, oid):
        df['oversample_id'] = oid
        return df
    if n_times > 0:
        df = pd.concat([add_oversample_id(df, 0)] + [
            add_oversample_id(df.query('cancer == 1'), i+1) for i in range(n_times)], axis=0)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='Baseline',
                        help="config name in configs.py")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--inference", action='store_true',
                        help="inference")
    parser.add_argument("--tta", action='store_true', 
                        help="test time augmentation ")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--silent", action='store_true')
    parser.add_argument("--progress_bar", action='store_true')
    parser.add_argument("--skip_existing", action='store_true')
    parser.add_argument("--calibrate", action='store_true')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--wait", type=int, default=0,
                        help="time (sec) to wait before execution")
    opt = parser.parse_args()
    pprint(opt)

    ''' Configure hardware '''
    opt.gpu = None # use all visible GPUs
    
    ''' Configure path '''
    cfg = eval(opt.config)
    export_dir = Path('results') / cfg.name
    export_dir.mkdir(parents=True, exist_ok=True)

    ''' Configure logger '''
    log_items = [
        'epoch', 'train_loss', 'train_metric', 'train_monitor', 
        'valid_loss', 'valid_metric', 'valid_monitor', 
        'learning_rate', 'early_stop'
    ]
    if opt.debug:
        log_items += ['gpu_memory']
    logger_path = f'{cfg.name}_nocv_{get_time("%y%m%d%H%M")}.log'
    LOGGER = TorchLogger(
        export_dir / logger_path, 
        log_items=log_items, file=not opt.silent
    )
    if opt.wait > 0:
        LOGGER(f'Waiting for {opt.wait} sec.')
        time.sleep(opt.wait)

    ''' Prepare data '''
    seed_everything(cfg.seed, cfg.deterministic)
    print_config(cfg, LOGGER)
    train = pd.read_csv(cfg.train_path)
    valid = train.query('split == "test"')
    train = train.query('split == "training"')
    if opt.debug:
        train = train.iloc[:1000]
    
    '''
    Training
    '''
    scores = []
    
    if opt.inference:
        pass

    else:
        fold = 0
        LOGGER(f'===== TRAINING FOLD {fold} =====')

        train_fold = train
        valid_fold = valid
        train_fold = oversample_data(train_fold, cfg.oversample_ntimes)

        train_data = cfg.dataset(
            df=train_fold,
            image_dir=cfg.image_dir,
            preprocess=cfg.preprocess['train'],
            transforms=cfg.transforms['train'],
            is_test=False,
            **cfg.dataset_params)
        valid_data = cfg.dataset(
            df=valid_fold,
            image_dir=cfg.image_dir,
            preprocess=cfg.preprocess['test'],
            transforms=cfg.transforms['test'],
            is_test=True,
            **cfg.dataset_params)

        train_loader = D.DataLoader(
            train_data, batch_size=cfg.batch_size, 
            shuffle=True, num_workers=opt.num_workers, pin_memory=True)
        valid_loader = D.DataLoader(
            valid_data, batch_size=cfg.batch_size, 
            shuffle=False, num_workers=opt.num_workers, pin_memory=True)

        model = cfg.model(**cfg.model_params)

        # Load snapshot
        if cfg.weight_path is not None:
            if cfg.weight_path.is_dir():
                weight_path = cfg.weight_path / f'fold{fold}.pt'
            else:
                weight_path = cfg.weight_path
            LOGGER(f'{weight_path} loaded.')
            weight = torch.load(weight_path, 'cpu')['model']
            fit_state_dict(weight, model)
            model.load_state_dict(weight, strict=False)
            del weight; gc.collect()

        if cfg.encoder_lr is not None:
            optimizer = cfg.optimizer(
                [
                    {"params": model.encoder.parameters(), "lr": cfg.encoder_lr},
                    {"params": model.decoder.parameters()},
                    {"params": model.segmentation_head.parameters()},
                ], 
                **cfg.optimizer_params)
        else:
            optimizer = cfg.optimizer(model.parameters(), **cfg.optimizer_params)
        scheduler = cfg.scheduler(optimizer, **cfg.scheduler_params)
        FIT_PARAMS = {
            'loader': train_loader,
            'loader_valid': valid_loader,
            'criterion': cfg.criterion,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'scheduler_target': cfg.scheduler_target,
            'batch_scheduler': cfg.batch_scheduler, 
            'num_epochs': cfg.num_epochs,
            'callbacks': deepcopy(cfg.callbacks),
            'hook': cfg.hook,
            'export_dir': export_dir,
            'eval_metric': cfg.eval_metric,
            'monitor_metrics': cfg.monitor_metrics,
            'fp16': cfg.amp,
            'parallel': cfg.parallel,
            'deterministic': cfg.deterministic, 
            'clip_grad': cfg.clip_grad, 
            'max_grad_norm': cfg.max_grad_norm,
            'random_state': cfg.seed,
            'logger': LOGGER,
            'progress_bar': opt.progress_bar, 
            'resume': opt.resume
        }
        try:
            trainer = TorchTrainer(model, serial=f'nocv', device=None)
            trainer.ddp_sync_batch_norm = convert_sync_batchnorm
            trainer.ddp_params = dict(
                broadcast_buffers=True, 
                find_unused_parameters=True
            )
            trainer.fit(**FIT_PARAMS)
        except Exception as e:
            err = traceback.format_exc()
            LOGGER(err)
            if not opt.silent:
                notify_me('\n'.join([
                    f'[{cfg.name}:nocv]', 
                    'Training stopped due to:', 
                    f'{traceback.format_exception_only(type(e), e)}'
                ]))
        del model, trainer, train_data; gc.collect()
        torch.cuda.empty_cache()
    
