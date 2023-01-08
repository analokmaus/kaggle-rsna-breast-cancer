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

from configs import *
from utils import print_config, notify_me


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='Baseline',
                        help="config name in configs.py")
    parser.add_argument("--only_fold", type=int, default=-1,
                        help="train only specified fold")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--inference", action='store_true',
                        help="inference")
    parser.add_argument("--tta", action='store_true', 
                        help="test time augmentation ")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--silent", action='store_true')
    parser.add_argument("--progress_bar", action='store_true')
    parser.add_argument("--skip_existing", action='store_true')
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
    if opt.only_fold >= 0:
        logger_path = f'{cfg.name}_fold{opt.only_fold}_{get_time("%y%m%d%H%M")}.log'
    else:
        logger_path = f'{cfg.name}_{get_time("%y%m%d%H%M")}.log'
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
    if opt.debug:
        train = train.iloc[:1000]
    splitter = cfg.splitter
    fold_iter = list(splitter.split(X=train, y=train[cfg.target_cols], groups=train[cfg.group_col]))
    
    '''
    Training
    '''
    scores = []
    for fold, (train_idx, valid_idx) in enumerate(fold_iter):
        
        if opt.only_fold >= 0 and fold != opt.only_fold:
            continue  # skip fold

        if opt.inference:
            continue

        if opt.skip_existing and (export_dir/f'fold{fold}.pt').exists():
            LOGGER(f'checkpoint fold{fold}.pt already exists.')
            continue

        LOGGER(f'===== TRAINING FOLD {fold} =====')

        train_fold = train.iloc[train_idx]
        valid_fold = train.iloc[valid_idx]

        train_data = cfg.dataset(
            df=train_fold,
            preprocess=cfg.preprocess['train'],
            transforms=cfg.transforms['train'],
            is_test=False,
            **cfg.dataset_params)
        valid_data = cfg.dataset(
            df=valid_fold, 
            preprocess=cfg.preprocess['test'],
            transforms=cfg.transforms['test'],
            is_test=False,
            **cfg.dataset_params)

        train_loader = D.DataLoader(
            train_data, batch_size=cfg.batch_size, shuffle=True,
            num_workers=opt.num_workers, pin_memory=False)
        valid_loader = D.DataLoader(
            valid_data, batch_size=cfg.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=False)

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
            trainer = TorchTrainer(model, serial=f'fold{fold}', device=opt.gpu)
            trainer.fit(**FIT_PARAMS)
        except Exception as e:
            err = traceback.format_exc()
            LOGGER(err)
            if not opt.silent:
                notify_me('\n'.join([
                    f'[{cfg.name}:fold{opt.only_fold}]', 
                    'Training stopped due to:', 
                    f'{traceback.format_exception_only(type(e), e)}'
                ]))
        del model, trainer, train_data, valid_data; gc.collect()
        torch.cuda.empty_cache()


    '''
    Inference
    '''
    outoffolds = np.full((len(train), 1), 0, dtype=np.float32)
    selfpreditions = np.full((cfg.cv, len(train), 1), 0, dtype=np.float32)
    for fold, (train_idx, valid_idx) in enumerate(fold_iter):

        if not (export_dir/f'fold{fold}.pt').exists():
            LOGGER(f'fold{fold}.pt missing. No target to predict.')
            continue

        LOGGER(f'===== INFERENCE FOLD {fold} =====')

        # inference_data = cfg.dataset(
        #     image_paths=train['image_paths'].values, 
        #     labels=train['label'].values, 
        #     preprocess=cfg.preprocess['test'],
        #     transforms=cfg.transforms['test'],
        #     is_test=False,
        #     **cfg.dataset_params)
        # inference_loader = D.DataLoader(
        #     inference_data,
        #     batch_size=min(4, cfg.batch_size//torch.cuda.device_count()), 
        #     shuffle=False,
        #     num_workers=opt.num_workers, pin_memory=False)

        # model = cfg.model(**cfg.model_params)
        checkpoint = torch.load(export_dir/f'fold{fold}.pt', 'cpu')
        scores.append(checkpoint['state']['best_score'])
        # fit_state_dict(checkpoint['model'], model)
        # model.load_state_dict(checkpoint['model'])
        # del checkpoint; gc.collect()
        # if cfg.parallel == 'ddp':
        #     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # trainer = TorchTrainer(model, serial=f'fold{fold}', device=opt.gpu)
        # trainer.register(hook=cfg.hook, callbacks=cfg.callbacks)

        # prediction = trainer.predict(inference_loader, progress_bar=opt.progress_bar)

        # outoffolds[valid_idx] = prediction[valid_idx]
        # selfpreditions[fold] = prediction

        # del model, trainer, inference_data; gc.collect()
        # torch.cuda.empty_cache()
    
    with open(str(export_dir/'predictions.pickle'), 'wb') as f:
        pickle.dump({
            'folds': fold_iter,
            'outoffolds': outoffolds,
            'predictions': selfpreditions
        }, f)

    LOGGER(f'scores: {scores}')
    LOGGER(f'mean +- std: {np.mean(scores):.5f} +- {np.std(scores):.5f}')
    if not opt.silent:
        notify_me('\n'.join([
            f'[{cfg.name}:fold{opt.only_fold}]',
            'Training has finished successfully.',
            f'mean +- std: {np.mean(scores):.5f} +- {np.std(scores):.5f}'
        ]))
