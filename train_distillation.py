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
from kuma_utils.utils import sigmoid
from timm.layers import convert_sync_batchnorm

from configs import *
from utils import print_config, notify_me
from metrics import Pfbeta
from training_extras import extend_df
from architectures import DistillationModel


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
    parser.add_argument("--only_fold", type=int, default=-1,
                        help="train only specified fold")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--inference", action='store_true', help="inference")
    parser.add_argument("--extended_inference", action='store_true')
    parser.add_argument("--per_site_inference", action='store_true')
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
    # opt.num_workers = min(cfg.batch_size, opt.num_workers)

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
    # data preprocessor
    if 'density' in train.columns:
        train['density'] = train['density'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3})
    if 'age' in train.columns:
        train['age'] = train['age'].fillna(train['age'].mean()) / 100.
    if 'aux_target_cols' in cfg.dataset_params.keys():
        if 'machine_id' in cfg.dataset_params['aux_target_cols']:
            train['machine_id'] = train['machine_id'].isin([93, 210, 216]).astype(float)
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
        if cfg.addon_train_path is not None:
            addon_train = pd.read_csv(cfg.addon_train_path)
            if 'fold' in addon_train.columns:
                addon_train = addon_train.loc[addon_train['fold'] == fold]
            train_data.update_df(addon_train)
        train_weights = train_data.get_labels().reshape(-1)
        valid_weights = valid_data.get_labels().reshape(-1)
        train_weights[train_weights == 1] = (train_weights == 0).sum() / (train_weights == 1).sum()
        train_weights[train_weights == 0] = 1
        if cfg.sampler is not None:
            sampler = cfg.sampler(train_weights.tolist(), len(train_weights))
        else:
            sampler = None
        LOGGER(f'train count: {len(train_data)} / valid count: {len(valid_data)}')
        LOGGER(f'train pos: {train_data.get_labels().reshape(-1).mean():.5f} / valid pos: {valid_weights.mean():.5f}')

        train_loader = D.DataLoader(
            train_data, batch_size=cfg.batch_size, 
            shuffle=True if cfg.sampler is None else False,
            sampler=sampler, num_workers=opt.num_workers, pin_memory=True)
        valid_loader = D.DataLoader(
            valid_data, batch_size=cfg.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True)

        teacher_models = []
        for t_cfg in cfg.teach_configs:
            model = t_cfg.model(**t_cfg.model_params)
            weight_path = Path(f'results/{t_cfg.name}/fold{fold}.pt')
            weight = torch.load(weight_path, 'cpu')['model']
            model.load_state_dict(weight, strict=False)
            teacher_models.append(model)
            LOGGER(f'teacher model: {t_cfg.name} loaded')
        
        student_model = cfg.model(**cfg.model_params)
        model = DistillationModel(teacher_models=teacher_models, student_model=student_model)

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
            'grad_accumulations': cfg.grad_accumulations, 
            'random_state': cfg.seed,
            'logger': LOGGER,
            'progress_bar': opt.progress_bar, 
            'resume': opt.resume
        }
        try:
            trainer = TorchTrainer(model, serial=f'fold{fold}', device=None)
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
                    f'[{cfg.name}:fold{opt.only_fold}]', 
                    'Training stopped due to:', 
                    f'{traceback.format_exception_only(type(e), e)}'
                ]))
        del model, trainer, train_data, valid_data; gc.collect()
        torch.cuda.empty_cache()


    '''
    Prediction and calibration
    '''
    outoffolds = []
    oof_targets = []
    thresholds = []
    sites = []
    eval_metric = Pfbeta(binarize=True, return_thres=True)
    
    for fold, (train_idx, valid_idx) in enumerate(fold_iter):

        if not (export_dir/f'fold{fold}.pt').exists():
            LOGGER(f'fold{fold}.pt missing. No target to predict.')
            continue

        LOGGER(f'===== INFERENCE FOLD {fold} =====')

        valid_fold = train.iloc[valid_idx]
        if opt.extended_inference:
            valid_fold = extend_df(valid_fold, 2)
        valid_data = cfg.dataset(
            df=valid_fold, 
            image_dir=cfg.image_dir,
            preprocess=cfg.preprocess['test'],
            transforms=cfg.transforms['test'],
            is_test=True,
            **cfg.dataset_params)
        valid_loader = D.DataLoader(
            valid_data, batch_size=cfg.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True)
    
        teacher_models = []
        for t_cfg in cfg.teach_configs:
            model = t_cfg.model(**t_cfg.model_params)
            teacher_models.append(model)
        model = DistillationModel(
            teacher_models=teacher_models,
            student_model=cfg.model(**cfg.model_params))
        
        checkpoint = torch.load(export_dir/f'fold{fold}.pt', 'cpu')
        fit_state_dict(checkpoint['model'], model)
        model.load_state_dict(checkpoint['model'])
        # clean up checkpoint
        if 'checkpoints' in checkpoint.keys():
            del checkpoint['checkpoints']
        model = model.student
        checkpoint['model'] = model.state_dict()
        torch.save(checkpoint, export_dir/f'fold{fold}.pt')
        del checkpoint; gc.collect()

        if cfg.parallel == 'ddp':
            model = convert_sync_batchnorm(model)
            inference_parallel = 'dp'
            valid_loader = D.DataLoader(
                valid_data, batch_size=cfg.batch_size*4, shuffle=False,
                num_workers=opt.num_workers, pin_memory=False)
        else:
            inference_parallel = None

        trainer = TorchTrainer(model, serial=f'fold{fold}', device=opt.gpu)
        trainer.logger = LOGGER
        trainer.register(hook=cfg.inference_hook, callbacks=cfg.callbacks)
        pred_logits = trainer.predict(valid_loader, parallel=inference_parallel, progress_bar=opt.progress_bar)
        target_fold = valid_data.get_labels()
        valid_sites = [valid_data.df_dict[valid_data.pids[i]]['site_id'].values[0] for i in range(len(valid_data))]

        if cfg.hook.__class__.__name__ == 'SingleImageAggregatedTrain': # max aggregation
            valid_fold['prediction'] = pred_logits.reshape(-1)
            agg_df = valid_fold.groupby(['patient_id', 'laterality']).agg(
                {'prediction': 'max', 'cancer': 'first'})
            pred_logits = agg_df['prediction'].values.reshape(-1, 1)
            target_fold = agg_df['cancer'].values.reshape(-1, 1)
        
        if opt.extended_inference:
            results = []
            for (_, pid, lat), p, t, s in zip(valid_data.pids, pred_logits.reshape(-1), target_fold.reshape(-1), valid_sites):
                results.append({'prediction_id': f'{pid}_{lat}', 'pred': p, 'target': t, 'site_id': s})
            results = pd.DataFrame(results).groupby('prediction_id').agg({'pred': 'mean', 'target': 'max', 'site_id': 'max'}).reset_index()
            pred_logits = results['pred'].values.reshape(-1, 1)
            target_fold = results['target'].values.reshape(-1, 1)
            valid_sites = results['site_id'].values.tolist()

        if opt.per_site_inference:
            sites.append(valid_sites)
            results = pd.DataFrame({
                'site_id': valid_sites, 
                'pred': pred_logits.reshape(-1), 
                'pred_bin': pred_logits.reshape(-1), 
                'label': target_fold.reshape(-1)})
            thres_fold = []
            for site in [1, 2]:
                score_site, thres_site = eval_metric(
                    torch.from_numpy(results.query('site_id == @site')['pred'].values),
                    torch.from_numpy(results.query('site_id == @site')['label'].values), )
                LOGGER(f'Site {site} PFbeta: {score_site:.5f} threshold: {thres_site:.5f}')
                results.loc[results['site_id'] == site, 'pred_bin'] = (sigmoid(results.loc[results['site_id'] == site, 'pred']) > thres_site).astype(float)
                thres_fold.append(thres_site)
            pred_logits = results['pred'].values.reshape(-1, 1)
            target_fold = results['label'].values.reshape(-1, 1)
            eval_score_fold = eval_metric.pfbeta(results['label'].values, results['pred_bin'].values)
            LOGGER(f'Overall PFbeta: {eval_score_fold:.5f}')
            thresholds.append(thres_fold)
        else:
            eval_score_fold, thres = eval_metric(torch.from_numpy(pred_logits), torch.from_numpy(target_fold))
            LOGGER(f'PFbeta: {eval_score_fold:.5f} threshold: {thres:.5f}')
            thresholds.append(thres)

        for im, metric_f in enumerate(cfg.monitor_metrics):
            LOGGER(f'Monitor metric {im}: {metric_f(torch.from_numpy(pred_logits), torch.from_numpy(target_fold)):.5f}')
        
        scores.append(eval_score_fold)
        outoffolds.append(pred_logits)
        oof_targets.append(target_fold)
        
        torch.cuda.empty_cache()
    
    with open(str(export_dir/f'predictions{"_ext" if opt.extended_inference else ""}.pickle'), 'wb') as f:
        pickle.dump({
            'folds': fold_iter,
            'outoffolds': outoffolds, 
            'targets': oof_targets,
            'sites': sites,
            'thresholds': thresholds
        }, f)

    LOGGER(f'scores: {scores}')
    LOGGER(f'mean +- std: {np.mean(scores):.5f} +- {np.std(scores):.5f}')
    if not opt.silent:
        notify_me('\n'.join([
            f'[{cfg.name}:fold{opt.only_fold}]',
            'Training has finished successfully.',
            f'mean +- std: {np.mean(scores):.5f} +- {np.std(scores):.5f}'
        ]))
