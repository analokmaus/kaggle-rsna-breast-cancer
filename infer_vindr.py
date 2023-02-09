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
    

    '''
    Prediction and calibration
    '''
    test = pd.read_csv('input/rsna-breast-cancer-detection/vindr_train.csv')
    predictions = []
    cfg.dataset_params['aux_target_cols'] = []
    test_data = cfg.dataset(
        df=test, 
        image_dir=cfg.image_dir,
        preprocess=cfg.preprocess['test'],
        transforms=cfg.transforms['test'],
        is_test=True,
        **cfg.dataset_params)
    test_loader = D.DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    for fold in range(cfg.cv):

        if not (export_dir/f'fold{fold}.pt').exists():
            LOGGER(f'fold{fold}.pt missing. No target to predict.')
            continue

        LOGGER(f'===== INFERENCE FOLD {fold} =====')

        model = cfg.model(**cfg.model_params)
        checkpoint = torch.load(export_dir/f'fold{fold}.pt', 'cpu')
        fit_state_dict(checkpoint['model'], model)
        model.load_state_dict(checkpoint['model'])
        if cfg.parallel == 'ddp':
            model = convert_sync_batchnorm(model)
            inference_parallel = 'dp'
            test_loader = D.DataLoader(
                test_data, batch_size=cfg.batch_size*4, shuffle=False,
                num_workers=opt.num_workers, pin_memory=True)
        else:
            inference_parallel = None

        trainer = TorchTrainer(model, serial=f'fold{fold}', device=opt.gpu)
        trainer.logger = LOGGER
        trainer.register(hook=cfg.hook, callbacks=cfg.callbacks)
        pred_logits = trainer.predict(test_loader, parallel=inference_parallel, progress_bar=opt.progress_bar)
        predictions.append(sigmoid(pred_logits))
        torch.cuda.empty_cache()
    predictions = np.stack(predictions, axis=0)
    
    with open(str(export_dir/'predictions_vindr.pickle'), 'wb') as f:
        pickle.dump({
            'predictions': predictions
        }, f)
    