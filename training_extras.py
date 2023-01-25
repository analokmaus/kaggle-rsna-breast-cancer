import torch
import pandas as pd
from torch.distributions.beta import Beta
from kuma_utils.torch.utils import freeze_module
from kuma_utils.torch.hooks import SimpleHook
from kuma_utils.torch.callbacks import CallbackTemplate


class MixupTrain(SimpleHook):

    def __init__(self, evaluate_in_batch=False, alpha=0.4, hard_label=False, lor_label=False):
        super().__init__(evaluate_in_batch=evaluate_in_batch)
        self.alpha = alpha
        self.beta = Beta(alpha, alpha)
        self.hard_label = hard_label
        self.lor_label = lor_label

    def forward_train(self, trainer, inputs):
        target = inputs[-1]
        bs = target.shape[0]
        lam = self.beta.rsample(sample_shape=(bs,)).to(target.device)
        idx = torch.randperm(bs).to(target.device)
        approx, lam = trainer.model(*inputs[:-1], lam=lam, idx=idx)
        if self.lor_label:
            target = target + (1 - target) * target[idx]
        else:
            target = target * lam[:, None] + target[idx] * (1-lam)[:, None]
            if self.hard_label:
                target = (target > self.hard_label).float()
        loss = trainer.criterion(approx, target)
        return loss, approx.detach()

    def forward_valid(self, trainer, inputs):
        return super().forward_train(trainer, inputs)

    def __repr__(self) -> str:
        return f'MixUp(alpha={self.alpha}, hard_label={self.hard_label}, lor_label={self.lor_label})'


class AuxLossTrain(SimpleHook):
    '''
    '''

    def __init__(self, evaluate_in_batch=False):
        super().__init__(evaluate_in_batch=evaluate_in_batch)
    
    def forward_train(self, trainer, inputs):
        targets = inputs[-1]
        approxs = trainer.model(*inputs[:-1])
        loss = trainer.criterion(approxs, targets)
        storage = trainer.epoch_storage
        storage['approx'].append(approxs[:, 0].view(-1, 1).detach())
        storage['target'].append(targets[:, 0].view(-1, 1).detach())
        return loss, approxs[:, 0].view(-1, 1).detach()

    forward_valid = forward_train

    def evaluate_batch(self, trainer, inputs, approx):
        pass

    def forward_test(self, trainer, inputs):
        approxs = trainer.model(*inputs[-1])
        return approxs[:, 0].view(-1, 1)

    def __repr__(self) -> str:
        return f'AuxLossTrain()'


class MultiLevelTrain(SimpleHook):
    '''
    '''

    def __init__(self, evaluate_in_batch=False):
        super().__init__(evaluate_in_batch=evaluate_in_batch)
    
    def forward_train(self, trainer, inputs):
        target = inputs[-1]
        approx0, approx1, approx2 = trainer.model(inputs[0])
        loss = trainer.criterion((approx0, approx1, approx2), target)
        return loss, approx0.detach()

    forward_valid = forward_train

    def forward_test(self, trainer, inputs):
        approx, _, _ = trainer.model(inputs[0])
        return approx

    def __repr__(self) -> str:
        return f'MultiLevelTrain()'


class SingleImageAggregatedTrain(SimpleHook):
    '''
    '''

    def __init__(self, evaluate_in_batch=False, multilevel=False):
        super().__init__(evaluate_in_batch=evaluate_in_batch)
        self.multilevel = multilevel

    def _evaluate(self, trainer, approx, target):
        group_ids = trainer.epoch_storage['group_id'].cpu().numpy().reshape(-1)
        lateralities = trainer.epoch_storage['laterality'].cpu().numpy().reshape(-1)
        approx = approx.cpu().float().numpy().reshape(-1)
        target = target.cpu().float().numpy().reshape(-1)
        agg_df = pd.DataFrame(
            {'patient_id': group_ids, 'laterality': lateralities, 
             'approx': approx, 'target': target}).groupby(['patient_id', 'laterality']).agg(
                {'approx': 'max', 'target': 'max'})

        if trainer.eval_metric is None:
            metric_score = None
        else:
            metric_score = trainer.eval_metric(
                torch.from_numpy(agg_df['approx'].values.reshape(-1, 1)), 
                torch.from_numpy(agg_df['target'].values.reshape(-1, 1)))
        monitor_score = []
        for monitor_metric in trainer.monitor_metrics:
            monitor_score.append(
                monitor_metric(
                    torch.from_numpy(agg_df['approx'].values.reshape(-1, 1)), 
                    torch.from_numpy(agg_df['target'].values.reshape(-1, 1))))
        return metric_score, monitor_score

    def evaluate_batch(self, trainer, inputs, approx):
        pass
    
    def forward_train(self, trainer, inputs):
        input_t, target_t, group_id, laterality = inputs
        storage = trainer.epoch_storage
        storage['group_id'].append(group_id)
        storage['laterality'].append(laterality)
        if self.multilevel:
            approx0, approx1, approx2 = trainer.model(input_t)
            loss = trainer.criterion((approx0, approx1, approx2), target_t)
        else:
            approx0 = trainer.model(input_t)
            loss = trainer.criterion(approx0, target_t)
        storage['approx'].append(approx0.detach())
        storage['target'].append(target_t)
        return loss, approx0.detach()

    forward_valid = forward_train

    def forward_test(self, trainer, inputs):
        if self.multilevel:
            approx, _, _ = trainer.model(inputs[0])
        else:
            approx = trainer.model(inputs[0])
        return approx

    def __repr__(self) -> str:
        return f'SingleImageAggregatedTrain()'


class StepDataset(CallbackTemplate):

    def __init__(self):
        super().__init__()
    
    def after_epoch(self, env, loader, loader_valid):
        loader.dataset.step()
