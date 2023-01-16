import torch
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


class StepDataset(CallbackTemplate):

    def __init__(self):
        super().__init__()
    
    def after_epoch(self, env, loader, loader_valid):
        loader.dataset.step()
