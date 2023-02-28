import torch
import torch.nn as nn
import torch.nn.functional as F
from global_objectives.losses import AUCPRLoss


def binary_cross_entropy(inputs, target, weight=None, reduction='mean', smooth_eps=None, from_logits=False):
    """cross entropy loss, with support for label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0
    if smooth_eps > 0:
        _target = target.clone().float()
        # _target.add_(smooth_eps).div_(2.)
        _target.mul_(1-smooth_eps).add(0.5*smooth_eps)
    else:
        _target = target
    if from_logits:
        return F.binary_cross_entropy_with_logits(inputs, _target, weight=weight, reduction=reduction)
    else:
        return F.binary_cross_entropy(inputs, _target, weight=weight, reduction=reduction)


def binary_cross_entropy_with_logits(inputs, target, weight=None, reduction='mean', smooth_eps=None, from_logits=True):
    return binary_cross_entropy(inputs, target, weight, reduction, smooth_eps, from_logits)


class BCELoss(nn.BCELoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', smooth_eps=None, from_logits=False):
        super(BCELoss, self).__init__(weight, size_average, reduce, reduction)
        self.smooth_eps = smooth_eps
        self.from_logits = from_logits

    def forward(self, input, target):
        return binary_cross_entropy(input, target,
                                    weight=self.weight, reduction=self.reduction,
                                    smooth_eps=self.smooth_eps, from_logits=self.from_logits)

    def __repr__(self):
        return f'BCELoss(smooth={self.smooth_eps})'


class BCEWithLogitsLoss(BCELoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', smooth_eps=None, from_logits=True):
        super(BCEWithLogitsLoss, self).__init__(weight, size_average,
                                                reduce, reduction, smooth_eps=smooth_eps, from_logits=from_logits)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if not isinstance(smoothing, torch.Tensor):
            self.smoothing = nn.Parameter(
                torch.tensor(smoothing), requires_grad=False)
        else:
            self.smoothing = nn.Parameter(
                smoothing, requires_grad=False)
        assert 0 <= self.smoothing.min() and self.smoothing.max() < 1
    
    @staticmethod
    def _smooth(targets:torch.Tensor, smoothing:torch.Tensor):
        with torch.no_grad():
            if smoothing.shape != targets.shape:
                _smoothing = smoothing.expand_as(targets)
            else:
                _smoothing = smoothing
            return targets * (1.0 - _smoothing) + 0.5 * _smoothing

    def forward(self, inputs, targets):
        targets = self._smooth(targets, self.smoothing)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1. - pt)**self.gamma * bce_loss
        return focal_loss.mean()

    def __repr__(self):
        return f'FocalLoss(smoothing={self.smoothing})'


class MultiLevelLoss(nn.Module):
    '''
    weights: (float, float, float) # concat, global, local
    '''
    def __init__(self, weights=(1., 1., 1.), pos_weight=None, loss_type='bce'):
        super().__init__()
        self.weights = weights
        self.pos_weight = pos_weight
        if loss_type == 'bce':
            self.loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        elif loss_type == 'aucpr':
            self.loss = AUCPRLoss()
        else:
            self.loss = loss_type
    
    def forward(self, inputs, target):
        loss = 0
        for i, w in enumerate(self.weights):
            loss += w * self.loss(inputs[i], target)
        return loss / sum(self.weights)
    
    def __repr__(self):
        return f'MultiLevel(weights={self.weights}, pos_weight={self.pos_weight})'


class MultiLevelLoss2(nn.Module):
    '''
    weights: (float, float, float) # concat, local, cam
    '''
    def __init__(self, weights=(1., 1., 1.), pos_weight=None):
        super().__init__()
        self.weights = weights

    def forward(self, inputs, target, target_cam):
        loss = self.weights[0] * binary_cross_entropy_with_logits(inputs[0], target)
        loss += self.weights[1] * binary_cross_entropy_with_logits(inputs[1], target)
        if inputs[2].shape[2:] != target_cam.shape[2:]:
            target_cam = F.interpolate(target_cam, size=inputs[2].shape[2:], mode="nearest")
        loss += self.weights[2] * F.mse_loss(inputs[2], target_cam)
        return loss / sum(self.weights)
    
    def __repr__(self):
        return f'MultiLevel2(weights={self.weights})'


class AuxLoss(nn.Module):
    '''
    '''
    def __init__(self, loss_types=('bce', 'mse'), weights=(1., 1.)):
        super().__init__()
        loss_dict = {
            'bce': binary_cross_entropy_with_logits,
            'mse': F.mse_loss,
            'aucpr': AUCPRLoss(),
        }
        self.loss_types = loss_types
        self.weights = weights
        self.loss_list = [loss_dict[l] for l in loss_types]
    
    def forward(self, inputs, targets):
        loss = 0
        for i, (f, w) in enumerate(zip(self.loss_list, self.weights)):
            loss += w * f(inputs[:, i], targets[:, i])
        return loss / sum(self.weights)
    
    def __repr__(self):
        return f'AuxLoss(loss_types={self.loss_types}, weights={self.weights})'
