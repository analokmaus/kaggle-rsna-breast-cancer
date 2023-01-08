import numpy as np
import torch.nn as nn
from kuma_utils.metrics import MetricTemplate


class Pfbeta(nn.Module):
    '''
    '''
    def __init__(self):
        super().__init__()

    def pfbeta(self, labels, predictions, beta=1.):
        y_true_count = 0
        ctp = 0
        cfp = 0

        for idx in range(len(labels)):
            prediction = min(max(predictions[idx], 0), 1)
            if (labels[idx]):
                y_true_count += 1
                ctp += prediction
            else:
                cfp += prediction

        beta_squared = beta * beta
        c_precision = ctp / (ctp + cfp)
        c_recall = ctp / max(y_true_count, 1)  # avoid / 0
        if (c_precision > 0 and c_recall > 0):
            result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
            return result
        else:
            return 0

    def optimal_f1(self, labels, predictions):
        thres = np.linspace(0, 1, 101)
        f1s = [self.pfbeta(labels, predictions > thr) for thr in thres]
        idx = np.argmax(f1s)
        return f1s[idx], thres[idx]

    def forward(self, approx, target):
        f1s, thres = self.optimal_f1(
            target.detach().cpu().numpy(), approx.sigmoid().detach().cpu().numpy())
        return f1s[0]
