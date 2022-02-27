# ----------------------------
# (adapted from SiamRPN++)
# ---------------------------
import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
import numpy as np

class LRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        if 'lr_spaces' not in self.__dict__:
            raise Exception('lr_spaces must be set in "LRSchduler"')
        super(LRScheduler, self).__init__(optimizer, last_epoch)

    def get_cur_lr(self):
        return self.lr_spaces[self.last_epoch]

    def get_lr(self):
        epoch = self.last_epoch
        return [self.lr_spaces[epoch] * pg['initial_lr'] / self.start_lr
                for pg in self.optimizer.param_groups]

    def __repr__(self):
        return "({}) lr spaces: \n{}".format(self.__class__.__name__,
                                             self.lr_spaces)


class LogScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.03, end_lr=5e-4,
                 epochs=50, last_epoch=-1):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.epochs = epochs
        self.lr_spaces = np.logspace(math.log10(start_lr),
                                     math.log10(end_lr),
                                     epochs)

        super(LogScheduler, self).__init__(optimizer, last_epoch)


class StepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=None,
                 step=10, mult=0.1, epochs=50, last_epoch=-1):
        if end_lr is not None:
            if start_lr is None:
                start_lr = end_lr / (mult ** (epochs // step))
            else:  # for warm up policy
                mult = math.pow(end_lr/start_lr, 1. / (epochs // step))
        self.start_lr = start_lr
        self.lr_spaces = self.start_lr * (mult**(np.arange(epochs) // step))
        self.mult = mult
        self._step = step

        super(StepScheduler, self).__init__(optimizer, last_epoch)


class MultiStepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=None,
                 steps=[10, 20, 30, 40], mult=0.5, epochs=50,
                 last_epoch=-1):
        if end_lr is not None:
            if start_lr is None:
                start_lr = end_lr / (mult ** (len(steps)))
            else:
                mult = math.pow(end_lr/start_lr, 1. / len(steps))
        self.start_lr = start_lr
        self.lr_spaces = self._build_lr(start_lr, steps, mult, epochs)
        self.mult = mult
        self.steps = steps

        super(MultiStepScheduler, self).__init__(optimizer, last_epoch)

    def _build_lr(self, start_lr, steps, mult, epochs):
        lr = [0] * epochs
        lr[0] = start_lr
        for i in range(1, epochs):
            lr[i] = lr[i-1]
            if i in steps:
                lr[i] *= mult
        return np.array(lr, dtype=np.float32)


class LinearStepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=0.005,
                 epochs=50, last_epoch=-1):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_spaces = np.linspace(start_lr, end_lr, epochs)
        super(LinearStepScheduler, self).__init__(optimizer, last_epoch)


class CosStepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=0.005,
                 epochs=50, last_epoch=-1):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_spaces = self._build_lr(start_lr, end_lr, epochs)

        super(CosStepScheduler, self).__init__(optimizer, last_epoch)

    def _build_lr(self, start_lr, end_lr, epochs):
        index = np.arange(epochs).astype(np.float32)
        lr = end_lr + (start_lr - end_lr) * \
            (1. + np.cos(index * np.pi / epochs)) * 0.5
        return lr.astype(np.float32)


class WarmUPScheduler(LRScheduler):
    def __init__(self, optimizer, warmup, normal, last_epoch=-1):
        warmup = warmup.lr_spaces  # [::-1]
        normal = normal.lr_spaces
        self.lr_spaces = np.concatenate([warmup, normal])
        self.start_lr = normal[0]

        super(WarmUPScheduler, self).__init__(optimizer, last_epoch)


LRs = {
    'log': LogScheduler,
    'step': StepScheduler,
    'multi-step': MultiStepScheduler,
    'linear': LinearStepScheduler,
    'cos': CosStepScheduler}


def _build_lr_scheduler(optimizer, lr_policy, start_lr, end_lr, epochs=50, last_epoch=-1):
    return LRs[lr_policy](optimizer, last_epoch=last_epoch,
                          start_lr=start_lr, end_lr=end_lr, epochs=epochs)

def _build_warm_up_scheduler(optimizer, cfg, epochs=50, last_epoch=-1):
    lr_policy = cfg["LEARNING_RATE"]["LR_POLICY"]
    start_lr = cfg["LEARNING_RATE"]["START_LR"]
    end_lr = cfg["LEARNING_RATE"]["END_LR"]
    warmup_lr_policy = cfg["LEARNING_RATE"]["WARMUP"]["LR_POLICY"]
    warmup_epoch = cfg["LEARNING_RATE"]["WARMUP"]["EPOCH"]
    warmup_start_lr = cfg["LEARNING_RATE"]["WARMUP"]["START_LR"]
    warmup_end_lr = cfg["LEARNING_RATE"]["WARMUP"]["END_LR"]
    
    sc1 = _build_lr_scheduler(optimizer, lr_policy=warmup_lr_policy, start_lr=warmup_start_lr,
             end_lr=warmup_end_lr, epochs=warmup_epoch, last_epoch=last_epoch)
    sc2 = _build_lr_scheduler(optimizer, lr_policy=lr_policy, start_lr=start_lr,
             end_lr=end_lr, epochs=(epochs-warmup_epoch), last_epoch=last_epoch)
    return WarmUPScheduler(optimizer, sc1, sc2, last_epoch)

def build_lr_scheduler(optimizer, cfg, last_epoch=-1):
    epochs = cfg["END_EPOCH"]
    if cfg["LEARNING_RATE"]["WARMUP"]["ENABLE"]:
        return _build_warm_up_scheduler(optimizer, cfg, epochs, last_epoch)
    else:
        return _build_lr_scheduler(optimizer, cfg, epochs, last_epoch)