import torch

class LinearWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, base_lr, max_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.max_lr = max_lr
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr = self.base_lr + (self.max_lr - self.base_lr) * (self.last_epoch / self.warmup_steps)
        else:
            lr = self.max_lr
        return [lr for _ in self.optimizer.param_groups]