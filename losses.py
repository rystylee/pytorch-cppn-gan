import torch
import torch.nn as nn


class GANLoss(object):
    def __init__(self):
        self.loss_fn = nn.BCELoss()

    def __call__(self, logits, loss_type):
        assert loss_type in ['g', 'd_real', 'd_fake']
        batch_size = len(logits)
        device = logits.device
        if loss_type == 'g':
            label = torch.ones(batch_size, 1).to(device)
            return self.loss_fn(logits, label)
        elif loss_type == 'd_real':
            label = torch.ones(batch_size, 1).to(device)
            return self.loss_fn(logits, label)
        elif loss_type == 'd_fake':
            label = torch.zeros(batch_size, 1).to(device)
            return self.loss_fn(logits, label)
