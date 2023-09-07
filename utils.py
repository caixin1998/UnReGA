import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import copy
import yaml
import torch.nn.functional as F
import random
import collections
import models

def seed_everything(seed):
    """
    Function to set random seeds for reproducibility.

    Args:
        seed (int): Random seed value.

    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class AverageMeter(object):
    """
    Computes and stores the average and current value.

    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the meter's values.

        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the meter with a new value.

        Args:
            val (float): New value to update the meter.
            n (int): Number of elements represented by the value.

        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def update_ema_params(model, ema_model, alpha, global_step):
    """
    Update the Exponential Moving Average (EMA) of model parameters.

    Args:
        model (nn.Module): Model whose parameters are being updated.
        ema_model (nn.Module): EMA model that stores the averaged parameters.
        alpha (float): EMA decay parameter.
        global_step (int): Current global step of the training.

    """
    alpha = min(1 - 1 / (global_step + 1), alpha)
    
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # Update EMA parameters with a weighted sum of current and EMA parameters
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def mean_models_params(models):
    """
    Compute the mean of model parameters from a list of models.

    Args:
        models (list): List of models to average parameters from.

    Returns:
        OrderedDict: Mean state_dict of model parameters.

    """
    worker_state_dict = [x.state_dict() for x in models]
    weight_keys = list(worker_state_dict[0].keys())
    fed_state_dict = collections.OrderedDict()

    for key in weight_keys:
        key_sum = 0
        for i in range(len(models)):
            key_sum = key_sum + worker_state_dict[i][key]
        fed_state_dict[key] = key_sum / len(models)

    return fed_state_dict

def torch_angular_error(a, b, sum=False):
    """
    Calculate the angular error between two sets of pitch-yaw angles.

    Args:
        a (Tensor): Tensor of pitch-yaw angles.
        b (Tensor): Tensor of pitch-yaw angles to compare against.
        sum (bool, optional): Whether to return the sum or mean of angular errors.

    Returns:
        float: Angular error or sum of angular errors.

    """
    def pitchyaw_to_vector(pitchyaws):
        sin = torch.sin(pitchyaws)
        cos = torch.cos(pitchyaws)
        return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], 1)

    def nn_angular_distance(a, b):
        sim = F.cosine_similarity(a, b, eps=1e-6)
        sim = F.hardtanh(sim, -1.0 + 1e-6, 1.0 - 1e-6)
        return torch.acos(sim) * 180.0 / np.pi

    y = pitchyaw_to_vector(a)
    y_hat = b

    if y_hat.shape[1] == 2:
        y_hat = pitchyaw_to_vector(y_hat)
        if sum:
            return torch.sum(nn_angular_distance(y, y_hat))
        else:
            return torch.mean(nn_angular_distance(y, y_hat))

    # Default case: Return the mean of angular errors
    return nn_angular_distance(y, y_hat).mean()

def build_adaptation_loss(loss, lamda_pseudo = 0.01):
    if loss == "uncertainty":
        adaptation_loss = models.UncertaintyLoss().cuda()
    elif loss == "wpseudo":
        adaptation_loss = models.PseudoLabelLoss().cuda()
    elif loss == "pseudo":
        adaptation_loss = models.WeightedPseudoLabelLoss().cuda()
    elif loss == "uncertain_pseudo":
        adaptation_loss = models.UncertaintyPseudoLabelLoss(lamda_pseudo).cuda()
    elif loss == "uncertain_wpseudo":
        adaptation_loss = models.UncertaintyWPseudoLabelLoss(lamda_pseudo).cuda()
    return adaptation_loss