import os
import torch
import numpy as np


def print_num_parameters(model):
    n_all_params = int(sum([np.prod(p.size()) for p in model.parameters()]))
    n_trainable_params = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))
    print(f"Number of parameters in {model.__class__.__name__}:  {n_trainable_params} / {n_all_params}")


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, specs):
        print(specs)
        self.initial = specs['initial']
        self.interval = specs['interval']
        self.factor = specs['factor']

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))


def adjust_learning_rate(lr_schedules, optimizer, epoch):
    if (type(lr_schedules)==list):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)
    else:
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules.get_learning_rate(epoch)


def get_learning_rates(optimizer):
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append(param_group['lr'])
    return lr_list


def print_learning_rates(optimizer):
    lr_txt = ""
    for param_group in optimizer.param_groups:
        lr_txt += " | " + str(param_group['lr'])
    print(lr_txt)

def weights_init(m):
    """
    initialize the weighs of the network for Convolutional layers and batchnorm layers
    """
    if isinstance(m, (torch.nn.modules.conv._ConvNd, torch.nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        torch.nn.init.constant_(m.bias, 0.0)
        torch.nn.init.constant_(m.weight, 1.0)

def clamp_gradient(model, clip):
    for p in model.parameters():
        torch.nn.utils.clip_grad_value_(p, clip)
