import functools
import logging
import math
import torch
import torch.nn as nn
from torch.nn import init
import pdb

from utils import *

def define_ESRCNN_S2self(opt):
    gpu_ids = opt['gpu_ids']
    net = ESRCNN_S2self()
    if opt['is_train']:
        net.weight_init(mean=0,std=0.01)
    if gpu_ids:
        assert torch.cuda.is_available()
        net = nn.DataParallel(net)
    return net

class ESRCNN_S2self(nn.Module):
    def __init__(self):
        super(ESRCNN_S2self, self).__init__()
        self.ESRCNNModule = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=9, stride=1, padding=4, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=5, stride=1, padding=2, bias=True)
        )

    def forward(self, inputs):
        outputs = self.ESRCNNModule(inputs)
        return outputs

    def weight_init(self,mean,std):
        for m in self._modules:
            normal_init(self._modules[m],mean,std)

def define_ESRCNN_S2L8_1(opt):
    gpu_ids = opt['gpu_ids']
    net = ESRCNN_S2L8_1()
    if opt['is_train']:
        net.weight_init(mean=0,std=0.01)
    if gpu_ids:
        assert torch.cuda.is_available()
        net = nn.DataParallel(net)
    return net

class ESRCNN_S2L8_1(nn.Module):
    def __init__(self):
        super(ESRCNN_S2L8_1, self).__init__()

        self.ESRCNNModule = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=9, stride=1, padding=4, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 7, kernel_size=5, stride=1, padding=2, bias=True)
        )

    def forward(self, inputs):
        outputs = self.ESRCNNModule(inputs)
        return outputs

    def weight_init(self,mean,std):
        for m in self._modules:
            normal_init(self._modules[m],mean,std)

def define_ESRCNN_S2L8_2(opt):
    gpu_ids = opt['gpu_ids']
    net = ESRCNN_S2L8_2()
    if opt['is_train']:
        net.weight_init(mean=0,std=0.01)
    if gpu_ids:
        assert torch.cuda.is_available()
        net = nn.DataParallel(net)
    return net

class ESRCNN_S2L8_2(nn.Module):
    def __init__(self):
        super(ESRCNN_S2L8_2, self).__init__()

        self.ESRCNNModule = nn.Sequential(
            nn.Conv2d(20, 64, kernel_size=9, stride=1, padding=4, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 7, kernel_size=5, stride=1, padding=2, bias=True)
        )

    def forward(self, inputs):
        outputs = self.ESRCNNModule(inputs)
        return outputs

    def weight_init(self,mean,std):
        for m in self._modules:
            normal_init(self._modules[m],mean,std)

def define_ESRCNN_S2L8_3(opt):
    gpu_ids = opt['gpu_ids']
    net = ESRCNN_S2L8_3()
    if opt['is_train']:
        net.weight_init(mean=0,std=0.01)
    if gpu_ids:
        assert torch.cuda.is_available()
        net = nn.DataParallel(net)
    return net

class ESRCNN_S2L8_3(nn.Module):
    def __init__(self):
        super(ESRCNN_S2L8_3, self).__init__()

        self.ESRCNNModule = nn.Sequential(
            nn.Conv2d(26, 64, kernel_size=9, stride=1, padding=4, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 7, kernel_size=5, stride=1, padding=2, bias=True)
        )

    def forward(self, inputs):
        outputs = self.ESRCNNModule(inputs)
        return outputs

    def weight_init(self,mean,std):
        for m in self._modules:
            normal_init(self._modules[m],mean,std)

def normal_init(m,mean,std):
    if isinstance(m,nn.ConvTranspose2d) or isinstance(m,nn.Conv2d):
        m.weight.data.normal_(mean,std)
        m.bias.data.zero_()
