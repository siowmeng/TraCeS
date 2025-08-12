import pickle
from typing import Any, Dict

import numpy as np
import torch
from torch import nn

epsilon = 1e-8
MAX_LAMBDA = 1e3

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device: str


def set_device(device_no=None):
    global device
    if device_no is None:
        device_no = '0'
    if device_no == 'cpu':
        device = 'cpu'
    else:
        device = 'cuda:' + str(device_no) if torch.cuda.is_available() else 'cpu'


def set_device_omnisafe(omnisafe_device):
    global device
    device = omnisafe_device


class NegationLayer(nn.Module):
    def forward(self, x):
        return -x
