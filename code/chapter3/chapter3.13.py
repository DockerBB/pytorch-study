from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <=1
    keep_prob = 1 - drop_prob
    #这种情况丢弃所有元素
    if keep_prob == 0:
        return torch.zeros_like(X)
    temp = torch.rand(X.shape)
    mask = (temp > keep_prob).float()
    return mask * X / keep_prob



