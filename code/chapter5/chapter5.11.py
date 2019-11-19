# 5.11 残差网络（ResNet）
# 理论上，原模型解的空间只是新模型解的空间的子空间.也就是说，如果我们能将新添加的
# 层训练成恒等映射[Math Processing Error]f(x)=x，新模型和原模型将同样有效。由于新
# 模型可能得出更优的解来拟合训练数据集， 因此添加层似乎更容易降低训练误差。然而在
# 实践中，添加过多的层后训练误差往往不降反升.即使利用批量归一化带来的数值稳定性使
# 训练深层模型更加容易,该问题仍然存在,针对这一问题,何恺明等人提出了残差网络(ResNet)。
# 它在2015年的ImageNet图像识别挑战赛夺魁，并深刻影响了后来的深度神经网络的设计。
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import os
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
#多块使用逗号隔开
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else :
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)