# 5.10 批量归一化
# 批量归一化的提出正是为了应对深度模型训练的挑战。在模型训练时，
# 批量归一化利用小批量上的均值和标准差，不断调整神经网络中间输出
# ，从而使整个神经网络在各层的中间输出的数值更稳定。批量归一化的
# 提出正是为了应对深度模型训练的挑战。在模型训练时，批量归一化利
# 用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个
# 神经网络在各层的中间输出的数值更稳定。

# 对全连接层和卷积层做批量归一化的方法稍有不同。
# 详情见https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter05_CNN/5.10_batch-norm

import time
import torch
from torch import nn, optim
import torch.nn.functional as F

import  sys
sys.path.append("..")
import d2lzh_pytorch as d2l
import os
#多块使用逗号隔开
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 判断当前模式是训练模式还是预测模式
    if not is_training:
        # 如果是预测模式，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维度上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else :
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        # 训练模式下用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 拉伸和偏移
    return Y, moving_mean, moving_var

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else :
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化为0，1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = torch.zeros(shape)
        self.movvin_var = torch.zeros(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在的显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.movvin_var = self.movvin_var.to(X.device)

        # 保存更新过的moving_mean和moving_avg，Module实例的traning属性为true，
        # 调用.eval()后设为false
        Y, self.moving_mean, self.movvin_var = batch_norm(self.training,
                                                          X, self.gamma, self.beta, self.moving_mean,
                                                          self.movvin_var, eps=1e-5, momentum=0.9
                                                          )
        return Y

# 使用批量归一化层的LeNet
# net = nn.Sequential(
#             nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
#             BatchNorm(6, num_dims=4),
#             nn.Sigmoid(),
#             nn.MaxPool2d(2, 2), # kernel_size, stride
#             nn.Conv2d(6, 16, 5),
#             BatchNorm(16, num_dims=4),
#             nn.Sigmoid(),
#             nn.MaxPool2d(2, 2),
#             d2l.FlattenLayer(),
#             nn.Linear(16*4*4, 120),
#             BatchNorm(120, num_dims=2),
#             nn.Sigmoid(),
#             nn.Linear(120, 84),
#             BatchNorm(84, num_dims=2),
#             nn.Sigmoid(),
#             nn.Linear(84, 10)
#         )


# batch_size = 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
#
# lr, num_epochs = 0.001, 5
# optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)



# 简洁实现
# 与我们刚刚自己定义的BatchNorm类相比，Pytorch中nn模块定义的BatchNorm1d和BatchNorm2d
# 类使用起来更加简单，二者分别用于全连接层和卷积层，都需要指定输入的num_features参数
# 值。下面我们用PyTorch实现使用批量归一化的LeNet。

net = nn.Sequential(
    nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
    nn.BatchNorm2d(6),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(6, 16, 5),
    nn.BatchNorm2d(16),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),
    d2l.FlattenLayer(),
    nn.Linear(16*4*4, 120),
    nn.BatchNorm1d(120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.BatchNorm1d(84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)































































