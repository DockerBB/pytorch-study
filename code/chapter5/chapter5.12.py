# 5.12 稠密连接网络(DenseNet)
# 与ResNet的主要区别在于，DenseNet里模块[Math Processing Error]B的输出不是像ResNet
# 那样和模块A的输出相加,而是在通道维上连结。这样模块A的输出可以直接传入模块B后面的
# 层。在这个设计里，模块A直接跟模块B后面的所有层连接在了一起。这也是它被称为“稠密
# 连接”的原因。
#
# DenseNet的主要构建模块是稠密块（dense block）和过渡层（transition layer）。前者
# 定义了输入和输出是如何连结的，后者则用来控制通道数，使之不过大。

# DenseNet使用了ResNet改良版的"批量归一化、激活和卷积"结构，我们首先在conv_block
# 函数里实现这个结构

import time
import torch
from torch import nn, optim
from torchsummary import summary
import torch.nn.functional as F
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels),
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    return blk

# 稠密块由多个conv_block组成，每块使用相同的输出通道数。但在前向计算时，我们将每块
# 的输入和输出在通道维上连结。

class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))

        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1) # 在通道维上将输入和输出连结
        return X

# 过渡层

# 由于每个稠密块都会带来通道数的增加，使用过多则会带来过于复杂的模型。过渡层用来控制
# 模型复杂度。它通过1×1卷积层来减小通道数，并使用步幅为2的平均池化层减半高和宽,从而
# 进一步降低模型复杂度。

def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2) # 将长宽减半
    )
    return blk


# DenseNet模型
# DenseNet首先使用同ResNet一样的单卷积层和最大池化层。
net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), # (n+1)/2
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # (n+1)/2
)

# 类似于ResNet接下来使用的4个残差块，DenseNet使用的是4个稠密块。同ResNet一样，我们可以
# 设置每个稠密块使用多少个卷积层。这里我们设成4，从而与上一节的ResNet-18保持一致。稠密
# 块里的卷积层通道数（即增长率）设为32，所以每个稠密块将增加128个通道。

# ResNet里通过步幅为2的残差块在每个模块之间减小高和宽。这里我们则使用过渡层来减半高和宽
# ，并减半通道数。

num_channels, growth_rate = 64, 32 # num_channels为当前的通道数
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    DB = DenseBlock(num_convs, num_channels, growth_rate)
    net.add_module("DenseBlock_%d" % i, DB)
    # 上一个稠密块的输出通道数
    num_channels = DB.out_channels
    # 在稠密块之间加入通道减半的过渡层
    if i != len(num_convs_in_dense_blocks) - 1:
        net.add_module("transition_block_%d" % i, transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

# 同ResNet一样，最后接上全局池化层和全连接层来输出。
net.add_module("BN", nn.BatchNorm2d(num_channels))
net.add_module("relu", nn.ReLU())
# GlobalAvgPool2d的输出: (Batch, num_channels, 1, 1)
net.add_module("global_avg_pool", d2l.GlobalAvgPool2d())
net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(num_channels, 10)))

# 我们尝试打印每个子模块的输出维度确保网络无误：
X = torch.rand((1, 1, 96, 96))
for name, layer in net.named_children():
    X = layer(X)
    print(name, ' output shape:\t', X.shape)


blk = DenseBlock(2, 3, 10)
X = torch.rand(4, 3, 8, 8)
Y = blk(X)
print(Y.shape)
# summary(blk, (3, 8, 8))

blk = transition_block(23,10)
print(blk(Y).shape)

batch_size = 256
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)










































