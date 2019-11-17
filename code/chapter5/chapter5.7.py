# 5.7 使用重复元素的网络 （VGG）
# AlexNet在LeNet的基础上增加了3个卷积层。但AlexNet作者对它们的卷积窗口、输出
# 通道数和构造顺序均作了大量的调整。虽然AlexNet指明了深度卷积神经网络可以取得
# 出色的效果，但并没有提供简单的规则指导后来的研究者如何设计新的网络。

# VGG提出了可以通过重复使用简单的基础块来构建深度模型的思路

# 对于给定的感受野（与输出有关的输入图片的局部大小），采用堆积的小卷积核优于
# 采用大的卷积核，因为可以增加网络深度来保证学习更复杂的模式，而且代价还比较
# 小（参数更少）。例如，在VGG中，使用了3个3x3卷积核来代替7x7卷积核，使用了2
# 个3x3卷积核来代替5*5卷积核，这样做的主要目的是在保证具有相同感知野的条件下
# ，提升了网络的深度，在一定程度上提升了神经网络的效果。对于给定的感受野（与
# 输出有关的输入图片的局部大小），采用堆积的小卷积核优于采用大的卷积核，因为
# 可以增加网络深度来保证学习更复杂的模式，而且代价还比较小（参数更少）。例如
# ，在VGG中，使用了3个3x3卷积核来代替7x7卷积核，使用了2个3x3卷积核来代替5*5
# 卷积核，这样做的主要目的是在保证具有相同感知野的条件下，提升了网络的深度，
# 在一定程度上提升了神经网络的效果。

import time
import torch
from torch import nn, optim
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)

# VGG网络
conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
# 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
fc_features = 512 * 7 * 7 # c * w * h
fc_hidden_units = 4096 # 任意

# VGG-11实现
def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_conv, in_channels, out_channels) in enumerate(conv_arch):
        # 每经过一个vgg_block都会使宽高减半
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_conv, in_channels, out_channels))

    # 全连接层
    net.add_module("fc", nn.Sequential(d2l.FlattenLayer(),
                                       nn.Linear(fc_features, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, 10)
                                       ))
    return net

net = vgg(conv_arch, fc_features, fc_hidden_units)
X = torch.rand(1, 1, 224, 224)
summary(net, (1, 224, 224))
# named_children获取一级子模块及其名字(named_modules会返回所有子模块,包括子模块的子模块)
for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape: ', X.shape)

ratio = 8
small_conv_arch = [(1, 1, 64//ratio), (1, 64//ratio, 128//ratio), (2, 128//ratio, 256//ratio),
                   (2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio)]
net = vgg(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
print(net)


if __name__ == '__main__':
    batch_size = 64
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)














































