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
from torchsummary import summary
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


# ResNet 模型
# ResNet的前两层跟之前介绍的GoogLeNet中的一样：在输出通道数为64、步幅为2的7×7卷积层
# 后接步幅为2的3×3的最大池化层。不同之处在于ResNet每个卷积层后增加的批量归一化层。

net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # (n+1)/2
)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert  in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block: # 第一个模块使用了步幅为二的最大池化层，所以不需要
                                       # 减小高和宽。所以第1个模块不会走这个分支，后面的
                                       # 模块第一层走这个分支，第二层走else分支
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
net.add_module("resnet_block2", resnet_block(64, 128, 2))
net.add_module("resnet_block3", resnet_block(128, 256, 2))
net.add_module("resnet_block4", resnet_block(256, 512, 2))

net.add_module("global_avg_pool", d2l.GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(512, 10)))


blk = Residual(3, 3, use_1x1conv=True, stride=2)
summary(blk, (3, 7, 7))

summary(net, (1, 224, 224))
X = torch.rand((1, 1, 224, 224))
for name, layer in net.named_children():
    X = layer(X)
    print(name, ' output shape:\t', X.shape)

batch_size = 256
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=48)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)





















































