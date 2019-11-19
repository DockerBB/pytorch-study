# 5.9 含并行连结的网络（GoogLeNet）

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

class Inception(nn.Module):
    # c1 - c4为每条线路里的层的输出通道数
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 线路1，单1*1卷积层
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # 线路2， 1*1卷积层后接3*3卷积层
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3， 1*1卷积层后接5*5卷积层
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4， 3*3最大池化层后接1*1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1,padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1) # ★★★★在通道维度连接输出★★★★

# GoogLeNet模型
# GoogLeNet跟VGG一样，在主体卷积部分中使用5个模块（block），每个模块之间
# 使用步幅为2的3×33×3最大池化层来减小输出高宽。第一模块使用一个64通道的
# 7×77×7卷积层。
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# 第二模块使用2个卷积层：首先是64通道的1×11×1卷积层，然后是将通道增大
# 3倍的3×33×3卷积层。它对应Inception块中的第二条线路。
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# 第三模块串联2个完整的Inception块。第一个Inception块的输出通道数为64+128+32+32=256,
# 其中4条线路的输出通道数比例为64:128:32:32=2:4:1:1。其中第二、第三条线路先分别将输
# 入通道数减小至96/192=1/2和16/192=1/12后，再接上第二层卷积层。第二个Inception块输出
# 通道数增至128+192+96+64=480，每条线路的输出通道数之比为128:192:96:64=4:6:3:2。其中
# 第二、第三条线路先分别将输入通道数减小至128/256=1/2和32/256=1/8。

b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 第四模块更加复杂。它串联了5个Inception块，其输出通道数分别是192+208+48+64=512、
# 160+224+64+64=512、128+256+64+64=512、112+288+64+64=528和256+320+128+128=832。
# 这些线路的通道数分配和第三模块中的类似，首先含3×33×3卷积层的第二条线路输出最
# 多通道，其次是仅含1×11×1卷积层的第一条线路，之后是含5×5卷积层的第三条线路和
# 含3×3最大池化层的第四条线路。其中第二、第三条线路都会先按比例减小通道数。这些
# 比例在各个Inception块中都略有不同。

b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 第五模块有输出通道数为256+320+128+128=832和384+384+128+128=1024的两个Inception块。
# 其中每条线路的通道数的分配思路和第三、第四模块中的一致，只是在具体数值上有所不同。
# 需要注意的是，第五模块的后面紧跟输出层，该模块同NiN一样使用全局平均池化层来将每个
# 通道的高和宽变成1。最后我们将输出变成二维数组后接上一个输出个数为标签类别数的全连
# 接层。

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   d2l.GlobalAvgPool2d())

net = nn.Sequential(b1, b2, b3, b4, b5,
                    d2l.FlattenLayer(), nn.Linear(1024, 10))

net = nn.Sequential(b1, b2, b3, b4, b5, d2l.FlattenLayer(), nn.Linear(1024, 10))
X = torch.rand(1, 1, 224, 224)
for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape: ', X.shape)

batch_size = 128
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)





















































































