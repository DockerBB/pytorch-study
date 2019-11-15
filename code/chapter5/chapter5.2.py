# 5.2填充和步幅
import torch
from torch import nn

# 定义一个函数来计算卷积层。对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    # (1, 1)代表批量大小和通道数均为1
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])


# 两侧分别填充1行或列
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

X = torch.rand(8, 8)
print(comp_conv2d(conv2d, X).shape)
# 当卷积核的高和宽不同时，我们也可以通过设置高和宽上不同的填充数使得输出输入具有相同的高和宽

# 使用高为5， 宽为3的卷积核。在高和宽的两侧的填充数分别为2和1（卷积核长宽分别减一除二）
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2,1))
print(comp_conv2d(conv2d, X).shape)

# 步幅
# 一般高上步幅sh,宽上步幅sw输出形状为（nh - kh + ph + sh）/sh  *  (nw - kw + pw + sw)/sw
# 若设置ph = kh - 1 , pw = kw - 1 则输出为（nh + sh -1)/sh  *  (nw + sw -1)/sw
# 更进一步若输入的高和宽分别能被高和宽上的步幅整除，则输出为（nh/sh） *  (nw/sw)

conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1), stride=(2, 2))
print(comp_conv2d(conv2d, X).shape)
