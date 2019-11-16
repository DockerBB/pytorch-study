# 5.3 多输入通道和多输出通道

# 当输入数据含多个通道时，我们需要构造一个输入通道数与输入数据的通道数相同
# 的卷积核，从而能够与含多通道的输入数据做互相关运算。假设输入数据的通道数
# 为ci,那么卷积核的输入通道数同样为ci。设卷积核窗口形状为kh * kw。当ci=1时，
# 我们知道卷积核只包含一个形状为kh * kw的二维数组。当ci>1时，我们将会为每
# 个输入通道各分配一个形状为kh * kw的核数组。把这ci个数组在输入通道维上连
# 结，即得到一个形状为ci * kh * kw的卷积核。由于输入和卷积核各有ci个通道，
# 我们可以在各个通道上对输入的二维数组和卷积核的二维核数组做互相关运算，再
# 将这ci个互相关运算的二维输出按通道相加，得到一个二维数组。这就是含多个通
# 道的输入数据与多输入通道的卷积核做二维互相关运算的输出。

import torch
from torch import nn
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

print(torch.__version__)

def corr2d_multi_in(X, K):
    # 沿着X和K的第0维（通道）分别计算求和
    res = d2l.corr2d(X[0,:,:], K[0,:,:])
    for i in range(1, X.shape[0]):
        res += d2l.corr2d(X[i,:,:],K[i,:,:])
    return res

X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

print(X.shape)
print(K.shape)
print(corr2d_multi_in(X, K))

# 多输出通道
def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历， 每次同输入X做互相关计算。所有结果使用stack函数合并在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K])

K = torch.stack([K, K+1, K+2])
print(K.shape)
print(corr2d_multi_in_out(X, K))


def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h * w)
    K = K.view(c_o, c_i)
    Y = torch.mm(K, X)  # 全连接层的矩阵乘法
    return Y.view(c_o, h, w)



X = torch.rand(3, 3, 3)
K = torch.rand(2, 3, 1, 1)

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

print((Y1 - Y2).norm().item() < 1e-6)





































