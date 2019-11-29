# 动量法
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
import torch
import matplotlib.pyplot

eta = 0.4 # 学习率
def f_2d(x1, x2):
    return  0.1 * x1 ** 2 + 2 * x2 ** 2

def gd_2d(x1, x2, s1, s2):
    return (x1 - eta*0.2*x1, x2 - eta*4*x2, 0, 0)


# 动量法
def momentum_2d(x1, x2, v1, v2):
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2

if __name__ == '__main__':
    print(torch.__version__)
    # 梯度下降的问题
    # 可以看到，同一位置上，目标函数在竖直方向（x2轴方向）比在水平方向（x1轴方向）的斜率的绝对值
    # 更大。因此，给定学习率，梯度下降迭代自变量时会使自变量在竖直方向比在水平方向移动幅度更大。
    # 那么，我们需要一个较小的学习率从而避免自变量在竖直方向上越过目标函数最优解。然而，这会造成
    # 自变量在水平方向上朝最优解移动变慢。
    d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
    # 下面我们试着将学习率调得稍大一点，此时自变量在竖直方向不断越过最优解并逐渐发散。
    eta = 0.6
    d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))

    # 动量法的提出是为了解决梯度下降的上述问题。由于小批量随机梯度下降比梯度下降更为广义，本章后续
    # 讨论将沿用7.3节（小批量随机梯度下降）中时间步t的小批量随机梯度gt的定义。设时间步t的自变量为xt
    #  ，学习率为ηt。 在时间步0，动量法创建速度变量v0，并将其元素初始化成0。在时间步t>0，动量法对
    #  每次迭代的步骤做如下修改：
    # v(t) <- γv(t-1) + η(t)g(t)
    # x(t) <- x(t-1) - v(t)
    # 超参数γ满足0<=γ<1。当γ=0时，动量法等价于小批量随机梯度下降。


eta, gamma = 0.4, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
# 可以看到使用较小的学习率η=0.4和动量超参数γ=0.5时，动量法在竖直方向上的移动更加平滑
# ，且在水平方向上更快逼近最优解。下面使用较大的学习率η=0.6，此时自变量也不再发散。
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
# 指数加权移动平均
























