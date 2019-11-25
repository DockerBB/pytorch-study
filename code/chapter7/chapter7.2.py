import numpy as np
import torch
import math
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
from matplotlib import pyplot

def gd(eta):
    x = 10
    results = [x]
    for i in range(10):
        x -= eta * 2 * x
        results.append(x)
    print("epoch 10, x:",x)
    return results


def show_trace(res):
    n = max(abs(min(res)), abs(max(res)), 10)
    f_line = np.arange(-n, n, 0.1)
    d2l.set_figsize()
    d2l.plt.plot(f_line, [x*x for x in f_line])
    d2l.plt.plot(res, [x*x for x in res], '-o')
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('f(x)')
    pyplot.show()

# learning rate
# 上述梯度下降算法中的正数η通常叫作学习率。这是一个超参数，需要人工设定。如果使用过小的学习
# 率，会导致x更新缓慢从而需要更多的迭代才能得到较好的解。


# 多维梯度下降
def train_2d(trainer):
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i+1, x1, x2))
    return results

def show_trace_2d(f, results):
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1,x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
    pyplot.show()

eta = 0.1

def f_2d(x1, x2):  # 目标函数
    return x1 ** 2 + 2 * x2 ** 2

def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 2 * x1, x2 - eta * 4 * x2, 0, 0)

# SGD随机梯度下降

def sgd_2d(x1, x2, s1, s2):
    return (x1 - eta * (2 * x1 + np.random.normal(0.1)),
            x2 - eta * (4 * x2 + np.random.normal(0.1)), 0, 0)


if __name__ == '__main__':
    res = gd(0.2)
    show_trace(res)
    # 下面展示使用学习率η=0.05时自变量x的迭代轨迹。可见，同样迭代10次后，当学习率过
    # 小时，最终x的值依然与最优解存在较大偏差。
    show_trace((gd(0.05)))

    # 如果使用过大的学习率，∣ηf′(x)∣可能会过大从而使前面提到的一阶泰勒展开公式不再成立：
    # 这时我们无法保证迭代x会降低f(x)的值。
    # 举个例子，当设学习率η=1.1时，可以看到x不断越过（overshoot）最优解x=0并逐渐发散。
    show_trace(gd(1.1))

    show_trace_2d(f_2d, train_2d(gd_2d))

    show_trace_2d(f_2d, train_2d(sgd_2d))





















