import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
from matplotlib import pyplot
from mpl_toolkits import mplot3d # 三维画图
import numpy as np

# 给定函数f(x) = x*cos(pi*x) -1.0<=x<=2.0
# 我们可以大致找出该函数的局部最小值和全局最小值的位置。需要注意的是，图中箭头所指示的只是大致位置。
def f(x):
    return x * np.cos(np.pi * x)

if __name__ == '__main__':
    d2l.set_figsize((4.5, 2.5))
    x = np.arange(-1.0, 2.0, 0.01)
    fig,  = d2l.plt.plot(x, f(x))
    fig.axes.annotate('local minimum', xy=(-0.3, -0.25), xytext=(-0.77, -1.0),
                      arrowprops=dict(arrowstyle='->'))
    fig.axes.annotate('global minimum', xy=(1.1, -0.95), xytext=(0.6, 0.8),
                      arrowprops=dict(arrowstyle='->'))
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('f(x)')
    pyplot.show()

    # 鞍点
    # 刚刚我们提到，梯度接近或变成零可能是由于当前解在局部最优解附近造成的。事实上，另一种可能性是当
    # 前解在鞍点（saddle point）附近。
    # 举个例子，给定函数f(x) = x^3 我们可以找出该函数的鞍点位置。
    x = np.arange(-2.0, 2.0, 0.1)
    fig, = d2l.plt.plot(x, x**3)
    fig.axes.annotate('saddle point', xy=(0, -0.2), xytext=(-0.52, -5.0),
                      arrowprops=dict(arrowstyle='->'))
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('f(x)')
    pyplot.show()

    # 再举个定义在二维空间的函数的例子，例如：f(x,y) = x^2 - y^2
    # 我们可以找出该函数的鞍点位置。也许你已经发现了，该函数看起来像一个马鞍，而鞍点恰好是马鞍上可坐
    # 区域的中心。
    x, y = np.mgrid[-1: 1: 31j, -1: 1: 31j]
    z = x ** 2 - y ** 2

    ax = d2l.plt.figure().add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, **{'rstride': 2, 'cstride': 2})
    ax.plot([0], [0], [0], 'rx')
    ticks = [-1, 0, 1]
    d2l.plt.xticks(ticks)
    d2l.plt.yticks(ticks)
    ax.set_zticks(ticks)
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('y')
    pyplot.show()