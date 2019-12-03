# 9.5多尺度目标检测
# 在9.4节（锚框）中，我们在实验中以输入图像的每个像素为中心生成多个锚框。这些锚框是对输入图像不同区
# 域的采样。然而，如果以图像每个像素为中心都生成锚框，很容易生成过多锚框而造成计算量过大。举个例子,
# 假设输入图像的高和宽分别为561像素和728像素，如果以每个像素为中心生成5个不同形状的锚框，那么一张图
# 像上则需要标注并预测200多万个锚框（561×728×5）。减少锚框个数并不难。一种简单的方法是在输入图像中
# 均匀采样一小部分像素，并以采样的像素为中心生成锚框。此外，在不同尺度下，我们可以生成不同数量和不同
# 大小的锚框。值得注意的是，较小目标比较大目标在图像上出现位置的可能性更多。举个简单的例子：形状为
# 1×1、1×2和2×2的目标在形状为2×2的图像上可能出现的位置分别有4、2和1种。因此，当使用较小锚框来检
# 测较小目标时，我们可以采样较多的区域；而当使用较大锚框来检测较大目标时，我们可以采样较少的区域。为
# 了演示如何多尺度生成锚框，我们先读取一张图像。它的高和宽分别为561像素和728像素。
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import torch

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

img = Image.open('img/catdog.jpg')
w, h = img.size # (728, 561)
# 我们在5.1节（二维卷积层）中将卷积神经网络的二维数组输出称为特征图。 我们可以通过定义特征图的形状
# 来确定任一图像上均匀采样的锚框中心。下面定义display_anchors函数。我们在特征图fmap上以每个单元（
# 像素）为中心生成锚框anchors。由于锚框anchors中x和y轴的坐标值分别已除以特征图fmap的宽和高，这些
# 值域在0和1之间的值表达了锚框在特征图中的相对位置。由于锚框anchors的中心遍布特征图fmap上的所有单
# 元，anchors的中心在任一图像的空间相对位置一定是均匀分布的。具体来说，当特征图的宽和高分别设为fmap_w
# 和fmap_h时，该函数将在任一图像上均匀采样fmap_h行fmap_w列个像素，并分别以它们为中心生成大小为s（假
# 设列表s长度为1）的不同宽高比（ratios）的锚框。
d2l.set_figsize()

def display_anchors(fmap_w, fmap_h, s):
    # 前两维的取值不影响输出结果(原书这里是(1, 10, fmap_w, fmap_h), 我认为错了)
    fmap = torch.zeros((1, 10, fmap_h, fmap_w), dtype=torch.float32)

    # 平移所有锚框使均匀分布在图片上
    offset_x, offset_y = 1.0/fmap_w, 1.0/fmap_h
    anchors = d2l.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5]) + \
        torch.tensor([offset_x/2, offset_y/2, offset_x/2, offset_y/2])

    bbox_scale = torch.tensor([[w, h, w, h]], dtype=torch.float32)
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)

display_anchors(fmap_w=3, fmap_h=2, s=[0.3])
plt.show()
display_anchors(fmap_w=2, fmap_h=1, s=[0.4])
plt.show()