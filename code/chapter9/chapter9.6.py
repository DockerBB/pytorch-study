# 目标检测数据集（皮卡丘）
# 在目标检测领域并没有类似MNIST或Fashion-MNIST那样的小数据集。为了快速测试模型，我们合成了一个小的
# 数据集。我们首先使用一个开源的皮卡丘3D模型生成了1000张不同角度和大小的皮卡丘图像。然后我们收集了
# 一系列背景图像，并在每张图的随机位置放置一张随机的皮卡丘图像。该数据集使用MXNet提供的im2rec工具将
# 图像转换成了二进制的RecordIO格式 [1]。该格式既可以降低数据集在磁盘上的存储开销，又能提高读取效率
# 。如果想了解更多的图像读取方法，可以查阅GluonCV工具包的文档 [2]。
from matplotlib import pyplot as plt
import os
import json
import numpy as np
import torch
import torchvision
from PIL import Image

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

data_dir = 'Dataset/pikachu'

assert os.path.exists(os.path.join(data_dir, "train"))
class PikachuDetDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, part, image_size=(256, 256)):
        assert part in ["train", "val"]
        self.image_size = image_size
        self.image_dir = os.path.join(data_dir, part, "images")

        with open(os.path.join(data_dir, part, "label.json")) as f:
            self.label = json.load(f)

        self.transform = torchvision.transforms.Compose([
            # 将 PIL 图片转换成位于[0.0, 1.0]的floatTensor, shape (C x H x W)
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        image_path = str(index + 1) + ".png"

        cls = self.label[image_path]["class"]
        label = np.array([cls] + self.label[image_path]["loc"],
                         dtype="float32")[None, :]

        PIL_img = Image.open(os.path.join(self.image_dir, image_path)
                             ).convert("RGB").resize(self.image_size)
        img = self.transform(PIL_img)

        sample = {
            "label": label, # shape: (1, 5) [class, xmin, ymin, xmax, ymax]
            "image": img    # shape: (3, *image_size)
        }

        return sample

def load_data_pikachu(batch_size, edge_size=256, data_dir = data_dir):
    image_size = (edge_size, edge_size)
    train_dataset = PikachuDetDataset(data_dir, 'train', image_size)
    val_dataset = PikachuDetDataset(data_dir, 'val', image_size)

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=4)
    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                           shuffle=False, num_workers=4)
    return train_iter, val_iter

# 下面我们读取一个小批量并打印图像和标签的形状。图像的形状和之前实验中的一样，依然是(批量大小,
# 通道数, 高, 宽)。而标签的形状则是(批量大小, m, 5)，其中m等于数据集中单个图像最多含有的边界
# 框个数。小批量计算虽然高效，但它要求每张图像含有相同数量的边界框，以便放在同一个批量中。由于
# 每张图像含有的边界框个数可能不同，我们为边界框个数小于m的图像填充非法边界框，直到每张图像均
# 含有m个边界框。这样，我们就可以每次读取小批量的图像了。图像中每个边界框的标签由长度为5的数组
# 表示。数组中第一个元素是边界框所含目标的类别。当值为-1时，该边界框为填充用的非法边界框。数组
# 的剩余4个元素分别表示边界框左上角的x和y轴坐标以及右下角的x和y轴坐标（值域在0到1之间）。这里的
# 皮卡丘数据集中每个图像只有一个边界框，因此m=1。

if __name__ ==  '__main__':
    batch_size, edge_size = 32, 256
    train_iter, _ = load_data_pikachu(batch_size, edge_size, data_dir)
    batch = iter(train_iter).next()
    print(batch["image"].shape, batch["label"].shape)

    imgs = batch["image"][0:10].permute(0,2,3,1) # 交换维度
    bboxes = batch["label"][0:10, 0, 1:]

    axes = d2l.show_images(imgs, 2, 5).flatten()
    for ax, bb in zip(axes, bboxes):
        d2l.show_bboxes(ax, [bb * edge_size], colors=['w'])
    plt.show()





























































