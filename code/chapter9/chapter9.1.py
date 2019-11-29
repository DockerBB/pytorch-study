# 9.1图像增广
from matplotlib import pyplot as plt
import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import os
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)
    plt.show()

def load_cifar10(is_train, augs, batch_size, root="~/Datasets/CIFAR"):
    dataset = torchvision.datasets.CIFAR10(root=root, train=is_train, transform=augs, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)

def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

def train_with_data_aug(train_augs, test_augs, lr=0.001):
    batch_size, net = 256, d2l.resnet18(10)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    train(train_iter, test_iter, net, loss, optimizer, device, num_epochs=10)


d2l.set_figsize()
img = Image.open('img/cat1.jpg')
d2l.plt.imshow(img)
plt.show()
# 翻转和裁剪
apply(img, torchvision.transforms.RandomHorizontalFlip())
apply(img, torchvision.transforms.RandomVerticalFlip())
# 在下面的代码里，我们每次随机裁剪出一块面积为原面积10 %∼100 %  的区域，且该区域的宽
# 和高之比随机取自0.5∼2，然后再将该区域的宽和高分别缩放到200像素。若无特殊说明，本节
# 中a和b之间的随机数指的是从区间[a, b]中随机均匀采样所得到的连续值。
shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
# 变化颜色
# 另一类增广方法是变化颜色。我们可以从4个方面改变图像的颜色：亮度（brightness）、对比
# 度（contrast）、饱和度（saturation）和色调（hue）。在下面的例子里，我们将图像的亮度
# 随机变化为原图亮度的50%（1−0.5）∼150%（1+0.5）。
apply(img, torchvision.transforms.ColorJitter(brightness=0.5))
# 我们也可以随机变化图像的色调
apply(img, torchvision.transforms.ColorJitter(hue=0.5))
#类似地，我们也可以随机变化图像的对比度。
apply(img, torchvision.transforms.ColorJitter(contrast=0.5))
# 我们也可以同时设置如何随机变化图像的亮度（brightness）、对比度（contrast）、饱和度
# （saturation）和色调（hue）。
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
# 叠加多个图像增广方法
# 实际应用中我们会将多个图像增广方法叠加使用。我们可以通过Compose实例将上面定义的多个图
# 像增广方法叠加起来，再应用到每张图像之上。
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
# 使用图像增广训练模型
all_imges = torchvision.datasets.CIFAR10(train=True, root="~/Datasets/CIFAR", download=True)
# all_imges的每一个元素都是(image, label)
show_images([all_imges[i][0] for i in range(32)], 4, 8, scale=0.8)
plt.show()
flip_aug = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()])

no_aug = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])

num_workers = 0 if sys.platform.startswith('win32') else 4
train_with_data_aug(flip_aug, no_aug)






















