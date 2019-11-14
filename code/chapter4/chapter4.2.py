# 4.2 模型参数的访问、初始化和共享
import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3,1)) #  Pytorch已进行默认初始化

print(net)
X = torch.rand(2, 4)
Y = net(X).mean()
print(X)
print(Y)

# 访问模型参数
print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param.size())

for name, param in net[0].named_parameters():
    print(name, param.size(), type(param))

class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20)) # 如果是Parameter类型的tensor回自动添加到模型的参数列表中#
        self.weight2 = torch.rand(20, 20)
    def forward(self, x):
        pass

n = MyModel()
for name, param in n.named_parameters():
    print(name)  #只有weight1

# 因为Parameter是Tensor，即Tensor拥有的属性它都有，比如可以根据data来访问参数数值，用grad来访问参数梯度
weight_0 = list(net[0].parameters())[0]
print(weight_0.data)
print(weight_0.grad)
Y.backward()
print(weight_0.grad)


# 初始化模型参数
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)

# 使用常数来初始化权重参数

for name, param in net.named_parameters():
    if 'bias' in name:
        init.constant_(param, val=0)
        print(name, param.data)

# 自定义初始化方法 令权重有50%概率为0，另外50%初始化为[-10,-5]和[5,10]两个区间里的均匀分布的随机数
def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        #print("\n")
        #print(tensor)
        tensor *= (tensor.abs() >=5).float()
        #print(tensor)

for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)

# 通过改变这些参数的data来改写模型参数值同时不会影响梯度
for name, param in net.named_parameters():
    if 'bias' in name:
        param.data += 1
        print(name, param.data)

# 共享模型参数
linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)

print(id(net[0]) == id(net[1]))
print(id(net[0].weight) == id(net[1].weight))

x = torch.ones(1, 1)
print(x)
y = net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad)























