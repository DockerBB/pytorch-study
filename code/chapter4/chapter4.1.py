import torch
from torch import nn
from collections import OrderedDict
class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例是还可以指定其他函数
        # 参数， 如“模型参数的访问，初始化和共享”一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

X = torch.rand(2, 784)
net = MLP()
print(net)
print(net(X))

class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):  # 如果传入的是一个OrderedDict
            for key, module in args[0].items():
                self.add_module(key, module)  # add_module方法会将module添加进self._modules(一个OrderedDict)
        else:  # 传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成员
        for module in self._modules.values():
            input = module(input)
        return input

net = MySequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
print(net)
print(net(X))

# ModuleList 类 接收一个子模块的列表作为输入，然后也可以类似List那样进行append和extend操作

net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10)) # 类似List的append操作
print(net[-1])
print(net)

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x

# ModuleDict类
# ModuleDict接收一个子模块的字典作为输入，然后也可以类似字典那样进行添加访问操作
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU()
})
net['output'] = nn.Linear(256, 10)
print(net['linear'])
print(net.output)
print(net)

# 构造复杂的模型
# 虽然上面介绍的这些类可以使模型构造更加简单，且不需要定义forward函数，但是直接继承Module类可以极大的拓展模型构造的
# 灵活性。下面构造一个稍微复杂点的网络FancyMLP。在这个网络中，我们通过get_constang函数创建训练中不被迭代的参数，即
# 常数参数。在前向计算中，除了使用创建的常数参数外，我们还使用Tensor的函数和Python的控制流，并多次调用相同的层

class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)

        self.rand_weight = torch.rand((20, 20), requires_grad=False) # 不可训练参数
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        # 使用创建的常参数， 以及nn.functional中的relu函数和mm函数
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)

        # 复用全连接层。 等价于两个全连接层共享参数
        x = self.linear(x)
        # 控制流，这里我们需要调用item函数来返回标量进行比较
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()

X = torch.rand(2, 20)
net = FancyMLP()
print(net)
print(net(X))

# 因为FancyMLP和Sequential类都是Moduled类的子类，所以我们可以嵌套调用他们
class NestMlp(nn.Module):
    def __init__(self, **kwargs):
        super(NestMlp, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())

    def forward(self, x):
        return self.net(x)

net = nn.Sequential(NestMlp(), nn.Linear(30,20), FancyMLP())

X = torch.rand(2, 40)
print(net)
print(net(X))

# 小结
# 可以通过继承Module类来构造模型。
# Sequential、ModuleList、ModuleDict类都继承自Module类。
# 与Sequential不同，ModuleList和ModuleDict并没有定义一个完整的网络，它们只是将不同的模块存放在一起，需要自己定义forward函数。
# 虽然Sequential等类可以使模型构造更加简单，但直接继承Module类可以极大地拓展模型构造的灵活性。