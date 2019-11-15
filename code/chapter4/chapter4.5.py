#读取和存储
import torch
from torch import nn

print(torch.__version__)
x = torch.ones(3)
torch.save(x, 'x.pt')


x2 = torch.load('x.pt')
print(x2)

y = torch.zeros(4)
torch.save([x, y],'xy.pt')
xy_list = torch.load('xy.pt')
print(xy_list)

torch.save({'x': x, 'y': y}, 'xy_dict.pt')
xy = torch.load('xy_dict.pt')
print(xy)

# 读写模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

net = MLP()
print(net.state_dict())
# 注意，只有具有可学习参数的层(卷积层、线性层等)才有state_dict中的条目。优化器(optim)也有一个state_dict，其中包含关于优化器状态以及所使用的超参数的信息。
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(optimizer.state_dict())

# 保存和加载模型
# 1 仅保存和加载模型参数state_dict
# 2 保存和加载整个模型

# 保存
#torch.save(model.state_dict(), PATH)
# 加载
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))


#保存和加载整个模型
# 保存
# torch.save(model,PATH)
# 加载
# model = torch.load(PATH)

X = torch.randn(2, 3)
Y = net(X)
PATH = "./net.pt"
torch.save(net.state_dict(), PATH)

net2 = MLP()
net2.load_state_dict(torch.load(PATH))
Y2 = net2(X)
print(Y == Y2)