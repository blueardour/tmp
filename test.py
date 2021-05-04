
import torch
import torch.nn.functional as F

# not in place
x = torch.tensor([-1.0]).requires_grad_().clone()
print(torch.autograd.grad(F.relu(x), (x, ))[0])
# tensor([0.])

# inplace
x = torch.tensor([-1.0]).requires_grad_().clone()
print(torch.autograd.grad(F.relu_(x), (x, ))[0])


import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(inplace = True)

    def forward(self, x):
        return self.dropout(self.relu(self.conv(x))).sum()

model = Net()
model.cuda()
model.train()

model(torch.autograd.Variable(torch.rand(1, 3, 16, 16).cuda().uniform_())).backward()



