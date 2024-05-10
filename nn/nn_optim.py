import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("../nnLogs")

dataset = torchvision.datasets.CIFAR10(root="../dataSet", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataload = DataLoader(dataset, batch_size=1)


class Nn(nn.Module):

    def __init__(self):
        super(Nn, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.max_pool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 10)

        self.run = nn.Sequential(
            self.conv1,
            self.max_pool1,
            self.conv2,
            self.max_pool2,
            self.conv3,
            self.max_pool3,
            self.flatten,
            self.fc1,
            self.fc2,
        )

    def forward(self, x):
        return self.run(x)


loss = nn.CrossEntropyLoss()
output = Nn()
optim = torch.optim.SGD(output.parameters(), lr=0.01)
# 训练20轮
for epoch in range(20):
    running_loss = 0.0
    for data in dataload:
        img, lab = data
        output2 = output(img)
        result = loss(output2, lab)
        # 梯度清0
        optim.zero_grad()
        # 反向传播 是按照他原有的路线 求出每个节点的梯度
        result.backward()
        # 进行参数调优
        optim.step()

        running_loss = running_loss + result
        print(output2)
        print(lab)
        print(result)
    print(running_loss)
print(output.shape)
