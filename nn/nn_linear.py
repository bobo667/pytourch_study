import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("../nnLogs")

dataset = torchvision.datasets.CIFAR10(root="../dataSet", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

data_load = DataLoader(dataset, batch_size=64,drop_last=True)


class DemoNn(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(196608, 10)

    def forward(self, x):
        return self.linear(x)


demoNn = DemoNn()

for data in data_load:
    imgs, targets = data
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    output = torch.flatten(imgs)
    print(output.shape)
    output = demoNn(output)
    print(output.shape)
