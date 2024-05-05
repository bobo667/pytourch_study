import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("../nnLogs")

dataset = torchvision.datasets.CIFAR10(root="../dataSet", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataload = DataLoader(dataset, batch_size=64)


class DemoNN(nn.Module):

    def __init__(self):
        super().__init__()
        # 彩色图像是三层所以 in_channels = 3
        self.conv1 = nn.Conv2d(3, 6, 3)

    def forward(self, x):
        x = self.conv1(x)
        return x


demoNN = DemoNN()
print(demoNN)

step = 0
for data in dataload:
    imgs, targets = data
    print(imgs.shape)
    output = demoNN(imgs)
    # torch.Size([64, 6, 30, 30])   64是代表着数量，因为定义的是batch_size = 64, 6 代表是输出的是6通道，30 ,30 图像的卷积之后宽高
    print(output.shape)
    writer.add_images("input", imgs, step)

    # shape 不知道填多少的时候可以填负一，他会根据后面的数进行运算
    output = torch.reshape(output, (-1, 3, 30, 30))

    writer.add_images("output", output, step)
    step = step + 1

writer.close()
