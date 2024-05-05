import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("../nnLogs")

dataset = torchvision.datasets.CIFAR10(root="../dataSet", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataload = DataLoader(dataset, batch_size=64)


#
# input = torch.tensor([
#     [1, 2, 0, 3, 1],
#     [0, 1, 2, 3, 1],
#     [1, 2, 1, 0, 0],
#     [5, 2, 3, 1, 1],
#     [2, 1, 0, 1, 1]], dtype=torch.float32)
#
# input = torch.reshape(input, (-1, 1, 5, 5))


class DemoNn(torch.nn.Module):

    def __init__(self):
        super(DemoNn, self).__init__()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        output = self.maxpool1(x)
        return output


demo = DemoNn()
# print(demo(input))
step = 0
for data in dataload:
    imgs, target = data
    writer.add_images("old_input", imgs, step)
    writer.add_images("new_out", demo(imgs), step)
    step = step + 1

writer.close()
