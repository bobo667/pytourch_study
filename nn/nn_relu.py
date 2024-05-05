import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("../nnLogs")

dataset = torchvision.datasets.CIFAR10(root="../dataSet", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataload = DataLoader(dataset, batch_size=64)

input = torch.tensor([
    [1, -0.5],
    [-1, 3]
])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)


class DemoNn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # output = self.relu(x)
        output = self.sigmoid(x)
        return output


demoNn = DemoNn()
# output = demoNn(input)
# print(output)
step = 0
for data in dataload:
    imgs, targets = data
    writer.add_images("input_relu", imgs, step)
    output = demoNn(imgs)
    writer.add_images("output_relu", output, step)
    step += 1

writer.close()
