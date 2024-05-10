import torch
import torch.nn as nn
import torch.nn.functional as F


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


input = torch.ones((64, 3, 32, 32))
output = Nn()(input)
print(output.shape)
