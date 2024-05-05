from torch import nn
import torch


class DemoNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output



demo_nn = DemoNN()
x = torch.tensor(1.0)
output = demo_nn(x)
print(output)
