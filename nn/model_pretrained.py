import torch
import torchvision
from torch import nn

# train_data = torchvision.datasets.ImageNet("../dataSet/ImageNet", split="train", download=True,
#                                            transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

# 在原有的网络中 加一层
vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))
# 修改原有的网络
vgg16_false.classifier[6] = nn.Linear(4096, 10)
