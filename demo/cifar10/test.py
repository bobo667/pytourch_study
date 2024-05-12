import os

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from model import Nn
from torch.utils.tensorboard import SummaryWriter


model_dict = torch.load("../../customerData/nn.pth")

nn_model = Nn()
nn_model.load_state_dict(model_dict)

img = Image.open("../../img/1715501394508.jpg").convert("RGB")
img = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
])(img)

img = torch.reshape(img, (1, 3, 32, 32))

output = nn_model(img)
print(output.argmax(1))
