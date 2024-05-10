import torch
import torchvision
from model_save import Nn

# 模型读取

# 方式1
model = torch.load("../data/vgg16.pth")

# 方式2 加载字典方式
vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16.load_state_dict(torch.load("../data/vgg16_state_dict.pth"))

# 陷阱
# AttributeError: Can't get attribute 'Nn' on <module '__main__' from 'F:\\xiangmu\\Python\\study_demo\\nn\\model_load.py'>
# 需要把模型的定义带过来 NN的class   from model_save import Nn
model = torch.load("../data/nn.pth")
print(model)
