import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)

# 模型保存 模型结构和参数
torch.save(vgg16, "../data/vgg16.pth")

# 保存方式2
# 把网络模型的参数保存下来 不保存结构 （官方推荐）
torch.save(vgg16.state_dict(), "../data/vgg16_state_dict.pth")



# 陷阱
class Nn(torch.nn.Module):
    def __init__(self):
        super(Nn, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        x = self.conv1(x)
        return x

nn = Nn()
torch.save(nn,"../data/nn.pth")
