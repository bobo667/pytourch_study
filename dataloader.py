from PIL import Image
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("dataLoadLogs")

test_data = torchvision.datasets.CIFAR10(root="./dataSet", train=False, transform=torchvision.transforms.ToTensor())

# 使用test_data  每次取64条数据，每次顺序进行打乱，采用一个主线程，最后一批次不够整除的不删除
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

img, target = test_data[0]

print(img.shape)
print(target)

index = 0
for (data) in test_loader:
    imgs, targets = data
    writer.add_images("dataloader.py", imgs, index)
    index = index + 1

writer.close()
