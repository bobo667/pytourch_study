import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)

summaryWriter = SummaryWriter("dataLoadLogs")

train_set = torchvision.datasets.CIFAR10(root="./dataSet", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataSet", train=False, transform=dataset_transform, download=True)
# img就是这个数据集的内容，target 就是这个数据集的标签分类
img, target = train_set[0]

#  train_set.classes 就是批数据集的分类
print(train_set.classes)
print(target)

for i in range(10):
    img, target = train_set[i]
    summaryWriter.add_image("torchvision_dataset", img, i)

summaryWriter.close()
