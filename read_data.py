import os

from torch.utils.data import Dataset
from PIL import Image


# 这个类一般需要实现两个方法，一个是__getitem__ 获取每个项，和__len__  这批数据集的长度
class MyDataset(Dataset):

    def __init__(self, root_dir: str, label_dir: str):
        all_path = os.path.join(root_dir, label_dir)
        self.label_dir = label_dir
        self.all_path = all_path
        self.img_path_list = os.listdir(all_path)

    def __getitem__(self, idx):
        img_name = self.img_path_list[idx]
        img_path: str = os.path.join(self.all_path, img_name)
        # 获取图片信息
        img_info = Image.open(img_path)
        return img_info, self.label_dir

    def __len__(self):
        return len(self.img_path_list)


# Image.open()


root_dir = "F:\\xiangmu\\tran_data\\hymenoptera_data\\hymenoptera_data\\train"
ants_label_dir = "ants"
bees_label_dir = "ants"

ants_dataset = MyDataset(root_dir, ants_label_dir)
bess_dataset = MyDataset(root_dir, bees_label_dir)

img, label = ants_dataset[0]

# 可以把两个数据集相加 ，然后这两个数据集就会合并
all_dataset = ants_dataset + bess_dataset

print(img, label)
