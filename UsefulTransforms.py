from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

pic_path = "F:\\xiangmu\\tran_data\\hymenoptera_data\\hymenoptera_data\\train\\ants\\0013035.jpg"
img_info = Image.open(pic_path)

# 转换图片类型 ToTensor的使用
tensor = transforms.ToTensor()
img_tensor_info = tensor(img_info)

# Normalize 归一化
"""
在数据处理和统计学中，归一化（Normalization）是一种常用的数据预处理技术，旨在将数据转换为标准形式，以便更有效地进行比较或分析。归一化的目的是消除不同特征之间的量纲差异，使得数据处于相似的数值范围内，有利于模型的收敛和提高算法的性能。

归一化通常应用于以下情况：

特征缩放：在机器学习中，特征的取值范围可能会相差较大，如果不进行归一化处理，可能导致某些特征对模型的影响过大，而其他特征影响较小。归一化可以使得所有特征都在同一数量级上，避免这种问题。
优化算法：在优化算法中，例如梯度下降法，归一化可以帮助算法更快地收敛，因为不同特征的梯度变化范围相近。
距离计算：在聚类算法和距离度量中，如K均值聚类、K近邻算法等，如果特征的数值范围不同，可能导致距离计算不准确，影响模型的性能。
"""
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_nor_tensor = normalize(img_tensor_info)

# Resize 等比例缩放或者扩大
print(img_info.size)
trans_resize = transforms.Resize((512, 512))
# 返回的输出是一个PIL的图片格式
img_resize = trans_resize(img_info)
print(img_resize)
img_resize = tensor(img_resize)
writer.add_image("img_resize", img_resize, 0)

# Compose 系列用法,他相当于一个调用链，这里的用法 其实等于 transforms.Resize((512, 512)) 然后再执行 tensor(img_resize),转换成tensor
# 这里后面的输入就需要是前面方法的输出
compose = transforms.Compose([
    transforms.Resize([512, 512]),
    tensor
])

compose = compose(img_info)
writer.add_image("compose_resize", compose, 0)

# RandomCrop 随机裁剪
random_crop = transforms.RandomCrop((50, 100))
compose2 = transforms.Compose([
    random_crop,
    tensor
])

for i in range(10):
    compose2_img = compose2(img_info)
    writer.add_image("RandomCrop", compose2_img, i)
