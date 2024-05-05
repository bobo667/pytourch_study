from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import cv2

"""
    通过 transforms.ToTensor  去解决两个问题
        
        2.Tensor 数据类型，相比于普通的数据类型有什么区别
            Tensor 类型有 
            反向传播属性 _backward_hooks 
            梯度       _grad
            梯度方法    _grad_fn
        其实他就是相比于普通的包装了神经网络的一些参数类型
"""

pic_path = "F:\\xiangmu\\tran_data\\hymenoptera_data\\hymenoptera_data\\train\\ants\\0013035.jpg"
img_info = Image.open(pic_path)

# 1.transforms 该如何使用
tensor = transforms.ToTensor()
img_tensor = tensor(img_info)
# cv2 使用方式
cv_img = cv2.imread(pic_path)

writer = SummaryWriter("logs")
writer.add_image("tensor_img2", img_tensor, 1)
writer.add_image("tensor_img2", cv_img, 2, dataformats="HWC")

writer.close()
