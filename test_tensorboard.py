from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter('logs')

"""
    tag,   标签 类似于title，他是需要唯一的
    scalar_value,    一个数值
    global_step=None, 这是第几步
    
    y轴 ：global_step
    X轴 ：scalar_value
"""

for i in range(10):
    writer.add_scalar("title", i, i)

"""
 writer.add_image() 
     tag,   标签 类似于title，他是需要唯一的
    img_tensor,    图片，仅支持 (torch.Tensor, numpy.ndarray, or string/blobname): Image data
    global_step=None, 这是第几步
    
    一般 PIL的image的类，输出的数据，无法满足img_tensor的类型限制，所以可以采用OpenCV的转换，他的类型就是numpy
    
    OpenCV安装  pip install opencv-python
    
    或者还可以直接用 numpy转换
"""

pic_path = "F:\\xiangmu\\tran_data\\hymenoptera_data\\hymenoptera_data\\train\\ants\\0013035.jpg"

# numpy 的转换
img_info = Image.open(pic_path)
np_img_array = np.array(img_info)

# writer.add_image("img1", np_img_array, 1)

""" 
到这一步运行报错，因为在文档说明
  Shape:
            img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
            convert a batch of tensor into 3xHxW format or call ``add_images`` and let us do the job.
            Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitable as long as
            corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.
            
    这里有说一个匹配数值 (3, H, W) 3代表的是通道，H代表的是高度，W代表的是宽度
"""
# 先检查这个图片的格式 输出 (512, 768, 3) （H,W,C） 这说明通道3是在最后一个，那么就需要更改一下格式
print(np_img_array.shape)
writer.add_image("img1", np_img_array, 1, dataformats="HWC")
writer.close()
