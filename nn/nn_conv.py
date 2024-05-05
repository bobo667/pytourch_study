import torch
import torch.nn.functional as F

input = torch.tensor([
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1]])

kernel = torch.tensor([
    [1, 2, 1],
    [0, 1, 0],
    [2, 1, 0]
])

print(input.shape)
print(kernel.shape)

# 尺寸变换
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input.shape)
print(kernel.shape)

# 代表每次对比跳的位数 stride
output = F.conv2d(input, kernel, stride=1, padding=0)
print(output)
# padding 指的是外层，比如说，上面的数据是 5 * 5 然后padding = 1 那么他的数据就成了 6 * 6，多出来的数据为0，
output2 = F.conv2d(input, kernel, stride=2, padding=0)
print(output2)
