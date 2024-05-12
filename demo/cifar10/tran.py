import torch
import torchvision
from torch.utils.data import DataLoader
from model import Nn
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('../../logs')

# 准备数据集
train_data = torchvision.datasets.CIFAR10("../../dataSet", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("../../dataSet", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())
# 五万
print(len(train_data))
# 一万
print(len(test_data))
print(F"训练数据集的长度:{len(train_data)},测试数据集是:{len(test_data)}")

# 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络
nn = Nn()

# 损失函数
lose_fn = torch.nn.CrossEntropyLoss()

# 优化器
# 1e -2 = 1 * (10)^(-2) = 1 / 100 = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(nn.parameters(), lr=learning_rate)

# 设置训练网络的参数
total_train_step = 0
# 设置测试的次数
total_test_step = 0
# 训练轮数
epoch = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nn.to(device)

nn.load_state_dict(torch.load("../../customerData/nn.pth"))
for i in range(epoch):
    print("----------第 {} 轮开始---------".format(i + 1))
    #  网络开始训练
    nn.train()

    for data in train_dataloader:
        imgs, labs = data
        imgs = imgs.to(device)
        labs = labs.to(device)
        outputs = nn(imgs)
        # 定义损失函数 计算误差
        loss = lose_fn(outputs, labs)
        # 梯度清0
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 使用优化器优化
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数:{},Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    #     進行測試
    #     没有梯度的时候
    # 开启训练模式
    nn.eval()
    loss_count = 0
    accuracy_count = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, labs = data
            imgs = imgs.to(device)
            labs = labs.to(device)
            outputs = nn(imgs)
            loss = lose_fn(outputs, labs)
            loss_count += loss.item()
            accuracy = (outputs.argmax(dim=1) == labs).sum()
            accuracy_count += accuracy

    print("整体测试集上的Loss ： {} ,正确率 : {}".format(loss_count, (accuracy_count / len(test_data))))
    writer.add_scalar("test_accuracy", accuracy_count / len(test_data), total_test_step)
    writer.add_scalar("test_loss", loss_count, total_test_step)
    total_test_step += 1

    torch.save(nn.state_dict(), "../../customerData/nn.pth")
    print("模型已保存")

writer.close()
