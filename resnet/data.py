import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


# 将所有4通道的图片统一为3通道图片
def deal_with_channel(path):
    for file_name in os.listdir(path):
        file = os.path.join(path, file_name)
        img = Image.open(file)
        img = img.convert("RGB")
        img.save(file)


mean = [0.485, 0.456, 0.406]  # 均值
std = [0.229, 0.224, 0.225]  # 标准差
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 256x256分辨率
    transforms.ToTensor(),  # 转换到张量
    transforms.Normalize(mean, std)  # 归一化
])


# 显示图像
def image_show(tensor):
    tensor = tensor.numpy().transpose((1, 2, 0))
    # 归一化的还原
    tensor = np.array(std) * tensor + np.array(mean)
    tensor = np.clip(tensor, 0, 1)
    plt.imshow(tensor)
    plt.show()


# 获得训练的数据集
def get_train_data(IS_DEBUG=False):
    data = ImageFolder('./data/train', transform=data_transform)

    if IS_DEBUG:
        print("标签数据：{}".format(data.class_to_idx))

    loader = DataLoader(data, batch_size=64, shuffle=True)
    return loader


# 获得测试的数据集
def get_test_data(IS_DEBUG=False):
    data = ImageFolder('./data/test', transform=data_transform)

    if IS_DEBUG:
        print("标签数据：{}".format(data.class_to_idx))

    loader = DataLoader(data, batch_size=64, shuffle=True)
    return loader
