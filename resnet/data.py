import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


# 将所有4通道的图片统一为3通道图片
def deal_with_channel(path):
    for file_name in os.listdir(path):
        file = os.path.join(path, file_name)
        img = Image.open(file)
        img = img.convert("RGB")
        img.save(file)


# 测试数据集加载器
class TestDataset(Dataset):
    # 初始化加载数据
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.transform = transform

        self.files = []
        # 遍历目录下的所有文件
        for file_name in os.listdir(dir):
            file = os.path.join(dir, file_name)
            if os.path.isfile(file):
                self.files.append(file)
        return

    # 返回所有数据的数量
    def __len__(self):
        return len(self.files)

    # 返回指定索引的数据集
    def __getitem__(self, idx):
        # 读取图像文件并将其转换为张量
        img = Image.open(self.files[idx]).convert("RGB")
        # 可选：应用数据转换
        if self.transform:
            img = self.transform(img)
        return img


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
    data = TestDataset('./data/test', transform=data_transform)
    loader = DataLoader(data, batch_size=1, shuffle=False)
    return loader
