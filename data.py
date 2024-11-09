import os

from PIL import Image
from torch.utils.data import Dataset

NUM_WORKERS = 8  # 数据读取线程数

# torch.Size([64, 3, 256, 256])
# torch.Size([64])


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

# 均值
mean = [-0.13227930850976577,0.0009032072751201647,0.000668213336575875,0.0010861044115739683,8.736464503700307e-05,-0.00042250074082551245,0.00033605961318379496,-0.00011272781509880225]
# 标准差
std = [0.021324149408854822,0.008439523316886572,0.01693572440418254,0.012510887614921454,0.020240274870162618,0.016078401353789913,0.01790334344388813,0.02377909824570553]


# 获得训练的数据集
def get_train_data(IS_DEBUG=False):
    pass


# 获得测试的数据集
def get_test_data(IS_DEBUG=False):
    pass
