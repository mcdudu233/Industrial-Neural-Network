import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# 数据读取线程数
NUM_WORKERS = 4

# torch.Size([64, 3, 256, 256])
# torch.Size([64])

# 类别
CLASSES = {
    "Health": 0,
    "Missing": 1,
    "Chipped": 2,
    "Surface": 3,
    "Root": 4,
}

# 均值
MEAN = [
    -0.13227930850976577,
    0.0009032072751201647,
    0.000668213336575875,
    0.0010861044115739683,
    8.736464503700307e-05,
    -0.00042250074082551245,
    0.00033605961318379496,
    -0.00011272781509880225,
]

# 标准差
STD = [
    0.021324149408854822,
    0.008439523316886572,
    0.01693572440418254,
    0.012510887614921454,
    0.020240274870162618,
    0.016078401353789913,
    0.01790334344388813,
    0.02377909824570553,
]


# 数据集加载器
class IndustrialDataset(Dataset):
    # 初始化加载数据
    def __init__(self, file):
        self.file = file
        self.data = pd.read_csv(file)

        # 数据集的标签
        # classes = self.data["Type"].drop_duplicates().values
        # self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        # print(self.class_to_idx)
        self.class_to_idx = CLASSES

    # 返回所有数据的数量
    def __len__(self):
        return int(self.data.count(axis=0).values.mean())

    # 返回指定索引的数据集
    def __getitem__(self, idx):
        tmp = self.data.iloc[idx, :]
        x = tmp[0:6].values
        x = x.astype(np.float32)
        y = self.class_to_idx[tmp["Type"]]
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)


# 获得训练的数据集
def get_train_data():
    data = IndustrialDataset("./data/train.csv")
    loader = DataLoader(data, batch_size=32, shuffle=True, num_workers=NUM_WORKERS)
    return loader


# 获得测试的数据集
def get_test_data():
    data = IndustrialDataset("./data/test.csv")
    loader = DataLoader(data, batch_size=32, shuffle=True, num_workers=NUM_WORKERS)
    return loader
