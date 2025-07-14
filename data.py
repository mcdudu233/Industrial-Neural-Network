import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 设置中文字体
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]

# 数据读取线程数
NUM_WORKERS = 8
# 每批次训练的数据
BATCH_SIZE = 32

# 类别
CLASSES = {
    "normal": 0,  # 正常齿轮（无缺陷）
    "hp_cm": 1,  # 健康部分-中心网格
    "hp_cd": 2,  # 健康部分-中心缺陷
    "kp": 3,  # 关键部分
}

# 图像预处理参数
IMAGE_SIZE = 224  # ResNet标准输入尺寸
MEAN = [0.485, 0.456, 0.406]  # 标准ImageNet均值
STD = [0.229, 0.224, 0.225]  # 标准ImageNet标准差


# COCO格式齿轮数据集加载器
class GearCocoDataset(Dataset):
    def __init__(self, data_dir, json_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # 加载COCO格式的JSON文件
        with open(os.path.join(data_dir, json_file), "r") as f:
            self.coco_data = json.load(f)

        # 创建图像ID到图像文件名的映射
        self.image_id_to_filename = {}
        for image_info in self.coco_data["images"]:
            self.image_id_to_filename[image_info["id"]] = image_info["file_name"]

        # 创建图像ID到类别的映射
        self.image_id_to_categories = {}
        for ann in self.coco_data["annotations"]:
            image_id = ann["image_id"]
            category_id = ann["category_id"]

            if image_id not in self.image_id_to_categories:
                self.image_id_to_categories[image_id] = []

            self.image_id_to_categories[image_id].append(category_id)

        # 创建类别ID到类别名称的映射
        self.category_id_to_name = {}
        self.category_name_to_id = {}
        for category in self.coco_data["categories"]:
            self.category_id_to_name[category["id"]] = category["name"]
            self.category_name_to_id[category["name"]] = category["id"]

        # 创建数据集索引
        self.image_ids = list(self.image_id_to_filename.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        filename = self.image_id_to_filename[image_id]

        # 加载图像
        image_path = os.path.join(self.data_dir, filename)
        image = Image.open(image_path).convert("RGB")

        # 获取图像的类别
        if image_id in self.image_id_to_categories:
            categories = self.image_id_to_categories[image_id]
            # 使用最常见的类别作为标签
            category_counts = {}
            for cat_id in categories:
                if cat_id not in category_counts:
                    category_counts[cat_id] = 0
                category_counts[cat_id] += 1

            # 获取最常见的类别
            most_common_category = max(category_counts.items(), key=lambda x: x[1])[0]
            category_name = self.category_id_to_name[most_common_category]
            # 将类别名称映射到我们的类别索引
            if category_name in CLASSES:
                label = CLASSES[category_name]
            else:
                # 如果类别名称不在我们的映射中，默认为正常类别
                label = CLASSES["normal"]
        else:
            # 如果图像没有标注，则认为是正常的
            label = CLASSES["normal"]

        # 应用数据转换
        if self.transform:
            image = self.transform(image)

        return image, label


# 测试集加载
class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # 加载所有图像文件名
        self.images = [f for f in os.listdir(data_dir) if f.endswith((".jpg", ".png"))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 加载图像
        image_path = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(image_path).convert("RGB")

        # 应用数据转换
        if self.transform:
            image = self.transform(image)

        return image


# 数据增强和预处理
def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )


# 获得训练的数据集
def get_train_data():
    transform = get_transforms(is_train=True)
    dataset = GearCocoDataset("./data/train", "train_coco.json", transform=transform)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    return loader


# 获得验证的数据集
def get_val_data():
    transform = get_transforms(is_train=False)
    # 如果有专门的验证集JSON文件，使用它
    # 否则，使用训练集的一部分作为验证集
    if os.path.exists("./data/val/val_coco.json"):
        dataset = GearCocoDataset("./data/val", "val_coco.json", transform=transform)
    else:
        dataset = GearCocoDataset(
            "./data/train", "train_coco.json", transform=transform
        )
        # 使用20%的训练数据作为验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        _, dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    return loader


# 获得测试的数据集
def get_test_data():
    transform = get_transforms(is_train=False)
    dataset = TestDataset("./data/val", transform=transform)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    return loader


# 显示图像
def image_show(tensor, title=None):
    # 转换tensor为numpy数组
    img = tensor.numpy().transpose((1, 2, 0))
    # 反归一化
    img = img * np.array(STD) + np.array(MEAN)
    img = np.clip(img, 0, 1)

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()
