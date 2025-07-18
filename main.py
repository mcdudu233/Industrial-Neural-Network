import os.path

import torch
from torch import nn, optim

from loader import CLASSES, get_train_data, get_val_data, get_test_data
from model import resnet_revise
from predict import predict
from train import train

# 全局参数区
IS_DEBUG = True  # 是否启用调试
IS_CUDA = False  # 是否使用CUDA
IS_TRAIN = False  # 是否训练模型 否则为评估
MODEL_PATH = "./model.pth"  # 模型存放位置


# 初始化检测环境
def init():
    global IS_DEBUG
    global IS_CUDA

    # 如果有cuda 则使用显卡的cuda加速
    if torch.cuda.is_available():
        IS_CUDA = True
        if IS_DEBUG:
            print("检测到显卡，将使用显卡：" + torch.cuda.get_device_name())


if __name__ == "__main__":
    init()

    # 超参数
    learning_rate = 0.0001  # 初始学习率
    learning_factor = 0.9  # 学习率调整因子

    # 创建齿轮缺陷检测模型
    num_classes = len(CLASSES)
    print(f"创建模型，类别数量: {num_classes}")
    model = resnet_revise(num_classes=num_classes)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

    # 学习率调整策略
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=learning_factor, patience=5  # 学习率调整器
    )

    # 加载训练好的模型（如果存在）
    if os.path.exists(MODEL_PATH):
        print(f"加载已有模型: {MODEL_PATH}")
        if IS_CUDA:
            model_state = torch.load(MODEL_PATH)
        else:
            model_state = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        model.load_state_dict(model_state)

    # 训练模型或者测试数据
    if IS_TRAIN:
        print("开始训练模型...")
        # 获取训练数据和验证数据
        train_loader = get_train_data()
        val_loader = get_val_data()

        # 训练模型
        model.train()
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            val_loader=val_loader,  # 添加验证集
            epochs=100,  # 训练轮数
            cuda=IS_CUDA,
        )

        # 保存训练好的模型
        print(f"保存模型到: {MODEL_PATH}")
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        print("开始评估模型...")
        test_loader = get_test_data()
        # 评估模型
        model.eval()
        predict(test_loader, model, cuda=IS_CUDA)
