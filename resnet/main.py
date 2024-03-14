import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from model import resnet34
from train import train
import data

# 全局参数区
IS_DEBUG = True  # 是否启用调试
IS_CUDA = False  # 是否使用CUDA
IS_TRAIN = True  # 是否训练模型 否则为评估
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


if __name__ == '__main__':
    init()

    # 超参数
    learning_rate = 0.01  # 初始学习率
    learning_factor = 0.978  # 学习率调整因子

    model = nn.Sequential(resnet34(2), nn.Softmax())  # 使用34层的resnet模型 加上softmax
    criterion = nn.CrossEntropyLoss()  # 损失计算器 交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 优化器 Adam优化
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,  # 学习率调整器 ReduceLROnPlateau lr=lr*factor
                                                     mode='min',
                                                     factor=learning_factor,
                                                     patience=5)

    # 训练模型或者测试数据
    if IS_TRAIN:
        # 获取训练数据
        train_data = data.get_train_data(IS_DEBUG)
        # 训练模型
        model.train()
        train(train_data, model, criterion, optimizer, scheduler, epochs=20, IS_DEBUG=IS_DEBUG, IS_CUDA=IS_CUDA)
        # 保存训练好的模型
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        # 加载训练好的模型和测试数据
        model.load_state_dict(torch.load(MODEL_PATH))
        test_data = data.get_test_data(IS_DEBUG)
        # 评估模型
        model.eval()
        classes = ['cat', 'dog']
        for _, image in enumerate(test_data):
            data.image_show(image[0])
            index = torch.argmax(model(image))
            print(classes[index])
