from time import sleep

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

from loader import STD, MEAN

# 设置中文字体
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]


def denormalize(numpy):
    numpy = numpy.transpose((1, 2, 0))
    numpy = numpy * np.array(STD) + np.array(MEAN)
    numpy = np.clip(numpy, 0, 1)
    return numpy


# 显示图像
def image_show(numpy0, numpy1, title0=None, title1=None):
    img0 = denormalize(numpy0)
    img1 = numpy1

    # 创建1x2子图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)  # 左图
    plt.imshow(img0)
    if title0:
        plt.title(title0)
    plt.axis("off")

    plt.subplot(1, 2, 2)  # 右图
    plt.imshow(img1)
    if title1:
        plt.title(title1)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def predict(loader, model, cuda=False):
    if cuda:
        model.cuda()

    # 类别名称
    classes = ["正常齿轮", "缺陷齿轮-划痕", "缺陷齿轮-划痕", "缺陷齿轮-点蚀"]

    # 设置模型为评估模式
    model.eval()

    # 不计算梯度
    with torch.no_grad():
        for images in loader:
            if cuda:
                images = images.cuda()

            # 前向传播
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # 显示预测结果 每批只有一张图片
            image = images[0].cpu()
            pred_label = predicted[0].item()

            # 显示图像和预测结果
            print(f"预测标签: {pred_label}, 类别: {classes[pred_label]}")
            cam = model.get_cam(pred_label, batch=0).cpu()
            image_show(
                image.numpy(), cam.numpy(), f"原图({classes[pred_label]})", "热力图"
            )

            # 睡眠
            sleep(1)
