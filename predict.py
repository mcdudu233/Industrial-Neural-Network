import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

from loader import STD, MEAN

# 设置中文字体
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]


# 显示图像
def image_show(tensor0, tensor1, title=None):
    # 转换tensor为numpy数组
    img = tensor0.numpy().transpose((1, 2, 0))
    # 反归一化
    img = img * np.array(STD) + np.array(MEAN)
    img = np.clip(img, 0, 1)

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis("off")
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
            title = f"预测: {classes[pred_label]}"
            image_show(image, title)
