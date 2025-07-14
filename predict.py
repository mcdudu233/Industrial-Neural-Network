import torch
from data import image_show


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

            # 显示预测结果
            for i in range(min(images.size(0), 4)):  # 每批次最多显示4张
                image = images[i].cpu()
                pred_label = predicted[i].item()

                # 显示图像和预测结果
                print(f"预测标签: {pred_label}, 类别: {classes[pred_label]}")
                title = f"预测: {classes[pred_label]}"
                image_show(image, title)
