import torch
import matplotlib.pyplot as plt
import numpy as np
from data import image_show


def predict(loader, model, cuda=False):
    # 类别名称
    classes = ["正常齿轮", "健康部分-中心网格", "健康部分-中心缺陷", "关键部分"]
    
    # 设置模型为评估模式
    model.eval()
    
    # 统计结果
    correct = 0
    total = 0
    class_correct = [0, 0, 0, 0]  # 更新为4个类别
    class_total = [0, 0, 0, 0]    # 更新为4个类别
    
    # 不计算梯度
    with torch.no_grad():
        for images, labels in loader:
            if cuda:
                images = images.cuda()
                labels = labels.cuda()
                
            # 前向传播
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # 统计整体准确率
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 统计每个类别的准确率
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
            
            # 显示第一个批次的一些图像和预测结果
            if total <= 8:  # 只显示前8张图片
                for i in range(min(images.size(0), 4)):  # 每批次最多显示4张
                    if total <= 8:  # 确保总共不超过8张
                        image = images[i].cpu()
                        pred_label = predicted[i].item()
                        true_label = labels[i].item()
                        
                        # 显示图像和预测结果
                        title = f"预测: {classes[pred_label]}, 实际: {classes[true_label]}"
                        image_show(image, title)
                        total += 1
    
    # 打印整体准确率
    print(f"测试集整体准确率: {100 * correct / total:.2f}%")
    
    # 打印每个类别的准确率
    for i in range(len(classes)):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f"类别 '{classes[i]}' 的准确率: {accuracy:.2f}%")
        else:
            print(f"类别 '{classes[i]}' 没有测试样本")
