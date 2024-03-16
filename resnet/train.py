import matplotlib.pyplot as plt
import torch


def train(loader, model, criterion, optimizer, scheduler, epochs=10, IS_DEBUG=False, IS_CUDA=False):
    if IS_CUDA:
        # 模型转交给显卡
        model.cuda()

    loss_seq = []  # 损失率序列
    accuracy_seq = []  # 准确率序列
    learning_seq = []  # 学习率序列
    for epoch in range(1, epochs + 1):
        total_loss = 0  # 总共的损失
        total_accuracy = 0  # 总共的准确率
        batch = 0  # 记录批次
        for _, data in enumerate(loader):
            input, actual = data
            if IS_CUDA:
                # 将数据转移到显卡上
                input = input.cuda()
                actual = actual.cuda()

            # 梯度参数清零
            optimizer.zero_grad()
            # 模型给出的预测值
            output = model(input)
            # 计算预测值和真实值的损失
            loss = criterion(output, actual)
            # 反向传播求梯度
            loss.backward()
            # 优化参数
            optimizer.step()
            # 学习率调整
            scheduler.step(loss)

            batch += 1
            size = input.size()[0]
            # 计算损失
            total_loss += loss.item() / size
            average_loss = total_loss / batch
            loss_seq.append(average_loss)
            # 计算准确率
            total_accuracy += torch.sum(torch.max(output, dim=1)[1] == actual) / size
            average_accuracy = total_accuracy / batch
            accuracy_seq.append(average_accuracy.item())
            # 计算学习率
            learning = optimizer.param_groups[0]['lr']
            learning_seq.append(learning)
            if IS_DEBUG:
                print("第{}批次的损失：{:.6f}\t学习率：{:.6f}\t准确率：{:.1f}%".format(batch, average_loss, learning,
                                                                                    average_accuracy * 100))

        # 计算平均损失
        average_loss = total_loss / batch
        average_accuracy = total_accuracy / batch
        if IS_DEBUG:
            print("*" * 20)
            print("完成第{}次训练：".format(epoch))
            print("损失：{}".format(average_loss))
            print("准确率：{}".format(average_accuracy))
            print("*" * 20)
    print("*" * 20)

    if IS_DEBUG:
        # 输出损失图像
        plt.plot(loss_seq, color='red')
        plt.plot(accuracy_seq, color='skyblue')
        plt.plot(learning_seq, color='pink')
        plt.legend()
        plt.show()
