import matplotlib.pyplot as plt
import torch
from matplotlib.font_manager import FontProperties


def train(
    loader,
    model,
    criterion,
    optimizer,
    scheduler,
    epochs=10,
    cuda=False,
):
    if cuda:
        # 模型转交给显卡
        model.cuda()

    loss_seq = []  # 损失率序列
    accuracy_seq = []  # 准确率序列
    learning_seq = []  # 学习率序列
    for epoch in range(1, epochs + 1):
        total_loss = 0  # 总共的损失
        total_accuracy = 0  # 总共的准确率
        for batch, data in enumerate(loader, 1):
            input, actual = data
            if cuda:
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

            size = input.size()[0]
            # 计算损失
            average_loss = loss.item() / size
            total_loss += average_loss
            # loss_seq.append(average_loss)
            loss_seq.append(total_loss / batch)
            # 计算准确率
            average_accuracy = torch.sum(torch.max(output, dim=1)[1] == actual) / size
            total_accuracy += average_accuracy
            # accuracy_seq.append(average_accuracy.item())
            accuracy_seq.append(total_accuracy.item() / batch)
            # 计算学习率
            learning = optimizer.param_groups[0]["lr"]
            learning_seq.append(learning)
            if IS_DEBUG:
                print(
                    "第{}批次的损失：{:.8f}\t学习率：{:.8f}\t准确率：{:.1f}%".format(
                        batch, average_loss, learning, average_accuracy * 100
                    )
                )

        # 计算平均损失
        average_loss = total_loss / batch
        average_accuracy = total_accuracy / batch
        print("*" * 20)
        print("完成第{}次训练：".format(epoch))
        print("损失：{}".format(average_loss))
        print("准确率：{}".format(average_accuracy))
        print("*" * 20)
    print("*" * 20)

    # 指定画图的字体
    font = FontProperties(fname="C:\Windows\Fonts\msyh.ttc")  # 指定字体路径

    # 输出损失图像
    plt.title("损失", fontproperties=font)
    plt.xlabel("批次", fontproperties=font)
    plt.ylabel("损失率", fontproperties=font)
    plt.plot(loss_seq[1:], color="red")
    plt.show()

    plt.title("学习率", fontproperties=font)
    plt.xlabel("批次", fontproperties=font)
    plt.ylabel("百分比", fontproperties=font)
    plt.plot(learning_seq, color="pink")
    plt.show()

    plt.title("准确度", fontproperties=font)
    plt.xlabel("批次", fontproperties=font)
    plt.ylabel("百分比", fontproperties=font)
    plt.plot(accuracy_seq, color="skyblue")
    plt.show()
