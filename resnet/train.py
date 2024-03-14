import matplotlib.pyplot as plt
import tqdm
from torch import nn
from torch.autograd import Variable


def train(loader, model, criterion, optimizer, scheduler, epochs=10, IS_DEBUG=False, IS_CUDA=False):
    if IS_CUDA:
        # 模型转交给显卡
        model.cuda()

    loss_seq = []  # 损失率序列
    for epoch in range(1, epochs + 1):
        total_loss = 0  # 总共的损失
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
            # TODO 转换成独热编码
            actual = nn.functional.one_hot(actual).float()
            # 计算预测值和真实值的损失
            loss = criterion(output, actual)
            # 反向传播求梯度
            loss.backward()
            # 优化参数
            optimizer.step()
            # 学习率调整
            scheduler.step(loss)

            batch += 1
            total_loss += loss.item() / input.size()[0]
            average_loss = total_loss / batch
            loss_seq.append(average_loss)
            if IS_DEBUG:
                print("第{}批次的损失：{}".format(batch, average_loss))

        # 计算平均损失
        average_loss = total_loss / batch
        if IS_DEBUG:
            print("*" * 20)
            print("完成第{}次训练：".format(epoch))
            print("损失：{}".format(average_loss))
    print("*" * 20)

    if IS_DEBUG:
        # 输出损失图像
        plt.plot(loss_seq)
        plt.show()
