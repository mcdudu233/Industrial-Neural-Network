import matplotlib.pyplot as plt
import torch
from matplotlib.font_manager import FontProperties


def evaluate(loader, model, criterion, cuda=False):
    """评估模型在验证集上的性能"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            if cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 统计损失和准确率
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += inputs.size(0)

    # 计算平均损失和准确率
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def train(
    loader,
    model,
    criterion,
    optimizer,
    scheduler,
    val_loader=None,
    epochs=10,
    cuda=False,
):
    """训练模型"""
    if cuda:
        # 模型转交给显卡
        model.cuda()

    # 记录训练过程
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    learning_rate_history = []

    best_val_acc = 0.0  # 记录最佳验证集准确率

    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for batch_idx, (inputs, targets) in enumerate(loader, 1):
            if cuda:
                # 将数据转移到显卡上
                inputs = inputs.cuda()
                targets = targets.cuda()

            # 梯度参数清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, targets)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            # 统计损失和准确率
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == targets).sum().item()
            total_samples += batch_size

            # 每20个批次打印一次训练信息
            if batch_idx % 20 == 0:
                batch_loss = loss.item()
                batch_acc = (predicted == targets).sum().item() / batch_size
                print(
                    f"Epoch: {epoch}/{epochs}, Batch: {batch_idx}/{len(loader)}, "
                    f"Loss: {batch_loss:.4f}, Acc: {batch_acc:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

        # 计算训练集上的平均损失和准确率
        train_loss = running_loss / total_samples
        train_acc = running_correct / total_samples
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        # 验证阶段
        if val_loader is not None:
            val_loss, val_acc = evaluate(val_loader, model, criterion, cuda)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)

            # 更新学习率
            scheduler.step(val_loss)

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            #     torch.save(model.state_dict(), "best_model.pth")
        else:
            # 如果没有验证集，使用训练损失更新学习率
            scheduler.step(train_loss)

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rate_history.append(current_lr)

        # 打印每个epoch的训练信息
        print("-" * 50)
        print(f"Epoch: {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        if val_loader is not None:
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        print("-" * 50)

    print("训练完成!")
    if val_loader is not None:
        print(f"最佳验证集准确率: {best_val_acc:.4f}")

    # 绘制训练过程图表
    plot_training_history(
        train_loss_history,
        train_acc_history,
        val_loss_history if val_loader else None,
        val_acc_history if val_loader else None,
        learning_rate_history,
    )


def plot_training_history(
    train_loss, train_acc, val_loss=None, val_acc=None, lr_history=None
):
    """绘制训练历史图表"""
    # 尝试获取中文字体，如果不存在则使用默认字体
    try:
        font = FontProperties(fname="C:\Windows\Fonts\msyh.ttc")
    except:
        font = None

    plt.figure(figsize=(15, 10))

    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_loss, label="训练损失", color="blue")
    if val_loss:
        plt.plot(val_loss, label="验证损失", color="red")
    plt.title("损失曲线", fontproperties=font)
    plt.xlabel("Epoch", fontproperties=font)
    plt.ylabel("损失", fontproperties=font)
    plt.legend(prop=font)
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(train_acc, label="训练准确率", color="blue")
    if val_acc:
        plt.plot(val_acc, label="验证准确率", color="red")
    plt.title("准确率曲线", fontproperties=font)
    plt.xlabel("Epoch", fontproperties=font)
    plt.ylabel("准确率", fontproperties=font)
    plt.legend(prop=font)
    plt.grid(True)

    # 绘制学习率曲线
    if lr_history:
        plt.subplot(2, 2, 3)
        plt.plot(lr_history, color="green")
        plt.title("学习率变化", fontproperties=font)
        plt.xlabel("Epoch", fontproperties=font)
        plt.ylabel("学习率", fontproperties=font)
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()
