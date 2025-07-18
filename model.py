import torch
import torch.nn as nn
import torchvision.models as models


# 基础残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# 瓶颈残差块（用于更深的网络）
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(
            out_channel,
            out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(
            out_channel, out_channel * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


# 齿轮缺陷检测ResNet模型
class GearResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(GearResNet, self).__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(
            3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# 使用预训练的ResNet模型
class PretrainedGearResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(PretrainedGearResNet, self).__init__()
        # 加载预训练的ResNet50模型
        self.model = models.resnet50(pretrained=pretrained)

        # 修改最后的全连接层以适应我们的分类任务
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# 使用预训练的ResNet50模型并支持标准CAM（GAP）
class CAMResNet50(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(CAMResNet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)

        # 采用全局平均池化（GAP）
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)

        # 全连接层
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes, bias=False)

        # softmax
        self.softmax = nn.Softmax(dim=1)

        # 保存最后一层卷积特征和类别
        self.feature_map = None
        self.model.layer4.register_forward_hook(self._save_output)

    def _save_output(self, module, input, output):
        self.feature_map = output.detach()

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x

    def get_cam(self, class_idx, batch=0):
        # 获取最后一层卷积特征图和全连接权重
        assert self.feature_map is not None, "请先前输入一张图片！"
        b, c, h, w = self.feature_map.shape
        feature = self.feature_map[batch]  # [2048, 7, 7]
        weight = self.model.fc.weight[class_idx]  # [2048]

        # 计算CAM (权重 * 特征图)
        cam = torch.zeros(feature.shape, dtype=torch.float32)
        for i, w in enumerate(weight):
            cam += w * feature[i]

        # ReLU
        cam = torch.relu(cam)
        # 归一化到[0,1]
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = cam.numpy()

        return cam


# 创建ResNet-18模型
def resnet18(num_classes=2):
    return GearResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# 创建ResNet-34模型
def resnet34(num_classes=2):
    return GearResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


# 创建ResNet-50模型
def resnet50(num_classes=2):
    return GearResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


# 创建预训练的ResNet模型
def pretrained_resnet(num_classes=2):
    return PretrainedGearResNet(num_classes=num_classes)


# 修改后的ResNet模型，用于齿轮缺陷检测
def resnet_revise(num_classes=2):
    # 默认使用支持CAM的ResNet50
    return CAMResNet50(num_classes=num_classes)
