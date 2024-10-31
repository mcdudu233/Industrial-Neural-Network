import torch
import torch.nn as nn


# 修改后的ResNet残差块
class BasicBlockRevise(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlockRevise, self).__init__()

        # 第一层3x3卷积层
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(32, out_channel)

        # 第二层3x3卷积层
        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.gn2 = nn.GroupNorm(32, out_channel)

        self.downsample = downsample

    def forward(self, x):
        # 残差块保留原始输入
        residual = x
        # 如果是虚线残差结构，则进行下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        # 第一层卷积运算
        out = self.conv1(x)
        out = self.gn1(out)
        out = nn.functional.relu(out)

        # 第二层卷积运算
        out = self.conv2(out)
        out = self.gn2(out)

        # 加上残差
        out += residual

        # 修改使用GeLU激活函数
        out = nn.functional.gelu(out)
        return out


# 定义ResNet类
class ResNetRevise(nn.Module):
    def __init__(
        self,
        block,
        blocks_num,
        num_classes=1000,
        include_top=True,
        groups=1,
        width_per_group=64,
    ):
        super(ResNetRevise, self).__init__()
        self.include_top = include_top
        # maxpool的输出通道数为64，残差结构输入通道数为64
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(
            3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 浅层的stride=1，深层的stride=2
        # block：定义的两种残差模块
        # block_num：模块中残差块的个数
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            # 自适应平均池化，指定输出（H，W），通道数不变
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            # 全连接层
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        # 遍历网络中的每一层
        # 继承nn.Module类中的一个方法:self.modules(), 他会返回该网络中的所有modules
        for m in self.modules():
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # kaiming正态分布初始化，使得Conv2d卷积层反向传播的输出的方差都为1
                # fan_in：权重是通过线性层（卷积或全连接）隐性确定
                # fan_out：通过创建随机矩阵显式创建权重
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    # 定义残差模块，由若干个残差块组成
    # block：定义的两种残差模块，channel：该模块中所有卷积层的基准通道数。block_num：模块中残差块的个数
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        # 如果满足条件，则是虚线残差结构
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channel,
                    channel * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(channel * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.in_channel,
                channel,
                downsample=downsample,
                stride=stride,
                groups=self.groups,
                width_per_group=self.width_per_group,
            )
        )
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(
                block(
                    self.in_channel,
                    channel,
                    groups=self.groups,
                    width_per_group=self.width_per_group,
                )
            )
        # Sequential：自定义顺序连接成模型，生成网络结构
        return nn.Sequential(*layers)

    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        # 清空上一次中间层的输出
        self.features = []

        # 无论哪种ResNet，都需要的静态层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 动态层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


# ResNet修改版本
def resnet_revise(num_classes=1000, include_top=True):
    return ResNetRevise(
        BasicBlockRevise, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top
    )
