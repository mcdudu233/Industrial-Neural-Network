#### 深度学习应用于工业零件的检测

#### 安装教程

1. 首先安装好git软件包
2. 输入 `git clone https://gitee.com/dudu233/industrial_neural_network.git` 克隆到某一目录
3. 使用PyCharm导入该项目

#### 文件结构

│ .gitignore -->忽略文件  
│ main.py -->项目main文件  
│ README.md -->项目介绍文件  
│  
├─data -->项目数据集  
│  
└─resnet -->resnet框架测试

#### 使用数据集

[齿轮检测训练集](https://drive.google.com/file/d/1CZo-Ab5BXkTjV-b1-NIFzYMjfJQMl4nG/view?usp=share_link)

[齿轮检测验证集](https://drive.google.com/file/d/1-0sSrmhElBseeZWICu77lzTxoOiRD8yG/view?usp=share_link)

下载后请解压到 `data` 目录下，训练集解压到 `data/train` （该目录下包含所有训练图片），验证集解压到 `data/val` 目录下。 

#### 参与贡献

1. Fork 本仓库
2. 新建 Feat_xxx 分支
3. 提交代码
4. 新建 Pull Request
