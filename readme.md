# 高性能图像分类系统

这是一个基于PyTorch的专业级深度学习图像分类系统，采用优化的ResNet架构，针对100个日常物品类别实现高精度图像识别。系统设计注重工程可靠性与扩展性，支持完整的训练-评估-推理流程，适用于学术研究、工业应用及教育场景。

附带预训练模型 -- 开箱即用📊 -- 生产环境级部署-- 可应用在工业物流等

## 技术亮点

- **优化的神经网络架构**：基于Bottleneck结构的自定义ResNet模型，在保证精度的同时优化了计算效率
- **完整的MLOps流程**：标准化数据处理、模型训练、评估与推理全链路支持
- **工业级优化技术**：集成混合精度训练、数据增强、断点续训等高级特性
- **灵活的配置系统**：支持超参数调整、多设备运行（CPU/GPU）、批处理推理
- **完善的错误处理**：包含数据校验、日志记录、异常捕获等可靠性保障机制

## 项目结构

```
├── Self/
│   ├── __pycache__/
│   ├── infere.py       # 推理相关代码
│   ├── model.py        # 模型定义
│   └── train.py        # 训练相关代码
├── dataset_split/
│   ├── test_files.txt  # 测试集文件列表
│   ├── train_files.txt # 训练集文件列表
│   └── val_files.txt   # 验证集文件列表
├── logs/
│   ├── app_20250829.log
│   └── app_20250830.log
├── main/
│   ├── logs/
│   └── main.py         # 主程序入口
├── model/
│   ├── best_resnet.pth # 最佳模型权重
│   └── classname.txt   # 类别名称文件
├── utils/
│   ├── __pycache__/
│   ├── data_loader.py  # 数据加载
│   ├── data_preprocess.py # 数据预处理
│   └── loggin_demo.py  # 日志配置
├── readme.md           # 项目说明文档
└── LICENSE             # 开源许可证
```

## 核心功能

- **端到端图像分类流水线**：从原始图像到分类结果的完整处理流程
- **可扩展模型架构**：基于Bottleneck结构的优化ResNet实现，支持不同深度配置
- **双模式运行系统**：批量化训练模式与交互式推理模式无缝切换
- **高级训练优化**：集成数据增强、自动混合精度训练(AMP)、学习率调度等技术
- **模型管理机制**：支持断点续训、最佳模型自动保存、权重导入导出
- **高效数据处理**：多进程数据加载、内存优化、图像验证与预处理

## 技术栈与依赖

| 技术/库 | 版本 | 用途 | 溯源 |
|---------|------|------|------|
| Python | 3.9+ | 核心编程语言 | <mcfile name="requirements.txt" path="/home/xpr/project/requirements.txt"></mcfile> |
| PyTorch | 2.3.0 | 深度学习框架 | <mcfile name="requirements.txt" path="/home/xpr/project/requirements.txt"></mcfile> |
| torchvision | 0.18.0 | 计算机视觉工具集 | <mcfile name="requirements.txt" path="/home/xpr/project/requirements.txt"></mcfile> |
| PIL (Pillow) | 9.0.0+ | 图像处理 | <mcfile name="requirements.txt" path="/home/xpr/project/requirements.txt"></mcfile> |
| NumPy | 1.20.0+ | 科学计算 | <mcfile name="requirements.txt" path="/home/xpr/project/requirements.txt"></mcfile> |
| scikit-learn | 1.0.0+ | 机器学习工具 | <mcfile name="utils/data_preprocess.py" path="/home/xpr/project/utils/data_preprocess.py"></mcfile> |
| tqdm | 4.60.0+ | 进度可视化 | <mcfile name="Self/train.py" path="/home/xpr/project/Self/train.py"></mcfile> |

## 安装说明

1. 克隆项目代码：
```bash
# 在GitCode上克隆项目
```

2. 安装依赖包：

可以直接使用项目提供的requirements.txt文件安装所有依赖：
```bash
pip install -r requirements.txt
```

或者手动安装主要依赖：
```bash
pip install torch==2.3.0 torchvision==0.18.0 Pillow numpy scikit-learn tqdm
```

3. 确保数据集已正确准备，并生成了相应的划分文件（train_files.txt、val_files.txt、test_files.txt）

## 使用方法

### 训练模式

使用以下命令启动模型训练：

```bash
cd /home/xpr/project
python main/main.py --mode train --filepath /home/xpr/project/model/ --data_path /path/to/your/dataset
```

参数说明：
- `--mode train`：指定运行模式为训练
- `--filepath`：指定模型权重保存目录
- `--data_path`：指定训练集数据目录

### 推理模式

使用以下命令启动模型推理：

```bash
cd /home/xpr/project
python main/main.py --mode infere --filepath /home/xpr/project/model/ --data_path None
```

参数说明：
- `--mode infere`：指定运行模式为推理
- `--filepath`：指定模型权重加载目录
- `--data_path`：推理模式下可设为None

在推理模式下，程序会提示输入图像文件路径，然后输出预测结果，包括预测类别、类别名称、置信度等信息。

## 项目配置

### 超参数配置

在`main.py`中的`train_main`函数内可以调整以下超参数：

- `LEARNING_RATE`：学习率，默认为0.001
- `WEIGHT_DECAY`：权重衰减，默认为1e-6
- `NUM_EPOCHS`：训练轮次，默认为50

### 数据加载配置

在`build_dataloaders`函数中可以调整以下参数：

- `batch_size`：批次大小，默认为16
- `num_workers`：数据加载进程数，默认为32

## 模型说明

项目使用了自定义的ResNet模型，基于Bottleneck结构实现。模型主要特点：

1. 初始卷积层：7×7卷积，步长2，输出通道数64
2. 四个残差块组：通道数分别为64、128、256、512
3. 全局平均池化层
4. 分类头：包含Dropout层和全连接层

## 数据集说明

项目使用的数据集包含100个类别的日常物品图像，类别名称定义在`model/classname.txt`文件中。数据集已划分为训练集、验证集和测试集，对应的文件列表分别存储在`dataset_split/`目录下。

## 日志说明

项目使用Python的logging模块记录日志，日志配置在`utils/loggin_demo.py`中定义。日志文件保存在`logs/`和`main/logs/`目录下。

## License

该项目使用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

