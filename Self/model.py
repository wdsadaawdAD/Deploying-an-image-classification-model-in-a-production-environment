import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

class Bottleneck(nn.Module):
    expansion = 4  # 瓶颈结构的通道扩展倍数
    # 新增：加入注意力机制（SE模块），增强关键特征权重
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_se=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 深度可分离卷积（groups=out_channels）：增强通道独立性，捕捉细粒度特征
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU6(inplace=True)  # ReLU6增强数值稳定性，适配深层网络
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        
        # 新增：SE注意力模块（Squeeze-and-Excitation）
        if self.use_se:
            se_channels = out_channels * self.expansion
            self.se_global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.se_fc1 = nn.Conv2d(se_channels, se_channels // 16, kernel_size=1, bias=True)  # 压缩通道
            self.se_fc2 = nn.Conv2d(se_channels // 16, se_channels, kernel_size=1, bias=True)  # 恢复通道
            self.se_sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x  # 保存原输入，用于后续残差连接

        # 1x1卷积：降维（减少计算量）→ 批归一化 → 激活
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3深度可分离卷积：提取局部特征 → 批归一化 → 激活
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1x1卷积：升维（恢复通道数）→ 批归一化（暂不激活，等残差相加后再激活）
        out = self.conv3(out)
        out = self.bn3(out)

        # SE注意力机制：对重要特征加权
        if self.use_se:
            se_out = self.se_global_avgpool(out)  # 全局平均池化：压缩空间维度为1x1
            se_out = self.se_fc1(se_out)          # 通道压缩：减少参数，增强泛化
            se_out = self.relu(se_out)            # 激活函数：引入非线性
            se_out = self.se_fc2(se_out)          # 通道恢复：匹配原特征通道数
            se_out = self.se_sigmoid(se_out)      # 归一化到0-1：得到特征权重
            out = out * se_out                    # 特征加权：突出重要特征

        # 残差连接：若输入输出维度不匹配，先通过downsample调整
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual  # 残差相加：主支路特征 + 调整后的原输入
        out = self.relu(out)  # 最终激活：引入非线性

        return out

class CustomResNet(nn.Module):
    # 新增参数：use_se（是否用SE注意力）、depth_multiplier（网络深度缩放）
    def __init__(self, block, layers, num_classes=100, width_multiplier=1.0, 
                 use_se=True, depth_multiplier=1.0):
        super(CustomResNet, self).__init__()
        # 1. 初始化通道数（按width_multiplier缩放，默认1.5倍提升容量）
        self.in_channels = int(64 * width_multiplier)
        self.use_se = use_se  # 保存SE注意力开关，供_make_layer调用
        
        # 2. 初始卷积层：3个3x3卷积替换1个7x7卷积（减少参数+保留细节）
        self.conv1 = nn.Sequential(
            # 第一层：降维（3→in_channels//2）+ 步幅2（缩小空间维度）
            nn.Conv2d(3, self.in_channels//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels//2),
            nn.ReLU6(inplace=True),
            # 第二层：保持通道数和空间维度，增强特征提取
            nn.Conv2d(self.in_channels//2, self.in_channels//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels//2),
            nn.ReLU6(inplace=True),
            # 第三层：升维（in_channels//2→in_channels），匹配后续残差块输入
            nn.Conv2d(self.in_channels//2, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(inplace=True)
        )
        # 最大池化：步幅1（避免过早丢失细节），保持空间维度
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        # 3. 构建残差块组（按depth_multiplier缩放残差块数量，提升网络深度）
        # 例：原[6,8,12,6] → depth_multiplier=1.2 → [7,10,14,7]
        layers = [int(layer * depth_multiplier) for layer in layers]
        self.layer1 = self._make_layer(block, int(64 * width_multiplier), layers[0], stride=1)  # 不缩小尺寸
        self.layer2 = self._make_layer(block, int(128 * width_multiplier), layers[1], stride=2) # 步幅2：缩小尺寸
        self.layer3 = self._make_layer(block, int(256 * width_multiplier), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(512 * width_multiplier), layers[3], stride=2)
        
        # 4. 分类头：2层全连接+BatchNorm（增强高层特征表达，抑制过拟合）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化：无论输入尺寸，输出1x1
        final_channels = int(512 * width_multiplier) * block.expansion  # 分类头输入维度
        self.fc = nn.Sequential(
            nn.Flatten(),  # 展平特征：(batch, C, 1, 1) → (batch, C)
            nn.Dropout(0.3),  # 第一层Dropout：抑制过拟合，保留更多特征
            nn.Linear(final_channels, final_channels // 2),  # 中间层：特征转换
            nn.BatchNorm1d(final_channels // 2),  # 1D批归一化：稳定训练，缓解梯度问题
            nn.ReLU6(inplace=True),  # 激活函数：引入非线性
            nn.Dropout(0.2),  # 第二层Dropout：进一步抑制过拟合
            nn.Linear(final_channels // 2, num_classes)  # 输出层：映射到类别数
        )

        # 权重初始化：针对不同层做针对性初始化，确保训练稳定性
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming初始化：适配ReLU/ReLU6，保证梯度传播稳定
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # BatchNorm初始化：权重=1（保证初始方差），偏置=0（保证初始均值）
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层初始化：正态分布（均值0，标准差0.01），避免初始值过大/过小
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """构建残差块组：堆叠blocks个Bottleneck，处理维度匹配"""
        downsample = None
        # 当步幅≠1（空间维度缩小）或输入通道≠输出通道（通道数变化）时，需要downsample调整残差
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                # 1x1卷积：调整通道数 + 步幅调整空间维度
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),  # 批归一化：保持数值稳定
            )

        layers = []
        # 第一个残差块：需传入stride和downsample（处理维度匹配）
        layers.append(block(self.in_channels, out_channels, stride, downsample, use_se=self.use_se))
        self.in_channels = out_channels * block.expansion  # 更新输入通道数，供后续块使用
        # 后续残差块：无需调整维度，直接堆叠
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, use_se=self.use_se))

        return nn.Sequential(*layers)  # 用Sequential包装成一个模块

    def forward(self, x):
        """前向传播：定义数据在网络中的流动路径"""
        # 初始特征提取：conv1 → maxpool
        x = self.conv1(x)
        x = self.maxpool(x)

        # 残差块组：逐层提取复杂特征
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 分类头：池化 → 展平 → 全连接
        x = self.avgpool(x)
        x = self.fc(x)  # fc已包含Flatten，无需单独调用torch.flatten

        return x

    @autocast()
    def inference(self, x):
        """推理专用方法：自动混合精度 + 关闭梯度 + 输出概率"""
        self.eval()  # 切换评估模式：关闭Dropout、BatchNorm用移动均值/方差
        with torch.no_grad():  # 关闭梯度计算：节省内存，加速推理
            outputs = self(x)  # 前向传播得到原始输出（logits）
            probabilities = F.softmax(outputs, dim=1)  # 转换为概率分布（0-1，每行和为1）
        return probabilities

    def get_optimizer(self, lr=0.001, weight_decay=5e-5):
        """获取优化器：分层学习率（高层学习率更高，适配深层网络）"""
        params = [
            # 底层（conv1、layer1）：学习率低（特征通用，少调整）
            {'params': self.conv1.parameters(), 'lr': lr * 0.5},
            {'params': self.layer1.parameters(), 'lr': lr * 0.5},
            # 中层（layer2、layer3）：学习率中等
            {'params': self.layer2.parameters(), 'lr': lr * 0.7},
            {'params': self.layer3.parameters(), 'lr': lr * 0.9},
            # 高层（layer4、fc）：学习率高（特征任务相关，多调整）
            {'params': self.layer4.parameters(), 'lr': lr},
            {'params': self.fc.parameters(), 'lr': lr * 1.2}
        ]
        # AdamW：带权重衰减的Adam，抑制过拟合
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def get_scheduler(self, optimizer):
        """获取学习率调度器：CosineAnnealingWarmRestarts（灵活调整，避免局部最优）"""
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,    # 初始周期（10个epoch）
            T_mult=2,  # 周期倍增（每次重启后周期×2）
            eta_min=1e-6,  # 最小学习率（避免学习率过低导致停滞）
            verbose=True  # 打印学习率调整信息
        )

# 模型实例化函数：默认参数提升容量，同时控制过拟合
def create_model(num_classes=100, width_multiplier=1.5, depth_multiplier=1.2, use_se=True):
    return CustomResNet(
        block=Bottleneck,  # 用Bottleneck作为基础残差块
        layers=[6, 8, 12, 6],  # 基础残差块数量（ResNet50风格）
        num_classes=num_classes,  # 分类类别数
        width_multiplier=width_multiplier,  # 通道数缩放（1.5倍）
        depth_multiplier=depth_multiplier,  # 残差块数量缩放（1.2倍）
        use_se=use_se  # 启用SE注意力
    )