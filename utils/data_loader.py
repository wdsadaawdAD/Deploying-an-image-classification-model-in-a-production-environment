import logging
import sys
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


'''工具函数，建立索引'''

def build_data_and_label(split_file_path):
    """
    解析划分文件，生成图像路径列表、标签列表、类别映射
    参数：split_file_path - 划分文件路径（如dataset_split/train_files.txt）
    返回：img_paths(图像路径列表), labels(数字标签列表), class_to_idx(类别→索引映射)
    """
    img_paths = []
    labels = []
    class_to_idx = {}  # 关键：确保所有集的类别映射一致（以训练集为准）
    
    with open(split_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 按\t分割“类别名”和“图像路径”（预处理时保存的格式）
            class_name, img_path = line.strip().split('\t')
            
            # 给新类别分配唯一索引（只在首次出现时分配）
            if class_name not in class_to_idx:
                class_to_idx[class_name] = len(class_to_idx)
            
            # 收集路径和标签
            img_paths.append(img_path)
            labels.append(class_to_idx[class_name])  # 转成数字标签
    
    return img_paths, labels, class_to_idx


'''这个类是 “图像文件→模型输入张量” 的转换器 -- 用来预处理,由DataLoader调用''' 
class ImageClassificationDataset(Dataset):
    def __init__(self, img_paths, labels, is_train=True):
        """
        参数：
            img_paths - 图像路径列表（从build_data_and_label获取）
            labels - 数字标签列表（与img_paths一一对应）
            is_train - 是否为训练集（训练集需数据增强，验证集不需要）
        """
        self.img_paths = img_paths
        self.labels = labels
        self.is_train = is_train
        
        # --------------- 关键：图像预处理（必须与模型适配）---------------
        if self.is_train:
            # 训练集：数据增强（提升泛化能力）+ 归一化
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 缩放到模型输入尺寸（你的模型默认224×224）
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
                transforms.RandomCrop(224, padding=16),  # 随机裁剪（避免过拟合）
                transforms.ToTensor(),  # 转成张量（HWC→CHW，值归一化到0-1）
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet均值（你的模型初始化用的这个）
                    std=[0.229, 0.224, 0.225]   # ImageNet标准差
                )
            ])
        else:
            # 验证/测试集：只做必要预处理（无数据增强，保证结果稳定）
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        # 返回数据集总样本数（DataLoader需要）
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 按索引加载单张图像和标签（DataLoader会批量调用这个方法）
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        
        try:
            # 加载图像（确保是RGB格式，模型输入是3通道）
            img = Image.open(img_path).convert('RGB')
            # 执行预处理
            img_tensor = self.transform(img)
            # 返回“图像张量+标签张量”（模型训练需要张量格式）
            return img_tensor, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            # 防止个别损坏图像中断训练，返回前一个有效样本（临时应急）
            logging.warning(f"加载图像 {img_path} 失败: {str(e)}，返回前一个样本")
            return self.__getitem__(max(0, idx-1))


'''dataloader在此'''

def build_dataloaders(train_file, val_file, batch_size=32, num_workers=4):
    """
    构建训练集和验证集的DataLoader
    参数：
        train_file/val_file - 训练/验证集划分文件路径
        batch_size - 批次大小（3080Ti建议16-32，显存够就设大）
        num_workers - 多线程数（服务器CPU核心多的话可设8）
    返回：train_loader, val_loader, num_classes（类别数）
    """
    # 1. 解析训练集数据，建立类别映射（以训练集为准！）
    train_img_paths, train_labels, class_to_idx = build_data_and_label(train_file)
    # 2. 解析验证集数据（用训练集的类别映射，避免标签不一致）
    val_img_paths, val_labels, _ = build_data_and_label(val_file)
    # 3. 计算类别数（模型输出层维度需要）
    num_classes = len(class_to_idx)
    logging.info(f"数据集类别数: {num_classes}，类别映射: {class_to_idx}")

    # 4. 创建Dataset实例
    train_dataset = ImageClassificationDataset(
        img_paths=train_img_paths, 
        labels=train_labels, 
        is_train=True  # 训练集启用数据增强
    )
    val_dataset = ImageClassificationDataset(
        img_paths=val_img_paths, 
        labels=val_labels, 
        is_train=False  # 验证集不启用数据增强
    )

    # 5. 创建DataLoader（核心：批量加载+打乱+多线程）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集必须打乱（保证泛化）
        num_workers=num_workers,
        pin_memory=True  # 锁定内存（GPU加速时更快，服务器必开）
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=num_workers,
        pin_memory=True
    )

    logging.info(f"DataLoader构建完成：训练集批次/轮次: {len(train_loader)}，验证集批次: {len(val_loader)}")
    return train_loader, val_loader, num_classes