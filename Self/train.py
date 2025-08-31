import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import os
import logging 
#from Self import create_model
from typing import Optional, Dict, Any  # 增加类型提示，提升可维护性

def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """训练一个epoch"""
    model.train()  # 切换到训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 混合精度训练
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # 反向传播和优化
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 统计损失和准确率
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """在验证集上评估"""
    model.eval()  # 切换到评估模式
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 关闭梯度计算
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast():  # 混合精度评估
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

'''主训练入口'''

def train_model(
    train_loader: DataLoader,          # 外部传入的训练集DataLoader（核心：不再内部加载）
    val_loader: DataLoader,            # 外部传入的验证集DataLoader
    model: nn.Module,                  # 外部传入的模型实例（支持任意符合规范的模型）
    criterion: nn.Module,              # 外部传入的损失函数（灵活适配不同任务）
    optimizer: torch.optim.Optimizer,  # 外部传入的优化器
    num_epochs: int = 50,              # 训练轮次（默认值保留，可外部调整）
    device: Optional[torch.device] = None,  # 设备（外部指定，支持CPU/GPU/多GPU）
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,  # 外部传入调度器
    scaler: Optional[GradScaler] = None,  # 混合精度缩放器（外部传入，支持自定义配置）
    save_path: str = "best_model.pth",  # 最佳模型保存路径
    log_interval: int = 10,            # 日志打印间隔（每多少批次打印一次训练进度）
    best_metric: str = "val_acc",      # 最优模型判定指标（支持"val_acc"准确率/"val_loss"损失）
    metric_higher_better: bool = True  # 指标是否“越高越好”（如acc越高越好，loss越低越好）
) -> Dict[str, Any]:
    """
    可伸缩性强的通用训练函数
    核心特点：
    1. 数据加载完全外部化：通过train_loader/val_loader参数传入，支持任意数据来源
    2. 核心组件可替换：模型、损失函数、优化器、调度器均由外部传入，适配不同任务
    3. 配置灵活：支持自定义设备、日志间隔、最优指标判定规则
    4. 完整返回训练记录：便于后续分析和可视化
    
    参数说明：
    - train_loader/val_loader: 训练/验证数据加载器，需返回 (input_tensor, label_tensor)
    - model: 待训练的模型（需已移动到指定device）
    - criterion: 损失函数（如CrossEntropyLoss、MSELoss等）
    - optimizer: 优化器（如AdamW、SGD等）
    - num_epochs: 总训练轮次
    - device: 训练设备（如torch.device("cuda:0")），默认自动检测
    - scheduler: 学习率调度器（如ReduceLROnPlateau、StepLR），可选
    - scaler: 混合精度训练缩放器，可选（默认自动创建）
    - save_path: 最佳模型保存路径
    - log_interval: 每多少个训练批次打印一次进度（如10表示每10批打印一次）
    - best_metric: 判定“最佳模型”的指标（"val_acc"或"val_loss"）
    - metric_higher_better: 指标是否“越高越好”（acc是，loss否）
    
    返回值：
    - 训练记录字典：包含每轮的训练/验证损失、准确率，以及最佳模型信息
    """
    # 1. 初始化默认参数（外部未传入时提供合理默认值，保证易用性）
    # 自动检测设备（外部未指定时）
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"训练设备: {device.type} (设备ID: {device.index if device.index is not None else 'N/A'})")
    
    # 自动创建混合精度缩放器（外部未传入时）
    if scaler is None:
        scaler = GradScaler() if device.type == "cuda" else None
        logging.info(f"自动初始化混合精度缩放器: {'启用' if scaler is not None else '禁用（非GPU环境）'}")
    
    # 2. 初始化训练记录（便于后续分析）
    training_history = {
        "train_loss": [],    # 每轮训练损失
        "train_acc": [],     # 每轮训练准确率
        "val_loss": [],      # 每轮验证损失
        "val_acc": [],       # 每轮验证准确率
        "lr": [],            # 每轮学习率（便于调试）
        "best_model": {
            "epoch": 0,
            "metric_value": 0.0,
            "save_path": save_path
        }
    }
    
    # 3. 初始化最佳指标（根据metric_higher_better设置初始值）
    if best_metric == "val_acc":
        best_metric_value = 0.0 if metric_higher_better else 1.0
    elif best_metric == "val_loss":
        best_metric_value = float("inf") if metric_higher_better is False else -float("inf")
    else:
        raise ValueError(f"不支持的最佳指标: {best_metric}，仅支持'val_acc'或'val_loss'")
    
    # 4. 核心训练循环（每轮epoch）
    for epoch in range(num_epochs):
        epoch_idx = epoch + 1  # 轮次从1开始计数（更符合直觉）
        logging.info(f"\n===== 训练轮次 {epoch_idx}/{num_epochs} =====")
        
        # -------------------------- 训练阶段 --------------------------
        model.train()  # 切换模型到训练模式（启用dropout、BN更新等）
        train_total = 0
        train_correct = 0
        train_running_loss = 0.0
        
        # 遍历训练集批次（外部传入的train_loader，内部不关心数据来源）
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch_idx} - Training")):
            # 数据移动到指定设备（确保与模型设备一致）
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 混合精度训练（仅GPU环境启用）
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(inputs)  # 模型前向传播（外部传入的模型，内部不关心结构）
                loss = criterion(outputs, labels)  # 计算损失（外部传入的损失函数）
            
            # 反向传播与参数更新
            if scaler is not None:
                scaler.scale(loss).backward()  # 缩放损失，避免梯度下溢
                scaler.step(optimizer)         # 更新参数
                scaler.update()                # 调整缩放器
            else:
                loss.backward()  # 普通反向传播
                optimizer.step() # 更新参数
            
            # 统计训练损失和准确率
            batch_size = inputs.size(0)
            train_running_loss += loss.item() * batch_size
            train_total += batch_size
            
            # 计算准确率（仅分类任务，若为回归任务可注释此部分）
            if outputs.ndim == 2 and outputs.size(1) > 1:  # 分类任务：输出为类别得分
                _, predicted = torch.max(outputs, 1)
                train_correct += predicted.eq(labels).sum().item()
            
            # 按间隔打印批次进度（避免日志刷屏）
            if (batch_idx + 1) % log_interval == 0:
                batch_loss = loss.item()
                batch_acc = predicted.eq(labels).sum().item() / batch_size if outputs.ndim == 2 else "N/A"
                logging.info(f"批次 {batch_idx+1}/{len(train_loader)} | 批次损失: {batch_loss:.4f} | 批次准确率: {batch_acc:.4f}")
        
        # 计算本轮训练的平均损失和准确率
        train_avg_loss = train_running_loss / train_total
        train_avg_acc = train_correct / train_total if train_total > 0 else 0.0
        training_history["train_loss"].append(train_avg_loss)
        training_history["train_acc"].append(train_avg_acc)
        logging.info(f"\n【训练轮次 {epoch_idx}】平均损失: {train_avg_loss:.4f} | 平均准确率: {train_avg_acc:.4f}")
        
        # 记录当前学习率（便于调试学习率策略）
        current_lr = optimizer.param_groups[0]["lr"]
        training_history["lr"].append(current_lr)
        logging.info(f"当前学习率: {current_lr:.6f}")
        
        # -------------------------- 验证阶段 --------------------------
        model.eval()  # 切换模型到评估模式（禁用dropout、固定BN）
        val_total = 0
        val_correct = 0
        val_running_loss = 0.0
        
        with torch.no_grad():  # 关闭梯度计算，节省内存和时间
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch_idx} - Validating"):
                # 数据移动到指定设备
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                # 混合精度评估
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # 统计验证损失和准确率
                batch_size = inputs.size(0)
                val_running_loss += loss.item() * batch_size
                val_total += batch_size
                
                # 计算准确率（仅分类任务）
                if outputs.ndim == 2 and outputs.size(1) > 1:
                    _, predicted = torch.max(outputs, 1)
                    val_correct += predicted.eq(labels).sum().item()
        
        # 计算本轮验证的平均损失和准确率
        val_avg_loss = val_running_loss / val_total
        val_avg_acc = val_correct / val_total if val_total > 0 else 0.0
        training_history["val_loss"].append(val_avg_loss)
        training_history["val_acc"].append(val_avg_acc)
        logging.info(f"【验证轮次 {epoch_idx}】平均损失: {val_avg_loss:.4f} | 平均准确率: {val_avg_acc:.4f}")
        
        # -------------------------- 学习率调度 --------------------------
        if scheduler is not None:
            # 支持两种调度器逻辑：基于验证损失的（如ReduceLROnPlateau）和基于轮次的（如StepLR）
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_avg_loss)  # 基于验证损失调度
            else:
                scheduler.step()  # 基于轮次调度
            logging.info(f"学习率调度后: {optimizer.param_groups[0]['lr']:.6f}")
        
        # -------------------------- 保存最佳模型 --------------------------
        # 判断是否更新最佳模型（根据指定的metric和高低规则）
        update_best = False
        if best_metric == "val_acc":
            if (metric_higher_better and val_avg_acc > best_metric_value) or (not metric_higher_better and val_avg_acc < best_metric_value):
                update_best = True
        elif best_metric == "val_loss":
            if (metric_higher_better and val_avg_loss > best_metric_value) or (not metric_higher_better and val_avg_loss < best_metric_value):
                update_best = True
        
        if update_best:
            best_metric_value = val_avg_acc if best_metric == "val_acc" else val_avg_loss
            # 保存模型（包含模型参数、优化器参数、训练状态，支持断点续训）
            torch.save({
                "epoch": epoch_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "best_metric": best_metric,
                "best_metric_value": best_metric_value,
                "training_history": training_history
            }, save_path)
            # 更新训练记录中的最佳模型信息
            training_history["best_model"] = {
                "epoch": epoch_idx,
                "metric": best_metric,
                "metric_value": best_metric_value,
                "save_path": save_path
            }
            logging.info(f"✅ 保存最佳模型到 {save_path} | 最佳{best_metric}: {best_metric_value:.4f}")
    
    # -------------------------- 训练结束 --------------------------
    logging.info(f"\n===== 训练完成 ======")
    logging.info(f"最佳模型轮次: {training_history['best_model']['epoch']}")
    logging.info(f"最佳{training_history['best_model']['metric']}: {training_history['best_model']['metric_value']:.4f}")
    logging.info(f"最佳模型路径: {training_history['best_model']['save_path']}")
    
    # 返回完整训练记录，便于后续分析（如绘制损失曲线、准确率曲线）