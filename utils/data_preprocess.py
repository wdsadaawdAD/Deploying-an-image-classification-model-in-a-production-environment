'''工具模块 -- 数据预处理 -- 加载模块
适配无GUI的Ubuntu服务器环境，去除可视化依赖
核心功能：路径校验、文件信息统计、图像有效性验证、数据集划分
'''
import logging
import sys
import os
import random
from datetime import datetime
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


def get_file_info(folder_path):       
    """工具函数：获取文件夹中所有文件的详细信息"""
    file_info_list = []
    # 处理路径可能存在的特殊字符（服务器环境常见问题）
    folder_path = os.path.abspath(folder_path)
    
    try:
        # 遍历文件夹（处理服务器环境下可能的权限问题）
        for entry in os.listdir(folder_path):
            entry_path = os.path.join(folder_path, entry)
            # 只处理文件，跳过子文件夹和隐藏文件（服务器常见隐藏文件如 .DS_Store）
            if os.path.isfile(entry_path) and not entry.startswith('.'):
                try:
                    file_stats = os.stat(entry_path)
                    # 整理文件核心信息（服务器场景无需冗余字段）
                    file_info = {
                        "文件名": entry,
                        "绝对路径": entry_path,
                        "大小(KB)": round(file_stats.st_size / 1024, 2),
                        "修改时间": datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                        "文件后缀": os.path.splitext(entry)[1].lower() if os.path.splitext(entry)[1] else "无后缀"
                    }
                    file_info_list.append(file_info)
                except PermissionError:
                    logging.warning(f"无权限访问文件: {entry}（跳过）")
                except Exception as e:
                    logging.error(f"处理文件 '{entry}' 时出错: {str(e)}（跳过）")
    except PermissionError:
        logging.error(f"无权限访问文件夹: {folder_path}（程序退出）")
        sys.exit(1)
    except FileNotFoundError:
        logging.error(f"文件夹 '{folder_path}' 不存在（程序退出）")
        sys.exit(1)
    
    return file_info_list

def is_valid_image(file_path):
    """服务器环境专用：验证图像文件有效性（轻量无GUI）"""
    # 只保留模型支持的图像格式（减少无效文件处理）
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # 先通过后缀快速过滤（服务器批量处理效率优先）
    if file_ext not in valid_extensions:
        return False, "不支持的格式"
    
    # 轻量验证图像完整性（避免服务器内存占用过高）
    try:
        with Image.open(file_path) as img:
            # 仅验证图像模式和尺寸（不加载完整像素，节省内存）
            if img.mode not in ['RGB', 'L']:  # 只支持RGB彩色和L灰度图（模型输入要求）
                return False, f"不支持的图像模式: {img.mode}"
            if img.size[0] < 32 or img.size[1] < 32:  # 过滤过小图像（无训练意义）
                return False, f"图像尺寸过小: {img.size}"
        return True, "有效图像"
    except Exception as e:
        # 服务器环境简化错误信息（避免冗余输出）
        return False, "文件损坏或无法解析"

def analyze_image_stats(image_paths):
    """服务器环境专用：图像统计分析（文本输出，无图形）"""
    if not image_paths:
        logging.warning("无有效图像可分析")
        return None
    
    # 统计核心维度（服务器场景关注实用信息）
    sizes = []
    modes = {}
    extensions = {}
    
    for path in image_paths:
        try:
            with Image.open(path) as img:
                sizes.append(img.size)
                # 统计图像模式分布
                mode = img.mode
                modes[mode] = modes.get(mode, 0) + 1
                # 统计格式分布
                ext = os.path.splitext(path)[1].lower()
                extensions[ext] = extensions.get(ext, 0) + 1
        except Exception:
            continue  # 跳过无效图像，不中断统计
    
    if not sizes:
        return None
    
    # 计算尺寸统计（为模型输入尺寸提供参考）
    widths, heights = zip(*sizes)
    stats = {
        "图像总数": len(sizes),
        "平均尺寸": (round(np.mean(widths), 1), round(np.mean(heights), 1)),
        "尺寸范围": f"宽 [{min(widths)}-{max(widths)}]px, 高 [{min(heights)}-{max(heights)}]px",
        "模式分布": modes,
        "格式分布": extensions
    }
    
    # 服务器友好的文本输出（清晰易读，适合终端查看）
    logging.info("\n=== 图像数据集统计报告 ===")
    for key, value in stats.items():
        logging.info(f"{key}: {value}")
    logging.info("==========================\n")
    
    return stats

def split_dataset(class_files, val_ratio=0.2, test_ratio=0.1, random_state=42):
    """服务器环境专用：数据集划分（按类别保持比例，支持大文件量）"""
    # 服务器环境固定随机种子（确保划分结果可复现，便于后续调试）
    random.seed(random_state)
    np.random.seed(random_state)
    
    split_result = {'train': {}, 'val': {}, 'test': {}}
    total_train = 0
    total_val = 0
    total_test = 0
    
    logging.info("\n=== 数据集划分结果 ===")
    for class_name, file_paths in class_files.items():
        # 跳过样本数过少的类别（避免划分后某集为空）
        if len(file_paths) < 5:
            logging.warning(f"类别 '{class_name}' 样本数不足5个（跳过划分，全部归入训练集）")
            split_result['train'][class_name] = file_paths
            total_train += len(file_paths)
            continue
        
        # 分两步划分：先分训练集和临时集，再分验证集和测试集（保证比例准确）
        train_paths, temp_paths = train_test_split(
            file_paths,
            test_size=val_ratio + test_ratio,
            shuffle=True,
            random_state=random_state
        )
        val_paths, test_paths = train_test_split(
            temp_paths,
            test_size=test_ratio / (val_ratio + test_ratio),
            shuffle=True,
            random_state=random_state
        )
        
        # 记录划分结果
        split_result['train'][class_name] = train_paths
        split_result['val'][class_name] = val_paths
        split_result['test'][class_name] = test_paths
        
        # 累计总数并输出单类别划分结果
        total_train += len(train_paths)
        total_val += len(val_paths)
        total_test += len(test_paths)
        logging.info(f"类别 '{class_name}': 训练集{len(train_paths)} | 验证集{len(val_paths)} | 测试集{len(test_paths)}")
    
    # 输出总统计
    logging.info(f"总计: 训练集{total_train} | 验证集{total_val} | 测试集{total_test}")
    logging.info("======================\n")
    
    return split_result

def save_split_indices(split_result, save_dir="dataset_split", overwrite=False):
    """服务器环境专用：保存划分结果（文本文件，便于后续加载）"""
    # 处理服务器路径（确保保存目录在当前工作目录下，避免权限问题）
    save_dir = os.path.join(os.getcwd(), save_dir)
    
    # 处理目录存在情况（服务器环境避免误删数据）
    if os.path.exists(save_dir):
        if overwrite:
            logging.warning(f"划分结果目录 '{save_dir}' 已存在（将覆盖内容）")
        else:
            logging.error(f"划分结果目录 '{save_dir}' 已存在（避免覆盖，程序退出）")
            return save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存每个集的文件路径（按类别记录，便于后续DataLoader加载）
    for split_type, class_data in split_result.items():
        file_path = os.path.join(save_dir, f"{split_type}_files.txt")
        # 服务器环境用UTF-8编码（避免中文路径乱码）
        with open(file_path, 'w', encoding='utf-8') as f:
            # 写入格式：类别名\t文件绝对路径（便于后续解析）
            for class_name, paths in class_data.items():
                for path in paths:
                    f.write(f"{class_name}\t{path}\n")
        logging.info(f"{split_type}集划分结果已保存至: {file_path}")
    
    return save_dir

def data_perp(data_path:str, val_ratio=0.2, test_ratio=0.1, save_split=True):
    """
    服务器环境数据预处理主函数（无GUI，批量处理友好）
    参数：
        data_path: 数据集根路径（要求一级子文件夹为类别）
        val_ratio: 验证集比例（默认0.2）
        test_ratio: 测试集比例（默认0.1）
        save_split: 是否保存划分结果（默认True）
    返回：
        预处理结果字典（包含类别、划分路径等）
    """
    # 1. 基础路径校验（服务器环境优先处理路径问题）
    data_path = os.path.abspath(data_path)
    if not os.path.exists(data_path):
        logging.error(f"错误：数据集路径 '{data_path}' 不存在（程序退出）")
        sys.exit(1)
    if not os.path.isdir(data_path):
        logging.error(f"错误：'{data_path}' 不是文件夹（程序退出）")
        sys.exit(1)
    
    # 2. 获取类别（服务器环境默认一级子文件夹为类别，符合图像分类常规结构）
    class_names = [
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d)) and not d.startswith('.')
    ]
    if not class_names:
        logging.error(f"错误：在 '{data_path}' 中未找到任何类别子文件夹（程序退出）")
        sys.exit(1)
    logging.info(f"发现 {len(class_names)} 个类别: {', '.join(class_names)}")
    
    # 3. 遍历每个类别，收集有效图像
    class_files = {}
    all_valid_images = []
    invalid_count = 0
    
    for class_name in class_names:
        class_path = os.path.join(data_path, class_name)
        # 获取该类别下所有文件信息
        file_info_list = get_file_info(class_path)
        if not file_info_list:
            logging.warning(f"类别 '{class_name}' 下无有效文件（跳过）")
            continue
        
        # 筛选有效图像
        valid_images = []
        for file_info in file_info_list:
            file_path = file_info["绝对路径"]
            is_valid, msg = is_valid_image(file_path)
            if is_valid:
                valid_images.append(file_path)
                all_valid_images.append(file_path)
            else:
                invalid_count += 1
                # 服务器环境减少冗余日志，每100个无效文件才提示一次
                if invalid_count % 100 == 0:
                    logging.warning(f"已发现 {invalid_count} 个无效文件（示例：{os.path.basename(file_path)} - {msg}）")
        
        class_files[class_name] = valid_images
        logging.info(f"类别 '{class_name}': 有效图像 {len(valid_images)} 张（原始文件 {len(file_info_list)} 个）")
    
    # 4. 输出无效文件统计（服务器环境需明确数据质量）
    if invalid_count > 0:
        logging.info(f"\n总计发现 {invalid_count} 个无效文件（已过滤）")
    if not all_valid_images:
        logging.error("错误：无任何有效图像数据（程序退出）")
        sys.exit(1)
    
    # 5. 图像统计分析（文本输出，无GUI）
    analyze_image_stats(all_valid_images)
    
    # 6. 数据集划分（按类别保持比例）
    split_result = split_dataset(class_files, val_ratio=val_ratio, test_ratio=test_ratio)
    
    # 7. 保存划分结果（服务器环境便于后续复用）
    save_dir = None
    if save_split:
        save_dir = save_split_indices(split_result)
    
    # 返回预处理结果（便于后续训练脚本调用）
    return {
        "class_names": class_names,
        "num_classes": len(class_names),
        "split_result": split_result,
        "split_save_dir": save_dir,
        "total_valid_images": len(all_valid_images)
    }

