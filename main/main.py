import argparse
import sys
import os
from typing import List, Dict, Any
import logging

import torch  # 若已导入则无需重复，nn 依赖 torch
import torch.nn as nn  # 关键：导入 torch.nn 并简写为 nn

# 先获取main.py所在目录的父目录（即project根目录）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将项目根目录添加到sys.path
sys.path.append(project_root)

from utils import loggin_demo,data_preprocess,data_loader
from Self import train,model,infere

'''用于classname - txt 提取的工具函数'''
def get_specific_lines(file_path, line_numbers):
    """
    提取文件中指定行数的文本
    :param file_path: txt文件路径
    :param line_numbers: 要提取的行数列表（如[1, 3, 5]，注意：第1行对应索引0）
    :return: 包含指定行文本的列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()  # 读取所有行到列表，每行包含换行符\n
            # 处理每行，去除换行符，并按指定行数提取
            result = []
            for num in line_numbers:
                # 检查行数是否有效（索引不能超过总行数-1）
                if 0 <= num < len(lines):
                    # strip() 可去除换行符和前后空格，根据需要选择是否保留
                    result.append(lines[num].strip())
                else:
                    result.append(f"行数 {num+1} 超出文件范围（文件共{len(lines)}行）")
            return result
    except FileNotFoundError:
        return f"错误：文件 {file_path} 不存在"
    except Exception as e:
        return f"错误：{str(e)}"


def parse_arguments() -> argparse.Namespace:
    """
    解析终端传入的命令行参数
    
    返回:
        argparse.Namespace: 包含所有解析后参数的命名空间对象
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='终端参数提取模板',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # 显示默认值
    )
    
    parser.add_argument(            #模型路径
        '-f', '--filepath',
        type=str,
        default="None",
        help='指定模型权重文件夹'
    )

    parser.add_argument(            #训练集路径
        '-d', '--data_path',
        type=str,
        default="None",
        help='指定训练集'
    )

    parser.add_argument(            #训练集路径
        '-m', '--mode',
        type=str,
        default="None",
        help='模式 -- 训练或推理'
    )
    
    # 解析参数
    args = parser.parse_args()
    
    return args

'''-----------------------------------------------------------------------'''

def train_main(model_filepath_train:str , data_path:str ):   #训练模式分支方法

    '''超参数'''
    LEARNING_RATE = 2e-4    # 核心调整：从3e-5提升至2e-4
    WEIGHT_DECAY = 2e-4     # 微调：从3e-4降至2e-4
    NUM_EPOCHS = 120        # 调整：从150减少至120
    ''''''

    '''断点加载'''
    # 断点文件路径（即你要加载的 best_resnet.pth）        
    resume_path = model_filepath_train + "best_resnet.pth"

    if  os.path.exists(resume_path):

        logging.info("模型文件存在")
        # 加载断点数据
        checkpoint = torch.load(resume_path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        model_s = model.create_model(num_classes = 100).to("cuda") #CustomResNet模型
        # 恢复模型权重
        model_s.load_state_dict(checkpoint["model_state_dict"])
        criterion = nn.CrossEntropyLoss()  # 无需修改，与CustomResNet的forward输出兼容


        # 5.4 优化器（调用CustomResNet自带的get_optimizer方法）
        optimizer = model_s.get_optimizer(
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        # 恢复优化器状态（关键：确保优化器“记住”上次的动量、学习率等）
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # 5.5 学习率调度器（调用CustomResNet自带的get_scheduler方法）
        scheduler = model_s.get_scheduler(optimizer=optimizer)
    else:

        logging.info("模型文件不存在")

        model_s = model.create_model(num_classes = 100).to("cuda") #CustomResNet模型
        criterion = nn.CrossEntropyLoss()  # 无需修改，与CustomResNet的forward输出兼容

         # 5.4 优化器（调用CustomResNet自带的get_optimizer方法）
        optimizer = model_s.get_optimizer(
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        scheduler = model_s.get_scheduler(optimizer=optimizer)

    class_model = data_preprocess.data_perp(data_path)    #数据预处理,划分,校验

    total_params = sum(p.numel() for p in model_s.parameters())
    logging.info(f"总参数量: {total_params / 1e6:.2f} M")  # 单位：百万

    '''验证模型文件夹'''
    if not os.path.exists(model_filepath_train):
        logging.error(f"错误：路径 '{model_filepath_train}' 不存在,程序退出")  #校验路径
        sys.exit()

    if not os.path.isdir(model_filepath_train):
        logging.error(f"错误：'{model_filepath_train}' 不是一个文件夹")  #校验路径对象是否为文件夹
        sys.exit()
    logging.info(f"模型文件夹正确存在")

    '''提取划分文件'''
    split_save_dir = class_model["split_save_dir"]  # 对应划分文件（train_files.txt等）的保存目录

    # 3. 拼接训练集和验证集的划分文件路径
    train_file_path = os.path.join(split_save_dir, "train_files.txt")  # 训练集划分文件
    val_file_path = os.path.join(split_save_dir, "val_files.txt")      # 验证集划分文件
    '''提取划分文件'''

    #正确闯入参数以获取预处理的图片张量
    train_loader, val_loader, num_classes = data_loader.build_dataloaders(
        train_file=train_file_path,  # 传入训练集划分文件路径
        val_file=val_file_path,      # 传入验证集划分文件路径
        batch_size=11,               # 可根据你的GPU显存调整
        num_workers=32                # 可根据服务器CPU核心数调整
    )

    #开始训练
    train.train_model(

        train_loader = train_loader,
        val_loader = val_loader,

        # 2. 模型与损失函数（从model.py初始化）
        model=model_s,
        criterion=criterion,
        
        # 3. 优化器与调度器（从CustomResNet方法获取）
        optimizer=optimizer,
        scheduler=scheduler,
        
        # 4. 训练配置（自定义参数）
        num_epochs=NUM_EPOCHS,
        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        save_path=model_filepath_train+"best_resnet.pth",
        log_interval=10,  # 每10个批次打印一次训练进度
        best_metric="val_acc",  # 按验证准确率判定最佳模型
        metric_higher_better=True  # 准确率越高越好

    )





'''-----------------------------------------------------------------------'''

def infere_main(model_path:str):    #推理模式模式分支方法

    classifier = infere.ImageClassifier(
        model_path=model_path + "best_resnet.pth",  # 相对路径（推荐，适配不同环境）
        # 或绝对路径：model_path="/home/xpr/project/model/best_resnet.pth"
        num_classes=100
    )

    '''获取类别编号映射'''
    img_paths, labels, class_to_idx = data_loader.build_data_and_label("/home/xpr/project/main/dataset_split/train_files.txt")  #获取类别编号映射

    while True:
        file_image = input("请您输入图片文件路径 -- 快速测试")

        result = classifier.predict(file_image)

        if result is not None:  # 确保预测成功（非None）
            predicted_class = result['predicted_class']  # 预测的类别（整数）
            confidence = result['confidence']            # 置信度（0~1之间的浮点数）
            all_probabilities = result['all_probabilities']  # 所有类别的概率列表（长度=类别数）

            # 关键：反转映射，得到 {索引: 类别名}
            idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
            predicted_class_name = idx_to_class[predicted_class]
            
            # 打印提取的结果
            print(f"预测类别: {predicted_class}")
            print(f"类别名称{predicted_class_name}")
            print(f"置信度: {confidence:.4f}（即 {confidence*100:.2f}%）")
            print(f"前5类概率: {all_probabilities[:5]}...")  # 示例：打印前5类概率
        else:
            print(f"图像 {image_path} 预测失败")


if __name__ == "__main__":    #主逻辑
    loggin_demo.setup_logging() #初始化日志
    args = parse_arguments()

    '''校验参数是否完整'''
    if args.filepath == "None" or args.data_path == "None" or args.mode == "None":
         logging.error(f"警告参数不完整,退出")    #报错
         sys.exit()

    logging.info(f"模型权重目录: {args.filepath}")
    logging.info(f"训练集目录: {args.data_path}")

    logging.info(f"程序启动")

    if args.mode == "train":

        logging.info(f"训练模式")
        train_main(model_filepath_train=args.filepath, data_path=args.data_path)

    elif args.mode == "infere":

        logging.info(f"推理模式")
        infere_main(args.filepath)
    
    else:
        logging.error(f"mode参数错误")
        sys.exit()

    