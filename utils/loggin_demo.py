'''工具模块 -- 日志'''
import logging
import os
from datetime import datetime

'''启动日志系统'''

def setup_logging():
    """配置日志系统"""
    # 创建日志文件夹（如果不存在）
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 日志文件名（包含当前日期）
    current_date = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"app_{current_date}.log")
    
    # 日志格式
    # 包含时间、日志级别、模块名、行号和日志消息
    log_format = "%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,  # 设置最低日志级别
        format=log_format,
        datefmt=date_format,
        handlers=[
            # 输出到控制台
            logging.StreamHandler(),
            # 输出到文件
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )

def demonstrate_logging():
    """演示不同级别的日志输出"""
    # 日志级别从低到高：DEBUG, INFO, WARNING, ERROR, CRITICAL
    logging.debug("这是DEBUG级别的日志 - 详细的调试信息")
    logging.info("这是INFO级别的日志 - 程序正常运行信息")
    logging.warning("这是WARNING级别的日志 - 潜在的问题")
    logging.error("这是ERROR级别的日志 - 发生错误但不影响程序继续运行")
    
    try:
        1 / 0
    except ZeroDivisionError:
        logging.critical("这是CRITICAL级别的日志 - 严重错误，可能导致程序终止", exc_info=True)
