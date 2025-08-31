import torch
from PIL import Image
import torchvision.transforms as transforms
#import model
import os
import sys
import numpy as np
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到 Python 搜索路径
sys.path.append(current_dir)
import model

class ImageClassifier:
    def __init__(self, model_path, num_classes, device=None):
        """初始化分类器"""
        self.device = device or (torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.num_classes = num_classes
        
        # 创建模型并加载权重
        self.model = model.create_model(num_classes=num_classes).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # 确保模型处于评估模式
        
        # 定义图像预处理（需与训练时保持一致）
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet均值
                                 std=[0.229, 0.224, 0.225])   # ImageNet标准差
        ])
    
    def predict(self, image_path):
        """预测单张图像的类别"""
        # 加载并预处理图像
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)  # 添加批次维度
        image = image.to(self.device)
        
        # 推理
        probabilities = self.model.inference(image)
        
        # 解析结果
        probabilities = probabilities.cpu().numpy()[0]
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': probabilities.tolist()
        }
    
    def batch_predict(self, image_paths):
        """批量预测图像类别"""
        # 加载并预处理所有图像
        images = []
        for path in image_paths:
            image = Image.open(path).convert('RGB')
            image = self.transform(image)
            images.append(image)
        
        # 堆叠成批次
        batch = torch.stack(images).to(self.device)
        
        # 推理
        probabilities = self.model.inference(batch)
        
        # 解析结果
        probabilities = probabilities.cpu().numpy()
        predicted_classes = np.argmax(probabilities, axis=1).tolist()
        confidences = np.max(probabilities, axis=1).tolist()
        
        return [
            {
                'image_path': path,
                'predicted_class': cls,
                'confidence': conf,
                'all_probabilities': prob.tolist()
            }
            for path, cls, conf, prob in zip(image_paths, predicted_classes, confidences, probabilities)
        ]
