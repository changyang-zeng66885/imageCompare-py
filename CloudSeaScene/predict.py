import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model import CNNClasification
import os

# 预测函数
def predict(image_path, model_path):
    # 加载模型
    model = CNNClasification()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置模型为评估模式

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) # Note。transform 操作会自动将pixel 值从 0-255 转换到 0-1
    # 这里transforms.Normalize 是将0-1 转换到-1到1 之间

    # 读取并处理图像
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # 添加 batch 维度

    # print(f"image size:{image.shape}")
    # image size:torch.Size([1, 3, 128, 128])

    # 进行预测
    with torch.no_grad():
        outputs = model(image)
        print(f"outputs:{outputs}")
        _, predicted = torch.max(outputs, 1)

    return predicted.item()  # 返回预测的类别

def grabCategoryNameDict(root_dir):
    # 收集所有图像路径和标签
    categoryNameDict = {}
    for label, class_name in enumerate(os.listdir(root_dir)):
        categoryNameDict[int(label)] = class_name
    return categoryNameDict

# 示例用法
if __name__ == "__main__":
    image_path = 'CloudSeaScene/images/cloudsea/15.jpg'  # 替换为实际图像路径
    model_path = 'CloudSeaScene/model_saved/CnnClasification_20240807_epoch5_acc09057.pth'  # 替换为实际模型路径
    root_dir = 'CloudSeaScene/images' # 图片路径
    labelDict = grabCategoryNameDict(root_dir) 
    predictResult = predict(image_path, model_path)
    print(f'Predict class: {labelDict[predictResult]}')