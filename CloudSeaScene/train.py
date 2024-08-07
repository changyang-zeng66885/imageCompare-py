import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from dataset import SceneImageDataset
from model import CNNClasification

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 创建数据集
data_dir = 'dataset'
debug = True # 在debug模式为true的情况下，每个类最多加载5张图片,这样能方便调试程序。
# 创建训练集和验证集
train_dataset = SceneImageDataset(root_dir=data_dir, transform=transform, train=True, debug=debug)
val_dataset = SceneImageDataset(root_dir=data_dir, transform=transform, train=False, debug=debug)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 初始化模型、损失函数和优化器
model = CNNClasification()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 验证模型
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

# 保存模型
modelSavePath = f"CnnClasification_20240807_epoch{num_epochs}.pth"
torch.save(model.state_dict(), modelSavePath)
print(f"训练完成，model saved at {modelSavePath}")