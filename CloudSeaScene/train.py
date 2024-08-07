import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from dataset import SceneImageDataset
from model import CNNClasification
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb # 训练过程可视化

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 创建数据集
data_dir = 'CloudSeaScene/images'
debug = False # 在debug模式为true的情况下，最多加载30张图片,这样能方便调试程序。
# 创建训练集和验证集
train_dataset = SceneImageDataset(root_dir=data_dir, transform=transform, train=True, debug=debug)
val_dataset = SceneImageDataset(root_dir=data_dir, transform=transform, train=False, debug=debug)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 初始化模型、损失函数和优化器
learning_rate = 0.001
num_epochs = 5
model = CNNClasification()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 创建 ReduceLROnPlateau 调度器
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.01, verbose=True)

wandb.init(
    # set the wandb project where this run will be logged
    project="SceneImageClassification-demo",
    config={
    "learning_rate": learning_rate,
    "architecture": "CNNClasification",
    "dataset": "SceneImageDataset",
    "epochs": num_epochs,
    }
)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)  
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        optimizer.zero_grad()

    # 验证模型
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 计算准确度
            _, predicted = torch.max(outputs.data, 1)            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.2f}%')
    wandb.log({"Epoch":epoch + 1,"Total Epoch":num_epochs, "Train Loss": train_loss, "Val Loss": val_loss, "Val Accuracy":accuracy})

# 保存模型
modelSavePath = f"CloudSeaScene/model_saved/CnnClasification_20240807_epoch{num_epochs}.pth"
torch.save(model.state_dict(), modelSavePath)
print(f"训练完成，model saved at {modelSavePath}")