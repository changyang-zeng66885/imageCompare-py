import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CNNClasification(nn.Module):
    def __init__(self):
        super(CNNClasification, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # 根据输入图像的大小调整
        self.fc2 = nn.Linear(128, 4)  # 4个类别

    def forward(self, x):
        x = self.pool(nn.ReLU(self.conv1(x)))
        x = self.pool(nn.ReLU(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = nn.ReLU(self.fc1(x))
        x = self.fc2(x)
        return x