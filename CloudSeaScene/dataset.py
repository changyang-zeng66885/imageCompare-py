import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SceneImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, train_ratio=0.8, debug=False):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 收集所有图像路径和标签
        for label, class_name in enumerate(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(class_dir, filename))
                        self.labels.append(label)
                        print(f"(dataset init) lable:{label},image:{os.path.join(class_dir, filename)}")

        # 划分训练集和验证集
        total_size = len(self.image_paths)
        train_size = int(total_size * train_ratio)
        val_size = total_size - train_size

        if train:
            self.image_paths = self.image_paths[:train_size]
            self.labels = self.labels[:train_size]
        else:
            self.image_paths = self.image_paths[train_size:]
            self.labels = self.labels[train_size:]

        # 在调试模式下限制加载的图像数量
        if debug:
            self.image_paths = self.image_paths[:5]
            self.labels = self.labels[:5]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label