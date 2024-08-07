import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from sklearn.model_selection import train_test_split

class SceneImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, train_ratio=0.9, debug=False):
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

        # 划分训练集和验证集
        # 打乱数据顺序
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)
        
        # 使用 train_test_split 进行随机划分
        self.train_image_paths, self.val_image_paths, self.train_labels, self.val_labels = train_test_split(
            self.image_paths, self.labels, test_size=1-train_ratio, random_state=42
        )

        if train:
            self.image_paths = self.train_image_paths
            self.labels = self.train_labels
        else:
            self.image_paths = self.val_image_paths
            self.labels = self.val_labels
                        


        # 在调试模式下限制加载的图像数量
        if debug:
            random.seed(42)
            # 随机选取若干个索引
            random_indices = random.sample(range(len(self.image_paths)), 30)
            # 根据选取的索引从两个列表中获取对应的元素
            selected_image_paths = [self.image_paths[i] for i in random_indices]
            selected_labels = [self.labels[i] for i in random_indices]
            self.image_paths = selected_image_paths
            self.labels = selected_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        #print(f"(getItem) lable:{label},image:{img_path}")
        return image, label