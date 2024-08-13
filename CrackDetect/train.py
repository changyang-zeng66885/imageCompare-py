import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from model import UNet
from dataset import CrackDataset
from tqdm import tqdm
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb # 训练过程可视化

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)

num_epochs = 1
batch_size = 8
learning_rate = 0.005
debug = True # 在 debug 模式下，只会加载前10条数据（方便调试）

train_dataset = CrackDataset(
    mode='train',
    image_dir='CrackDetect/imageSrc/images',
    mask_dir='CrackDetect/imageSrc/masks',
    transform=transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()]),
    debug=debug
)
val_dataset = CrackDataset(
    mode='val',
    image_dir='CrackDetect/imageSrc/images',
    mask_dir='CrackDetect/imageSrc/masks',
    transform=transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()]),
    debug=debug
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Model, loss function, and optimizer
model = UNet(in_channels=3, out_channels=1).to(device)
# model.load_state_dict(torch.load("CrackDetect/model_saved/best_model_imagesrcBig_20240808_epoch_30.pth")) #加载保存的模型参数
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 创建 ReduceLROnPlateau 调度器
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.01, verbose=True)

wandb.init(
    # set the wandb project where this run will be logged
    project="CrackDetectByUnet-demo",
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "CNN",
    "dataset": "CrackDataset",
    "epochs": num_epochs,
    }
)

# Training loop
best_val_loss = float('inf')
print("Begin Train")
for epoch in range(num_epochs):
    print(f"Epoch {epoch}/{num_epochs}")
    # Train
    model.train()
    train_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # print(f"Train loss: {train_loss}")

    train_loss /= len(train_loader)
    scheduler.step(train_loss)

    # Validate
    model.eval()
    val_loss = 0.0
    all_outputs = []
    all_masks = []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)

            loss = criterion(outputs, masks)
            val_loss += loss.item()

            # Convert logits to probabilities
            probabilities = torch.sigmoid(outputs)
            # Flatten tensors and collect results
            all_outputs.append(probabilities.cpu().numpy().flatten())
            all_masks.append(masks.cpu().numpy().flatten())

    val_loss /= len(val_loader)

    # Concatenate all results
    all_outputs = np.concatenate(all_outputs)
    all_masks = np.concatenate(all_masks)
    # Calculate AUC score
    auc_score = roc_auc_score(all_masks, all_outputs)

    # Print training and validation loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} ,AUC: {auc_score:.4f}')
    wandb.log({"Train Loss": train_loss, "Val Loss": val_loss,'AUC':auc_score})

saveModelPath = f'CrackDetect/model_saved/best_model_imagesrcBig_epoch_{num_epochs}.pth'
torch.save(model.state_dict(),saveModelPath )
print('Saved best model at: '+ saveModelPath )

wandb.finish()


