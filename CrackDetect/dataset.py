import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# 自定义数据集类
class CrackDataset(Dataset):
    def __init__(self, mode,image_dir, mask_dir,transform=None, test_size=0.2, random_state=42,debug=False):
        self.mode = mode
        self.transform = transform
        self.test_size = test_size
        self.random_state = random_state
        self.debug = debug

        # self.image_dir = 'CrackDetect/imageSrc/images'
        # self.mask_dir = 'CrackDetect/imageSrc/masks'
        # self.image_dir = 'imageCompare-py/CrackDetect/bailushuyuan/images'
        # self.mask_dir = 'imageCompare-py/CrackDetect/bailushuyuan/labeled_images'
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.all_image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        self.all_mask_files = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.png')])

        if len(self.all_image_files) != len(self.all_mask_files):
            raise ValueError("Number of images and masks must be equal.")

        # Split the data into train and test sets
        self.train_image_files, self.test_image_files, self.train_mask_files, self.test_mask_files = train_test_split(
            self.all_image_files, self.all_mask_files, test_size=self.test_size, random_state=self.random_state
        )

        if self.mode == 'train':
            self.image_files = self.train_image_files
            self.mask_files = self.train_mask_files
        elif self.mode == 'val':
            self.image_files = self.test_image_files
            self.mask_files = self.test_mask_files
        else:
            raise ValueError("Invalid mode. Must be 'train' or 'val'.")

        # Limit the number of samples in debug mode
        if self.debug:
            self.image_files = self.image_files[:10]
            self.mask_files = self.mask_files[:10]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])
        mask_path = os.path.join(self.mask_dir, self.mask_files[index])

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask