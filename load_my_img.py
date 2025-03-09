#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
        自定义dataset，加载图片数据
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = []
        self.label_list = []

        # 遍历每一个类别文件夹
        for label in os.listdir(root_dir):
            print("process dir label {}".format(label))
            class_dir = os.path.join(root_dir, label)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    self.image_list.append(os.path.join(class_dir, img_name))
                    self.label_list.append(label)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.label_list[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label
        
import torchvision.transforms as transforms

# 对图像进行一些预处理和数据增强， 便于后期使用
my_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
    transforms.Resize((128, 128)),      # Resize to 128x128
    transforms.ToTensor(),              # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize, 对图像进行归一化处理，即减去均值并除以标准差
]) 



from torch.utils.data import DataLoader

# 创建 CustomDataset 实例，加载训练数据
dir="./my_dataset/train"
print("load my dataset in {}".format(dir))
train_dataset = CustomDataset(root_dir=dir, transform=my_transform)
# 创建 DataLoader 实例
print("build my data loader")
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
# 迭代加载数据
for images, labels in train_loader:
    # 在这里可以对 images 和 labels 进行处理
    print(images.shape, labels)
    
    
