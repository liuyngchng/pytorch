#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor


class CustomDataset(Dataset):
    """
        自定义dataset，加载本地图片数据作为训练的样本数据
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.image_list = []
        self.label_list = []

        # 遍历每一个类别文件夹
        for label in os.listdir(root_dir):
            print("process dir label {}".format(label))
            class_dir = os.path.join(root_dir, label)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    # add image file 
                    self.image_list.append(os.path.join(class_dir, img_name))
                    # add label to image, must be number, not string, remember it, string can't be the element in a matrix
                    # self.label_list.append(label)
                    self.label_list.append(self.class_to_idx[label])  # 存储数值型标签

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.label_list[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class NeuralNetwork(nn.Module):
    """
    # Define a neural network as subclass of nn.Module used to train
    """

    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)


def train(dataloader, model, loss_fn, optimizer):
    """
    train the model use data from dataloader,
    use loss_fn as loss function
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # 在train函数内添加
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == "__main__":

def train_my_dt():
    train_img_dir = "./my_dataset/train"
    print("start load my dataset in {}".format(train_img_dir))
    # 对图像进行一些预处理和数据增强， 便于后期使用
    my_transform = transforms.Compose([
        transforms.RandomResizedCrop(128),  # 随机裁剪缩放
        transforms.ColorJitter(0.4, 0.4, 0.4),  # 颜色抖动
        transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
        transforms.Resize((128, 128)),  # Resize to 128x128
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize, 对图像进行归一化处理，即减去均值并除以标准差
    ])
    # 创建 CustomDataset 实例，加载训练数据
    train_dataset = CustomDataset(root_dir=train_img_dir, transform=my_transform)
    # 创建 DataLoader 实例
    print("build my data loader")
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    # to check data loaded to train_loader
    for images, labels in train_data_loader:
        print(images.shape, labels)

    for X, y in train_data_loader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"y: {y}")
        break

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    print("move neural network instance to the device")
    model = NeuralNetwork().to(device)
    print("model is:".format(model))
    print("get a loss function")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_data_loader, model, loss_fn, optimizer)
    torch.save(model.state_dict(), "model.pth")
    print("saved PyTorch Model State to model.pth")
