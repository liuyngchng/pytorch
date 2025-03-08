#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch import nn
import logging.config
from transformers import AutoModel

# 加载配置
logging.config.fileConfig('logging.conf')

# 创建 logger
logger = logging.getLogger(__name__)


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
            logger.info("process dir label {}".format(label))
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

def get_transformer():
    # 对图像进行一些预处理和数据增强， 便于后期使用
    my_transform = transforms.Compose([
        transforms.RandomResizedCrop(128),  # 随机裁剪缩放
        transforms.ColorJitter(0.4, 0.4, 0.4),  # 颜色抖动
        transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
        transforms.Resize((128, 128)),  # Resize to 128x128
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize, 对图像进行归一化处理，即减去均值并除以标准差
    ])
    return my_transform

def train(data_loader, model, loss_fn, optimizer, device):
    """
    train the model use data from dataloader,
    use loss_fn as loss function
    """
    size = len(data_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(data_loader, model, loss_fn, device):
    """
    test the model use data from data_loader
    :param data_loader:
    :param model:
    :param loss_fn:loss function
    :param device: GPU or CPU
    :return:
    """
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    logger.info(f"test err: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train_my_model():
    """
    train model with localized data
    :return:
    """
    train_img_dir = "./my_dataset/train"
    logger.info("start load my dataset in {}".format(train_img_dir))
    # 创建 CustomDataset 实例，加载训练数据
    train_dataset = CustomDataset(root_dir=train_img_dir, transform=get_transformer())
    # 创建 DataLoader 实例
    logger.info("build my data loader")
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    # to check data loaded to train_loader
    for images, labels in train_data_loader:
        logger.info(images.shape, labels)

    for X, y in train_data_loader:
        logger.info(f"Shape of X [N, C, H, W]: {X.shape}")
        logger.info(f"y: {y}")
        break

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device} device")
    logger.info("move neural network instance to the device")
    model = NeuralNetwork().to(device)
    # model = AutoModel.from_pretrained("base_model_name")  # 例如 "bert-base-uncased"
    logger.info("model is:".format(model))
    logger.info("get a loss function")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    epochs = 50
    pth_model = "train_classify_model.pth"
    for t in range(epochs):
        logger.info(f"Epoch {t + 1}\n-------------------------------")
        train(train_data_loader, model, loss_fn, optimizer, device)
    torch.save(model.state_dict(), pth_model)
    logger.info("saved PyTorch model state to file: {}".format(pth_model))


def test_my_model():
    """
    test the model have been trained with test data
    :return:
    """
    test_img_dir = "./my_dataset/val"
    logger.info("start load test dataset in {}".format(test_img_dir))
    test_dataset = CustomDataset(root_dir=test_img_dir, transform=get_transformer())
    test_data_loader = DataLoader(test_dataset, batch_size=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNetwork().to(device)              # 先定义网络结构并迁移到设备
    pth_model = "train_classify_model.pth"
    model.load_state_dict(torch.load(pth_model, map_location=device))  # 保持设备一致性
    # 优化模型加载方式
    try:
        model.load_state_dict(torch.load("train_classify_model.pth", map_location=device))
    except RuntimeError as e:
        logger.error("error occurred in load model: {}".format(str(e)))
    model.eval()  # 显式设置评估模式
    loss_fn = nn.CrossEntropyLoss()
    test(test_data_loader, model, loss_fn, device)


if __name__ == "__main__":
    # 训练模型
    # train_my_model()
    # 使用测试数据查验模型预测的准确度
    test_my_model()
