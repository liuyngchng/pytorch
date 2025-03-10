#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#https://blog.51cto.com/u_13317/7419054

import torch
from torchvision import transforms
from PIL import Image
 
# 指定一个变换流程，比如调整大小和转换为Tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小
    transforms.ToTensor(),         # 转换为Tensor
])
 
# 加载你的图片
image = Image.open('./my_dataset/train/cat/1.jpeg')
 
# 应用变换
image_tensor = transform(image)
 
# 增加一个batch维度，因为大多数PyTorch模型期望批量输入
image_tensor = image_tensor.unsqueeze(0)  # 变成 [C, H, W] 到 [1, C, H, W]


from torchvision import models

# 加载预训练的ResNet模型
model = models.resnet18(pretrained=True)
model.eval()  # 设置模型为评估模式



#推理模式

with torch.no_grad():  # 在推理模式下，不需要计算梯度
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs.data, 1)
print("Predicted class:", predicted.item())


# 训练示例
from torch.utils.data import DataLoader, TensorDataset

# 创建一个TensorDataset
label="cat"
dataset = TensorDataset(image_tensor, torch.tensor([label]))  # label是你图片的标签

# batch_size=1，即使只有一个样本也要这样设置，因为大多数模型期望批量输入。
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  

# 在训练循环中使用dataloader
for inputs, labels in dataloader:
    optimizer.zero_grad()         # 清空梯度
    outputs = model(inputs)        # 前向传播
    loss = loss_fn(outputs, labels)  # 计算损失
    loss.backward()               # 反向传播
    optimizer.step()              # 更新权重


