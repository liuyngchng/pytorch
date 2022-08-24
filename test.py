#!/usr/bin/python3
import torch,torchvision

from torch import nn, optim

model = torchvision.models.resnet18(pretrained=True)

# Freeze all the parameters in the network
i=0
for param in model.parameters():
    # print(f"param={param}")
    i+=1
    param.requires_grad = False
print(f"i={i}")
