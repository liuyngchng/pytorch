#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolo26n.pt")

# 训练
model.train(
    data="/home/rd/Downloads/yolo_data_input/data.yaml",  # 指向刚创建的 data.yaml
    epochs=50,
    imgsz=640,
    device='cpu'  # 因为你没有显卡
)