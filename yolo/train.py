#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ultralytics
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

# 加载模型
model = YOLO("yolo26n.pt")
# 加载数据集
dataset_path = "/home/rd/Downloads/coco128_with_yaml/coco128.yaml"

try:
    data_info = check_det_dataset(dataset_path)
    print("✅ 数据集验证通过！")
    print(f"训练图片路径: {data_info['train']}")
    print(f"类别数量: {data_info['nc']}")
    print(f"类别名称: {data_info['names']}")
except Exception as e:
    print(f"❌ 数据集验证失败: {e}")

# 训练
# model.train(
#     data=dataset_path,
#     epochs=5,
#     imgsz=640,
#     device='cpu'
# )


# 查看模型信息
trained_model = "/home/rd/workspace/pytorch/yolo/runs/detect/train5/weights/best.pt"

import torch
from ultralytics.nn.tasks import DetectionModel  # 导入需要的类


ckpt = torch.load(trained_model, map_location='cpu', weights_only=False)


# 如果是完整模型保存（YOLO 默认方式）
if 'model' in ckpt:
    print(f"模型类型: {type(ckpt['model'])}")
    print(f"保存时的 epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"最佳 fitness: {ckpt.get('best_fitness', 'N/A')}")

# 如果是仅权重保存
if 'model_state_dict' in ckpt:
    print(f"权重层数: {len(ckpt['model_state_dict'].keys())}")