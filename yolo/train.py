#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ultralytics
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

# 加载模型，会自动进行模型下载，如果已经下载过，则不会重复下载
# 也可以手动下载模型，https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
model = YOLO("yolo26n.pt")
# 加载数据集，这里可以使用自己已经标注好的数据集合，也可以使用公共数据集
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
model.train(
    data=dataset_path,
    epochs=5,
    imgsz=640,
    # 缩放图像的范围，默认值为 0.5 到 1.5
    multi_scale=True,       # 开启多尺度训练， 这个很重要，
    # 如果设置为 True，则会随机选择一个尺度范围，在该范围内随机缩放图像，然后再进行训练。
    # 你还可以指定尺度范围，例如 scale=0.5 代表在 0.5x 到 1.5x 的基准尺寸之间随机缩放
    scale=0.5,
    # 旋转增强参数
    degrees=180,             # 允许旋转的角度范围（0-180），180代表任意角度
    # 配合使用的其他增强
    flipud=0.5,              # 上下翻转概率
    fliplr=0.5,              # 左右翻转概率

    # 其他有用的增强
    hsv_h=0.015,             # 色调变化（模拟不同光照）
    hsv_s=0.7,               # 饱和度变化
    hsv_v=0.4,               # 明度变化
    perspective=0.0005,      # 透视变换（模拟不同视角）
    cache='ram',             # 缓存方式，可选 'ram  ' 或 'disk'，让旋转和翻转的图像缓存到内存中，以加快训练速度
    # 训练时使用的设备，可以是 'cpu' 或 'cuda'
    device='cpu'
)


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