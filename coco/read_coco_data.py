#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import pandas as pd
import numpy as np
from PIL import Image
import io

# 1. 读取一个Parquet文件
df = pd.read_parquet('/home/rd/Downloads/coco/data/train-00027-of-00040-c99041dbf751516d.parquet')
df = df.reset_index(drop=True)
# 创建必要的文件夹
os.makedirs('images', exist_ok=True)
os.makedirs('labels', exist_ok=True)

# 2. 遍历每一行数据
for index, row in df.iterrows():
    # 3. 解码图片并保存
    img = Image.open(io.BytesIO(row['image']['bytes']))
    # 获取图像尺寸，用于归一化
    img_width, img_height = img.size
    img.save(f'./images/{row["image_id"]}.jpg')

    # 4. 解析标注信息，转换为YOLO格式
    with open(f'./labels/{row["image_id"]}.txt', 'w') as f:
        # 获取 objects 数据
        objects = row['objects']
        # 获取各个字段（它们都是 numpy 数组）
        bbox_list = objects['bbox']  # 每个元素是一个包含4个值的数组
        class_ids = objects['category']  # 类别ID数组
        # 遍历每个标注框
        for i, bbox in enumerate(bbox_list):
            # bbox 是 [x_min, y_min, width, height] 格式
            # 注意：根据日志，bbox 可能是一个数组，需要确保是数值
            if isinstance(bbox, (list, np.ndarray)):
                x_min, y_min, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
            else:
                # 如果 bbox 是标量（你的日志显示可能是对象数组），需要特殊处理
                # 从你的日志看：bbox 数组的每个元素是 array([...])
                # 所以这里直接取值
                x_min, y_min, width, height = bbox[0], bbox[1], bbox[2], bbox[3]

            class_id = class_ids[i]

            # 转换为 YOLO 格式：中心点坐标 + 宽高（全部归一化）
            x_center = (x_min + width / 2) / img_width
            y_center = (y_min + height / 2) / img_height
            width_norm = width / img_width
            height_norm = height / img_height

            # 写入文件
            f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
    # 可选：每处理100张图片打印一次进度
    if (index + 1) % 100 == 0:
        print(f"已处理 {index + 1} 张图片")