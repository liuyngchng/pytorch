#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import pandas as pd
from PIL import Image
import io

# 1. 读取一个Parquet文件
df = pd.read_parquet('/home/rd/Downloads/train-00027-of-00040-c99041dbf751516d.parquet')

# 创建必要的文件夹
os.makedirs('images', exist_ok=True)
os.makedirs('labels', exist_ok=True)

# 2. 遍历每一行数据
for index, row in df.iterrows():
    # 3. 解码图片并保存
    img = Image.open(io.BytesIO(row['image']['bytes']))
    img.save(f'/home/rd/Downloads/images/{row["image_id"]}.jpg')

    # 4. 解析标注信息，并保存为YOLO格式的txt文件
    # 注意：这部分需要根据具体列名编写转换逻辑
    # with open(f'labels/{row["image_id"]}.txt', 'w') as f:
    #     for bbox in row['objects']:
    #         # 将COCO格式的bbox (x, y, w, h) 转为YOLO格式 (center_x, center_y, width, height)
    #         # 写入文件
    #         f.write(f"{bbox['class_id']} {x_center} {y_center} {width} {height}\n")