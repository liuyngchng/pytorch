#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import os

# 读取 classes.txt
classes_file = '/home/rd/Downloads/yolo_data_input/classes.txt'
if not os.path.exists(classes_file):
    print(f"❌ 文件不存在: {classes_file}")
    exit(1)
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# 获取当前目录的绝对路径
current_path = os.path.abspath('.')

# 构建 data.yaml 内容
data_config = {
    'path': current_path,      # 数据集根目录
    'train': 'images',          # 训练图片目录
    'val': 'images',            # 验证图片目录（暂时和训练一样）
    'nc': len(classes),         # 类别数量
    'names': classes            # 类别名称列表
}

# 写入 data.yaml
with open('data.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(data_config, f, sort_keys=False, allow_unicode=True)

print(f"✅ 成功创建 data.yaml")
print(f"   - 类别数量: {len(classes)}")
print(f"   - 类别名称: {classes}")