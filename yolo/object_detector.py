#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ultralytics import YOLO
model = YOLO("./yolo26n.pt")

# results = model("https://ultralytics.com/images/bus.jpg")
# 检测图片 456.jpg, 置信度设置为 0.6
results = model("123.png", conf=0.2)
print(f"检测到 {len(results[0].boxes)} 个物体")
results[0].show()  # 这会弹窗显示带标注框的图片