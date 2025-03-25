#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cleantext import clean
import json

if __name__ == "__main__":

    with open("1.txt") as f:
        text = clean(f.read()) # 清洗
        chunks = [text[i:i+4096] for i in range(0, len(text), 4096)] # 分块

    with open("output.jsonl", "w") as f:
        for chunk in chunks:
            f.write(json.dumps({"text": chunk}) + "\n") # 输出JSONL