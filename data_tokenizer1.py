#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union

from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from transformers import AutoTokenizer
import logging.config

# 加载配置
logging.config.fileConfig('logging.conf')

# 创建 logger
logger = logging.getLogger(__name__)


def token_json(model: str, data_files: str)-> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    logger.info("load localized dataset for json")
    tokenizer = AutoTokenizer.from_pretrained(model)
    my_dataset = load_dataset("json", data_files=data_files)
    def tokenize_func(example):
        text = f"问：{example['q']}\n答：{example['a']}"
        return tokenizer(text,
             max_length=512,
             padding=True,
             truncation=True
        )

    my_dataset = my_dataset.map(tokenize_func, batched=True)
    return my_dataset

if __name__ == "__main__":
    model_name = "../DeepSeek-R1-Distill-Qwen-1.5B"
    # model_name = "../DeepSeek-R1-Distill-Llama-8B"
    # dt = token_jsonl(model_name, "1.jsonl")
    dt =  token_json(model_name, "my.json")
    logger.info(f"data structure: {dt}, data sample: {dt[0]}")
    #
    # dt = convert_to_json("my.txt")
    # logger.info(f"{dt}")

