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

def token_jsonl(model: str, data_files: str)-> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    logger.info("load localized dataset for jsonl")
    # 加载分词器，确保路径正确
    tokenizer = AutoTokenizer.from_pretrained(model)
    # 加载数据集
    my_dataset = load_dataset("json", data_files=data_files, split="train")
    # 定义处理函数
    def tokenize_fn(x):
        text = f"Instruction:{x['instruction']}\nInput:{x['input']}\nOutput:{x['output']}"
        return tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",  # 统一填充到max_length，这种处理方法能够保证数据统一，但可能会导致浪费存储，padding为统一的长度;更好的做法是在使用数据training的时候进行动态填充
            return_tensors="pt"  # 返回PyTorch张量（根据框架可选）
        )
    # 处理数据集（默认batched=False，逐条处理）
    my_dataset1 = my_dataset.map(tokenize_fn, batched=False)
    logger.debug(f"data structure: {my_dataset1}, data sample: {my_dataset1[0]}")
    return my_dataset1


def convert_to_json(file_path):
    with open(file_path) as f:
        return [{"instruction": "生成燃气服务回答", "input": line.strip(), "output": ""} for line in f]


def token_txt(model: str, data_files: str)-> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    logger.info("load localized dataset for txt")
    tokenizer = AutoTokenizer.from_pretrained(model)
    my_dataset = load_dataset("text", data_files=data_files)["train"]
    def tokenize_fn(x):
        return tokenizer(
            x["text"],
            truncation=True,
            max_length=512,
            return_overflowing_tokens=True  # 启用文本分块
        )

    my_dataset1 = my_dataset.map(tokenize_fn, batched=True)
    logger.debug(f"data structure: {my_dataset1}, data sample: {my_dataset1[0]}")
    # my_dataset = my_dataset.map(
    #     lambda x: tokenizer(x["text"], truncation=True, max_length=512, return_overflowing_tokens=True), batched=True)
    return my_dataset1

def token_json(model: str, data_files: str)-> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    """
    在训练的时候，需要对数据进行动态填充
    from transformers import DataCollatorForLanguageModeling

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # 自动填充到8的倍数，提升GPU计算效率
    )

    # 在Trainer中启用
    trainer = Trainer(
        ...
        data_collator=data_collator,
    )
    这样做的优势，每个batch自动填充到该batch内最长序列长度，兼顾内存效率和训练效果
    """
    logger.info("load localized dataset for json")
    tokenizer = AutoTokenizer.from_pretrained(model)
    my_dataset = load_dataset("json", data_files=data_files)["train"]
    logger.info(f"data structure: {my_dataset}, data sample: {my_dataset[0]}")
    def tokenize_func(example):
        text = [f"问：{q}\n答：{a}" for q, a in zip(example['q'], example['a'])]
        return tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_overflowing_tokens=True
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

