#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging.config
import torch
from transformers import AutoModelForCausalLM


# 训练输入的大模型
model_name = "../DeepSeek-R1-Distill-Llama-8B"

# 加载配置
logging.config.fileConfig('logging.conf')

# 创建 logger
logger = logging.getLogger(__name__)

def get_model(name:str):
    """
    build a base model which would be trained
    :param name: model name/path in local
    :return: a base model
    """
    logger.info(f"loading model {name}")
    _model_instance = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16,  # 优先 float32 > bfloat16 > float16
        # device_map="auto",
        device_map={"": 0},
    )


    # 降低精度，节约显存
    # _model_instance = AutoModelForCausalLM.from_pretrained(
    #     name,
    #     torch_dtype=torch.float16,          # 降低精度，减少显存消耗量
    #     # device_map="auto",                  # 自动分配设备
    #     # attn_implementation="flash_attention_2",    # pip install flash_attn
    #     device_map = {"":0},                # 强制使用单一设备
    #     quantization_config=BitsAndBytesConfig(
    #         load_in_4bit=True,                # 启用4bit量化缓解显存压力（适合24GB显存）
    #         bnb_4bit_compute_dtype=torch.float16
    #     )
    # )
    logger.info(f"_model_instance.dtype: {_model_instance.dtype}")
    return _model_instance

if __name__ == "__main__":
    model = get_model(model_name)