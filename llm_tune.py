#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过 nvidia-smi -L 获取指定 GPU 的 UUID

watch -n 1 nvidia-smi 观察 GPU 加载情况
训练前执行 sudo nvidia-smi -pm 1启用持久模式
使用nohup后台运行避免ssh中断影响
添加try-except块捕捉CUDA错误并自动重试
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]='GPU-99b29e6e-b59b-2d02-714f-16bc83525830'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, pipeline, \
    DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
import logging.config

model_name = "../DeepSeek-R1-Distill-Qwen-1.5B"
# model_name = "../DeepSeek-R1-Distill-Llama-8B"

# 加载配置
logging.config.fileConfig('logging.conf')

# 创建 logger
logger = logging.getLogger(__name__)

def train():
    """
    fine tune a LLM with localized knowledge
    若需高效微调可集成peft库实现LoRA。数据准备质量对最终效果影响最大，建议确保训练文本包含明确的问答/指令结构
    :return:
    """
    # 加载本地模型和分词器
    logger.info("load local model and tokenizer")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # 优先float32 > bfloat16 > float16
        device_map="auto"
    )

    # 降低精度，节约显存
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.float16,          # 降低精度，减少显存消耗量
    #     device_map="auto",                  # 自动分配设备
    #     # attn_implementation="flash_attention_2",    # pip install flash_attn
    #     # device_map = {"":0},                # 强制使用单一设备
    #     quantization_config=BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_compute_dtype=torch.float16
    #     )
    # )
    # PEFT 微调
    logger.info("parameter efficient fine-tuning")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)  # 原模型需先加载量化
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 加载训练数据
    logger.info("load localized dataset")
    # dataset = load_dataset("csv", data_files="train_data.csv")
    dataset = load_dataset("text", data_files="1.txt")
    dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=512), batched=True)

    # 设置训练参数
    logger.info("set training args")
    training_args = TrainingArguments(
        output_dir="./txt_trainer",
        num_train_epochs=300,
        per_device_train_batch_size=1,  # 1, 2, 4 值越大，训练速度越快，同时可能提升模型稳定性，进而可能提高精度
        gradient_accumulation_steps=2,
        # gradient_checkpointing=True,
        learning_rate=3e-5,
        save_total_limit=2,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    # 开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )
    logger.info("start train")
    trainer.train()
    logger.info(f"save model to dir: {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)  # 确保测试时加载一致的分词器


def test():
    logger.info("load base model")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # 优先 float32 > bfloat16 > float16
        device_map="auto"
    )
    logger.info("PEFT base model")
    peft_model = (PeftModel.from_pretrained(base_model, "./txt_trainer")
                  .merge_and_unload())  # 合并LoRA权重提升推理速度
    logger.info("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained('./txt_trainer')  # 加载微调后的分词器
    logger.info("build test pipeline")
    generator = pipeline('text-generation', model=peft_model,
                         tokenizer=tokenizer,                   # 加载保存的分词器
                         # device=0,                              # 指定 GPU
                         temperature=0.1,                       # 降低随机性
                         top_p=0.9                              # 提高生成聚焦度
                         )
    logger.info("trigger test")
    result = generator("昆仑燃气燃气缴费如何操作", max_length=200)
    logger.info(f"test result: {result[0]['generated_text']}")


if __name__ == "__main__":
    train()
    test()
