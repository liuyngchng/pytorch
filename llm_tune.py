#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
（1）硬件
        1）Geforce RXT 3090Ti，功率消耗350W（额定450W）, 显存占用 16GB(共24GB)
        2）Thinkpad T14 笔记本外接显卡, 1th Gen Intel® Core™ i7-1165G7 × 8， 16GB 内存，
（2）运行
        1）通过 nvtop(sudo apt-get install nvtop) 或 nvidia-smi -L 获取指定 GPU 的 UUID

        2）watch -n 1 nvidia-smi 观察 GPU 加载情况，实施检测功率、显存占用情况

        3）训练前执行 sudo nvidia-smi -pm 1启用持久模式

        4）使用nohup后台运行避免ssh中断影响
        5）添加try-except块捕捉CUDA错误并自动重试
        6）1.json format { "instruction": "your instruction", "input": "you input txt", "output": "something want tobe outputed" }
"""
import os
gpu_UUID = 'GPU-99b29e6e-b59b-2d02-714f-16bc83525830'
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_UUID
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, pipeline, \
    DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
import logging.config

model_name = "../DeepSeek-R1-Distill-Llama-8B"

# 加载配置
logging.config.fileConfig('logging.conf')

# 创建 logger
logger = logging.getLogger(__name__)


def check_gpu():
    """
    check gpu whether it is available
    :return:
    """
    if not torch.cuda.is_available():
        raise "gpu not available err"
    for i in range(torch.cuda.device_count()):
        dev = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: UUID[{dev.uuid}], name[{dev.name}], mem[{dev.total_memory / 1024 ** 3:.1f}GB]")

def build_model(name:str):
    """
    build a base model which would be trained
    :param name: model name/path in local
    :return: a base model
    """
    # my_model = AutoModelForCausalLM.from_pretrained(
    #     name,
    #     torch_dtype=torch.bfloat16,  # 优先 float32 > bfloat16 > float16
    #     # device_map="auto",
    #     device_map={"": 0},
    # )

    # 降低精度，节约显存
    my_model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16,          # 降低精度，减少显存消耗量
        # device_map="auto",                  # 自动分配设备
        # attn_implementation="flash_attention_2",    # pip install flash_attn
        device_map = {"":0},                # 强制使用单一设备
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,                # 启用4bit量化缓解显存压力（适合24GB显存）
            bnb_4bit_compute_dtype=torch.float16
        )
    )
    return my_model

def train():
    """
    fine tune a LLM with localized knowledge
    若需高效微调可集成peft库实现LoRA。数据准备质量对最终效果影响最大，建议确保训练文本包含明确的问答/指令结构
    :return:
    """
    # 加载本地模型和分词器
    logger.info("load local model and tokenizer")
    model = build_model(model_name)
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
    my_dataset = load_dataset("text", data_files="1.txt")["train"]

    # def tokenize_fn(x):
    #     return tokenizer(
    #         x["text"],
    #         truncation=True,
    #         max_length=512,
    #         return_overflowing_tokens=True  # 启用文本分块
    #     )
    #
    # my_dataset = my_dataset.map(tokenize_fn, batched=True)
    my_dataset = my_dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=512, return_overflowing_tokens=True), batched=True)

    # 设置训练参数
    logger.info("set training args")
    training_args = TrainingArguments(
        output_dir="./txt_trainer",
        num_train_epochs=10,
        per_device_train_batch_size=4,  # 1, 2, 4 值越大，训练速度越快，同时可能提升模型稳定性，进而可能提高精度
        gradient_accumulation_steps=4,
        # gradient_checkpointing=True,
        learning_rate=2e-4,
        save_total_limit=2,
        fp16=True,                  # 启用混合精度训练
        logging_steps=50,            # 添加训练监控
        report_to="tensorboard",    # 强化日志监控
        logging_dir="./logs",
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    # 开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=my_dataset,
        data_collator=data_collator,
    )
    logger.info("start train")
    trainer.train()
    logger.info(f"save model to dir: {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)  # 确保测试时加载一致的分词器


def test():
    logger.info("load base model")
    base_model = build_model(model_name)
    logger.info("PEFT base model")
    peft_model = (PeftModel.from_pretrained(base_model, "./txt_trainer")
                  .merge_and_unload())  # 合并LoRA权重提升推理速度
    logger.info("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained('./txt_trainer')  # 加载微调后的分词器
    logger.info("build test pipeline")
    generator = pipeline('text-generation', model=peft_model,
                         tokenizer=tokenizer,                   # 加载保存的分词器
                         # device=0,                             # 指定 GPU
                         temperature=0.1,                       # 降低随机性
                         top_p=0.9,                             # 提高生成聚焦度
                         repetition_penalty=1.2,                # 添加重复惩罚提升生成质量
                         do_sample=True
                         )
    logger.info("trigger test")
    result = generator("昆仑燃气如何缴费", max_length=200)
    logger.info(f"test result: {result[0]['generated_text']}")


if __name__ == "__main__":
    check_gpu()
    train()
    test()
