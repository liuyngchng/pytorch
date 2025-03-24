#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过 nvidia-smi -L 获取指定 GPU 的 UUID

watch -n 1 nvidia-smi 观察 GPU 加载情况
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]='GPU-99b29e6e-b59b-2d02-714f-16bc83525830'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, pipeline, \
    DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

def train():
    """
    fine tune a LLM with localized knowledge
    若需高效微调可集成peft库实现LoRA。数据准备质量对最终效果影响最大，建议确保训练文本包含明确的问答/指令结构
    :return:
    """
    # 加载本地模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        "../DeepSeek-R1-Distill-Qwen-1.5B",
        torch_dtype=torch.float16,          # 降低精度，减少显存消耗量
        load_in_4bit=True,                  # 4bit量化（需bitsandbytes库）
        device_map="auto"                   # 自动分配设备
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)  # 原模型需先加载量化
    tokenizer = AutoTokenizer.from_pretrained("../DeepSeek-R1-Distill-Qwen-1.5B")

    # 加载训练数据
    # dataset = load_dataset("csv", data_files="train_data.csv")
    dataset = load_dataset("text", data_files="1.txt")
    dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=512), batched=True)

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./txt_trainer",
        num_train_epochs=3,
        per_device_train_batch_size=2,
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
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)  # 确保测试时加载一致的分词器


def test():
    generator = pipeline('text-generation', model='./txt_trainer',
                         tokenizer='./txt_trainer',     # 加载保存的分词器
                         device=0                           # 指定 GPU
                         )
    result = generator("如何缴费？：", max_length=300)
    print(result[0]['generated_text'])


if __name__ == "__main__":
    train()
    test()
