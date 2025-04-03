#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
（1）硬件
        1）Geforce RXT 3090Ti，功率消耗400W（额定450W）, 显存占用 23GB(共24GB)
        2）ThinkPad_T14_Gen_2_Intel(https://psref.lenovo.com/WDProduct/ThinkPad/ThinkPad_T14_Gen_2_Intel)
            通过Thunderbolt4(雷电4)接口外接显卡, CPU 为 11th Gen Intel® Core™ i7-1165G7 × 8， 内存 16GB
 (2) python env
    pip install torch peft datasets transformers tensorboardX
    # 离线环境下可通过以下方式安装，严格保证离线和在线环境下的python，pip版本完全相同
        cd my_whl_dir
        # 有线环境下载 whl包
        pip download torch peft datasets transformers
        # 离线环境安装
        pip install torch peft datasets transformers tensorboardX --no-index --find-links=/a/b/c/my_whl_dir
（3）运行
        1）通过 nvtop(sudo apt-get install nvtop) 或 nvidia-smi -L 获取指定 GPU 的 UUID

        2）watch -n 1 nvidia-smi 观察 GPU 加载情况，实施检测功率、显存占用情况
            watch -n 1 "nvidia-smi -i GPU-99b29e6e-b59b-2d02-714f-16bc83525830 --query-gpu=utilization.gpu,memory.used --format=csv"

        3）训练前执行 sudo nvidia-smi -pm 1 启用持久模式

        4）使用nohup后台运行避免ssh中断影响
        5）添加try-except块捕捉CUDA错误并自动重试
"""
import json
import os
from typing import Union

# 对于多GPU 环境，此处需要修改成自己的 gpu_UUID，或者 GPU index
gpu_UUID = 'GPU-99b29e6e-b59b-2d02-714f-16bc83525830'
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_UUID
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, pipeline, \
    DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from peft import LoraConfig, get_peft_model, PeftModel
import logging.config

# 训练输入的大模型
model_name = "../DeepSeek-R1-Distill-Llama-8B"
# 训练输出的大模型
model_output_dir="./txt_trainer"
# 通过 tensorboard 查看训练过程的日至目录
tensorboard_log_idr = "./logs"
# 训练输入的语料库素材
# local_dataset = "my.txt"
local_dataset = "my.json"

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
        logger.info(f"GPU {i}: UUID[{dev.uuid}], name[{dev.name}], "
                    f"mem[{dev.total_memory / 1024 ** 3:.1f}GB]")
        logger.info(
            f"当前显存占用: {torch.cuda.memory_allocated() / 1024 ** 3:.1f}GB "
            f"/ {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.1f}GB"
        )




def token_json(model: str, data_files: str)-> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    """
    data_files：内容格式如下 [{"q": "xxx?", "a": "xxx"}, {"q": "xxx?", "a": "xxx"}]
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
    logger.info(f"data structure: {my_dataset}, data sample: {my_dataset[0]}")
    return my_dataset

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

def get_trainer(model, tokenizer, train_dataset, output_dir: str):
    # 设置训练参数

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,            # 数字较大可能会导致过拟合
        per_device_train_batch_size=4,  # 1, 2, 4 值越大，训练速度越快，同时可能提升模型稳定性，进而可能提高精度
        gradient_accumulation_steps=4,
        # gradient_checkpointing=True,
        learning_rate=2e-4,
        save_total_limit=2,
        fp16=True,                  # 启用混合精度训练
        logging_steps=50,            # 添加训练监控
        report_to="tensorboard",    # 强化日志监控
        logging_dir=tensorboard_log_idr,
    )
    logger.debug(f"set training args as {training_args}")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # 自动填充到8的倍数，提升GPU计算效率
    )
    # 开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    return trainer

def peft_train():
    """
    fine tune a LLM with localized knowledge
    若需高效微调可集成peft库实现LoRA。数据准备质量对最终效果影响最大，建议确保训练文本包含明确的问答/指令结构
    :return:
    """
    # 加载本地模型和分词器
    logger.info(f"start peft_train, load local model and tokenizer from {model_name}")
    model = get_model(model_name)
    # PEFT 微调
    logger.info("parameter efficient fine-tuning")
    peft_config = LoraConfig(
        r=8,        # 表示低秩矩阵的秩（rank），控制适配层的表达能力。数值越大适配能力越强，但计算量也越大（典型值范围8-64）
        lora_alpha=32,# 缩放系数，控制低秩矩阵对原始参数的干预强度。与r共同决定最终权重调整幅度（计算公式：ΔW = α/r * (A·B)）
        # q_proj：查询（Query）投影矩阵
        # v_proj：值（Value）投影矩阵
        # 表示仅修改Transformer中这两个核心注意力层的参数
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)  # 原模型需先加载量化
    logger.debug(f"model.peft_config after peft_config: {model.peft_config}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 加载训练数据
    logger.info(f"load local dataset from {local_dataset}")
    my_dataset = token_json(model_name, local_dataset)
    # logger.info(f"data structure:{my_dataset}, sample data: {my_dataset[0]}")
    logger.info("start training")
    trainer = get_trainer(model, tokenizer, my_dataset, model_output_dir)
    for attempt in range(3):
        try:
            trainer.train()
            break
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            # 动态调整batch_size
            trainer.args.per_device_train_batch_size = max(1, trainer.args.per_device_train_batch_size // 2)
            logger.error(f"error {e}, retry with trainer.args.per_device_train_batch_size ={trainer.args.per_device_train_batch_size} ")
    # trainer.train()
    logger.debug(f"model.peft_config after trainer.train(): {model.peft_config}")
    model = model.merge_and_unload()
    logger.info(f"save merged model to {model_output_dir}")
    model.save_pretrained(
        model_output_dir,
        safe_serialization=True,
        save_adapter=True,          # 显式保存adapter
        is_main_process=True        # 保存config

    )
    logger.info(f"tokenizer.save_pretrained({model_output_dir})")
    tokenizer.save_pretrained(model_output_dir)  # 确保测试时加载一致的分词器
    # need to run 'sudo apt-get install tree' first
    check_cmd = f"tree {model_output_dir}"
    logger.info(f"run cmd {check_cmd}")
    os.system(check_cmd)


def test_model():
    # logger.info(f"start test_model, load base model {model_output_dir}")
    # base_model = get_model(model_output_dir)
    # logger.info(f"PEFT base model {model_output_dir}")
    # peft_model = PeftModel.from_pretrained(base_model, model_output_dir)
    # peft_model = peft_model.merge_and_unload()  # 合并LoRA权重提升推理速度
    # assert "lora" not in str(peft_model.state_dict().keys()), "LoRA权重未合并"

    logger.info(f"load peft model {model_output_dir}")
    peft_model = get_model(model_output_dir)
    logger.info(f"load tokenizer {model_output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_output_dir)  # 加载微调后的分词器
    logger.info("build test pipeline")
    """
    text-generation 文本生成
    text-classification 文本分类
    question-answering 问答
    summarization 摘要生成
    translation_xx_to_yy 翻译(如zh_to_en)
    ner 命名实体识别
    fill-mask 完形填空
    """
    generator = pipeline('text-generation', model=peft_model,
                         tokenizer=tokenizer,                   # 加载保存的分词器
                         # device=0,                             # 指定 GPU
                         temperature=0.1,                       # 降低随机性
                         top_p=0.9,                             # 提高生成聚焦度
                         repetition_penalty=1.2,                # 添加重复惩罚提升生成质量
                         do_sample=True
                         )

    prompt = "问：如何成为昆昆燃气的服务客户？\n答："  # 匹配训练时的"问-答"模板
    logger.info(f"trigger test {prompt}")
    result = generator(prompt, max_length=1024)
    answer = result[0]['generated_text'].split("[output]")[-1].strip()
    logger.info(f"test result: {answer}")

def init_env():
    check_gpu()
    # 需管理员权限
    shell_cmd = "sudo nvidia-smi -pm 1"
    logger.info(f"start execute {shell_cmd}")
    os.system(shell_cmd)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    prompt = "instruction(使用方法说明)：\n\t1 - for LLM train task(启动模型训练)\n\t2 - for LLM  test task(启动模型测试)\n\tother(其他选项) - nothing done(程序退出)"
    print(prompt)
    task = input("请输入要执行的任务(1/2):")
    if task == "1":
        logger.info("启动模型训练")
        init_env()
        peft_train()
    elif task == "2":
        logger.info("启动模型测试")
        init_env()
        test_model()
    else:
        logger.error(f"nothing done, input '1' or '2', you must make a choice.")