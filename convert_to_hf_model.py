#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC
from torch import nn
import torch
from transformers import BertConfig, BertModel, PretrainedConfig
from transformers import PreTrainedModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# BertModel config
config = BertConfig(
    vocab_size_or_config_json_file=30522,  # 根据你的模型调整这些参数
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    num_classes=2                       # 分类任务中的类别数量，根据所进行的实际任务进行配置
)


class CustomModel(PreTrainedModel, ABC):
    """
    a customized model with super class PreTrainedModel
    """
    config_class = BertConfig  # 使用你的配置类
    pretrained_model_archive_map = {}  # 如果需要可以从网上加载，否则留空或填充适当的URLs
    base_model_prefix = "bert"  # 根据你的模型架构调整

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.bert = BertModel(config)  # 使用BertModel或其他合适的模型架构

        num_classes = 2
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 根据你的任务调整分类器层
        self.init_weights()  # 初始化权重

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        return logits


if __name__ == "__main__":
    # 将自定义pytorch 模型转换为Hugging Face格式并保存,还没有调通
    custom_model = CustomModel(config)
    state_dict = torch.load('./train_classify_model.pth')
    custom_model.load_state_dict(state_dict, strict=False)  # 加载权重, strict=False 表示可以部分加载
    custom_model.save_pretrained('./custom_model.hf')  # 保存模型和配置到指定目录


    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained('./custom_model.hf')
    model.save_pretrained('./hf_formatted_model.hf')  # 重新生成标准格式
    print(model)
    # # 这儿会请求网路，从 huggingface.co 下载 bert-base-uncased 模型，如果本地没有的话
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # 根据你的任务选择合适的tokenizer
    print("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("../bge-large-zh-v1.5")
