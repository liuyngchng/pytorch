from datasets import load_dataset
from transformers import AutoTokenizer
import logging.config

# 加载配置
logging.config.fileConfig('logging.conf')

# 创建 logger
logger = logging.getLogger(__name__)

model_name = "../DeepSeek-R1-Distill-Qwen-1.5B"

def token_json():
    # 加载分词器，确保路径正确
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 加载数据集
    my_dataset = load_dataset("json", data_files="1.jsonl", split="train")


    # 定义处理函数
    def tokenize_fn(x):
        text = f"Instruction:{x['instruction']}\nInput:{x['input']}\nOutput:{x['output']}"
        return tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",  # 统一填充到max_length
            return_tensors="pt"  # 返回PyTorch张量（根据框架可选）
        )


    # 处理数据集（默认batched=False，逐条处理）
    my_dataset1 = my_dataset.map(tokenize_fn, batched=False)

    # 保存处理后的数据
    my_dataset1.save_to_disk("processed_dataset")

    # 打印示例
    print(my_dataset1)

def token_txt():
    logger.info("load localized dataset")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    my_dataset = load_dataset("text", data_files="my.txt")["train"]
    logger.info(f"my_dataset数据集结构: {my_dataset}")
    logger.info(f"my_dataset首条样本: {my_dataset[0]}")
    def tokenize_fn(x):
        return tokenizer(
            x["text"],
            truncation=True,
            max_length=512,
            return_overflowing_tokens=True  # 启用文本分块
        )

    my_dataset1 = my_dataset.map(tokenize_fn, batched=True)
    # my_dataset = my_dataset.map(
    #     lambda x: tokenizer(x["text"], truncation=True, max_length=512, return_overflowing_tokens=True), batched=True)

    logger.info(f"my_dataset1数据集结构: {my_dataset1}")
    logger.info(f"my_dataset1首条样本: {my_dataset1[0]}")
if __name__ == "__main__":
    # token_json()
    token_txt()