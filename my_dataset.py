from datasets import load_dataset
from transformers import AutoTokenizer

if __name__ == "__main__":
    # 加载分词器，确保路径正确
    tokenizer = AutoTokenizer.from_pretrained("../DeepSeek-R1-Distill-Qwen-1.5B")

    # 加载数据集
    my_dataset = load_dataset("json", data_files="1.jsonl", split="train")


    # 定义处理函数
    def process_function(x):
        text = f"Instruction:{x['instruction']}\nInput:{x['input']}\nOutput:{x['output']}"
        return tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",  # 统一填充到max_length
            return_tensors="pt"  # 返回PyTorch张量（根据框架可选）
        )


    # 处理数据集（默认batched=False，逐条处理）
    my_dataset1 = my_dataset.map(process_function, batched=False)

    # 保存处理后的数据
    my_dataset1.save_to_disk("processed_dataset")

    # 打印示例
    print(my_dataset1)
