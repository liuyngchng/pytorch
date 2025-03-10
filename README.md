# 1. tutorial

https://pytorch.org/tutorials/index.html

# 2. test
进行分类的模型训练和测试
./train_classify.py

# 3. 模型转换

```$xslt
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
python3 convert_hf_to_gguf.py \
    --vocab-type bpe \
    --outfile output.gguf \
    input.pth
```
