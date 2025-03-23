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
# 4. download model

```shell
#  确保 git-lfs 被安装
sudo apt-get install git git-lfs
git clone https://www.modelscope.cn/deepseek-ai/DeepSeek-R1-Distill-Llama-8B.git
```

# 5. run
硬件环境 一张 RTX 3090Ti ，显存 24GB
--tensor-parallel-size ，如果有多张显卡，可以设置大于1
```shell
# 通过 CUDA_VISIBLE_DEVICES 指定 GPU
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B --tensor-parallel-size 1 --max-model-len 8096 --gpu-memory-utilization 0.8 --enforce-eager
```