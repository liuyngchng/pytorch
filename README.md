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

# --gpu-memory-utilization 0.4 out of memory
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization 0.5 --enforce-eager

```
# 6. train

```sh
watch -n 1 nvidia-smi

pip install flash-attn --no-build-isolation
sudo apt install nvidia-cuda-toolkit
```

训练显存储占用情况
```text

nvidia-smi
Tue Mar 25 10:13:02 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.120                Driver Version: 550.120        CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce MX450           Off |   00000000:01:00.0 Off |                  N/A |
| N/A   51C    P8             N/A /    9W |       1MiB /   2048MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 3090 Ti     Off |   00000000:52:00.0 Off |                  Off |
| 53%   56C    P2            172W /  450W |    4161MiB /  24564MiB |     39%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    1   N/A  N/A      5970      C   python3                                      4154MiB |
+-----------------------------------------------------------------------------------------+

```