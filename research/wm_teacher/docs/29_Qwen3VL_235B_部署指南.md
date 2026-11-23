# Qwen3-VL-235B-A22B Caption 部署指南

> 本文档供在其他服务器上部署 Qwen3-VL-235B 进行视频 caption 打标使用。

---

## 一、模型信息

| 项 | 值 |
|---|---|
| 模型 | Qwen3-VL-235B-A22B-Instruct |
| 架构 | MoE，235B 总参数，22B 激活参数 |
| 精度 | bf16（默认） |
| 磁盘大小 | 439GB（96 个 safetensors） |
| 权重显存 | ~470GB（bf16） |
| 上下文长度 | 256K |
| 本地缓存 | `.cache/models/Qwen3-VL-235B-A22B-Instruct/`（439GB） |

---

## 二、GPU 需求

### 2.1 方案对比

| 方案 | GPU 需求 | 显存/GPU | 总显存 | 说明 |
|------|---------|---------|--------|------|
| **bf16 + TP=8** | 8× H20-96GB | 58.75GB 权重 + 37GB KV | 768GB | **最安全，推荐** |
| **bf16 + TP=4 + EP** | 4× H20-96GB | ~117GB→EP 降至 ~60GB | 384GB | 需 vLLM 支持 MoE EP |
| **FP8 + TP=4** | 4× H20-96GB | 29.4GB 权重 + 66GB KV | 384GB | 需 FP8 版模型 |
| **bf16 + TP=4** | 4× H100-80GB | 117.5GB | 320GB | ❌ 超显存 |
| **bf16 + TP=8** | 8× H100-80GB | 58.75GB 权重 + 21GB KV | 640GB | ✅ 可用 |
| **bf16 + TP=8** | 8× A100-80GB | 58.75GB 权重 + 21GB KV | 640GB | ✅ 可用 |
| **bf16 + TP=16** | 16× A100-40GB | 29.4GB 权重 + 10.6GB KV | 640GB | ✅ 可用但 KV 紧张 |

### 2.2 推荐配置

#### 最小可用（1 副本）
```bash
# 8× H20-96GB, TP=8, bf16
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
    --tensor-parallel-size 8 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 32 \
    --enforce-eager --trust-remote-code
```

#### 高吞吐（2 副本，需 16× H20-96GB）
```bash
# 副本 1: GPU 0-7, TP=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve ... --port 8000 &
# 副本 2: GPU 8-15, TP=8
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 vllm serve ... --port 8001 &
```

#### FP8 降配（4× H20-96GB）
```bash
# 需下载 FP8 版本：Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
    --tensor-parallel-size 4 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 32 \
    --trust-remote-code
```

### 2.3 不可用的配置

| 配置 | 原因 |
|------|------|
| 4× H20-96GB bf16 TP=4 | 117.5GB/GPU > 96GB 显存 |
| 6× A100-40GB bf16 TP=6 | 78.3GB/GPU > 40GB 显存 |
| 4× A100-40GB bf16 TP=4 | 117.5GB/GPU > 40GB 显存 |

---

## 三、推理吞吐

| 配置 | 副本数 | 吞吐 | 每天 |
|------|--------|------|------|
| 8× H20-96GB (TP=8) | 1 | ~10-15 videos/min | ~14-22K |
| 16× H20-96GB (2×TP=8) | 2 | ~20-30 videos/min | ~29-43K |
| 32× H20-96GB (4×TP=8) | 4 | ~40-60 videos/min | ~58-86K |
| 4× H20-96GB FP8 (TP=4) | 1 | ~10-15 videos/min | ~14-22K |

---

## 四、Caption 工作量

| 批次 | 数据集 | 视频数 | 1 副本预计 | 2 副本预计 |
|------|--------|--------|-----------|-----------|
| 1 | DexYCB + H2O | 2,460 | ~2h | ~1h |
| 2 | + Charades 增强 | 12,308 | ~14h | ~7h |
| 3 | + OpenVidHD 10K | 22,308 | ~25h | ~13h |
| 4 | + Ego-Exo4D + Matrix | ~80K | ~5 天 | ~2.5 天 |
| 5 | + HOIGen/NoXi/Seamless 增强 | ~200K | ~10 天 | ~5 天 |

**首轮训练所需（批次 1-3，~22K）**：1 副本 ~25h 或 2 副本 ~13h

---

## 五、部署步骤（在其他服务器上）

### 5.1 环境准备

```bash
# Python 环境
conda create -n vlm python=3.11 -y
conda activate vlm
pip install vllm==0.8.5 transformers torch
pip install huggingface_hub

# 登录 HF（用你的 token）
hf auth login --token <HF_TOKEN>
```

### 5.2 下载模型

```bash
# 方式 1：从 HF 下载（~439GB，需较长时间）
hf download Qwen/Qwen3-VL-235B-A22B-Instruct --local-dir /path/to/models/Qwen3-VL-235B

# 方式 2：从已缓存的服务器 rsync（如果两台服务器网络互通）
rsync -avP source_server:/srv/wm_teacher/.cache/models/Qwen3-VL-235B-A22B-Instruct/ /path/to/models/Qwen3-VL-235B/
```

### 5.3 启动 vLLM 服务

```bash
# 8× H20-96GB, TP=8
vllm serve /path/to/models/Qwen3-VL-235B \
    --tensor-parallel-size 8 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 32 \
    --enforce-eager --trust-remote-code \
    --port 8000
```

### 5.4 下载 manifest 和脚本

```bash
# 从 HF 下载项目 manifest + 脚本
hf download NZC415/wm-dataset --repo-type dataset --local-dir wm_dataset_meta
```

### 5.5 运行 caption

```bash
# 修改 caption_videos.py 中的 API_URLS 指向新服务器
# 然后运行
python3 caption_videos.py
```

---

## 六、注意事项

1. **vLLM 版本**：0.8.5 已验证支持 Qwen3-VL-235B。更高版本可能也兼容。
2. **视频帧采样**：每个视频采样 8 帧（ffmpeg），编码为 base64 发送给 VLM。
3. **断点续传**：caption_videos.py 支持断点续传（已 caption 的视频会跳过）。
4. **代理**：如果服务器需要代理访问 HF，设置 `HF_ENDPOINT=https://hf-mirror.com` 或 `HTTPS_PROXY`。
5. **模型加载慢**：如从共享 FS 加载，建议先拷贝到本地 NVMe 或 /dev/shm（439GB 拷贝约需 30-60min）。
6. **FP8 版本**：如需 4 卡部署，下载 `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8`（~220GB），质量略降但显存减半。
