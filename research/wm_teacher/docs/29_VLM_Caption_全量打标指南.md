# VLM Caption 全量打标指南与进度

> 最后更新：2026-07-26 17:00
> 状态：MIRA caption 进行中（57%），ETA ~1天11小时

---

## 一、当前进度

### 总览

| 数据集 | 总视频数 | 已完成 caption | 失败 | 状态 |
|---|---|---|---|---|
| easycom | 3,920 | 3,913 | 7 | ✅ 完成（7 个文件损坏） |
| noxi | 41,744 | 41,744 | 0 | ✅ 完成 |
| charades | 73,603 | 73,601 | 2 | ✅ 完成（2 个文件损坏） |
| mira | 2,073,297 | 1,188,624 | 37,642 | 🔄 57% 进行中 |
| **总计** | **2,192,164** | **1,307,682** | 37,651 | — |

### MIRA 详情
- 已完成：1,188,624 / 2,073,297（**57%**）
- 失败：37,642（空文件，已跳过）
- 当前速度：~7 vid/s（双节点合计，60秒采样）
- ETA：约 **1 天 11 小时**（明天完成）
- Node A (7B) 贡献：765,287 ok
- Node B (235B) 贡献：261,169 ok

### 已合并到 unified manifest
- easycom/noxi/charades 全量 caption 已合并回 `unified_train.jsonl` / `unified_val.jsonl`
- 旧 caption 保留在 `prompt_legacy` 字段
- unified_train: 51,044 行，5 空 prompt (0.01%)
- MIRA caption 完成后需再次合并

### 系统健康（2026-07-26 17:00）
- Node A: 8/8 副本 HTTP 200 ✅，GPU 利用率 100%（8 卡），显存 ~87GB/卡
- Node B: HTTP 200 ✅

---

## 二、系统架构

### 双节点部署

| 节点 | IP | 模型 | 部署方式 | 角色 |
|---|---|---|---|---|
| Node A | PRIVATE_NODE_ADDRESS | Qwen2.5-VL-7B-Instruct | 8 副本 × TP=1, --enforce-eager, port 8000-8007 | caption (7B) |
| Node B | PRIVATE_NODE_ADDRESS | Qwen3-VL-235B-A22B-Instruct-FP8 | 1 副本 × TP=8, --async-scheduling, port 8000 | caption (235B) + 占卡守护 |

### 为什么 Node A 用 7B 而非 235B

Node A 部署 235B 时遇到两个叠加问题：
1. **MQ 连接问题**：`get_ip()` 返回外网 IP `PRIVATE_NODE_ADDRESS`（未绑定本地网卡），worker 连接 engine core 被代理拦截。修复：`VLLM_HOST_IP=127.0.0.1`
2. **FlashInfer AllReduce CUDA 错误**：graph capture 阶段 GPU worker 崩溃（GPU 5 在 TP=8 时崩溃，GPU 3 在 TP=4 时崩溃）。这是硬件级问题，多张 GPU 在高负载下不稳定。

**解决方案**：改用 Qwen2.5-VL-7B-Instruct（dense 模型，无 MoE AllReduce），8 副本 TP=1（每 GPU 1 个独立副本），`--enforce-eager` 禁用 CUDA graphs。质量略低于 235B 但仍可用，通过 `model` 字段区分来源。

---

## 三、脚本使用指南

### 3.1 目录结构

```
/srv/wm_teacher/
├── deploy_qwen3vl.sh              # 235B 部署脚本（Node B 用，含占卡启停逻辑）
├── scripts/
│   ├── recaption_unified.py       # 主 caption 脚本（支持多端点负载均衡、断点续传）
│   ├── merge_recaptioned.py       # 合并新 caption 回 unified manifest
│   ├── generate_dataset_manifest.py  # 生成数据集完整 manifest
│   ├── repair_mira_empty.py       # 修复 MIRA 空文件（从 tar 重新提取+转码）
│   └── stage2_auto_caption.sh     # 阶段二自动衔接（NoXi→Charades→MIRA）
├── training_metadata/
│   ├── unified_train.jsonl        # Wan2.2 训练 manifest（51,044 行）
│   ├── unified_val.jsonl          # 验证 manifest（5,573 行）
│   ├── manifest_mira_full.jsonl   # MIRA 完整 manifest（2,073,297 行）
│   ├── manifest_noxi_full.jsonl   # NoXi 完整 manifest（41,744 行）
│   ├── manifest_charades_full.jsonl  # Charades 完整 manifest（73,603 行）
│   └── recaptioned/               # 新 caption 输出目录
│       ├── mira_recaptioned.jsonl
│       ├── noxi_recaptioned.jsonl
│       ├── charades_recaptioned.jsonl
│       └── easycom_recaptioned.jsonl
└── models/
    ├── Qwen3-VL-235B-A22B-Instruct-FP8/  # 235B 模型（222GB）
    └── Qwen2.5-VL-7B-Instruct/            # 7B 模型（~16GB）
```

### 3.2 部署 vLLM

#### Node B: Qwen3-VL-235B（TP=8 单副本）

```bash
# 停止占卡 → 部署 vLLM → 退出时自动恢复占卡
bash /srv/wm_teacher/deploy_qwen3vl.sh

# 或后台运行
nohup bash deploy_qwen3vl.sh > /tmp/vllm_qwen3vl.log 2>&1 & disown

# 健康检查
curl http://PRIVATE_NODE_ADDRESS:8000/health
```

**关键配置**（已写入 deploy_qwen3vl.sh）：
- `VLLM_HOST_IP=127.0.0.1` — 修复 MQ 连接问题
- `--mm-encoder-tp-mode data` — 视觉编码器数据并行
- `--enable-expert-parallel` — MoE 专家并行
- `--async-scheduling` — 重叠调度与解码
- 占卡逻辑：启动前 `stop_occupy`，退出时 `trap ... EXIT start_occupy`

#### Node A: Qwen2.5-VL-7B（8 副本 TP=1）

```bash
# 停止占卡
pkill -f "gpu_occupier|gpu_guard"; sleep 3
for pid in $(fuser /dev/nvidia0 2>/dev/null); do kill -9 $pid; done

# 启动 8 个副本（每 GPU 1 个，port 8000-8007）
MODEL="/srv/wm_teacher/models/Qwen2.5-VL-7B-Instruct"
for i in 0 1 2 3 4 5 6 7; do
  port=$((8000 + i))
  VLLM_HOST_IP=127.0.0.1 CUDA_VISIBLE_DEVICES=$i \
  /root/.venv/bin/vllm serve "${MODEL}" \
      --tensor-parallel-size 1 \
      --host 0.0.0.0 --port ${port} \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.90 \
      --max-num-seqs 32 \
      --enforce-eager \
      --trust-remote-code &
done

# 健康检查
for p in 8000 8001 8002 8003 8004 8005 8006 8007; do
  curl -s -o /dev/null -w "port ${p}: %{http_code}\n" http://localhost:${p}/health
done
```

### 3.3 运行 Caption

#### 单节点 caption（Node B 235B）

```bash
cd /srv/wm_teacher

/root/.venv/bin/python scripts/recaption_unified.py \
    --manifest training_metadata/manifest_mira_full.jsonl \
    --datasets mira \
    --output-dir training_metadata/recaptioned \
    --api-urls http://PRIVATE_NODE_ADDRESS:8000/v1/chat/completions \
    --concurrent 48
```

#### 双节点并行 caption（Node A 7B + Node B 235B）

两个进程写同一个输出文件（`mira_recaptioned.jsonl`），通过 `done` 集合去重，断点续传。

```bash
# Node B 进程（235B）
nohup /root/.venv/bin/python scripts/recaption_unified.py \
    --manifest training_metadata/manifest_mira_full.jsonl \
    --datasets mira \
    --output-dir training_metadata/recaptioned \
    --api-urls http://PRIVATE_NODE_ADDRESS:8000/v1/chat/completions \
    --concurrent 48 \
    > /tmp/recaption_mira_nodeB.log 2>&1 & disown

# Node A 进程（7B，8 端点）
nohup /root/.venv/bin/python scripts/recaption_unified.py \
    --manifest training_metadata/manifest_mira_full.jsonl \
    --datasets mira \
    --output-dir training_metadata/recaptioned \
    --api-urls http://localhost:8000/v1/chat/completions \
             http://localhost:8001/v1/chat/completions \
             http://localhost:8002/v1/chat/completions \
             http://localhost:8003/v1/chat/completions \
             http://localhost:8004/v1/chat/completions \
             http://localhost:8005/v1/chat/completions \
             http://localhost:8006/v1/chat/completions \
             http://localhost:8007/v1/chat/completions \
    --concurrent 128 \
    --model /srv/wm_teacher/models/Qwen2.5-VL-7B-Instruct \
    > /tmp/recaption_mira_nodeA.log 2>&1 & disown
```

**关键参数**：
- `--api-urls`：多个端点自动 round-robin 负载均衡
- `--concurrent`：并发请求数（235B 用 48，7B 用 128）
- `--model`：必须匹配 vLLM serve 的 model id（本地路径）
- `--limit N`：限制每数据集 N 个（冒烟测试用）
- 断点续传：从输出文件加载 `done` 集合，跳过已完成的

#### 自动衔接（stage2_auto_caption.sh）

```bash
nohup bash scripts/stage2_auto_caption.sh > /tmp/stage2_auto.log 2>&1 & disown
```

按顺序执行：NoXi → Charades → MIRA（含 manifest 生成）。

### 3.4 合并 Caption 回 unified manifest

```bash
cd /srv/wm_teacher

/root/.venv/bin/python scripts/merge_recaptioned.py \
    --manifest training_metadata/unified_train.jsonl \
                   training_metadata/unified_val.jsonl \
    --recaption-dir training_metadata/recaptioned \
    --datasets mira noxi charades easycom
```

- 旧 prompt 保存到 `prompt_legacy` 字段
- 自动备份：`unified_train.jsonl.bak.<timestamp>`
- 验证填空率

### 3.5 修复 MIRA 空文件

```bash
# 预览（dry run）
/root/.venv/bin/python scripts/repair_mira_empty.py --dry-run

# 实际修复（4 并行 worker）
/root/.venv/bin/python scripts/repair_mira_empty.py --workers 4
```

从原始 tar 包重新提取视频并转码到 720p。

### 3.6 GPU 占卡

```bash
# 启动占卡
rm -f /srv/PRIVATE_STORAGE
nohup /root/.venv/bin/python /srv/PRIVATE_STORAGE > /tmp/gpu_occupy.log 2>&1 & disown

# 启动 25 分钟自动守护
nohup bash /srv/PRIVATE_STORAGE > /tmp/gpu_guard.log 2>&1 & disown

# 停止占卡
pkill -f "gpu_occupier|gpu_guard"; sleep 3
for pid in $(fuser /dev/nvidia0 2>/dev/null); do kill -9 $pid; done
```

---

## 四、Caption 质量对比

### 统一 Prompt 设计

所有数据集使用同一个训练导向 prompt（`recaption_unified.py` 中的 `UNIFIED_PROMPT`）：
- 3-5 句英文描述（60-120 词）
- 覆盖：主体、动作、环境、镜头/视觉
- 跨数据集一致，避免 prompt embedding 分布偏移

### 质量提升（阶段一，unified_train 子集）

| 数据集 | 旧平均长度 | 新平均长度 | 提升 | 旧样本 | 新样本 |
|---|---|---|---|---|---|
| charades | 133 | 582 | 4.4x | "holding medicine, sneezing" | "rainbow tie-dye t-shirt, dimly lit room..." |
| easycom | 85 | 690 | 8.1x | "Two people having a conversation, video_id: xxx" | "grey polo, plaid shirt, dark wood table, modern office..." |
| mira | 86 | 616 | 7.1x | "Rocket League player jumping, boosting" | "orange rocket-powered car, futuristic soccer field, scoreboard 0-2..." |
| noxi | 276 | 617 | 2.2x | "male aged 36-45, interviewee, Arabic" | "young woman, dark hair, whiteboard, classroom..." |

### 7B vs 235B 质量

- 235B caption 更详细（~600 字符），视觉细节更丰富
- 7B caption 略短（~200 字符），但仍包含场景、主体、动作描述
- 通过 `model` 字段区分来源，可后续按需过滤或重打

---

## 五、已知问题与解决方案

### 5.1 MIRA 空文件（~37,649 个，1.8%）

**原因**：特定 tar 包（2026-05-08T21-23）在下载/解压时损坏，导致 0 字节 mp4 文件。
**影响**：空文件无法提取帧，caption 失败（"no frames"）。
**修复**：
- `recaption_unified.py` 已加快速预检查（`os.path.getsize(vp) < 1024` 跳过）
- `repair_mira_empty.py` 可从原始 tar 重新提取并转码

### 5.2 Node A 235B 部署失败

**原因**：FlashInfer AllReduce CUDA 错误 + 多 GPU 不稳定（GPU 5/3 在 graph 编译时崩溃）。
**修复**：改用 Qwen2.5-VL-7B dense 模型，8 副本 TP=1 + `--enforce-eager`。

### 5.3 vLLM MQ 连接问题

**原因**：`get_ip()` 返回外网 IP（未绑定本地网卡），worker 连接被代理拦截。
**修复**：`VLLM_HOST_IP=127.0.0.1`（已写入 deploy_qwen3vl.sh）。

### 5.4 PyAV 依赖

**问题**：ffmpeg/ffprobe 不在 PATH 中。
**修复**：使用 PyAV（`av` 18.0.0）进行帧采样，无外部依赖。

### 5.5 Squid 代理拦截 localhost

**问题**：`http_proxy` 环境变量导致 localhost 请求被 Squid 拦截。
**修复**：脚本内 `NO_PROXY_DICT = {"http": None, "https": None}`。

---

## 六、环境信息

| 组件 | 版本 |
|---|---|
| GPU | 8× NVIDIA H20-96GB × 2 节点 |
| CUDA Driver | 535.161.08 |
| vLLM | 0.16.1rc1.dev256 |
| torch | 2.10.0+cu129 |
| transformers | 4.57.6 |
| PyAV | 18.0.0 |
| Python | 3.13（/root/.venv） |
| ffmpeg/ffprobe | 不在 PATH（用 PyAV 替代） |
| Squid proxy | `http://star-proxy.oa.com:3128`（需 NO_PROXY=localhost 绕过） |

---

## 七、后续步骤

1. **MIRA caption 完成**（~22 小时后）→ 合并回 unified manifest
2. **修复 MIRA 空文件** → `repair_mira_empty.py`
3. **验证 unified manifest** → 确认填空率 99.99%+
4. **启动 Wan2.2 训练** → unified manifest 已就绪

---

## 八、常用监控命令

```bash
# caption 进度
wc -l /srv/wm_teacher/training_metadata/recaptioned/*.jsonl

# Node A 7B 速度
grep -aoE "ok=[0-9]+" /tmp/recaption_mira_nodeA.log | tail -1

# Node B 235B 速度
grep -aoE "ok=[0-9]+" /tmp/stage2_auto.log | tail -1

# vLLM 健康
curl -s -o /dev/null -w "%{http_code}\n" http://PRIVATE_NODE_ADDRESS:8000/health  # Node B
for p in 8000 8001 8002 8003 8004 8005 8006 8007; do
  curl -s -o /dev/null -w "port ${p}: %{http_code}\n" http://localhost:${p}/health  # Node A
done

# GPU 状态
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```
