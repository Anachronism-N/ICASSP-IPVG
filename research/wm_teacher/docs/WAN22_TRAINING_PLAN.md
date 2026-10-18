# Wan2.2 训练数据准备方案

> 最后更新：2026-07-12
> 目标：将下载的数据集整理为 Wan2.2 可用的训练格式
>
> **状态提示：本文保留早期方案。当前数据实况、Caption 质检和首轮训练步骤以 [24_Wan2.2首轮训练执行计划.md](./24_Wan2.2首轮训练执行计划.md) 为准。**

---

## 一、Wan2.2 训练背景

Wan2.2 是开源的视频生成模型（5B-14B），支持 Text-to-Video、Image-to-Video、Text-Image-to-Video。
官方不提供训练脚本，社区方案为 **DiffSynth-Studio**（支持 LoRA 训练和全量微调）。

### 训练数据格式要求

| 维度 | 推荐值 |
|------|--------|
| 视频格式 | MP4（H.264/H.265） |
| 分辨率 | 1280×704（720P 横屏）或 704×1280（竖屏） |
| 帧率 | 24 FPS |
| 单段时长 | 2-10 秒（太短无效，太长 OOM） |
| 标注格式 | JSON/CSV：`{video_path, caption, [condition_data]}` |
| VAE 压缩 | 16×16×4（T×H×W）— 推理时自动处理 |

---

## 二、数据集适配分析

### MIRA Rocket Science — 高价值，可直接使用（660GB）

| 维度 | 详情 |
|------|------|
| 格式 | 328 个 tar 包，每个含 ~496 MP4 + ~992 JSONL |
| 视频估算 | ~162,000 段 MP4，每段 ~3MB，约 30 秒 |
| 标注 | JSONL 含 step/frame/action/physics 等结构化数据 |
| Wan2.2 适配 | 可直接用 `action` 字段构造条件 prompt |

**处理流程**：
1. 批量解包 tar → 提取所有 MP4 + 对应 JSONL
2. 用 FFmpeg 统一转码为 720p/24fps
3. 用 LLM 根据 JSONL 生成自然语言 caption
4. 构建训练 manifest

### Charades — 可用，需补充 caption（31GB）

| 维度 | 详情 |
|------|------|
| 格式 | 9,848 个 MP4（480p） |
| 标注 | 动作类别标签，无自然语言 caption |
| Wan2.2 适配 | 需用 VLM 生成 caption |

**处理流程**：
1. 已有 480p MP4，需 FFmpeg 升采样到 720p
2. 用 VLM（如 InternVL/Qwen-VL）为每段生成描述
3. 构建训练 manifest

### EasyCom — 可用，需提取（66GB tar.gz）

| 维度 | 详情 |
|------|------|
| 格式 | tar.gz 内含 3,421 文件（含视频、音频、校准数据） |
| 标注 | 内置标注文件 |
| Wan2.2 适配 | 解压后提取视频 + 对话文本 |

**处理流程**：
1. 解压 tar.gz
2. 定位视频文件目录
3. 提取对话文本作为 caption
4. FFmpeg 统一转码

### Action100M — 元数据，非视频（37GB）

| 维度 | 详情 |
|------|------|
| 格式 | 120 个 Parquet 文件 |
| 内容 | 视频 URL（YouTube）+ 分层动作标签 |
| Wan2.2 适配 | 需先通过 yt-dlp 下载视频 |

**处理流程**：
1. 读取 Parquet，提取 YouTube URL
2. 用 yt-dlp 批量下载视频
3. 生成 caption
4. 暂不推荐优先处理（视频下载依赖网络）

### AgiBot World — 需转换（165GB）

| 维度 | 详情 |
|------|------|
| 格式 | tar.gz 内含 Parquet 格式的 episode 数据 |
| 内容 | 机器人摄像头帧 + 关节状态 + 动作指令 |
| Wan2.2 适配 | Parquet 中图片需解码导出为 MP4 |

**处理流程**：
1. 解压 tar.gz → Parquet
2. 从 Parquet 中提取图像序列 → 合成 MP4
3. 关节/动作数据生成 caption
4. 暂不优先处理（转换复杂）

### 帧级数据集 — 图片序列 → 视频转换（通用方案）

> 多项数据集（DexYCB、VFHQ、HOI4D、H2O 等）以帧级图片序列（PNG/JPG）形式提供。
> 需通过 FFmpeg 将图片序列合成为 MP4 视频后用于 Wan2.2 训练。

| 数据集 | 当前状态 | 图片数 | 组织结构 | 预估视频数 |
|--------|---------|--------|---------|-----------|
| **DexYCB** | 3/10 subjects | 348K | `{session}/{camera_id}/*.jpg` | ~300/主题 |
| **VFHQ** | ⏳ bypy 下载中 | 15K 文件夹 | `{clip_id}/*.png` | ~15K |
| **H2O** | 84GB 下载中 | 待确认 | tar.gz 解压后确认 | 待确认 |
| **HOI4D** | 需百度网盘下载 | 待确认 | 待确认 | 待确认 |

**图片序列 → MP4 转换通用命令：**

```bash
# 通用 FFmpeg 图片序列转视频
ffmpeg -framerate 30 -pattern_type glob -i '{image_dir}/*.jpg' \
  -vf "scale=1280:704:force_original_aspect_ratio=decrease,pad=1280:704:(ow-iw)/2:(oh-ih)/2" \
  -r 24 -c:v libx264 -preset fast -crf 23 -an output.mp4

# 批量转换脚本（Python）
# 遍历所有序列目录，对每个目录生成一个 MP4
```

**DexYCB 具体方案：**

DexYCB 结构为 `{subject}/{session}/{camera_id}/{frame}.jpg`。
每个 session 的每个 camera_id 是一段连续视频，需按相机分别合成。

```python
# 伪代码
for subject_dir in subjects:
    for session_dir in sessions:
        for camera_dir in cameras:
            frames = sorted(glob(f"{camera_dir}/*.jpg"))
            if len(frames) >= 10:  # 至少 10 帧才合成
                ffmpeg_images_to_video(frames, output_mp4)
```

**VFHQ 具体方案：**

VFHQ 为 15K+ 个独立文件夹，每个文件夹是一个高质量的采访人脸片段（~100 帧）。

```python
# 每个文件夹 → 一个 MP4
for clip_dir in vfhq_clips:
    frames = sorted(glob(f"{clip_dir}/*.png"))
    ffmpeg_images_to_video(frames, f"vfhq_{clip_name}.mp4")
```

> **状态：已记录方案，暂不执行转换。等 VFHQ 下载完成 + DexYCB 全部 subjects 就绪后统一处理。**

---

## 三、数据处理流水线

### Phase 1：MIRA 数据处理（最高优先级）

```
MIRA tarballs (328个, 660GB)
    ↓ tar 批量解包
MP4 + JSONL (162K对)
    ↓ FFmpeg 转码 720p/24fps
统一格式 MP4
    ↓ LLM caption生成 (基于 JSONL action 数据)
{mp4_path, caption} 对
    ↓ 构建 manifest
training.jsonl
```

### Phase 2：Charades 数据处理

```
Charades MP4 (9,848个, 31GB)
    ↓ FFmpeg 升采样 480p→720p/24fps
统一格式 MP4
    ↓ VLM caption生成
{mp4_path, caption} 对
    ↓ 构建 manifest
training.jsonl
```

### Phase 3：EasyCom 数据处理

```
EasyCom tar.gz (66GB, 3,421文件)
    ↓ tar 解压
音频 + 视频 + 标注
    ↓ 提取视频 + 对话文本
{mp4_path, caption} 对
    ↓ FFmpeg 统一转码
training.jsonl
```

---

## 四、Wan2.2 Training Manifest 格式

```json
[
  {
    "video_path": "mira_processed/clip_00001.mp4",
    "caption": "A Rocket League match from player 0's perspective. The car boosts forward, jumps to hit the ball, and performs an aerial flip. Score is 2-1.",
    "source": "mira_rocket_science",
    "resolution": "1280x704",
    "fps": 24.0,
    "duration_sec": 8.2,
    "num_frames": 197
  },
  {
    "video_path": "charades_processed/VSJ05.mp4",
    "caption": "A person sitting on a couch, reaching for a book on the coffee table, opening it and starting to read.",
    "source": "charades",
    "resolution": "1280x704",
    "fps": 24.0,
    "duration_sec": 5.5,
    "num_frames": 132
  }
]
```

---

## 五、所需工具

| 工具 | 用途 | 安装 |
|------|------|------|
| FFmpeg | 视频转码/分割 | 系统安装 |
| Python PIL/OpenCV | 帧操作 | `pip install opencv-python pillow` |
| huggingface_hub | 下载模型 | `pip install huggingface_hub` |
| VLM (InternVL/Qwen) | 自动 caption | HF 下载模型权重 |
| DiffSynth-Studio | Wan2.2 训练 | `git clone` |

---

## 六、下一步执行计划

| 步骤 | 任务 | 预计时间 | 优先级 |
|------|------|---------|--------|
| 1 | 安装 FFmpeg | 5min | P0 |
| 2 | MIRA tar 批量解包 + 统计 | 30min | P0 |
| 3 | MIRA 视频 FFmpeg 转码 | 2-4h | P0 |
| 4 | Charades 视频 FFmpeg 升采样 | 1-2h | P1 |
| 5 | EasyCom tar.gz 解压 + 视频提取 | 15min | P1 |
| 6 | 安装 DiffSynth-Studio | 5min | P1 |
| 7 | LLM caption 生成（批量） | 按需 | P2 |
| 8 | 构建 training manifest | 10min | P2 |
