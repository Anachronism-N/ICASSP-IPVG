# Wan2.2 训练数据集整理与 Caption 状态

> 最后更新：2026-07-20
>
> **状态提示：当前 MIRA 2,026,000 条最终 Caption、数据质量审计和首轮训练步骤以 [24_Wan2.2首轮训练执行计划.md](./24_Wan2.2首轮训练执行计划.md) 为准。**

---

## 一、Wan2.2 已转换数据

| 数据集 | 视频数 | 大小 | Caption | 方案 |
|--------|--------|------|---------|------|
| **MIRA** | ~1.3M | ~3T | JSONL action | LLM 将 action→自然语言（已修复） |
| **Charades** | 9,848 | 32G | 动作标签→自然语言 | 基于官方 annotations 生成 caption |
| **EasyCom** | 271 | 2.3G | 对话文本 | 已有 |
| **RAVDESS** | 2,880 | 1G | 情感标签 | 已删除/低优先级 |
| **DexYCB** | 2,400 | 19M | ✅ 已转换 | 图片序列→视频 (1280x704, 24fps) |
| **H2O** | 60 | — | ✅ 已转换 | 图片序列→视频 |
| **EPIC** | 3 | 295M | 动作标注 | 场景单一/暂缓 |

### MIRA Caption 修复

已解决的 3 个问题：

1. **arena 字段修复**：原先 100% 条目显示 "Unknown" → `build_template_caption()` 默认值改为 "Rocket League Arena"
2. **time_start 字段修复**：原先 75.3% 条目为 0.0 → 模板现在使用游戏时钟/回合数据替代原始 time_start
3. **Caption 长度优化**：原先平均 757 字符（首轮训练过长）→ 模板重写为 80-200 字符
4. **新增 caption_i2v 字段**：图像转视频格式，更短（40-80 字符）
5. **性能优化**：SQLite 缓存构建完成 (`dataset/mira_jsonl_cache/cache.db`, 2GB, 2.2M summaries)，速度提升 168x（1.5→252 captions/sec，15 天→2 小时）

### Charades Caption 修复

1. **问题**：原先 9,848 条 caption 均为 "A person performing a daily activity..."（通用占位文本）
2. **修复**：从 AI2 S3 下载官方 Charades annotations（157 个动作类别：taking_bag, sitting_at_table, opening_door 等）
3. **新 caption 格式**："In a bedroom, a person is taking a bag from somewhere, sitting at a table..."
4. **I2V 格式**："Person taking a bag and sitting at a table"
5. **输出**：`dataset/wan22_training/charades_processed/manifest.jsonl`

---

## 二、原始数据待处理

| 数据集 | 大小 | 格式 | Caption | 方案 |
|--------|------|------|---------|------|
| **OpenVidHD** | 3.0T | 14 zips, 300K+ MP4 | ❌ | **需 VLM 打 caption** |
| **HOI4D** | 3.3G | 下载中 | 标注存在 | 视频含 3D 标注 |
| **EPIC-KITCHENS** | 8.7G | 28 MP4 | 动作标注 | 场景单一/暂缓 |
| **EasyCom** | 102G | 已提取 | 对话文本 | 已有 |

---

## 三、新下载数据（待处理）

| 数据集 | 大小 | Caption | 方案 |
|--------|------|---------|------|
| **HOIGen-1M** | 235G, 10 zips | CSV caption | 已提取 106,100 视频，caption 来自数据集 metadata |
| **Seamless Interaction** | 50G, 50 tars | 待确认 | 已从 50 tars 提取 476 视频，caption 来自数据集 metadata |
| **Matrix Dataset** | 下载中 | 游戏控制信号 | 控制信号→通用 caption |
| **CelebV-HQ** | 76 videos | 属性标注 | 属性→自然语言 |
| **Action100M** | 278 videos | 动作标签 | 标签→自然语言 |

---

## 四、Caption 需求总结

| 需打 Caption | 视频数 | 打标方式 |
|-------------|--------|---------|
| **OpenVidHD** | 300K+ | VLM (InternVL/Qwen-VL) |
| **MIRA** | 1.3M | ✅ 已用 JSONL action 模板生成 |
| **Charades** | 9,848 | ✅ 已用官方 annotations 生成 |
| **DexYCB** | 2,400 | ✅ 图片序列已转视频 |
| **H2O** | 60 | ✅ 图片序列已转视频 |

| 已有 Caption | 视频数 | 说明 |
|-------------|--------|------|
| **HOIGen-1M** | 106,100 | 已提取，caption 来自 dataset metadata |
| **Seamless Interaction** | 476 | 已提取，caption 来自 dataset metadata |
| **EasyCom** | 323 | 对话文本 |
| **RAVDESS** | 2,880 | 情感标签（低优先级） |

---

## 五、打 Caption 方案

1. **VLM 方案**：用 InternVL/Qwen-VL 对视频帧采样生成自然语言描述
2. **MIRA 模板方案**：✅ JSONL 含 action/physics → 已用 LLM 模板转为自然语言
3. **HOIGen-1M**：✅ 已提取 106,100 视频，直接使用 CSV caption，无需额外处理
4. **Seamless Interaction**：✅ 已从 50 tars 提取 476 视频，caption 来自 metadata
