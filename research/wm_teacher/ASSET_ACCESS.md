# 数据、训练权重及其他资产访问说明

更新时间：2026-11-18。本文只说明访问位置和当前备份状态，不包含任何凭据、cookies、内部节点地址或私有日志。私有仓库需由仓库所有者授予读取权限后，用自己的 Hugging Face 登录访问。

## 1. 代码与文档

| 内容 | 位置 | 状态 |
|---|---|---|
| Teacher、免训练交互世界模型原型、harness、测试、评测协议、进度文档 | GitHub `Anachronism-N/ICASSP-IPVG`，分支 `wm-teacher-experiments-20261018`，目录 `research/wm_teacher/` | 已发布；文件哈希见 `PUBLICATION_MANIFEST.json` |
| 上述公开快照的私有备份 | HF dataset `NZC415/wm-dataset`（私有）`backups/20261116/published_teacher/` | 已逐文件哈希验收 |
| 项目脚本、测试、文档、训练元数据（脱敏） | `NZC415/wm-dataset` `backups/20261116/project/` | 已验收；凭据被替换为 `<REDACTED_CREDENTIAL>` |
| DiffSynth-Studio 改动、数据脚本、demo 源码（1203 个文件） | `NZC415/wm-dataset` `backups/20261116/restoration/source_remainder.tar.gz` | 已实际下载并逐成员验收 |
| 实验文本记录（3513 份：评测结果、配置、报告） | `NZC415/wm-dataset` `backups/20261116/experiment_records/` | 已实际下载并逐成员验收 |
| 统一恢复索引（每个文件的固定 revision、路径、SHA256） | `NZC415/wm-dataset` `backups/20261118/BACKUP_RESTORE_INDEX.json`、`RESTORE_BACKUP.md` | 已验收 |

## 2. 训练数据

| 内容 | 位置 | 规模/状态 |
|---|---|---|
| Wan2.2 5 秒训练窗口（视频分片 + metadata） | HF dataset `nzc-415/wan22-5s`（公开） | 955 个分片，约 7.79 TB |
| 上述数据的补充分片 | HF dataset `NZC415/wan22-5s-trainval-overflow`（公开） | 10 个分片，约 23.27 GB |
| 已有分片覆盖 | 历史上传状态与远端分片比对 | 约 2,507,162 个窗口有对应远端分片 |
| 大型 manifest/caption（46 个） | `NZC415/wm-dataset` `backups/20261116/large_metadata/`；`final_5s_landscape.jsonl` 与 `nzc-415/wan22-5s/metadata/` 同 SHA；`captions_5s.jsonl` 与 `mira_captions_final.jsonl` 的 gzip 在 `zcoutlook/wm-backup-private` `large_metadata/` | **46/46 已验收** |
| 原始数据源、处理中间产物、旧上传状态 | 团队集群共享存储的 `wm_dataset/` 项目根目录；具体挂载路径见私有备份中的 `ASSET_ACCESS_PRIVATE.md` | 仅本地，未全部上传 |

下载示例（私有仓库需要已授权账号）：

```bash
huggingface-cli login            # 使用你自己的账号，不要把 token 写进脚本
huggingface-cli download nzc-415/wan22-5s --repo-type dataset --include "metadata/*" --local-dir ./wan22-5s
huggingface-cli download NZC415/wm-dataset backups/20261118/BACKUP_RESTORE_INDEX.json --repo-type dataset --local-dir ./restore
```

恢复时以 `BACKUP_RESTORE_INDEX.json` 里的固定 revision 和 SHA256 为准，不依赖 main 的当前状态；tar 包只接受索引列出的普通文件，恢复到新的空目录。

**许可说明**：部分来源（如 MIRA、NoXi、Seamless Interaction、YouTube 来源视频等）有禁止再分发或版权限制。公开仓库中已存在的数据不等于获得了再分发许可；使用前请核对各数据集原始条款。

## 3. 模型权重

| 内容 | 位置 | 状态 |
|---|---|---|
| 基座模型 Wan2.2-TI2V-5B 等上游权重 | 从官方发布渠道下载 | 不在本项目仓库中 |
| 本项目训练的 checkpoint（full1_5b、ft2_5b、merge 插值、round1–4 LoRA、e3_arms、smoke 等，共 65 个 safetensors，约 217.25 GB） | HF dataset `zcoutlook/wm-backup-private`（私有）`checkpoints/models/train/...`，路径与集群 `wm_dataset/models/train/` 一致 | **65/65 已上传**，逐文件远端 SHA256 验收 |
| HF 账号下的模型仓库 `NZC415/causal-forcing-stage2-ode-i2v`、`NZC415/m2d` | HF model（公开） | 不是本项目 Teacher checkpoint |

## 4. 评测媒体及其他资产

| 内容 | 位置 | 状态 |
|---|---|---|
| 评测视频、帧图、demo 媒体（约 20,891 个文件，约 29.75 GB） | `zcoutlook/wm-backup-private` `media/eval_demo/evalmedia_0000{1..5}.tar`（每个 tar 附 `.index.json` 成员 SHA256）；其余在集群 `wm_dataset/eval/`、`demo/` | 6 个分片中 5 个已上传（13,975 个文件，约 22.7 GiB）；`evalmedia_00000` 因额度满未传 |
| 补充源码/运行脚本/旧节点文档（9296 个文件，脱敏） | HF dataset `zcoutlook/wm-backup-private`（私有）`restoration/supplemental_source_runtime.tar.gz` | 已上传并哈希验收 |
| 运行环境、旧集群迁移包（16 个 tar，约 264.2 GB） | 集群共享存储中的迁移备份目录 | 未上传；只有本地 SHA256 验收记录 |
| 人工审阅私密记录、凭据、cookies | 不上传 | 按设计排除 |

## 5. 存储状态与已满位置（2026-11-18 核查）

| 存储 | 状态 |
|---|---|
| HF 账号 `NZC415` **私有存储** | **已满**。LFS 上传预检返回 403 `Private storage limit reached`；该账号私有仓库合计约 128.8 GB。小型文本提交仍可成功，大文件无法再上传 |
| HF 账号 `zcoutlook` 私有存储 | **已满**（2026-11-19 返回 403 `Private storage limit reached`）。已存约 237.5 GiB：补充归档、2 份大 caption、65 个权重、5 个评测媒体分片、5 个 extra 视频分片 |
| HF 公开仓库 | 可写；但公开发布涉及数据许可和模型发布决定，未擅自使用 |
| 团队集群共享存储 | 约 87% 已用，仍有约 245 TB 可用；所有原始数据和权重仍在此 |
| 当前节点本地盘 | 约 7% 已用 |

`NZC415`、`zcoutlook` 两个账号私有存储均已满。截至 2026-11-19 仍未上传：

- 评测媒体分片 `evalmedia_00000`（约 5 GB）
- 历史上传状态中尚未上传的 257,333 个 5 秒窗口（约 900 GB，计划在私有 `zcoutlook/wan22-5s-remaining` `shards_legacy/`，0/215 分片）
- `wan22_5s_extra` 新增窗口：已上传 6,000 个（`zcoutlook/wan22-5s-remaining` `shards_extra/extra_0000{0..4}.tar`），剩余约 22,500 个（19/24 分片）
- 旧集群迁移包（约 264 GB）与各数据集原始下载文件

评测媒体及原始数据涉及来源许可，暂不上传；若 `zcoutlook` 也满额，继续上传需要以下任一条件：升级私有存储额度、在已有额度的另一授权账号/组织中提供写权限，或由所有者明确决定哪些内容可以公开发布、哪些旧私有仓库可以清理。
