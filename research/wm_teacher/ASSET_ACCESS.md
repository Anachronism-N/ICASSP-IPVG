# 数据、训练权重及其他资产访问说明

更新时间：2026-11-20。本文只说明访问位置和当前备份状态，不包含任何凭据、cookies、内部节点地址或私有日志。私有仓库需由仓库所有者授予读取权限后，用自己的 Hugging Face 登录访问。

## 当前续传范围（2026-11-19）

私有备份保留不变：65 个训练权重、两份大 caption、补充源码记录已完成；评测媒体 5/6 分片、extra 窗口 6000 个已完成。

公开续传已改为逐来源许可筛选，而非将全部私有缺口直接公开。前一公开任务只完成 17 个元数据对象，视频分片未验收完成。已经私有备份的 6000 个 extra 窗口不再次上传。

| 本轮范围 | 公开位置 | 状态 |
|---|---|---|
| RoboMIND 6000 个 5 秒窗口 | `zcoutlook/wan22-5s-remaining-public/eligible_shards/robomind/` | **5/5 分片、6000/6000 窗口已独立验收**；实际来源为 Traly/RoboMIND-lerobot，CC BY-NC 4.0，附署名、许可链接和处理说明 |
| UK-Road-DashCam 4510 个 5 秒窗口 | `zcoutlook/wan22-5s-remaining-public/eligible_shards/ukdash/` | **4/4 分片、4510/4510 窗口已独立验收**；源数据卡声明 MIT，附源出处和许可说明 |
| DexYCB 2400 个旧预处理片段 | `zcoutlook/wan22-4s-clips/eligible_shards/dexycb/` | **2/2 分片、2400/2400 片段已独立验收**；CC BY-NC 4.0，保留原始长度信息，不全部标为 4 秒 |
| 257333 个旧缺失窗口 | 集群原件保留 | 未传。MIRA、NoXi、Seamless 等在来源条款中禁止再分发；VFHQ 等仍需核实条款 |
| OpenVidgen 12000 个 extra 窗口 | 集群原件保留 | 未传；视频版权及公开再分发授权待确认 |
| 混合评测分片 evalmedia_00000、旧迁移包、其他原始数据 | 集群原件保留 | 未传；不能将来源视频帧、私有日志或未经审查的包整体公开 |

更正：旧文档把 Charades 简写为 CC BY-NC 并不准确。其数据许可明确不授予公开发布修改数据或向第三方分发数据的权利，本轮不公开上传该来源。RoboMIND 的本地下载来源是 Traly 转换版（CC BY-NC），不能以另一个官方仓库的 Apache 标签替代该版本许可。

分片完成以远端文件大小、SHA256 和成员索引验收为准；“上传中”不表示已经全部完成。受限来源继续备份需符合原条款的授权私有存储，或提供另外取得的公开再分发授权；仅允许将仓库公开不等于取得来源授权。

### 本轮完成验收（2026-11-19）

本轮 **11/11 个分片、12,910/12,910 个视频**完成上传及独立校验，tar 合计 **50.50 GB**。这仅代表本轮来源许可允许的子集完成，**不是全部数据资产已上传**。

- 在每个分片固定 commit 上核对远端字节数和 LFS SHA256；实际下载 .tar.index.json 并与冻结计划逐成员比较。
- 本地已上传 tar 的完整 SHA256 与远端一致，全部 12,910 个成员逐一核对字节数与 SHA256，无遗漏、重复或缺失源；没有再次全量下载约 50 GB 的 tar。
- 已私有上传的 6000 个 extra 窗口与本轮成员集合无交集，未重复上传。
- 两个公开仓库均有 eligible_shards/VERIFICATION_20261119.json，记录每个分片的固定 revision、tar SHA256、成员索引 SHA256 及验收范围。
- **仍未上传**：257,333 个受限/待核实来源的旧 5 秒窗口、12,000 个 OpenVidgen extra 窗口、混合评测包 evalmedia_00000，以及其他原始数据和旧迁移包；许可和私有容量限制未解除。

### 追加续传批次（2026-11-19）

本批 **934/934 个文件、115.36 GB 已完成上传**；2026-11-20 已按冻结计划独立核对全部远端文件大小与 LFS SHA256，并下载 17 份批次索引验证无遗漏或重复。此前逐文件提交触发的 429 限流已通过冷却后每 50 个文件批量提交解决，旧错误记录保留；本批后续无新错误。

| 内容 | 公开位置（HF dataset） | 数量及大小 | 状态 |
|---|---|---|---|
| 未备份评测包中核对 jobs 输出映射的模型生成视频，仅使用许可核实的 RoboMIND/UKdash 输入 | `zcoutlook/wan22-5s-remaining-public/generated_media/licensed_inputs/` | 216 个，约 0.293 GB | **216/216 已完成并独立验收**；不是源视频/GT |
| 本地已下载的 Traly/RoboMIND-lerobot 原始 MP4 子集 | `zcoutlook/wan22-5s-remaining-public/original_sources/robomind/` | 718 个，约 115.07 GB | **718/718 已完成并独立验收**；CC BY-NC 4.0，保留出处与许可说明 |

每个文件以本地计算出的 SHA256 和远端 LFS SHA256/大小一致后记为完成。全部 934 个文件的远端快照验收通过；没有再次下载 115 GB 视频。验收清单位于 `zcoutlook/wan22-5s-remaining-public/original_sources/robomind/ADDITIONAL_ASSETS_VERIFICATION_20261120.json`，包含固定 revision、逐文件路径、SHA256 与批次索引证据。两类文件不是此前已上传的 6000 个 extra 窗口，不重复传该窗口集。RoboMIND 原始文件是本地选取的子集，不宣称完整上游数据集全部备份。

其余未传资产继续单独核实：257,333 个旧窗口、12,000 个 OpenVidgen 窗口、剩余混合评测媒体、绝大部分旧训练预处理数据、原始下载与旧迁移包仍未全部上传。来源明确禁止再分发、授权待核实或含凭据的内容不整体转公开。

### 继续补齐原始视频与旧预处理片段（2026-11-20）

以下两批均已独立验收完成；**不代表所有数据资产全部完成**：

| 内容 | 公开仓库/目录 | 范围 | 状态 |
|---|---|---|---|
| 本地 UK-Road-DashCam 原始 MP4 | `zcoutlook/wan22-5s-remaining-public/original_sources/ukdash/` | 274 个、174.67 GB | **274/274 原始视频已上传并独立SHA验收**；20 文件一提交 |
| RoboMIND 元数据、DexYCB 训练 manifest 与源许可说明 | `zcoutlook/wan22-5s-remaining-public/original_sources/{robomind,dexycb}/` | 23 个元数据文件、3 个许可说明；约 91.03 MB | **26/26 已上传并独立SHA验收**；文本快照移除集群绝对路径和凭据，不修改源文件 |
| RAVDESS 本地预处理 MP4 | `zcoutlook/wan22-4s-clips/eligible_shards/ravdess/` | 2880 个、约 1.04 GB，4 个分片 | **4/4 已上传并独立验收，2880/2880 成员SHA通过**；保留 CC BY-NC-SA 4.0、署名和相同许可要求 |

更正早前遗漏：UK-Road-DashCam 的本地原始文件后缀是大写 .MP4，此前只检查小写 .mp4 导致错误判断为无原始视频；本批使用大小写无关的文件类型检查。已完成的 4510 个 UKdash 5 秒窗口与这 274 个原始文件是不同资产，不把原始视频备份记为窗口重复传输。

RAVDESS 来源为 Livingstone 与 Russo 的官方发布，处理子集保持非商业、署名、相同许可条件。只上传本地2880个预处理视频，不把其数量外推为完整上游音频/歌曲数据集覆盖。

**300/300 个文件、174.76 GB 已全部上传并独立验收**：冻结计划与远端路径、大小及SHA256一一匹配，实际下载15份批次索引验证无遗漏或重复，未再次下载174GB视频。验收证据为 zcoutlook/wan22-5s-remaining-public/original_sources/NEXT_ASSETS_20261120/VERIFICATION.json。4个RAVDESS分片已经按远端固定revision、tar SHA256、实际下载的成员索引及全部2880个本地tar成员SHA256独立验收通过。遇额度、限流或校验错误停止，不改公开性、不删除源文件、不启动GPU。

其余受限旧窗口、OpenVidgen、混合评测残余、其他原始数据和旧迁移包仍未全部上传，原件保留。重新处理不解除已知来源许可；全量备份完成标记仍为false。

### DexYCB原始压缩包追加备份（2026-11-20）

**3/3 个原始压缩包、36.61 GB 已完成上传并独立验收**，普通文件成员共 **522,936 个**。公开位置：

`zcoutlook/wan22-4s-clips/original_sources/dexycb_archives/`

本批是本地已下载的原始 RGB、深度、标签数据，而不是此前已经备份的2400个处理视频片段。来源为DexYCB，保留原作者署名和CC BY-NC 4.0非商业许可；没有修改原始压缩包。

| 本地保留文件名 | 压缩包实际内部根目录 | 状态 |
|---|---|---|
| 20200709-subject-01.tar.gz | 20200709-subject-01 | **已完整审计、上传并独立SHA验收** |
| 20200709-subject-02.tar.gz | 20200813-subject-02 | **已完整审计、上传并独立SHA验收** |
| 20200709-subject-03.tar.gz | 20200820-subject-03 | **已完整审计、上传并独立SHA验收** |

每个包已完整解码gzip并校验所有普通成员路径、大小及SHA256，远端压缩包大小及LFS SHA256与审计缓存一致；实际下载 .members.jsonl.gz 成员索引和 .audit.json 审计记录，核对522,936个成员计数、总字节量与真实subject根无遗漏。独立验收证据：`original_sources/dexycb_archives/VERIFICATION_20261120.json`。拒绝路径穿越、链接/设备成员、凭据内容或损坏包，不把目录名称或上传进程存活当完成证据。三个包只代表本地下载的部分subject，不是完整DexYCB数据集。

已传的UK原始视频、RoboMIND、DexYCB/RAVDESS处理片段与65权重不重传。其余受限来源与未安全审阅的旧迁移包仍需适用授权和足够的私有存储；当前全部资产完成标记保持false。

### 补齐此前大小门排除的节点manifest（2026-11-20）

旧节点源码/运行记录归档中有4份manifest因单文件约48.61 MB超过文本大小门而未包含。现已确认4份源文件SHA256完全相同，去重为一个**50000行、压缩约6.56 MB的脱敏私有快照**，上传并实际下载解压校验通过。

位置：`NZC415/wm-dataset/backups/20261120/excluded_node_manifest/`；包含manifest.sanitized.jsonl.gz与RESTORE_INDEX.json。固定revision：b6f80f67abbfff0d47215f831aac8ce69bbc4b70。原始文件未修改，备份说明包含源SHA、存储SHA与4个节点的相同内容引用；这不代表其引用的视频或50000条训练任务均已备份/运行。

### EgoDex既有上游副本身份核验（2026-11-20）

本地下载根目录 `dataset/new_sources/egodex/` 下的 **989 个文件、约91.41 GB** 已逐一确认与既有公开上游副本字节一致：731个LFS对象以本地SHA256对照上游LFS SHA256，258个普通Git元数据以本地Git blob SHA1对照上游对象；无缺失或不一致。

- 上游公开位置：`griffinlabs/EgoDex-LeRobot-v3.0`；固定revision：41d60b449629b2181ff5b735d31c2a2cf8b3cad8。
- 身份及恢复引用证据（私有）：`NZC415/wm-dataset/backups/20261120/EGODEX_EXISTING_REMOTE_REFERENCE.json`；固定revision：1af35218de934dc45d5abf872b037bb60ee13525。
- **这是确认第三方现有副本可用于恢复，不是将91GB原始数据重新上传到本人账号，也不是独立自有备份。** 上游仓库的可访问性由其所有者决定。
- 本地来源为声明CC BY-NC-ND 4.0的转换版；本次不改变视频、重新裁切或公开再发布。先前6000个私有EgoDex窗口不因此计为完整原始数据备份。

全项目仍未备份完成：剩余257333旧窗口、12000个OpenVidgen、6637个混合评测文件及其余原始数据/迁移包仍未形成完整自有远端覆盖。两个私有账号5GiB容量预检仍返回403，现无可用的其他私有对象存储配置。已有公开副本或重新处理不能替代受限来源的再分发授权；需要所有者提供适用许可允许的私有存储或额外来源授权。

### Physics-IQ Original混合评测缺口补齐（2026-11-20）

从未传混合评测包中单独筛选并完成 **4个分片、2248个文件（2050个既有模型生成视频和198张原版switch-frame输入图），源字节约1.08GB**。位置：`zcoutlook/wan22-5s-remaining-public/generated_media/physics_original/`；完成证据为同目录VERIFICATION_20261120.json。

来源为Physics-IQ Original公开基准，已核对官方代码仓库固定commit3aa11416b94e32723a7eda6719a58193d90138f9的“All other materials”CC BY 4.0原文，附作者署名、许可与改动说明。并非Physics-IQ Verified gated数据，不包含其他数据集来源，未向榜单提交实验。

全部视频路径及输入关系已与既有jobs清单关联；每个tar的远端固定revision、LFS SHA和大小核验通过，实际下载成员索引对照冻结计划；上传的本地tar所有成员SHA复核通过，无遗漏重复。模型输出不被标记为GT或原始拍摄视频，备份不证明基准分数或模型质量。

先前混合评测包剩余6637个文件中，本批补齐2248个；该包仍剩4389个文件，约3.995GB（以最新差集清单为准），不能视为整个evalmedia_00000已完成。受限旧窗口、OpenVidgen、其他原始数据及迁移包仍未形成全量备份。

### Physics-IQ E2输出重复副本去重（2026-11-20）

继续从未传混合评测包中核实198个E2模型输出及198个相同内容的本地副本：生成任务、原版Physics-IQ输入帧、模型base_s12c5、seed/CFG/shift映射已核对，两套文件的SHA256和大小逐一相同。

现已只上传**198个规范视频（tar约92.78 MB）**，成员索引记录相同本地别名，能够恢复**396条本地文件路径**，没有重复传输185.26 MB两份相同视频，也没有删除或修改本地副本。

公开位置：`zcoutlook/wan22-5s-remaining-public/generated_media/physics_original_e2/`；包含canonical_outputs.tar、.index.json、NOTICE.md、VERIFICATION_20261120.json。198个成员SHA及固定revision远端tar SHA通过，成员索引实际下载与冻结计划完全一致。这些是模型生成输出，不是源拍摄视频或GT，也没有重新执行GPU实验或提交榜单。

本批去重后，混合评测残余为**3993个文件、3.8099 GB**。257333个旧窗口、12000个OpenVidgen及其他原始数据、迁移包仍未完成；许可和私有容量阻塞不因此解除。

## 1. 代码与文档

| 内容 | 位置 | 状态 |
|---|---|---|
| Teacher、免训练交互世界模型原型、harness、测试、评测协议、进度文档 | GitHub `Anachronism-N/ICASSP-IPVG`，分支 `wm-teacher-experiments-20261018`，目录 `research/wm_teacher/` | 已发布；文件哈希见 `PUBLICATION_MANIFEST.json` |
| 上述公开快照的私有备份 | HF dataset `NZC415/wm-dataset`（私有）`backups/20261116/published_teacher/` | 已逐文件哈希验收 |
| 项目脚本、测试、文档、训练元数据（脱敏） | `NZC415/wm-dataset` `backups/20261116/project/` | 已验收；凭据被替换为 `<REDACTED_CREDENTIAL>` |
| DiffSynth-Studio 改动、数据脚本、demo 源码（1203 个文件） | `NZC415/wm-dataset` `backups/20261116/restoration/source_remainder.tar.gz` | 已实际下载并逐成员验收 |
| 实验文本记录（3513 份：评测结果、配置、报告） | `NZC415/wm-dataset` `backups/20261116/experiment_records/` | 已实际下载并逐成员验收 |
| 统一恢复索引（每个文件的固定 revision、路径、SHA256） | `NZC415/wm-dataset` `backups/20261120/CHECKED_CROSS_ACCOUNT_INDEX_V3.json`；旧文本恢复说明仍在 `backups/20261118/RESTORE_BACKUP.md` | 最新跨账号索引已上传并重新下载验证，固定 revision `80cd6fb97e208b155ccee0d32b1e3fd2bf644591`；包括新增Physics资产、EgoDex第三方引用与未传阻塞，**不表示全量已完成** |

## 2. 训练数据

| 内容 | 位置 | 规模/状态 |
|---|---|---|
| Wan2.2 5 秒训练窗口（视频分片 + metadata） | HF dataset `nzc-415/wan22-5s`（公开） | 955 个分片，约 7.79 TB |
| 上述数据的补充分片 | HF dataset `NZC415/wan22-5s-trainval-overflow`（公开） | 10 个分片，约 23.27 GB |
| 已有分片覆盖 | 历史上传状态与远端分片比对 | 约 2,507,162 个窗口有对应远端分片 |
| 大型 manifest/caption（46 个） | `NZC415/wm-dataset` `backups/20261116/large_metadata/`；`final_5s_landscape.jsonl` 与 `nzc-415/wan22-5s/metadata/` 同 SHA；`captions_5s.jsonl` 与 `mira_captions_final.jsonl` 的 gzip 在 `zcoutlook/wm-backup-private` `large_metadata/` | **46/46 已验收** |
| 4 秒预处理片段（`wan22_training/`，Round1 用其前 49 帧训练；约 159 万段 + MIRA，含中间产物共约 10.3 TB） | 仅集群共享存储 `wm_dataset/dataset/wan22_training/` | **整体未完成；DexYCB 2400 个片段和 RAVDESS 2880 个片段已公开备份**；HF 上 `wan22-processed-clips` 已于 2026-10-01 清空（当时认为被 5 秒数据取代）。Round1 训练清单与 caption 在私有备份 `training_metadata/` |
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
| HF 公开仓库 | 用户已授权使用；只继续上传经核实可再分发的来源，私有仓库保持私有 |
| 团队集群共享存储 | 约 87% 已用，仍有约 245 TB 可用；所有原始数据和权重仍在此 |
| 当前节点本地盘 | 约 7% 已用 |

`NZC415`、`zcoutlook` 两个账号私有存储均已满。截至 2026-11-19 仍未上传：

- 评测媒体分片 `evalmedia_00000`（约 5 GB）
- 历史上传状态中尚未上传的 257,333 个 5 秒窗口（约 900 GB，计划在私有 `zcoutlook/wan22-5s-remaining` `shards_legacy/`，0/215 分片）
- `wan22_5s_extra` 新增窗口：已上传 6,000 个（`zcoutlook/wan22-5s-remaining` `shards_extra/extra_0000{0..4}.tar`），本轮另在公开仓库验收 RoboMIND 6000 个和 UKdash 4510 个，累计覆盖 16,510/28,510 个；剩余 OpenVidgen 12,000 个未传
- 旧集群迁移包（约 264 GB）与各数据集原始下载文件

评测媒体及原始数据涉及来源许可，暂不上传；若 `zcoutlook` 也满额，继续上传需要以下任一条件：升级私有存储额度、在已有额度的另一授权账号/组织中提供写权限，或由所有者明确决定哪些内容可以公开发布、哪些旧私有仓库可以清理。

## 6. 全部数据存储位置总表（2026-11-19 核查）

### Hugging Face

| 账号 | 仓库 | 可见性 | 大小 | 内容 |
|---|---|---|---|---|
| nzc-415 | `wan22-5s` | 公开 | 7.79 TB | 5 秒训练窗口 955 个 tar 分片（2,483,000 个窗口）+ `metadata/`（final manifest、caption） |
| NZC415 | `wan22-5s-trainval-overflow` | 公开 | 23.3 GB | Round2/3 训练与验证用到但主仓库缺的 24,162 个 5 秒窗口 |
| zcoutlook | `wan22-5s-remaining-public` | 公开 | 本轮视频约 49.52 GB | eligible_shards/robomind/ 5 分片 + eligible_shards/ukdash/ 4 分片，共 10,510 窗口；附成员索引和验收凭证 |
| zcoutlook | `wan22-4s-clips` | 公开 | 本轮视频约 0.98 GB | eligible_shards/dexycb/ 2 分片、2400 个 DexYCB 片段；其他来源没有在本轮公开上传 |
| zcoutlook | `wan22-5s-remaining` | 私有 | 12.0 GB | `shards_extra/`：`wan22_5s_extra` 新增窗口 5/24 分片（6,000 个） |
| zcoutlook | `wm-backup-private` | 私有 | 243 GB | 65 个训练权重、评测媒体 5/6 分片、2 份大 caption、补充源码/运行记录 |
| NZC415 | `wm-dataset` | 私有 | 3.0 GB | 代码/文档/实验记录/元数据备份、44 份大型 manifest、恢复索引与访问说明 |
| NZC415 | `CelebV-HQ-archives`、`VFHQ-videos`、`VFHQ-videos-sharded`、`VFHQ-videos-archives`、`direct-downloads-sharded`、`direct-downloads-archives` | 公开/私有混合 | 约 512 GB | 部分原始数据源下载包（CelebV-HQ、VFHQ、直链下载类） |
| zcoutlook | `wan22-4s-clips` | 公开 | 新增原始包36.61 GB | `original_sources/dexycb_archives/`：3个DexYCB原始压缩包（原始RGB/深度/标签），522,936普通文件成员已审计；另有4个RAVDESS处理分片，共2880视频 |
| NZC415 | `dataset-metadata`、`action100m`、`source-json`、`url-datasets` | 私有 | 约 2.5 GB | 数据源元数据、下载 URL 清单 |
| NZC415 / nzc-415 | `wan22-processed-clips`、`wan22-clips-public` | 公开 | 0 | 空仓库（旧 4 秒片段已清空） |

账号私有存储：NZC415 与 zcoutlook 均已满（LFS 403）。nzc-415 的公开存储也曾在 2026-10-12 触发上限。

### 只在集群共享存储（未上传）

| 内容 | 规模 |
|---|---|
| 原始数据源下载（egocentric_interaction、general_action、general_video、digital_human、new_sources） | 约 27.7 TB（部分来源在上表原始数据仓库中有副本） |
| 4 秒预处理片段及中间产物 `wan22_training/` | 约 10.3 TB（MIRA 约 6.0 TB，CelebV-HQ 约 2.7 TB） |
| 5 秒窗口中未上传部分：旧缺失 257,333 个（MIRA 208,749、NoXi 22,399、Seamless 16,622、VFHQ 9,536、OpenVidHD 27），extra 剩余 12,000 个（OpenVidgen） | 约 1 TB |
| 评测媒体分片 evalmedia_00000 | 约 5 GB |
| 上游基座模型权重、旧集群迁移包（环境/缓存/日志） | 约 0.5 TB + 264 GB |

### GitHub

`Anachronism-N/ICASSP-IPVG` 分支 `wm-teacher-experiments-20261018` 的 `research/wm_teacher/`：代码、测试、文档；不含数据和权重。
