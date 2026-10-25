# 完整插值评测验收更新

> 2026-10-18 23:13:15 CST：原3033视频完整解码、三模型全部WM/公共VBench8维、GTref108、Physics198/66、WMB350与三份固定120loss已严格通过；summary和Teacher页重建完成。见docs/32_插值Teacher完整评测结论.md。实验执行/覆盖完成不等于Teacher质量自动或人工验收通过；下文旧快照按时间解释。

# 项目进展全记录

> 最后更新：2026-10-18 21:38 CST；旧机01:21暂停仅历史。

## 当前状态

- 三插值3033视频全部生成且完整解码；三模型GTref108、Physics198/66与固定120loss均已独立严格验证。数据、checkpoint、prompt、seed和Phase E3913输出及raw/po评分保留。
- Physics raw A03/A05/A07=29.634157/28.169852/26.875848；DreamSim=0.16940009/0.16556586/0.15975603；loss=0.19359857/0.19132043/0.18940758。全部是固定seed本地实测，配对区间见eval/diag/merge_all_analysis_20261018.json，物理与领域收益有tradeoff。
- 三模型WM8维评分完整，公共VBench和三节点WMB继续并行；final gate仍pending，Teacher质量/用户review未通过，不能开始Student/大训练。当前资源与日志看docs/31顶部和monitor JSON。
- 持久化评分/监控/验收及安全公开导出已回归验证。GitHub代理与授权权限已解决，发布到指定ICASSP-IPVG独立分支及research/wm_teacher子目录，不携带原仓cookies历史。首批快照已正常push并核验远端040814269512ddad93cc9734811940bd8de54630；最终全部评分验收后追加报告并再次核对远端，不销毁旧机。

## 2026-10-18 私有本地资产导出（12:15历史验证记录）

> 以下记录当时的导出边界；新机之后已恢复所需生成/评分资产，当前状态见上文。

- 用户要求旧机资产私下导出到仓库外共享目录：`PRIVATE_MIGRATION_BACKUP`，根目录700、文件600，不公开上传。
- 现有会话BatchMode SSH仍可访问旧worker，无需保存或重新获取密码；新服务器密码认证失败不影响本次旧会话导出。
- **2026-10-18 12:15:23 CST完整私有备份验证通过**：worker6份约94.36GB、launcher9份约169.12GB，加VBench外链真实权重约0.72GB，总16归档264,197,908,480字节。全归档SHA256、15个源目录对比及额外目标对比PASS；207外链均覆盖，pip/conda导出成功，无partial，700/600权限核验通过。
- 完成证据`manifests/backup_complete.json`及`manifests/final_verification.log`；恢复说明`RESTORE_PRIVATE.md`。新Agent从共享盘恢复即可，不再需要密码登录旧机下载本次列出的资产。回收旧机前仍要确认新机可访问共享备份和项目ROOT、没有其他任务/资产依赖；本次未关机或实际解包，插值实验仍暂停未完成。

## 2026-10-18 旧服务器暂停交接（01:21历史冻结快照）

> 以下183/3033、2850待恢复与旧15卡信息仅描述旧机冻结时刻；新集群已接续，不是当前数量或待办。

- 用户明确要求停止旧机推理，覆盖此前持续运行目标。两节点本项目调度/15个生成进程/后续等待评分/demo自动刷新均已停止，GPU显存已释放；没有关机或销毁旧机器。
- Phase E的3,913输出及14组评分完整覆盖PASS，summary/Teacher页已保存，质量gate仍未通过。
- 插值已保存183/3,033个视频（α0.5 WM99/108，VBench84/355），其余2,850任务待新机完成，评分及固定val loss未完成，不能当作实验完成。
- 恢复清单`eval/diag/jobs_merge_pending_20261018.jsonl`；暂停状态/完成输出/15个残留claim见`eval/diag/migration_pause_20261018.json`。完整覆盖仍验证`eval/jobs_merge_all.jsonl`。
- 原始输出、jobs、prompt和模型未删除或更改。新机确认旧任务退出后再处理记录中的残留claim，不批量删活跃锁。释放旧机器前需私下备份所需`/dockerdata`模型/环境/benchmark和日志，GitHub尚未成功推送。

---

## 0、本轮推进记录（2026-10-09，迁移到 2×8 A100-40GB；历史记录）

> 本节旧双节点/15卡、当时运行状态和调度关系均为历史。当前新4节点/32卡进度以上方快照和交接文档顶部为准。

### 环境
- 节点：launcher `PRIVATE_NODE_ADDRESS`（rank0）+ worker `PRIVATE_NODE_ADDRESS`（rank1），无 IB，NCCL 走 eth1
- worker 需手动挂载 `/srv/PRIVATE_STORAGE`（taiji_client mount，见用户指令）
- **worker GPU7 有 Xid 31 MMU 硬件故障**（vLLM / 训练均触发 illegal memory access）→ 训练用 15 卡
- 共享盘上的 conda env 经 fuse import 极慢 → 改用两节点本地 `/opt/conda/envs/torch-base`（torch 2.7.1, flash_attn 2.8.1），权重复制到 `/dockerdata/models`
- fuse 会给新进程返回刚编辑脚本的旧内容 → 所有长任务脚本先复制到 `/dockerdata/wm_tmp` 再运行

### 5s 数据集（121 帧窗口）收尾 ✅
- reslice 补切：+2,775 窗口；30,865 窗口源文件损坏（egoexo4d/mira/celebvhq 等），不可恢复
- caption 补齐：Qwen3.5-35B-A3B，vLLM 0.31.0 TP=4（A100 上 bf16 67GB 单卡放不下），补 2,565 条，12 条顽固失败
- `finalize_5s.py` 完成：`dataset/wan22_5s/final_5s_{landscape,portrait}.{jsonl,csv}`，缺 caption 仅 13 条
- 总计 sliced 2,764,495 窗口

### Round2 结果 ✅（2026-10-11 完成）
- 训练：run1 step 0–4500（10-10 16:03 因共享盘短时读失败崩溃于 4946 步）+ run2 从 step-4500 续训剩余 29,055 未训样本，1937 步，10-11 03:09 正常结束；最终 ckpt `models/train/round2_5s_lora_r32/run2/step-1937.safetensors`（全局约 step 6437）
- 修复：`DiffSynth-Studio/diffsynth/core/data/unified_dataset.py` 数据读取改为 12 次指数退避重试（原 3 次无等待）
- 留出 val loss（120 条 5s 片段，121 帧，固定 5 个时间步 + 固定噪声，`eval/loss_*.json`）：

| 模型 | val loss | vs base | 逐片段胜率 |
|---|---|---|---|
| Wan2.2-TI2V-5B base | 0.1975 | — | — |
| Round1（49 帧） | 0.1922 | −2.7% | 119/120 |
| Round2 中间（全局 step 3000） | 0.1884 | −4.6% | — |
| **Round2 最终** | **0.1879** | **−4.9%** | **120/120**（vs Round1 也是 120/120，−2.3%） |

  9 个数据集全部单调改善，mira（−8.5%）、direct（−9.1%）、noxi（−5.6%）提升最大；后半程（step 3000→最终）只再降 0.3%，已接近饱和
- 生成样例：`eval/gen_{base,round1,round2}/`，8 个数据集各 1 条，121 帧 1280×704，seed 0；Review 页面 `demo/`（`scripts/serve_demo.py`，端口 8765）

### Round2 训练（过程记录）
- 数据：`training_metadata/round2/round2_train.csv` 96,555 条 / val 1,735 条（按 source_sequence 留出，去 scene_cut，13 数据集配比见 `scripts/build_round2_mixture.py`）
- 配置：Wan2.2-TI2V-5B LoRA r32，**121 帧** 1280×704，lr 1e-4，1 epoch，从 Round1 `step-14921` 初始化（2s→5s curriculum）
- 单卡 A100-40GB 实测峰值 37.6GB，~19.6 s/step；15 卡 6,437 步，预计 ~35h
- 输出：`models/train/round2_5s_lora_r32/run1/`，每 500 步存 ckpt
- 启动：`scripts/launch_round2.sh`（torchrun c10d，需 `--local_addr`）；守护：`scripts/watchdog_round2.sh`（崩溃/挂死自动从最新 ckpt 续训到新 runN）
- 训练后评测：`scripts/eval_round2.sh`（固定 t/噪声的 val loss：base vs round1 vs round2 + 121 帧生成样例）

### Round3 / Round4 与 Teacher 验收
- **决策（10-13）**：双向 Teacher 效果经确认之后，才开始流式 / 因果 Student 的训练
- Round3（lr 退火，从 Round2 继续训）✅：val loss 0.1876（比 Round2 低 0.14%，t=−7.5）；GT 参考指标 DreamSim 0.1577，运动误差 0.730，均为目前最好
- Round4（按美学分筛选数据，从 Round3 继续训）🔄：`scripts/pipeline_round4.sh`，日志 `dataset/download_logs/pipeline_round4.log`
- 验收页 `demo/teacher.html`（`scripts/build_teacher_review.py`）：
  - 自动指标：val loss、GT 参考、VBench-I2V，逐项给出 PASS/FAIL
  - 盲评：24 条片段，比较候选 Teacher、base、最强基线
  - 两条流水线结束时会自动重建该页面

### 全参训练 FT1 / FT2（10-15 ~ 10-16，`scripts/pipeline_ft2.sh`，日志 `dataset/download_logs/pipeline_ft2.log`）
- FT1 ✅：从 base 起步并融合 R4 LoRA 后全参训练，使用 round5 数据（33k 条），lr 5e-6，cosine，2203 步，ckpt 为 `models/train/full1_5b/run1/step-2203.safetensors`
  - 实测（原始 prompt，1 个 seed）：val loss 0.1880（R4 为 0.1876）；WM DreamSim 0.1596（R4 为 0.1569）；运动误差 0.672（目前最好）；WM VBench 86.62；**Physics-IQ 24.8**（base 30.1，R4 27.3）；**WMB 7.09**（base 8.30，R4 7.10）
  - 实测结论仅限本次配方：在R4初始化、round5数据和当前超参下，全参训练未恢复通用指标。不能推广为所有原领域全参训练均无效。
- 补充数据：ukdash 4,510、robomind 6,000、egodex 6,000、openvidgen 12,000 条，全部生成 dense caption（Qwen3.5-35B-A3B，均 155 词，失败 0）
- FT2 ✅：从 base 起步全参训练，使用 `training_metadata/ft2/ft2_train.csv`（32,365 条，补充数据占 39%，原领域数据剔除镜头运动最大的片段，短 caption 0），lr 5e-6，2158 步，10-16 22:04 结束
  - 实测：val loss 0.1967，WM DreamSim 0.1722，运动误差 0.922，三项都接近 base，**领域收益基本消失**；**Physics-IQ 28.3**（base 30.1，R4 27.3，FT1 24.8）；**WMB 8.15**（base 8.30，R4 7.10）
  - WMB 各问题通过率（base / R4 / FT2）：时序一致性 0.72 / 0.41 / 0.66，形变 0.71 / 0.43 / 0.67，美学 0.81 / 0.62 / 0.79；分领域：机器人 8.08 / 5.64 / 7.76，驾驶 8.14 / 5.88 / 7.78。注意 FT2 训练数据里本身就有机器人和驾驶片段
  - FT2 原始 VBench-I2V 已完成：89.82（base 89.86，R4 89.47），WM-Bench Total 87.11（base 87.06）。这些为去camera后8维Total*；DreamX同8维重算90.79，快照原始含camera Total为90.49，不能混比。原始 prompt 的 7 个非 dynamic 维度对 DreamX 为 2/7，尚未达到目标。
  - 解释限制：FT2 同时改变初始化、数据混合、镜头筛选、第一人称比例和caption策略，不能把结果归因于“通用数据必然稀释领域收益”。需要单因素消融。
  - 10-17 已更新 `eval/summary.md`、`eval/summary.json` 和 `demo/teacher.html`。FT2 的自动验收均未通过，盲评待定；phase E 的改写prompt版与merge评测仍未完成。
- 更正：FT1 用的 round5 数据里有 9,846 条（30%）短 caption，是在"训练只用 dense caption"这条规定之前生成的
- 训练数据诊断（`scripts/scan_cuts.py`，结果在 `eval/diag/`）：训练片段几乎没有硬切；转码时帧率按时间重采样，训练和推理的 timestep shift 都是 5，两者一致；matrix 和 mira 的帧间变化分别为 8.5 和 5.3，其余来源都 ≤2.4。R4 / FT1 生成视频的帧间变化比 base 多约 40%
- 权重插值 🔄（参考 Cosmos-Predict2.5 的 model soup 做法）：`scripts/merge_ckpt.py`，θ = base + α·(FT1 − base)，α 取 0.3 / 0.5 / 0.7，checkpoint 存于 `models/train/merge/`；`scripts/eval_merge.sh` 负责评测 WM、VBench、Physics-IQ、WMB，共 3,033 个视频，排在 phase E 之后
- 备选：FT3 降低高运动数据源比例（`training_metadata/ft3/ft3_cand_motion.csv`，matrix ×0.3，mira ×0.5）；推理扫参（`gen_jobs.py` 支持按任务设置 cfg / shift / steps / neg）；VCD 一致性奖励 LoRA
- phase E（10-18 00:51:51）已完整结束：3913生成视频、14组benchmark-model严格覆盖PASS，摘要及Teacher页面更新。po同协议：VBench base89.79/R4 89.60/FT1 89.54/FT2 89.78；Physics-IQ base32.48/R4 32.47/FT1 31.29/FT2 31.43；WMB base8.86/R4 8.40/FT1 8.30/FT2 8.82。prompt语义偏移仍保留，不把扩写涨分认作模型训练收益。插值评测继续，未完成。
- FT1最终checkpoint已保存，但收尾exit=127；运行中覆盖shell脚本是可能解释，尚未独立复现。训练运行期间不要覆盖脚本。FT2正常exit=0。
- `gen_jobs.py` 改动：（1）每个视频生成前先原子认领，空闲的分片会接手其他分片剩余的任务，避免单个慢分片拖住整体；（2）若同一张卡上有 `eval_val_loss.py` 在跑，先等它结束，避免 OOM

### 迁移交接

- 供新服务器coding agent首读：`docs/31_服务器迁移与CodingAgent交接.md`。
- 包含目标与Teacher验收约束、历史实验与已验证结论、动态任务快照、模型/数据/依赖迁移清单、恢复命令、GitHub未推送状态与安全注意事项。
- 当前新4节点已实际接续生成和CPU评分，旧服务器本项目任务已停止；本节原运行关系为历史。接手前重新核查最新监控、日志与进程，不能重启旧pipeline或重复启动现有托管任务。

### HF 上传 ✅（受配额限制，部分）
- 旧 `{NZC415,nzc-415}/wan22-processed-clips` 已于 10-01 重建为空仓库（4s clips 被 5s 数据取代）
- `nzc-415/wan22-5s`：955 shards / 7.79TB，覆盖 2,483,000 / 2,764,481 个 final 5s 窗口；10-12 18:03 触发免费账号公共存储上限（403）后停止
  - 未上传 281,495 个窗口：mira 210,835、noxi 33,001、seamless 19,853、vfhq 17,487、openvidhd 319（清单 `/dockerdata/wm_logs/hf_missing_windows.txt`，launcher 本地）
- `NZC415/wan22-5s-trainval-overflow`：补传其中 Round2/Round3 训练+验证用到的 24,162 个窗口（10 shards，`index.jsonl` 给出窗口→shard 映射，`scripts/upload_trainval_overflow.py`）→ **Round2/3 训练与验证数据已可完整从 HF 复现**
- 剩余 ~25.7 万个非训练窗口需 PRO 账号或其他存储

---

## 0、本轮推进记录（2026-08-18）

### 数据处理全部完成

14 个数据集全部完成转码/切分为 Wan2.2-TI2V 格式（1280×704/704×1280, 24fps, 4s clips, libopenh264）：

| 数据集 | clips | 状态 |
|---|---|---|
| MIRA | 2,073,297 | ✅ 无需转码（已有 caption_i2v） |
| Ego-Exo4D | 22,161 | ✅ 转码完成（50,383 视频 → 22,161 有效 clips） |
| CelebV-HQ | 1,277,996 | ✅ 转码完成（12,988 视频 → 1.28M clips） |
| HOIGen | 231,084 | ✅ |
| Charades | 73,603 | ✅ |
| NoXi | 41,744 | ✅ |
| VFHQ | 34,173 | ✅ |
| Seamless | 25,432 | ✅ |
| Direct (Project Aria) | 25,312 | ✅ |
| Matrix | 14,124 | ✅ |
| EasyCom | 3,920 | ✅ |
| DexYCB | 2,400 | ✅ |
| OpenVidHD | 1,317 | ✅ |
| H2O | 297 | ✅ |
| **合计** | **~3.83M clips** | |

### HF 上传进度

仓库 `NZC415/wan22-processed-clips`（已改为公开仓库，无存储限制）

| 数据集 | shards | 状态 |
|---|---|---|
| H2O | 1 | ✅ |
| OpenVidHD | 1 | ✅ |
| DexYCB | 1 | ✅ |
| EasyCom | 2 | ✅ |
| Matrix | 5 | ✅ |
| Direct | 8 | ✅ |
| Seamless | 4 | ✅ |
| VFHQ | 9 | ✅ |
| NoXi | 4 | ✅ |
| Charades | 14 | ✅ |
| HOIGen | 24/61 | 🔄 上传中 |
| Ego-Exo4D | ~3 | ⏳ |
| CelebV-HQ | ~160 | ⏳ |
| MIRA | ~1013 | ⏳ |
| **已上传** | **73 shards (~584GB)** | |

### Caption 进度

| 数据集 | clips | 已 caption | 状态 |
|---|---|---|---|
| MIRA | 2,073,297 | 1,216,449 | ✅ 已有 caption_i2v |
| HOIGen | 231,084 | 231,083 | ✅ |
| CelebV-HQ | 1,277,996 | 0 | ❌ 待 caption（queue 已就绪） |
| Charades | 73,603 | 73,601 | ✅ |
| NoXi | 41,744 | 41,744 | ✅ |
| VFHQ | 34,173 | 34,173 | ✅ |
| Seamless | 25,432 | 25,432 | ✅ |
| Direct | 25,312 | 25,310 | ✅ |
| Ego-Exo4D | 22,161 | 22,161 | ✅ |
| Matrix | 14,124 | 14,124 | ✅ |
| EasyCom | 3,920 | 3,913 | ✅ |
| DexYCB | 2,400 | 2,400 | ✅ |
| OpenVidHD | 1,317 | 1,317 | ✅ |
| H2O | 297 | 297 | ✅ |
| **合计** | **~3.83M** | **~1.69M (44%)** | |

### Caption Queue 文件

位置：`training_metadata/caption_queues/`

| 文件 | clips | 说明 |
|---|---|---|
| `celebvhq.jsonl` | 1,277,996 | 待 caption，最大批次 |
| `egoexo4d_full.jsonl` | 22,161 | 已完成 ✅ |
| `matrix_full.jsonl` | 14,124 | 已完成 ✅ |
| `caption_queue_all.jsonl` | 469,007 | 合并清单（不含 CelebV-HQ/MIRA） |

### 新增数据集（本轮）

| 数据集 | 来源 | clips | 说明 |
|---|---|---|---|
| CelebV-HQ | `NZC415/CelebV-HQ-archives` (HF) | 1,277,996 | 数字人脸视频 |
| VFHQ | `NZC415/VFHQ-videos-archives` (HF) | 34,173 | 人脸视频 |
| Direct (Project Aria) | `NZC415/direct-downloads-archives` (HF) | 25,312 | ADT/HOT3D/AEA 等第一人称 |

### 仍在进行

| 任务 | 状态 |
|---|---|
| HOIGen HF 上传 | 🔄 24/61 shards |
| HoloAssist 修复 | 39.7%（~33 天，不阻塞） |
| CelebV-HQ caption | ❌ 待 caption（128 万 clips） |

---

## 0、本轮推进记录（2026-08-12）

### 数据下载

#### 新增下载（从 HF 私有仓库）
| 数据集 | 体积 | 来源 | 状态 |
|---|---|---|---|
| **CelebV-HQ** | 303GB (38 tars) | `NZC415/CelebV-HQ-archives` | ✅ 下载完成，🔄 提取中 |
| **VFHQ** | 20GB (3 tars) | `NZC415/VFHQ-videos-archives` | ✅ 下载完成，🔄 提取中 (3,717 MP4) |
| **Direct-downloads** | 61GB (8 tars) | `NZC415/direct-downloads-archives` | ✅ 下载完成，🔄 提取中 |
| **Wan2.2-TI2V-5B** | 34GB | ModelScope | ✅ 下载完成 (825 tensors 校验通过) |
| **Qwen2.5-VL-7B** | 16GB | ModelScope | ✅ 下载完成 |

#### Direct-downloads 内容（之前 CDN 403 无法下载的 Project Aria 系列）
| 子数据集 | 文件数 | 类型 |
|---|---|---|
| Hot3DQuest | 226 | 手物交互（Quest 版） |
| ADT (Aria ADT) | 219 | 日常活动 |
| Hot3DAria | 192 | 手物交互（Aria 版） |
| AriaEverydayActivities | 129 | 日常行为 |
| 合计 | 768 | 第一人称视角 |

#### 下载/修复中
- **HoloAssist 修复**：ahat_depth 38.4%（230.8GB/601.8GB），~137KB/s
- **Matrix part_1000/1100**：ModelScope 上 0 字节文件，无法下载（2/39 缺失，可忽略）

### 数据处理（转码+切分为 Wan2.2 格式）

#### 已完成（11 个数据集，统一 manifest train 70,151 / val 7,783）
| 数据集 | clips | 格式 | manifest |
|---|---|---|---|
| MIRA | ~2,000,000 | 1280×704 24fps 4s | ✅ |
| HOIGen | 231,084 | 1280×704 24fps 4s | ✅ |
| Charades | 73,603 | 1280×704 24fps 4s | ✅ |
| NoXi | 41,744 | 1280×704 24fps 4s | ✅ |
| Seamless | 25,432 | 704×1280 24fps 4s (竖屏) | ✅ |
| Ego-Exo4D | ~15,600+ (🔄 增长中) | 1280×704 24fps 4s | ✅ |
| Matrix | 14,124 | 1280×704 24fps 4s | ✅ |
| EasyCom | 3,920 | 1280×704 24fps 4s | ✅ |
| DexYCB | 2,400 | 1280×704 24fps 2.5s | ✅ |
| OpenVidHD | 1,317 | 704×1280 24fps 4s (竖屏) | ✅ |
| H2O | 297 | 1280×704 24fps 4s | ✅ |

#### 处理中
| 任务 | 进度 | 预计完成 |
|---|---|---|
| Ego-Exo4D 转码 | 500/50,383 (1%), 38,925 clips | ~33h |
| VFHQ 提取 | 3,717 MP4 (3 tars) | ~10min |
| Direct-downloads 提取 | 153 MP4 (8 tars, 768 expected) | ~30min |
| CelebV-HQ 提取 | shard 1/38 | ~2h |
| HoloAssist 修复 | 38.4% | ~33 天 |

#### HF 上传
- 仓库 `NZC415/wm-dataset`（私有）已上传：
  - 统一 manifest（train 70,151 / val 7,783，JSONL + CSV）
  - 11 个数据集的 `*_wan22/manifest_wan22.jsonl`
  - 全部处理脚本
  - 训练配置 + 审计报告 + 文档

### Caption 进度

- **在其他节点进行**（用户自行部署）
- Node B: Qwen3-VL-235B-A22B-Instruct-FP8, TP=8
- Node A: Qwen2.5-VL-7B-Instruct, 8 副本
- 详见 `docs/29_Qwen3VL_235B_部署指南.md` 和 `docs/29_VLM_Caption_全量打标指南.md`
- 截至 2026-07-26：已完成 1,307,682 / 2,192,164（59.6%）

### 新增文档
- `docs/26_数据集收集与许可.md` — 全部数据集下载状态、许可分析、HF 策略
- `docs/27_数据预处理与训练格式适配.md` — Wan2.2 格式要求、帧采样逻辑、转码/切分方案
- `docs/28_数据总量与项目目标评估.md` — 数据总量、HF 上传可行性、项目目标覆盖度
- `docs/29_Qwen3VL_235B_部署指南.md` — 235B 模型 GPU 需求、vLLM 部署、caption 流程

### 新增脚本
- `scripts/transcode_and_clip.py` — 批量转码+切分（libopenh264, 多进程, 4s 段）
- `scripts/build_unified_manifest.py` — 11 数据集统一 manifest 构建
- `scripts/generate_wan22_manifest.py` — 为 *_wan22 目录生成 manifest
- `scripts/decode_audit.py` — 随机抽样解码+首帧审计
- `scripts/filter_existing_videos.py` — 全量路径存在性校验
- `scripts/caption_with_transformers.py` — Qwen2.5-VL transformers 直驱 caption
- `scripts/extract_matrix_zips.py` — 从 Matrix zip 抽样提取视频

### 关键技术决策
- **帧采样**：Wan2.2 从 frame 0 取 49 连续帧（~2s），无随机起始 → 段长设 4s（浪费最少）
- **编码器**：环境无 libx264（--disable-gpl），改用 libopenh264 + -b:v 5M
- **切分**：>5s 视频切为 4s 段；2-5s 不切分；<2s 自动降帧
- **分辨率**：首轮仅横屏（1280×704），竖屏（704×1280）后续加入
- **235B 部署**：最少 8× H20-96GB（bf16 TP=8）或 4× H20-96GB（FP8 TP=4）

---

## 0、本轮推进记录（2026-07-26）

### VLM Caption 全量打标（阶段二进行中）

详见 `docs/29_VLM_Caption_全量打标指南.md`

#### 双节点并行部署
- **Node B (29.127.33.93)**：Qwen3-VL-235B-A22B-Instruct-FP8，TP=8 单副本，vLLM 0.16.1
- **Node A (29.191.211.184)**：Qwen2.5-VL-7B-Instruct，8 副本 × TP=1，--enforce-eager（235B 因 FlashInfer AllReduce 硬件问题无法部署，改用 7B dense 模型）

#### Caption 进度（2026-07-26 17:00 更新）
| 数据集 | 总视频数 | 已完成 | 失败 | 状态 |
|---|---|---|---|---|
| easycom | 3,920 | 3,913 | 7 | ✅ 完成 |
| noxi | 41,744 | 41,744 | 0 | ✅ 完成 |
| charades | 73,603 | 73,601 | 2 | ✅ 完成 |
| mira | 2,073,297 | 1,188,624 | 37,642 | 🔄 57% 进行中，ETA ~1天11h |
| **总计** | **2,192,164** | **1,307,682** | 37,651 | — |

- Node A (7B): ok=765,287，8 副本全健康，GPU 利用率 100%
- Node B (235B): ok=261,169，HTTP 200
- 双节点合计速度：~7 vid/s

#### 已合并到 unified manifest
- easycom/noxi/charades 全量 caption 已合并回 `unified_train.jsonl` / `unified_val.jsonl`
- 旧 caption 保留在 `prompt_legacy` 字段
- unified_train: 51,044 行，5 空 prompt (0.01%)

#### 新增脚本
- `scripts/recaption_unified.py`：主 caption 脚本（多端点负载均衡、断点续传、空文件跳过）
- `scripts/merge_recaptioned.py`：合并新 caption 回 unified manifest
- `scripts/generate_dataset_manifest.py`：生成数据集完整 manifest
- `scripts/repair_mira_empty.py`：修复 MIRA 空文件（从 tar 重新提取+转码）
- `scripts/stage2_auto_caption.sh`：阶段二自动衔接（NoXi→Charades→MIRA）
- `gpu_guard_25min.sh`：GPU 占卡 25 分钟自动守护

#### 关键问题与解决
- Node A 235B 部署失败：FlashInfer AllReduce CUDA 错误 + GPU 5/3 崩溃 → 改用 7B
- vLLM MQ 连接问题：`VLLM_HOST_IP=127.0.0.1`
- MIRA 空文件 ~37K（1.8%）：集中在 2026-05-08T21-23 tar 包，脚本已加快速跳过

---

## 0、本轮推进记录（2026-07-22）

### Qwen3-VL-235B-A22B-Instruct-FP8 部署与全量 Caption 完成

详见 `docs/28_Qwen3-VL-235B_Caption完成报告.md`

#### 模型部署
- **Qwen3-VL-235B-A22B-Instruct-FP8** 下载完成：`models/Qwen3-VL-235B-A22B-Instruct-FP8/`（222GB，24 分片）
- **vLLM 部署**：单节点 8× H20-96GB，TP=8 单副本，FP8 + expert-parallel + async-scheduling
- **启动耗时**：~29 分钟（权重 16min + CUDA graph 编译 1min + DeepGEMM warmup 2min + graph 捕获 10min）

#### VLM Caption 生成（8,155 视频）
| 数据集 | 视频数 | 成功 | 失败 | Caption 来源 |
|---|---|---|---|---|
| DexYCB | 2,119 | 2,119 | 0 | qwen3-vl-235b-a22b-fp8 |
| H2O | 60 | 60 | 0 | qwen3-vl-235b-a22b-fp8 |
| OpenVidHD | 1,000 | 1,000 | 0 | qwen3-vl-235b-a22b-fp8 |
| Matrix | 5,000 | 4,976 | 24 | qwen3-vl-235b-a22b-fp8 |

#### 已有 Caption 合并（195,347 条）
| 数据集 | 合并数 | 来源 |
|---|---|---|
| HOIGen-1M | 171,775 | CSV caption_info |
| Charades | 73,603 | 官方标注 manifest.jsonl |
| NoXi | 41,744 | 官方标注 manifest.jsonl |
| Matrix | 14,074 | qwen3-vl-235b（本轮生成） |
| EasyCom | 3,920 | CSV caption |
| DexYCB | 2,400 | manifest_captioned.jsonl |
| H2O | 297 | manifest_captioned.jsonl |

#### Unified Manifest 最终状态
| 指标 | 本轮前 | 本轮后 |
|---|---|---|
| unified_train 行数 | 51,044 | 51,044 |
| 空 prompt | 42,072 (82.5%) | **5 (0.01%)** |
| 已填 prompt | 8,972 (17.5%) | **51,039 (99.99%)** |

分布：hoigen1m 35.2% / charades 17.6% / mira 17.6% / noxi 9.1% / matrix 8.8% / easycom 6.9% / dexycb 4.3% / h2o 0.5%

### 新增脚本
- `scripts/generate_openvidhd_manifest.py`：从文件名解析 OpenVidHD 视频，生成 manifest.jsonl
- `scripts/caption_with_vllm.py`：vLLM API 调用 caption，支持 4 数据集 + 断点续传 + PyAV 帧采样
- `scripts/merge_captions_to_wan22.py`：合并各类 caption 源到 wan22 manifest

### 修改脚本
- `deploy_qwen3vl.sh`：TP=4×2 副本 → TP=8×1 副本 + FP8 优化 flags
- `scripts/build_unified_manifest.py`：SOURCE_CONFIGS 已含 dexycb/h2o/matrix

### 环境实况（覆盖旧文档）
- vLLM 0.21.0+cu129 **已安装**（旧文档说"未安装"已过时）
- transformers 5.10.1 + torch 2.11.0
- ffmpeg/ffprobe **不在 PATH 中**，用 PyAV（`av` 13.1.0）替代
- Squid proxy `http://star-proxy.oa.com:3128` 需 `NO_PROXY=localhost` 绕过

---

## 一、Wan2.2 训练数据总览（2026-07-22 更新）

### Caption 状态总览

| 数据集 | 视频数 | Caption 来源 | 状态 |
|---|---|---|---|
| HOIGen-1M | 231,084 | CSV caption_info | ✅ 已合并 |
| MIRA | 1,918,956 | JSONL→LLM 模板 | ✅ 已修复 |
| Charades | 73,603 | 官方标注 | ✅ 已合并 |
| NoXi | 41,744 | 官方标注 | ✅ 已合并 |
| Matrix | 14,124 | qwen3-vl-235b | ✅ 已生成 |
| EasyCom | 3,920 | 官方标注 CSV | ✅ 已合并 |
| DexYCB | 2,400 | qwen3-vl-235b | ✅ 已生成 |
| OpenVidHD | 1,000 | qwen3-vl-235b | ✅ 已生成 |
| H2O | 297 | qwen3-vl-235b | ✅ 已生成 |

### 已转换为 720p/24fps MP4

| 数据集 | 视频数 | 大小 | Caption | 用途 |
|--------|--------|------|---------|------|
| **MIRA processed** | 2,041,279 | 5.4TB | JSONL→LLM (caption修复完成) | 游戏多视角交互 |
| Charades | 9,848 | 32GB | 官方标注→自然语言caption (已完成) | 日常居家活动 |
| EasyCom | 271 | 2.3GB | 对话文本(已有) | AR多人对话 |
| RAVDESS | 2,880 | 1GB | 低优先级 | 情感表达(已排除) |
| OpenVidHD | 1,000 | 1.1GB | →VLM (已完成 qwen3-vl-235b) | 通用高清视频 |
| DexYCB | 2,400 | ~? | →VLM (已完成 qwen3-vl-235b) | 手物抓取 |
| EPIC | 3 | 295MB | 暂缓 | 厨房HOI |

### 原始数据

| 数据集 | 大小 | 格式 | 说明 |
|--------|------|------|------|
| MIRA Rocket Science | 8.3TB | 4,797 tars, 2.1M MP4 | 游戏多视角，持续处理中 |
| OpenVidHD | 3.0TB | 14 zips, 300K+ MP4 | 通用高清视频 |
| H2O | 177GB | tar.gz, 183K img | 双手操作图片序列 (60/61序列已转MP4, 397MB) |
| DexYCB | 71GB | 3 subjects, 348K img | 手物抓取图片序列 (已转MP4: 2,400视频, 1280x704, 24fps) |
| EasyCom | 102GB | 3,421 files | AR对话(已提取) |
| EPIC-KITCHENS | 8.7GB | 28 MP4 | 厨房HOI |
| Charades | 31GB | 9,848 MP4 | 日常活动 |

### 新下载数据

| 数据集 | 大小 | 状态 | Caption | 类型 |
|--------|------|------|---------|------|
| **HOIGen-1M** | 234GB | 106,100视频已提取 | CSV自带 | 人-物交互 |
| **Seamless Interaction** | ~? | 476视频已提取 | 待检查 | 人人交互 |
| **Matrix Dataset** | ~890GB | 碎片整理完成, ~213GB释放 | qwen3-vl-235b (已完成) | 游戏场景(Forza+Cyberpunk) |
| Action100M (yt-dlp) | 278 videos | cookies过期 | 动作标签 | 人类动作 |
| CelebV-HQ (yt-dlp) | 76 videos | cookies过期 | 属性标注 | 数字人脸 |

---

## 二、下载管线状态

| 管线 | 状态 | 进度 |
|------|------|------|
| MIRA → Wan2.2 | 活跃 | 1,918,956 视频 |
| Matrix git clone | 完成 (碎片整理+去重) | ~890GB, 释放~213GB |
| yt-dlp Action100M | BLOCKED (cookies过期) | 278/60,000 |
| yt-dlp CelebV-HQ | BLOCKED (cookies过期) | 76/13,844 |
| OpenVidHD 补全 | 完成 | 33分片 (829.6GB) 从HF下载 |
| HoloAssist 修复 | 进行中 (wget --continue) | 3个tar修复, 889GB, ETA ~5天 |
| NoXi 下载 | 完成 (per-session zips, 4并行workers) | 已下载 |
| Ego-Exo4D 下载 | 进行中 | ~11TB |
| Assembly101 | BLOCKED | 需HuggingFace token |
| Nymeria | BLOCKED | Facebook CDN 403 |
| Inter-X | BLOCKED | 需Google Drive URLs (skeletons.zip, texts.zip) |
| Wan2.2 模型 | 完成 | 从ModelScope下载 (34GB) |
| Qwen3-VL-235B-FP8 | 完成 | 从ModelScope下载 (222GB) |
| HOI4D bypy | 已停 | 3.3GB |
| VFHQ bypy | 不可用 | 需百度客户端 |
| HOT3D/Aria | 不可用 | Facebook CDN 不可达 |

---

## 三、Caption 方案（2026-07-22 更新）

### VLM 部署

| 项 | 值 |
|---|---|
| 模型 | Qwen3-VL-235B-A22B-Instruct-FP8 |
| 部署 | vLLM 0.21.0+cu129, TP=8 单副本 |
| 硬件 | 8× H20-96GB |
| 量化 | FP8 (282GB) |
| 吞吐 | ~1-6 vid/s（取决于视频长度） |

### 已完成 Caption

| 数据集 | 视频数 | Caption 方式 | 状态 |
|--------|--------|-------------|------|
| HOIGen-1M | 231,084 | CSV 自带 | 已就绪 |
| MIRA | 1.9M | JSONL→LLM 模板 | 已修复 |
| Charades | 73,603 | 官方标注 | 已合并 |
| NoXi | 41,744 | 官方标注 | 已合并 |
| Matrix | 14,124 | qwen3-vl-235b-fp8 | 已完成 |
| EasyCom | 3,920 | 官方标注 CSV | 已合并 |
| DexYCB | 2,400 | qwen3-vl-235b-fp8 | 已完成 |
| OpenVidHD | 1,000 | qwen3-vl-235b-fp8 | 已完成 |
| H2O | 297 | qwen3-vl-235b-fp8 | 已完成 |

---

## 四、Wan2.2 训练基础设施

| 项目 | 状态 | 详情 |
|------|------|------|
| DiffSynth-Studio | 已部署 | 训练框架就绪 |
| 训练配置 | 已创建 | Wan2.2-TI2V-5B, LoRA rank 32 |
| Smoke test | 就绪 | 5,476 samples |
| Wan2.2 模型权重 | 完成 | 从ModelScope下载 (34GB) |
| Qwen3-VL-235B-FP8 | 完成 | 从ModelScope下载 (222GB) |
| Unified manifest | 就绪 | 51,044 train + 5,573 val, 99.99% 有 caption |
| 训练文档 | 已创建 | docs/25_Wan22_Training_Infra.md |

---

## 五、数据集全列表 (90个)

### 已下载 (14个)
MIRA, OpenVidHD, Charades (caption已完成), EasyCom, H2O (已转MP4), DexYCB (已转MP4), EPIC-KITCHENS, RAVDESS, CREMA-D, InterHuman, HoloAssist (tar修复中), Action100M, CRAMA-D, BEAT2

### 下载中 (5个)
HOIGen-1M (视频提取完成), Seamless Interaction (视频提取完成), Matrix Dataset (碎片整理完成), HoloAssist (tar修复), NoXi (下载完成)

### 下载中 (大型下载)
Ego-Exo4D (11TB)

### 已申请 (8个)
Ego4D, OpenHumanVid, Assembly101 (BLOCKED: HF token), Nymeria (BLOCKED: CDN 403), LRS2, LRS3, MultiMediate, UDIVA

### 额外阻塞项
Inter-X (BLOCKED: 需skeletons.zip/texts.zip URLs), yt-dlp (BLOCKED: 需新YouTube cookies)

### 已排除 (21个)
AMASS, Human3.6M, HumanML3D, Motion-X, AIST++, BEAT, TalkSHOW, Trinity, GRAB, CMU Panoptic, ARCTIC, Something-Something V2, ViCo, AgiBot World, CALVIN, LIBERO, Habitat, AI2-THOR, ALFRED, BEHAVIOR, VirtualHome, TEACh, RoboCasa365, CHiME-6

### 暂缓 (37个)
其余模拟器/机器人/评测/音频数据集
