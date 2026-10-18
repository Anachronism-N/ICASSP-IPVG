# 完整插值评测验收更新

> 2026-10-18 23:13:15 CST：原3033视频完整解码、三模型全部WM/公共VBench8维、GTref108、Physics198/66、WMB350与三份固定120loss已严格通过；summary和Teacher页重建完成。见docs/32_插值Teacher完整评测结论.md。实验执行/覆盖完成不等于Teacher质量自动或人工验收通过；下文旧快照按时间解释。

# 30 Teacher 模型（Wan2.2-TI2V-5B 双向）训练实验记录

> 更新至2026-10-18：当前新4节点32卡已恢复生成/评分资产并实际接续，α0.3/α0.5各1011条完整解码、GTref108与Physics66场景CPU覆盖已核验，α0.7仍生成；完整WM/VBench、WMB、3份120条loss与Teacher gate/用户review未齐。旧183/3033暂停属于01:21历史冻结边界，不是当前进度。准确18:54:37 CST快照、托管任务与JSON指针统一见`docs/31_服务器迁移与CodingAgent交接.md`顶部；动态增量读`eval/diag/merge_monitor_latest.json`，恢复清单读`eval/diag/merge_resume_20261018.json`（旧stage字段不代表当前）。Phase E3913输出/14组评分已结束并保留。凡写“实测”的数字来自`eval/summary.md`或原始结果；“推断”尚未验证。
> 目标：只用 Wan2.2-TI2V-5B 训练一个双向 Teacher，要求：
> - 保住领域内收益；
> - 在 VBench-I2V 的多个非 dynamic 维度上胜过 DreamX-World-1.0等头部模型，dynamic维度允许落后。DreamX快照原始Total90.49%，本项目去camera后的同8维Total*重算90.79%；两者不能混称榜单原始分。
>
> Teacher 效果确认之后，才开始流式 / 因果 Student 的训练。

---

## 1. 通用设置

| 项目 | 设置 |
|---|---|
| 模型 | Wan2.2-TI2V-5B（DiT + Wan2.2 VAE + umT5），图+文 → 视频 |
| 分辨率 / 帧数 | 1280×704，121 帧（5s @24fps）；Round1 为 49 帧 |
| 文本token上限 | 同一umt5 tokenizer，whitespace清洗后512 tokens。R4超限1/24650、FT1超限5/33040、FT2超限5/32365；三组评测扩写均0超限。大规模caption被截断不是当前证据支持的原因，详见`eval/diag/caption_token_lengths.json` |
| 训练框架 | DiffSynth-Studio 2.0.17（本地改动见 `patches/diffsynth_studio_fb337fb.patch`：数据读取指数退避重试、cosine LR、DeepSpeed 下按优化器步计数） |
| 历史训练硬件 | R2–FT2使用旧2节点×8 A100-40GB，共15健康卡（旧worker GPU7有Xid31故障），无IB，NCCL走eth1；当前插值生成在新4节点32卡，不改变原训练batch/配方 |
| 训练目标 | flow matching，timestep shift=5（与推理一致） |
| LoRA 设置 | r32，target `q,k,v,o,ffn.0,ffn.2`，gradient checkpointing |
| 全参设置 | DeepSpeed ZeRO-2 + DeepSpeedCPUAdam（优化器状态和 fp32 master 放 CPU），bf16，micro-batch每GPU1、梯度累积1（global batch15），gradient_clipping=1.0，weight_decay=0.01，`overlap_comm=false`，bucket 5e7。约 33 s/step |
| 推理 | 50 步，CFG 5，shift 5，官方负面提示词，seed 0 |
| 训练 / 评测脚本 | `scripts/train_round2_16gpu.sh`（LoRA）、`scripts/train_full_16gpu.sh`（全参）、`scripts/launch_round2.sh`、`scripts/pipeline_*.sh` |

### 1.1 评测体系

| 评测 | 内容 | 脚本 |
|---|---|---|
| val loss | 120 条留出 5s 片段（按 source sequence 留出），固定 5 个时间步和固定噪声 | `scripts/eval_val_loss.py` |
| WM-Bench | 108 条留出领域片段，用 GT 首帧 + caption 生成，跑 VBench 各维度 | `scripts/build_eval_jobs.py`、`scripts/vbench_eval.py` |
| GT 参考指标 | 生成视频与真实后续画面比：DreamSim 距离（均值 / 末帧）、运动量误差 \|log(gen/GT)\| | `scripts/gt_ref_metrics.py` |
| VBench-I2V | 官方 355 对图文（16:9），8 维 Total（不含 camera_motion）；本地 1 seed，榜单为 5 seed | `scripts/vbench_eval.py`、`scripts/score_pool.py` |
| Physics-IQ | Original 集（公开 GCS），198 条，i2v 协议 | `scripts/build_extra_bench_jobs.py`、`scripts/score_extra.sh` |
| WorldModelBench | 350 条、7 个领域，VILA judge（vila-ewm-qwen2-1.5b）；总分 = 指令遵循 /3 + 物理 5 问 + 常识 2 问 | 同上 |
| prompt 优化版（_po） | 只在评测时用 Qwen3.5-35B-A3B 把短 benchmark prompt 改写成训练风格的 dense caption；训练数据不改动 | `scripts/rewrite_prompts_vbench.py` |

---

## 2. 各轮实验

### 2.1 总表（实测，原始 prompt，1 seed）

| 轮次 | 初始化 | 方法 | 数据 | 步数 | val loss↓ | WM DreamSim↓ | 运动误差↓ | VBench-I2V | Physics-IQ↑ | WMB↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| base | — | — | — | — | 0.1975 | 0.1746 | 0.994 | **89.86** | **30.1** | **8.30** |
| Round1 | base | LoRA r32，lr 1e-4，49 帧 | 59,682 条（12 个数据集，dataset_repeat=2） | 14,921 | 0.1922 | 0.1599 | 0.881 | — | — | — |
| Round2 | R1 | LoRA r32，lr 1e-4，121 帧 | 96,555 条 | 约 6,437 | 0.1879 | 0.1607 | 0.743 | 89.53 | — | — |
| Round3 | R2 | LoRA，lr 5e-5，cosine 退火 | 30,001 条全新窗口 | 2,001 | **0.1876** | 0.1577 | 0.730 | 89.42 | — | — |
| Round4 | R3 | LoRA，lr 3e-5，cosine | 24,650 条，按美学分筛选 | 1,644 | **0.1876** | **0.1569** | 0.730 | 89.47 | 27.3 | 7.10 |
| FT1 | base + R4 LoRA 融合 | **全参**，lr 5e-6，cosine，warmup 100 | round5：33,040 条（其中 30% 是短 caption，见 2.6） | 2,203 | 0.1880 | 0.1596 | **0.672** | 原始版未测 | 24.8 | 7.09 |
| FT2 | **base** | **全参**，lr 5e-6，cosine，warmup 100 | ft2：32,365 条，补充数据占 39%，全部 dense caption | 2,158 | 0.1967 | 0.1722 | 0.922 | **89.82** | 28.3 | 8.15 |

FT2 原始 prompt 的 7 个非 dynamic 维度（base / FT2 / DreamX 官方榜单）：

| 维度 | base | FT2 | DreamX |
|---|---|---|---|
| i2v_subject | 97.83 | 97.87 | 98.27 |
| i2v_background | 99.09 | 99.08 | 99.04 |
| subject_consistency | 95.45 | 95.10 | 95.35 |
| background_consistency | 97.54 | 97.19 | 97.24 |
| motion_smoothness | 98.71 | 98.80 | 98.52 |
| aesthetic_quality | 61.97 | 61.70 | 63.70 |
| imaging_quality | 71.06 | 70.93 | 71.46 |

FT2 原始 prompt 对 DreamX 为 2/7，尚未达到多个非 dynamic 维度领先头部模型的目标。小幅分差尚无多 seed 显著性验证。FT2 的 WM-Bench Total 为 87.11（base 87.06，R4 86.56），说明画面一致性有部分收益；不能把 DreamSim 接近 base 概括成所有领域指标均无收益。

参考：
- VBench-I2V快照逐维重算（与本地一样去camera的Total*）：DreamX-World-1.0 90.79，Magi-1 90.41，Wan2.2-TI2V-5B（官方 prompt 优化版）89.94。快照原始Total分别90.49、89.28、88.80，含camera维度；不要直接混比两个Total口径。
- Physics-IQ 官方榜单：Wan2.2 38.3，Sora2 42.3。
- WMB 官方榜单：KLING 9.10，CogVideoX-I2V 7.08。

WM-Bench 上相关基线的 DreamSim（实测）：CogVideoX-5B-I2V 0.1940，Wan2.1-I2V-14B 0.2182，HunyuanVideo-I2V 0.1980。从 R1 开始，我们的模型在这一项上都优于这三者。

### 2.2 Round1：领域适配（2s，49 帧）
- **做法**：
  - 12 个实际来源：HOIGen、EgoExo4D、Charades、NoXi、Direct、MIRA、CelebV-HQ、Matrix、VFHQ、DexYCB、EasyCom、H2O；用VLM生成dense caption，见 `scripts/build_round1_mixture.py`。Round2改为13源（移除DexYCB，加入Seamless和OpenVidHD）；
  - 单个数据集最多占 25%；按 source sequence 轮转取样，保证来源多样性；
  - 按 sequence 留出 val 集；
  - LoRA r32，lr 1e-4，8张H20；`dataset_repeat=2`，同一epoch内每条样本重复两次（等效119,364次取样），因此总步数为ceil(59,682×2/8)=14,921。
- **结果（实测）**：val loss −2.7%，在 120 条留出片段上 119 条变好；DreamSim 0.1746→0.1599。

### 2.3 Round2：延长到 5s（121 帧）
- **做法**：
  - 从 R1 继续训练，形成 2s→5s 的课程式训练；
  - 数据是 96,555 条 5s 窗口，见 `scripts/build_round2_mixture.py`，去掉了含 scene cut 的窗口；
  - 中途因共享盘短时读失败崩溃一次，从 step-4500 续训剩余样本；
  - 为此在 DiffSynth 的数据读取中加了 12 次指数退避重试。
- **结果（实测）**：
  - val loss 0.1879（相对 base −4.9%），120 条留出片段全部变好，9 个数据集单调改善；
  - 后半程只再降 0.3%，已接近饱和；
  - VBench-I2V 从 89.86 降到 89.53，通用能力开始下降。

### 2.4 Round3：学习率退火
- **做法**：从 R2 继续，换成 30,001 条从未用过的新窗口，lr 5e-5，cosine 调度，见 `scripts/build_round3_mixture.py`。
- **结果（实测）**：
  - 领域指标小幅改善：val loss 0.1876，DreamSim 0.1577，运动误差 0.730；
  - VBench 降到 89.42，主要是 dynamic_degree 从 56.10 降到 54.07。

### 2.5 Round4：按美学分筛选数据
- **做法**：
  - 每个数据集内，先去掉运动量最低的一部分窗口；
  - 剩下的按 LAION 美学分从高到低取，每个 sequence 最多取若干条（`scripts/build_round4_mixture.py`）；
  - 从 R3 继续，lr 3e-5。
- **结果（实测）**：
  - 领域指标最好：DreamSim 0.1569；
  - VBench 89.47，比 R3 只回升 0.05；
  - 第一次测通用 bench：Physics-IQ 27.3（base 30.1），WMB 7.10（base 8.30），明显退化。
- **用 prompt 优化版和 DreamX 对比**：
  - 比较的是 7 个非 dynamic 维度；
  - 胜 4 项：i2v_background、subject_consistency、background_consistency、motion_smoothness；
  - 负 3 项：i2v_subject、aesthetic、imaging；
  - base 的 prompt 优化版也是 4/7，可见这 4 项优势来自 base 本身，不是训练带来的。

### 2.6 Round5（LoRA）→ 改为 FT1（全参）
- round5 数据（`scripts/build_round5_mixture.py`）：
  - 选样分数为 z(美学分) + w·z(log 运动量)，目的是拉高 dynamic；
  - 混入 30% 短 caption（9,846 条），短 caption 由 vLLM 生成。
- 计划调整：R5 的 LoRA 没有训练，直接用这批数据做全参训练 FT1。
  - 初始化：把 R4 的 LoRA 融合进 base；
  - lr 5e-6；
  - 第一次保存 checkpoint 时 NCCL OOM，处理方法是关闭 `overlap_comm`、把 bucket 缩小、把 lr 从 1e-5 降到 5e-6。
- **结果（实测）**：
  - 领域指标和 R4 持平，运动误差 0.672，是目前最好；
  - 通用能力更差：Physics-IQ 24.8，WMB 7.09。
- **更正**：之后定下了"训练只用完整 dense caption"的规定，FT1 的数据是在此之前生成的，含 30% 短 caption，FT1 的结论受这一点影响。

### 2.7 FT2：从 base 开始，加入补充数据
- **做法**（`scripts/build_ft2_mixture.py`）：
  - 原领域数据占 61%：
    - 每个数据集去掉镜头运动最大的 20%，绝对上限为 8px / 4 帧，用 phase correlation 估计，见 `scripts/score_camera.py`；
    - 第一人称数据（direct、egoexo4d）减半；
    - 不按运动量做筛选。
  - 补充数据占 39%：
    - OpenVid 通用 1080p 6,600 条；
    - 英国行车记录 ukdash 1,610 条；
    - RoboMIND 2,244 条；
    - EgoDex 2,112 条。
    - 全部用 Qwen3.5-35B-A3B 打 dense caption，平均 155 词，见 `scripts/caption_5s_extra.py`。
  - 短 caption 0 条。
  - RoboMIND/EgoDex 的许可带有非商业等限制（EgoDex含ND）；本次不上传其原始/处理视频，也不将受限数据纳入GitHub提交。是否允许其他再分发须核对具体许可，不能仅据NC/ND推导所有公开上传均禁止。
- **结果（实测）**：
  - 通用能力明显恢复：Physics-IQ 28.3，WMB 8.15；
  - 真实后续匹配收益大幅缩小：val loss 0.1967，DreamSim 0.1722，都接近 base；但领域 WM-Bench Total 为 87.11，略高于 base 87.06。
  - 不能据此认定“通用数据必然稀释领域收益”：FT2 同时改变了初始化（R4→base）、数据混合、镜头筛选、第一人称比例和短 caption 策略。这不是单因素消融，尚不能区分各项贡献。

### 2.8 WMB 细分（实测）

| 领域 | base | R4 | FT1 | FT2 |
|---|---|---|---|---|
| animation | 8.06 | 7.24 | 7.26 | 8.00 |
| autonomous vehicle | 8.14 | 5.88 | 5.90 | 7.78 |
| human | 7.20 | 6.48 | 6.26 | 7.26 |
| industrial | 8.76 | 8.18 | 8.30 | 8.96 |
| natural | 9.50 | 9.48 | 9.56 | 9.56 |
| robotics | 8.08 | 5.64 | 5.50 | 7.76 |
| video games | 8.34 | 6.80 | 6.86 | 7.70 |

| 各问题通过率 | base | R4 | FT1 | FT2 |
|---|---|---|---|---|
| 形变（违反质量守恒 / 固体本构） | 0.71 | 0.43 | 0.47 | 0.67 |
| 非物理穿透 | 0.80 | 0.62 | 0.63 | 0.79 |
| 美学 | 0.81 | 0.62 | 0.61 | 0.79 |
| 时序一致性（闪烁 / 突变） | 0.72 | 0.41 | 0.40 | 0.66 |
| 指令遵循（满分 3） | 2.27 | 2.05 | 2.00 | 2.25 |

牛顿定律、流体、重力三项，当前 judge 的通过率都 ≥0.98，未检测到明显退化；这不等于已证实生成视频符合这些物理规律。FT2 数据包含机器人和驾驶，领域覆盖改善是可能原因，但其贡献尚未由消融实验确认。

---

## 3. 配对统计不确定性（2026-10-17，单 seed）

固定本次生成seed，对相同案例做5,000次配对bootstrap（VBench按355个case联合重采样，保留维度间相关性和109/246的维度适用子集）。95%区间仅表示案例抽样不确定性，不包括跨seed、judge变化或训练随机性；不能用于因果归因。

| FT2减对照 | 平均差 | 95%配对bootstrap区间 | 解释 |
|---|---|---|---|
| VBench Total减base（分） | -0.04 | [-0.231, +0.150] | 包含0，不能声称FT2优于或显著差于base |
| VBench Total减R4（分） | +0.352 | [+0.086, +0.616] | 本组case支持回升 |
| WMB减base | -0.151 | [-0.283, -0.029] | 尚未完全恢复到base |
| WMB减R4 | +1.046 | [+0.869, +1.220] | 回升证据较强 |
| DreamSim减base | -0.00243 | [-0.00572, +0.00079] | 小幅接近改善未稳定区分于0 |
| DreamSim减R4 | +0.01529 | [+0.00669, +0.02468] | 真实后续匹配仍差于R4（距离越低越好） |

Physics-IQ同样做了5,000次配对bootstrap，但抽样单位是66个物理场景（每场景3视角，共198视频），不是198条独立样本。按官方Original公式重算，每帧list先按官方解析器保留4位小数，再跨视角聚合和物理方差归一化；重算点估计与官方metrics JSON一致。

| Physics-IQ差值（分） | 平均差 | 95%场景级配对区间 | 解释 |
|---|---|---|---|
| FT2 raw − base raw | -1.792 | [-2.801, -0.810] | 本组场景支持仍低于base |
| FT2 raw − R4 raw | +1.011 | [-1.560, +3.657] | 虽均值回升，区间包含0，不能认定可靠超过R4 |
| R4 po − base po | -0.012 | [-2.138, +2.211] | 本组场景不能认定两者存在可靠差异 |
| FT2 po − base po | -1.055 | [-1.838, -0.328] | 同po协议仍低于base |
| FT1 po − base po | -1.195 | [-3.169, +0.814] | 区间包含0，不支持明确优劣 |
| FT1 po − FT1 raw | +6.467 | [+3.899, +9.138] | 输入增强差异，不是训练参数收益 |
| FT2 po − FT2 raw | +3.088 | [+0.822, +5.353] | 输入增强差异，仍保留语义偏移限制 |

完整WMB扩写配对结果（`eval/diag/wmb_full_raw_po_bootstrap.json`）：R4_po−base_po=−0.460，95%区间[−0.603,−0.317]；FT1_po−base_po=−0.560，区间[−0.711,−0.411]；FT2_po−base_po=−0.034，区间[−0.137,+0.069]包含0。FT2在扩写WMB已接近base，R4/FT1仍有残余退化；这与原始协议、Physics-IQ结论分别报告，不能归因于输入扩写提高模型参数能力。

详见 `eval/diag/physiq_paired_bootstrap.json` 与 `eval/diag/ft2_paired_bootstrap.json`；以上均不包含多seed波动。2026-10-17 16:32还对当时已完成的2,841个phase E视频逐个解码计帧，全部满足121帧、24fps、1280×704，未发现损坏。该检查不评价语义/物理正确性，也不涵盖审计之后才生成的视频。21:55再检查当时全部3,736个已生成输出，均121帧/24fps/1280×704且零损坏，记录`eval/diag/phase_e_integrity_followup.json`；23:03:05全部3,913输出最终解码检查完成，零损坏，记录`eval/diag/phase_e_integrity_final.json`。评分完整性和语义/物理质量仍是另外的检查。

额外留出核查（10-17）：FT2训练32,365行没有重复视频；与val loss使用的120条片段精确视频路径重叠为0；映射到原数据manifest后，训练15,810个来源序列与验证110个来源序列交集为0，所有验证片段均有映射。训练caption词数最小61、中位数145、最大403。该核查不涵盖与公开benchmark的近重复内容或跨数据集相同原视频。

## 4. 退化原因诊断

1. **生成视频变得不稳定（实测）**：用 `scripts/scan_cuts.py` 扫描 WMB 生成视频（64×36 灰度帧差）：

   | 模型 | 含跳变的视频比例 | flicker | 帧间变化中位数 |
   |---|---|---|---|
   | base | 7.1% | 1.02 | 3.71 |
   | FT1 | 10.0% | 1.31 | 5.04 |
   | R4 | 12.3% | 1.64 | 5.57 |
   | FT2 | 6.6% | 0.90 | 3.41 |

   FT2原始WMB350条在同一proxy下恢复到接近或低于base，支持“时序变化收敛”的观察；运动量减少不一定是更真实的运动，也不能替代指令/物理评分。结果`eval/diag/cuts_wmb_ft2.jsonl`。

2. **暂未发现普遍硬切，不能据此排除数据质量问题（实测）**：
   - 低分辨率帧差检测不是可靠的镜头边界分类器。全量扫描发现 openvidgen 2.3%、robomind 4.3%、ukdash 2.5%、egodex 1.0% 的片段超过跳变阈值，其余来源接近 0；仅人工抽查的一条 robomind 被确认是真实快速运动。
   - 输出端 `-r 24` 和部分脚本的 `fps=24` 按时间重采样；每来源抽查片段均为 24fps、121帧。尚未逐条核查原始源时间戳、重复帧、caption准确性和数据泄漏。
   - 当前代码中训练与推理默认 timestep shift 都为 5；这只排除了显式 shift 不一致，不能证明时间步采样/损失权重最优。
3. **大运动集中在两个数据源（实测）**：matrix 帧间变化 8.47，mira 5.28，其余来源都 ≤2.4。这两个源在 FT2 里合计占 14%。
4. **推断（未验证）**：模型从领域数据里学到了"更大的运动"。在领域内这让运动量更接近 GT（运动误差 0.994→0.730），但到没见过的场景（机器人、驾驶、游戏），就变成了时序不稳定和形变。另外，LoRA 和全参训练都只在领域数据上训，模型对这些场景的先验被冲淡了。
5. **prompt 的影响取决于benchmark，且当前扩写存在语义偏移（实测）**：此前VBench上base改写89.86→89.79、R4改写89.47→89.60，提升有限。Physics-IQ扩写条件下，base30.13→32.48、R4 27.33→32.47、FT1 24.82→31.29、FT2 28.34→31.43。相同po协议下R4与base仅差−0.01分；FT1/FT2仍分别低于base1.19/1.05分，不能宣称超过base。R4恢复与FT1/FT2残余退化必须区分，不能将所有raw退化都称为参数遗忘。扩写改变输入，不是模型参数改善，且以下语义偏移限制仍需保留。
   - 10条固定随机样本抽查发现：0013原描述仅指定释放篮球，扩写增加压缩并停在箱内的结局；0138原描述仅指定释放纸团，扩写增加下沉至杯底的结局；0150原文灰色网球和白色灯罩被改为棕色球和白色盒子。
   - 扩写脚本只检查字数，没有逐项验证对象属性、动作约束和终局。当前po应视为额外条件/语义可能变化的探索结果，而非严格等价的官方原始benchmark结果。保留原raw数据和现po文件，不覆盖已完成实验。
   - 后续严格改写应只补充首帧可见场景，完整保留原动作和对象属性，禁止增加未给定的碰撞、沉浮或终局；使用独立版本并重新验证。当前四模型同po对照已在Phase E完整完成；后续报告继续保留此限制，不重跑或覆盖旧po结果。

---

### 4.1 对DreamX目标差距的分解

去camera同8维Total*中，FT2比DreamX低0.969分：dynamic贡献−0.691分，其他7维合计−0.278分（不是重新定义的“无动态总分”）。FT2的美学差−2.002个百分点（Total*贡献−0.182）、i2v_subject差−0.401个百分点（−0.117）、imaging差−0.530个百分点（−0.048）；motion_smoothness优势+0.284个百分点（+0.088）。因此允许dynamic落后时，优先应验证主体保持与美学/画质优化，同时避免跨帧稳定性再退化。仅把motion_smoothness抬高并不能满足多个非dynamic维度最优的要求。详见`eval/diag/vbench_target_gap.json`，该分解不提供跨seed显著性。

### 4.2 最大物理退化场景的定性核查

针对FT2相对base总分贡献最差的4个场景，取center视角GT/base/FT2的0/0.5/1.5/3/4.5秒接触表（`eval/diag/physiq_review_*.jpg`），未改输入或分数：
- ball-hits-duck：GT球接近鸭子后移至后方；FT2在0.5秒出现两个球状物及不自然竖向运动，1.5秒还出现细长拖影。base同样未准确复现碰撞时序，但本例FT2存在更明显的对象增生。
- ball-ramp：GT球很快通过斜坡并离开画面；base/FT2均在后续几秒反复出现坡上球，未匹配真实轨迹和消失时机。
- potato-in-water：GT土豆快速进入液体并到杯底；两个生成都存在形状/折射表现和入水时序偏差，不能仅凭画面平滑认为物理正确。
- marble-run-x：生成里的小球主要停留在上方轨道，未表现原描述完整沿坡运动；低分辨率接触表难精确追踪GT小球，不据此做强模型排序。

这是4个场景/1视角/5时间点的定性观察，非完整因果诊断。它提示残余问题包含小物体轨迹、碰撞时序与对象保持，并不都是全局镜头闪烁；不能将帧差降低直接当作这些能力改善。

### 4.3 固定验证噪声时间步的分解

同120条片段和相同噪声，在训练scheduler表位置0.1/0.3/0.5/0.7/0.9计算MSE。R4相对base变化为−7.78%/−6.49%/−4.99%/−3.59%/−2.52%，FT1为−7.37%/−6.29%/−4.84%/−3.44%/−2.37%，FT2为−0.28%/−0.49%/−0.53%/−0.54%/−0.43%。因此FT2的匹配收益缩小贯穿这5个位置，不只是某一个噪声点。位置比例不是线性sigma，正向MSE不等于guided采样质量，详见`eval/diag/validation_noise_analysis.json`。

### 4.4 参数漂移的描述性统计

完整CPU读取base和FT1/FT2的825个同名DiT tensor（4,999,787,712参数）验证keys/shape及有限性。相对L2更新‖θ−θ_base‖/‖θ_base‖：FT1=0.03313、FT2=0.01421。该量只说明FT2距离base较近，不证明欠拟合或遗忘根因；FT1还包含R4 LoRA初始化的累积位移，不能拿二者比作同起点同数据训练更新量。逐block结果`eval/diag/parameter_drift.json`，未改任何权重。

### 4.5 开发子集与caption语义检查

已准备但未启动16条round2留出开发样本（排除R2/R3/R4/FT1/FT2训练精确视频路径、原WM108和loss120），3个模型×4个CFG/Zero*方案共192条jobs，见`eval/diag/inference_dev_jobs.jsonl`。它们不是公开benchmark调参样本，但未核查R1来源重叠与跨数据集近重复。首条CelebV-HQ抽查为真实学术幻灯片，其caption错写标题和部分内容，说明“dense/非空”不等于caption正确；须人工review开发样本后才能用于挑推理方案，当前未执行。

### 4.6 需要验证的 CFG 放大效应（代码事实与假设分开）

代码事实：当前SFT以完整正向caption训练（`DiffSynth-Studio/examples/wanvideo/model_training/train.py:84`），loss只做正向model_fn（`DiffSynth-Studio/diffsynth/diffusion/loss.py:25`）；尚无caption dropout或负向分支的显式监督。推理则使用同一个微调DiT算正/负两路（`DiffSynth-Studio/diffsynth/diffusion/base_pipeline.py:322`），CFG5组合为 `5*v_pos - 4*v_neg`。这本身不是实现错误，也是常见微调设置。

若微调正/负预测相对base偏移分别为δ_pos、δ_neg，guided输出偏移为 `5*δ_pos - 4*δ_neg`。验证loss仅观察正向预测，不能评价负向变化或guided采样稳定性。这是代数和代码层面的事实，不是已证明的退化根因。可在固定首帧/caption/seed上扫CFG1/3/5或测试CFG-Zero*，并同时看时序一致性、指令遵循和真实后续匹配；不能靠降低CFG制造静止视频来宣称质量提升。训练短caption策略保持不变：仍只使用完整dense caption。

## 5. 调研：可借鉴的方法

| 方法 | 出处 | 实测结论 | 对我们的用处 |
|---|---|---|---|
| Model Soup / 权重插值 | Cosmos-Predict2.5（arXiv 2511.00062，§4.2.1） | 5 个领域分别 SFT，各 30k iter、batch256；2B lr3e-5、14B lr1.3e-5。合并多个领域模型与 cooldown 模型后，人评显示保留特化能力且减轻通用退化；Soup/TIES/DARE-TIES 接近，DARE-Linear较差。未公开最终系数 | 为合并方向提供视频模型证据；我们的 base↔FT1 插值与其多领域 Soup 不同，效果仍待实测 |
| WiSE-FT | Wortsman et al., CVPR 2022 | θ=(1−α)θ_base+αθ_ft，分类任务上 ID 和 OOD 同时提升 | 同上 |
| VCD 一致性奖励 | arXiv 2510.19193，§4.1/Table2 | 文中 Wan2.1-1.3B-I2V：subject一致性96.40→97.12，background一致性98.87→99.02，Temporal Flickering97.90→98.18，smooth98.61→98.84；dynamic43.82→39.27，imaging69.89→69.86。实验368×272、25帧、30采样步、AdamW lr2e-4。LoRA/末步截断反传明确用于Open-Sora；不能直接当作Wan配方 | 非dynamic一致性有实测支持，但对Wan2.2-5B适用性及退化风险仍需验证 |
| Flow-DPO | VideoAlign（arXiv 2501.13918，Table3） | 常数β时VBench Total83.19→83.41，但Quality84.37→84.19略降；SFT Total82.31、Quality83.13。时间缩放β(1−t)²出现reward hacking。使用内部视频模型，不是本项目的VBench-I2V口径 | 可评估偏好微调；不能把其Total涨幅当作本项目质量提升承诺 |
| DreamX-World 1.0 | arXiv 2606.16993，§3.1/3.3/3.5 | 相机阶段冻结DiT，只训E-PRoPE分支；事件阶段全参微调DiT并混入非事件数据，采用保守更新/梯度裁剪；蒸馏后DiffusionNFT RL加KL。正文未证实分支零初始化，未给lr/裁剪阈值/VBench-I2V数值 | 分阶段保先验、受约束后训练值得借鉴，不需先照搬相机/流式架构 |
| CFG-Zero* | arXiv 2503.18886；官方WeichenFan/CFG-Zero-star | 每样本最优缩放s*=〈v_c,v_u〉/(‖v_u‖²+1e-8)，guided=s*v_u+CFG·(v_c−s*v_u)，并将初始若干ODE步预测置零。官方代码/例子针对Wan2.1-14B，I2V默认zero_init_steps=0仍表示首步置零 | 可做独立推理消融；未验证Wan2.2-TI2V-5B适配，且零化步数应明确定义，不能照搬索引参数 |
| VideoREPA | arXiv 2505.23656 | 物理分（VideoPhy PC）31.4→40.1，没有报告 VBench | 备选 |
| Wan2.2 reward-LoRA工程参考 | aigc-apps/VideoX-Fun：scripts/wan2.2_fun/train_reward_lora.py | 代码有boundary_type=full的5B分支，以及LoRA和采样步反传；默认lr1e-4、rank128/alpha64、49帧、384×672。没有提供本项目TI2V5B/121帧实测收益和显存结果；默认路径是14B低噪专家，且训练循环未证实首帧图像条件 | 不直接套用默认脚本；先验证5B VAE、图像条件、batch CFG拆分与有限梯度。代码支持不等于benchmark提升证据 |

---

## 6. 当前进行中与下一步（2026-10-18；当前快照见交接顶部）

- **Phase E已完整结束并保留**（2026-10-18 00:51:51）：3913视频及14个benchmark-model组独立最终严格覆盖PASS，`eval/summary.{md,json}`和`demo/teacher.html`已更新；不用重跑旧pipeline。FT2自动验收仍未通过，人工盲评待定。po同协议WMB为base8.86/R4 8.40/FT1 8.30/FT2 8.82；VBench为base89.79/R4 89.60/FT1 89.54/FT2 89.78，Physics-IQ为base32.48/R4 32.47/FT1 31.29/FT2 31.43。保留原始raw及po语义偏移限制，不将扩写涨分归为训练参数收益。
- **权重插值仍在独立全评测阶段**：θ = base + α·(FT1 − base)，α为0.3/0.5/0.7；每模型WM108、公共VBench355、Physics198、WMB350，共1011条/模型、3033条总视频。旧机01:21冻结183条后，新4节点32卡已实际接续，同模型/prompt/seed不变。

| 模型 | 全生成/CPU覆盖 | DreamSim↓ | Physics原始↑ | 固定120 loss↓ | 尚未完整 |
|---|---|---:|---:|---:|---|
| α0.3 | 1011完整逐帧解码；GTref108；Physics198/66 | 0.16940009 | 29.634157 | 0.19359857 | 公共VBench、WMB和final gate |
| α0.5 | 1011完整逐帧解码；GTref108；Physics198/66 | 0.16556586 | 28.169852 | 0.19132043 | 公共VBench、WMB和final gate |
| α0.7 | 1011完整逐帧解码；GTref108；Physics198/66 | 0.15975603 | 26.875848 | 0.18940758 | 公共VBench、WMB和final gate |

- 当前核验时间2026-10-18 21:38 CST。三模型WM-Bench8维完整case评分已出；公共355case VBench和三节点WMB并行运行。固定120loss保持原base视频集合、顺序、5时间位置/噪声，逐clip/per-t/per-dataset/overall严格核验均通过。
- 全部三模型CPU配对分析代码为scripts/analyze_merge_metrics.py，诊断eval/diag/merge_all_analysis_20261018.json；两模型历史报告merge_a03_a05_analysis_20261018.json保留。A03−A05物理+1.464305pp CI[0.030637,3.105535]，对FT2+1.296169 CI[0.064674,2.496352]，均边缘正且多比未调整；对base/R4 CI跨0。A03末帧与运动误差相对A05更差，因此不是全面收益。
- GTref和Physics区间均为固定生成seed/案例场景配对bootstrap，GTref非来源聚类，未包含训练/seed/judge方差，多重比较未调整；原始与扩写输入协议分开，不能据单项均值宣布SOTA、Pareto最优或Teacher通过。
- 三份1011完整解码和CPU覆盖、三固定120loss是完成的子阶段；最终门禁仍需48维评分、350题judge以及两full manifests全部通过，随后才重建报告。Teacher自动gate与用户review仍是独立条件。

- **之后的执行顺序**：
  1. 根据插值曲线，选领域收益和通用能力最平衡的 α；
  2. 推理扫参：CFG 3/4/5 × shift 3/5/7，加 CFG-Zero*，负面提示词加强闪烁 / 形变（`gen_jobs.py` 已支持按任务设置 cfg / shift / steps / neg）。CFG-Zero*实现已作为默认关闭的可选框架补丁接入；job字段`cfg_zero_star=true`与`cfg_zero_steps=0/1/2`分别用于仅最优缩放/额外置零首1或2步，语义不同于官方旧版zero_init_steps索引。CPU测试通过，GPU效果未测，不纳入现有模型分数；
  3. 先验证 VCD/受约束微调在 Wan2.2-5B 上的小样本配方，再决定是否训练；并非已经启动或保证有效。
  4. 前面都不够时，用 FT3：降低高运动数据源的比例（`training_metadata/ft3/ft3_cand_motion.csv`，matrix ×0.3，mira ×0.5），或者把领域数据和通用数据分别训成两个模型再做 soup。

## 7. GitHub 发布状态

用户已授权实验方案、代码与分析发布到新目标`Anachronism-N/ICASSP-IPVG`。旧机冻结时本地提交`c038304`、旧origin/main `e329f76`仅历史记录，旧push认证失败，不代表新目标已有当前代码。新机git-lfs3.4.1已恢复，hook不能跳过；当前代理网络和认证推送权限已经核验，独立分支wm-teacher-experiments-20261018首批选择性脱敏快照已push，**远端hash040814269512ddad93cc9734811940bd8de54630已核验**。完整GPU结果后续追加，不把初次发布当实验完成。公开使用独立脱敏快照，保留本地运维交接与未提交改动；数据、权重、cookies、token、私钥、私有迁移备份/历史日志和受限视频不加入公共提交，不宣称发布或完整实验成功。

## 8. 工程问题记录

- fuse 曾返回旧脚本内容，因此运行前复制到本地。不要覆盖正在运行的shell脚本。FT1结尾`exit=127`发生在最终checkpoint已保存之后；被覆盖脚本导致偏移读取是可能解释，未独立复现，不能当作已证实根因。FT2正常exit=0。
- 10-17修复后续评测调度：merge仅在phase E已进入打分且3913输出均存在/非空后启动，防止主流水线的广义gen_jobs等待误等merge。VBench池避开正在运行WMB judge的GPU，并按GPU互斥，避免同卡多评分模型峰值显存相加。
- Teacher页面修复：Full-FT中文/英文标签不再用`startswith(ft2)`匹配，且分维度表的模型名称不覆盖盲评KEYMAP映射。自动FAIL必须保留，不因网页更新宣布验收完成。可显式`--candidate m_full1_a05`选择插值模型，默认仍FT2；未测指标保持pending。
- 固定迁移摘要`eval/diag/migration_checksums.json`包含10个关键权重/输入/patch文件的SHA256、大小和mtime，约50.2GB；生成摘要本身不是迁移证明。后续新四节点实际缓存/评分恢复与smoke已完成，证据见文档31顶部/里程碑，不再称当前“未迁移”。
- 完整性审计：旧VBench仅按JSON存在skip且会跳过缺失输入，容易让子集评分被误当全量。已要求所有case视频存在/非空并验证每维结果数量。公开355个case中，subject相关维度适用246条、background相关维度109条，美学/画质355条；不是所有维度都应有355条。
- 最终覆盖入口：`python3 scripts/summarize_eval.py --validate-jobs eval/jobs_ft2_all.jsonl --validate-jobs eval/jobs_merge_all.jsonl`，按清单验证输出、逐维case集合及唯一性/有限分数、WMB350题及1750/700条物理/常识记录、Physics-IQ66场景×3视角、WM108条GT参考的必需数值；未完成则失败并不重写摘要。已有FT2 raw的1011个输出和4组benchmark通过校验，模拟丢video_results/GTref空条目/重复case均被拒绝。2026-10-18 00:51:51整phase E最终覆盖已通过（3,913输出、14组benchmark-model）；merge尚未完成，仍需独立验证，不将其写成已完成。
- claim审计：旧机冻结claim与当前新32分片活跃claim分开核查；18:54:37监控有32活跃、0过期，无发现超40分钟残留阻塞。旧`all claims exist`快捷跳过可绕过stale回收，因此遇到缺失输出时必须检查claim age与真正的生成进程，不能仅按进程退出宣布完成。
- torchrun 在本环境下会解析出 FQDN，导致 rendezvous 失败 → 显式加 `--local_addr`。
- 生成分片有长尾，且曾与 val loss 抢同一张卡导致 OOM → `gen_jobs.py` 改为原子认领任务 + 空闲分片接手剩余任务；同卡有 `eval_val_loss.py` 时先等待。
- 历史旧worker GPU7是坏卡，仅适用于旧节点；新4节点32卡已通过生成smoke。旧apex不支持sm_80，在独立venv屏蔽；VILA依赖用`--no-deps`安装并配triton3.1.0，不向当前生成环境混装。
