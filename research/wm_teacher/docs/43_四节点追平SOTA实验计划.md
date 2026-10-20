# 43 四节点追平SOTA实验计划（2026-10-21）

> 用户要求：充分利用4个节点（32×A100-40GB），在各benchmark上尽量追平SOTA；方案、代码、分析和结论推送GitHub。本文件是预注册计划，不是结果。

## 1. 差距诊断（来自已有完整评测，未新增实验）

VBench-I2V（355条，本地1 seed；榜单5 seed，协议不同）：

| 维度 | 本地base raw | 本地base po | 官方Wan2.2-5B po | DreamX-World-1.0（Total第一） | Magi-1 |
|---|---:|---:|---:|---:|---:|
| dynamic_degree | 56.10 | 46.75 | 52.85 | **72.11** | 68.21 |
| aesthetic_quality | 61.97 | 62.43 | 64.01 | 63.70 | **64.74** |
| imaging_quality | 71.06 | 71.23 | 70.70 | 71.46 | 69.71 |
| i2v_subject | 97.83 | 98.19 | 98.43 | 98.27 | 98.39 |
| subject_consistency | 95.45 | 95.91 | 94.95 | 95.35 | 93.96 |

- 相对DreamX的Total*差−0.93，其中dynamic_degree贡献−0.73、非dynamic合计仅−0.20（eval/diag/vbench_target_gap.json）。**主要差距是运动幅度，其次是美学分。**
- 已有训练（Round4、FT1、FT2、插值）都没有改善这两项，FT1/Round4反而降低物理和WMB分；Prompt-optimized提高美学/主体，但dynamic降9pp。
- Physics-IQ base 30.13%为本地最好；WMB base 8.297最好。训练候选在这两个基准上全部低于base。

结论：最直接、成本最低的杠杆是**推理采样配置**（sigma shift、CFG、提示词运动描述），其次才是面向运动/美学的数据训练。

## 2. 防止骗分的硬约束

- 不在VBench 355、WM108、Physics、WMB或fixed val120上选参数。选参只用独立开发集（第3节）；冻结后每个基准只完整跑一次。
- dynamic_degree只是光流阈值计数，抖动或镜头乱晃也能拉高它。所以选择规则同时要求：motion_smoothness不降、主体/背景一致性不降超过阈值、与真实后续视频的DreamSim和运动量误差不变差。
- raw与po两种提示词的结果都保留并分别报告，不混算。

## 3. E1：独立开发集推理配置消融（4节点，立即启动）

- 开发集：从训练候选池按17个数据集分层，每个数据集约6个窗口（约100个），只取首帧和真实后续视频。剔除与VBench、WM108、fixed val120、P1/C1案例同父来源的窗口。base模型从未在这些数据上训练过，所以对base不存在训练泄漏。
- 配置网格：sigma_shift ∈ {3, 5, 8, 12} × cfg ∈ {4, 5, 6}，共12组；提示词用原caption（raw），seed=0，50步，121帧，1280×704。
- 评分：VBench同实现的8个维度（含dynamic），加GT DreamSim、末帧DreamSim和运动量误差。
- 预注册选择规则：在开发集上按本地Total*计算方式求分。候选需同时满足：motion_smoothness ≥ 默认配置−0.3pp；subject_consistency ≥ 默认−1.0pp；GT DreamSim ≤ 默认+0.005；运动量误差 ≤ 默认+0.05。满足条件的配置中取Total*最高者；若都不满足，保留默认配置shift5/cfg5。
- 预算：生成上限140 allocated GPUh，评分上限40 GPUh；失败也计入，不自动扩大预算。

## 4. E2：冻结配置后的完整基准（4节点）

用E1选出的配置跑一次完整基准：VBench 355（raw和po）、WM108、Physics 198、WMB 350，评分算法与已有结果完全一致，并与base默认配置配对比较。上限120 GPUh。

## 5. E3：面向运动与美学的训练（E1/E2之后，按证据决定）

- 数据：在训练池中用光流运动量和美学分筛选“高运动、高美学”子集；caption使用经委托审阅/独立复核确认过的修正规则重标注。
- 臂：base初始化LoRA，分为原caption、修正caption、修正caption+基座保真项三个臂，对应docs/40。每个节点一个臂，第4节点负责参考与评分。
- 只在开发集上比较，冻结一个候选后再跑一次完整基准。Student/蒸馏/AR不在本计划内。

## 6. 产出与发布

每阶段的代码、配置、预注册规则、无身份聚合结果和结论推送GitHub。视频、权重、caption原文和凭据不公开。
