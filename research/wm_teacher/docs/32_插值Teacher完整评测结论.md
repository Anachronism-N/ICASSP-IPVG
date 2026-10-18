# 三个权重插值模型：完整评测与可复核结论

- 完整覆盖门禁：PASS（严格 merge_final_gate；不是质量门禁）。
- Teacher 自动条件：FAIL（按每个候选的全部自动条件判定）。
- 人工盲评 / 用户门禁：pending，尚未完成；不得宣称 Teacher 通过或 SOTA。
- 报告完成不授权因果/流式 Student、FT3 训练。

## 协议与单位

所有本地比较只用原始 raw prompt；不混入 _po。VBench 公共集355条，WM108条，Physics198视角/66完整场景，WMB350条，固定验证120条/121帧/5时间步。
VBench/WM Total* 严格调用原 summarize_eval.total：排除 camera_motion；dynamic 权重0.5，仍计入 Total*；七个 non-dynamic 单项用于榜首操作性条件。不是官方原始 Total。
imaging_quality 原值/归一化保持原脚本处理，不额外 clamp 或缩放；表中 VBench/WM 含该项统一乘100显示，异常幅度不隐藏。Physics 原始分数0–1与100×百分比分开；WMB约0–10，loss/DreamSim/motionErr非百分数。未测不补零、不推断。

## 公开 VBench-I2V raw

| 模型 | Total* | I2V* | Quality* | i2v_subject | i2v_background | subject_consistency | background_consistency | motion_smoothness | aesthetic_quality | imaging_quality | dynamic_degree（不计head wins） |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Wan2.2-TI2V-5B (ours, base) [base] | 89.86% | 98.11% | 81.61% | 97.83% | 99.09% | 95.45% | 97.54% | 98.71% | 61.97% | 71.06% | 56.10% |
| Round4 LoRA (aes-filtered) [round4] | 89.47% | 97.81% | 81.12% | 97.62% | 98.83% | 95.03% | 97.16% | 98.70% | 61.70% | 70.90% | 53.66% |
| Full-FT 5B (FT1) [full1] | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| Full-FT 5B (FT2, base init + general data) [ft2] | 89.82% | 98.13% | 81.51% | 97.87% | 99.08% | 95.10% | 97.19% | 98.80% | 61.70% | 70.93% | 56.91% |
| 0.7·base + 0.3·FT1 (weight merge) [m_full1_a03] | 89.81% | 98.13% | 81.49% | 97.97% | 99.00% | 95.42% | 97.43% | 98.77% | 61.85% | 70.93% | 55.28% |
| 0.5·base + 0.5·FT1 (weight merge) [m_full1_a05] | 89.89% | 98.13% | 81.66% | 97.96% | 99.00% | 95.36% | 97.29% | 98.80% | 61.77% | 70.85% | 57.72% |
| 0.3·base + 0.7·FT1 (weight merge) [m_full1_a07] | 89.77% | 98.09% | 81.45% | 97.86% | 99.04% | 95.13% | 97.20% | 98.79% | 61.75% | 70.80% | 56.50% |
| Wan2.1-I2V-14B-720P [wan21] | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| HunyuanVideo-I2V [hunyuan] | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| CogVideoX-5B-I2V [cogvideox] | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |

## 领域内 WM-Bench raw

| 模型 | Total* | I2V* | Quality* | i2v_subject | i2v_background | subject_consistency | background_consistency | motion_smoothness | aesthetic_quality | imaging_quality | dynamic_degree（不计head wins） |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Wan2.2-TI2V-5B (ours, base) [base] | 87.06% | 97.32% | 76.81% | 97.63% | 98.09% | 92.20% | 94.68% | 98.73% | 48.10% | 64.11% | 60.19% |
| Round4 LoRA (aes-filtered) [round4] | 86.56% | 96.48% | 76.64% | 96.74% | 97.62% | 92.01% | 95.07% | 98.69% | 47.73% | 63.37% | 60.19% |
| Full-FT 5B (FT1) [full1] | 86.62% | 96.41% | 76.83% | 96.70% | 97.56% | 92.01% | 94.85% | 98.67% | 47.92% | 63.23% | 62.96% |
| Full-FT 5B (FT2, base init + general data) [ft2] | 87.11% | 97.68% | 76.53% | 97.92% | 98.37% | 92.38% | 94.61% | 99.00% | 47.75% | 63.73% | 56.48% |
| 0.7·base + 0.3·FT1 (weight merge) [m_full1_a03] | 86.90% | 97.07% | 76.74% | 97.31% | 98.00% | 92.22% | 94.96% | 98.69% | 48.03% | 63.98% | 59.26% |
| 0.5·base + 0.5·FT1 (weight merge) [m_full1_a05] | 87.06% | 97.17% | 76.94% | 97.32% | 98.14% | 92.24% | 95.11% | 98.69% | 48.09% | 63.90% | 61.11% |
| 0.3·base + 0.7·FT1 (weight merge) [m_full1_a07] | 86.91% | 96.86% | 76.96% | 97.11% | 97.86% | 92.30% | 95.07% | 98.72% | 48.01% | 63.93% | 61.11% |
| Wan2.1-I2V-14B-720P [wan21] | 86.63% | 95.54% | 77.72% | 96.30% | 96.61% | 89.96% | 93.04% | 98.54% | 48.36% | 69.22% | 70.37% |
| HunyuanVideo-I2V [hunyuan] | 85.44% | 96.55% | 74.32% | 97.29% | 97.25% | 92.48% | 94.59% | 99.36% | 46.66% | 57.80% | 43.52% |
| CogVideoX-5B-I2V [cogvideox] | 84.00% | 93.59% | 74.40% | 95.48% | 94.45% | 92.67% | 94.30% | 98.34% | 45.92% | 58.06% | 52.78% |

## GT参考、物理基准与验证loss

| 模型 | DreamSim↓ | DreamSim末帧↓ | motionErr↓ | Physics raw↑ | Physics %↑ | WMB /10↑ | 指令 /3 | 物理 /5 | 常识 /2 | fixed120 loss↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| Wan2.2-TI2V-5B (ours, base) [base] | 0.1746 | 0.2087 | 0.9942 | 0.301304 | 30.13% | 8.297 | 2.266 | 4.509 | 1.523 | 0.197533 |
| Round4 LoRA (aes-filtered) [round4] | 0.1569 | 0.1837 | 0.7298 | 0.273267 | 27.33% | 7.100 | 2.049 | 4.023 | 1.029 | 0.187596 |
| Full-FT 5B (FT1) [full1] | 0.1596 | 0.1861 | 0.6719 | 0.248192 | 24.82% | 7.091 | 1.997 | 4.080 | 1.014 | 0.188037 |
| Full-FT 5B (FT2, base init + general data) [ft2] | 0.1722 | 0.2077 | 0.9216 | 0.283380 | 28.34% | 8.146 | 2.251 | 4.451 | 1.443 | 0.196675 |
| 0.7·base + 0.3·FT1 (weight merge) [m_full1_a03] | 0.1694 | 0.2047 | 0.9408 | 0.296342 | 29.63% | 8.031 | 2.251 | 4.343 | 1.437 | 0.193599 |
| 0.5·base + 0.5·FT1 (weight merge) [m_full1_a05] | 0.1656 | 0.1952 | 0.9010 | 0.281699 | 28.17% | 7.814 | 2.194 | 4.274 | 1.346 | 0.191320 |
| 0.3·base + 0.7·FT1 (weight merge) [m_full1_a07] | 0.1598 | 0.1880 | 0.8054 | 0.268758 | 26.88% | 7.511 | 2.100 | 4.183 | 1.229 | 0.189408 |
| Wan2.1-I2V-14B-720P [wan21] | 0.2182 | 0.2544 | 0.8213 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| HunyuanVideo-I2V [hunyuan] | 0.1980 | 0.2438 | 0.8585 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| CogVideoX-5B-I2V [cogvideox] | 0.1940 | 0.2415 | 0.8301 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |

motionErr 为原GTref motion_ratio字段（绝对log运动量误差），不解释为运动质量或 dynamic 单项。

## Teacher 操作性自动条件与人工pending

| 候选 | loss胜过base条数 /120（≥108） | GT优于base及已有相关基线 | 实际GT对照 | head wins /7（≥3） | 领先维度 | 全部自动 | 人工 |
|---|---|---|---|---|---|---|---|
| m_full1_a03 | 120 | FAIL | base, wan21, hunyuan, cogvideox | 0 | 无 | FAIL | pending |
| m_full1_a05 | 120 | FAIL | base, wan21, hunyuan, cogvideox | 0 | 无 | FAIL | pending |
| m_full1_a07 | 120 | PASS | base, wan21, hunyuan, cogvideox | 0 | 无 | FAIL | pending |

| non-dynamic维度 | m_full1_a03 | m_full1_a05 | m_full1_a07 | DreamX-World-1.0 | ToMoviee-2.0 | Steamer-I2V | Magi-1 | Pi | JT-CV |
|---|---|---|---|---|---|---|---|---|---|
| i2v_subject | 97.97% | 97.96% | 97.86% | 98.27% | 99.05% | 98.02% | 98.39% | 98.67% | 99.14% |
| i2v_background | 99.00% | 99.00% | 99.04% | 99.04% | 99.64% | 99.37% | 99.00% | 98.87% | 99.58% |
| subject_consistency | 95.42% | 95.36% | 95.13% | 95.35% | 95.12% | 93.45% | 93.96% | 96.87% | 95.87% |
| background_consistency | 97.43% | 97.29% | 97.20% | 97.24% | 97.10% | 96.99% | 96.74% | 97.50% | 97.48% |
| motion_smoothness | 98.77% | 98.80% | 98.79% | 98.52% | 99.10% | 98.47% | 98.68% | 99.18% | 99.25% |
| aesthetic_quality | 61.85% | 61.77% | 61.75% | 63.70% | 64.00% | 63.94% | 64.74% | 61.91% | 62.37% |
| imaging_quality | 70.93% | 70.85% | 70.80% | 71.46% | 70.15% | 70.48% | 69.71% | 72.86% | 70.77% |

head wins仅指候选维度≥上述六个原榜单头部该维度最大值，不代表配对显著性、总体胜出或SOTA。本地固定1 seed与榜单5 seed协议不同；人工盲评始终独立pending。

## 描述性权衡与Pareto

Teacher GT条件沿用原脚本：比较base及已有相关基线；未测相关基线不扩大为新增评分任务，也不宣称优于未测者。

- 三插值候选的VBench Total*最高：m_full1_a05；仅当前固定输出描述。
- 三插值候选的DreamSim最低：m_full1_a07；仅当前固定输出描述。
- 三插值候选的Physics raw最高：m_full1_a03；仅当前固定输出描述。
- 三插值候选的WMB最高：m_full1_a03；仅当前固定输出描述。
- 三插值候选的验证loss最低：m_full1_a07；仅当前固定输出描述。
- Pareto非支配集合（仅三个完整raw插值候选）：m_full1_a03, m_full1_a05, m_full1_a07。
- 比较方向：VBench Total*↑、DreamSim↓、Physics↑、WMB↑、val↓；不以dynamic单项作为择优依据。
- 未齐这五项而排除的插值候选：无；controls/相关baseline只作表格对照，不参与候选Pareto。
- 不设任意跨基准权重，不据单seed分差宣称通用最优；不同指标存在描述性权衡。

## 固定seed配对不确定性（已有分析，不重新估计）

固定输出seed；同一案例/三视角完整场景配对 percentile 95% CI；bootstrap=5000，分析RNG seed=20261018。
CI不包含来源cluster、生成seed、训练或judge变异；多重比较未校正，不做因果结论或跨seed外推。Physics差值已是百分比点（pp），不再次乘100。

### dreamsim

| 候选 minus 对照 | Δ | 95% CI |
|---|---|---|
| m_full1_a03 minus base | -0.005238 | [-0.008755, -0.002250] |
| m_full1_a03 minus round4 | 0.012482 | [0.004530, 0.020721] |
| m_full1_a03 minus full1 | 0.009768 | [0.001376, 0.018307] |
| m_full1_a03 minus ft2 | -0.002809 | [-0.006653, 0.000773] |
| m_full1_a05 minus base | -0.009072 | [-0.014715, -0.003860] |
| m_full1_a05 minus round4 | 0.008648 | [0.003025, 0.014616] |
| m_full1_a05 minus full1 | 0.005933 | [-0.000098, 0.011949] |
| m_full1_a05 minus ft2 | -0.006644 | [-0.012697, -0.000624] |
| m_full1_a07 minus base | -0.014882 | [-0.022741, -0.007593] |
| m_full1_a07 minus round4 | 0.002838 | [-0.001430, 0.006914] |
| m_full1_a07 minus full1 | 0.000124 | [-0.003533, 0.003799] |
| m_full1_a07 minus ft2 | -0.012453 | [-0.020828, -0.004853] |
| m_full1_a03 minus m_full1_a05 | 0.003834 | [-0.000280, 0.008116] |
| m_full1_a03 minus m_full1_a07 | 0.009644 | [0.003105, 0.016242] |
| m_full1_a05 minus m_full1_a07 | 0.005810 | [0.002198, 0.009700] |

### dreamsim_last

| 候选 minus 对照 | Δ | 95% CI |
|---|---|---|
| m_full1_a03 minus base | -0.003969 | [-0.010054, 0.001560] |
| m_full1_a03 minus round4 | 0.020989 | [0.010443, 0.032023] |
| m_full1_a03 minus full1 | 0.018654 | [0.007448, 0.029749] |
| m_full1_a03 minus ft2 | -0.003002 | [-0.009173, 0.002886] |
| m_full1_a05 minus base | -0.013459 | [-0.021452, -0.005781] |
| m_full1_a05 minus round4 | 0.011500 | [0.004432, 0.019031] |
| m_full1_a05 minus full1 | 0.009164 | [0.000340, 0.017767] |
| m_full1_a05 minus ft2 | -0.012491 | [-0.021447, -0.004201] |
| m_full1_a07 minus base | -0.020677 | [-0.030874, -0.010880] |
| m_full1_a07 minus round4 | 0.004281 | [-0.002098, 0.010440] |
| m_full1_a07 minus full1 | 0.001946 | [-0.004317, 0.008425] |
| m_full1_a07 minus ft2 | -0.019710 | [-0.030506, -0.009954] |
| m_full1_a03 minus m_full1_a05 | 0.009489 | [0.002680, 0.016713] |
| m_full1_a03 minus m_full1_a07 | 0.016708 | [0.007480, 0.026765] |
| m_full1_a05 minus m_full1_a07 | 0.007219 | [0.001498, 0.013187] |

### motion_ratio

| 候选 minus 对照 | Δ | 95% CI |
|---|---|---|
| m_full1_a03 minus base | -0.053385 | [-0.095227, -0.014313] |
| m_full1_a03 minus round4 | 0.210975 | [0.110299, 0.322298] |
| m_full1_a03 minus full1 | 0.268853 | [0.158094, 0.391277] |
| m_full1_a03 minus ft2 | 0.019201 | [-0.023500, 0.064194] |
| m_full1_a05 minus base | -0.093208 | [-0.153863, -0.034127] |
| m_full1_a05 minus round4 | 0.171152 | [0.071079, 0.279204] |
| m_full1_a05 minus full1 | 0.229029 | [0.120133, 0.351673] |
| m_full1_a05 minus ft2 | -0.020623 | [-0.075966, 0.037620] |
| m_full1_a07 minus base | -0.188768 | [-0.292930, -0.091860] |
| m_full1_a07 minus round4 | 0.075592 | [0.013101, 0.141291] |
| m_full1_a07 minus full1 | 0.133469 | [0.063415, 0.212707] |
| m_full1_a07 minus ft2 | -0.116183 | [-0.208631, -0.029352] |
| m_full1_a03 minus m_full1_a05 | 0.039824 | [0.008450, 0.071944] |
| m_full1_a03 minus m_full1_a07 | 0.135384 | [0.055456, 0.227687] |
| m_full1_a05 minus m_full1_a07 | 0.095560 | [0.024336, 0.182144] |

### Physics（pp）

| 候选 minus 对照 | Δ | 95% CI |
|---|---|---|
| m_full1_a03 minus base | -0.496221 | [-1.552731, 0.508867] |
| m_full1_a03 minus round4 | 2.307460 | [-0.229477, 4.942136] |
| m_full1_a03 minus full1 | 4.814920 | [2.740768, 7.114853] |
| m_full1_a03 minus ft2 | 1.296169 | [0.064674, 2.496352] |
| m_full1_a05 minus base | -1.960527 | [-3.532250, -0.581663] |
| m_full1_a05 minus round4 | 0.843154 | [-1.409251, 3.061349] |
| m_full1_a05 minus full1 | 3.350614 | [1.798039, 5.137659] |
| m_full1_a05 minus ft2 | -0.168136 | [-1.930147, 1.462418] |
| m_full1_a07 minus base | -3.254530 | [-5.154293, -1.555244] |
| m_full1_a07 minus round4 | -0.450849 | [-2.343860, 1.169192] |
| m_full1_a07 minus full1 | 2.056611 | [0.979000, 3.257116] |
| m_full1_a07 minus ft2 | -1.462140 | [-3.408828, 0.271565] |
| m_full1_a03 minus m_full1_a05 | 1.464305 | [0.030637, 3.105535] |
| m_full1_a03 minus m_full1_a07 | 2.758309 | [0.908961, 4.696823] |
| m_full1_a05 minus m_full1_a07 | 1.294004 | [0.075703, 2.595429] |

## 可复核来源

只列相对路径和SHA256，不导出prompt、私有日志、节点地址、凭据或人工身份。source_sha256是本次报告输入快照；checkpoint下表是原迁移清单hash，当前仅复查存在/大小，未重新计算10GB checkpoint hash；base外部预训练权重原清单未记录，不能造hash。

### source_sha256

| 相对来源 | SHA256 |
|---|---|
| Physics-IQ/descriptions/descriptions_original.csv | 66dbe342affda20d4abb76579a36d985cb7e221b5b6eae571069f50f155c37fc |
| WorldModelBench/worldmodelbench.json | 56cf38e7fc8cb3a485d71de890a38d2857752cc168addda8900b89f9809a842b |
| eval/diag/decode_m_full1_a03_20261018.json | ff9349f36b422d27d0d4a11702ac759a5e5da614c38795571ea400735348fd95 |
| eval/diag/decode_m_full1_a05_20261018.json | 721a9fc49c3b5c88d829bab1c02264c849983ed0642918e24682de67cb135379 |
| eval/diag/decode_m_full1_a07_20261018.json | 7d7a791b181705141ce29167f05d41aa753415854040aa57fc7c629671f75c4c |
| eval/diag/merge_all_analysis_20261018.json | ef17e416ecf9561a4c7d7b7206a1ee3e3521ce15c56708c1c7c3f86f86fa5a9a |
| eval/diag/migration_checksums.json | 25c7c3e932b6bc13c37d34407b5b5b54c50e628c8e1228a36e8892d71bf6bc98 |
| eval/jobs_ft2_all.jsonl | c5ba30c79e87e3b1d2a127a1c108ede8f9018af7e33ed50531bd4dd12944dda9 |
| eval/jobs_merge_all.jsonl | 2c865909865d5bcb0111d394d00a36c4547dd24521626f92fea5acdec0634930 |
| eval/physiq/results/physics-IQ-benchmark/results/base-op-run_01.csv | a4de9545792faa3d39e98905887cb2921a5d48fb73defee13d06f5bea479e606 |
| eval/physiq/results/physics-IQ-benchmark/results/base-op-run_01_metrics.json | 9ee444561e04dd3fede81cee516a78f04263064e47b2152e10cacfe730ed126b |
| eval/physiq/results/physics-IQ-benchmark/results/ft2-op-run_01.csv | dade0240d116ac743d1f7a7e97fd02d2159d3573085d545f75a469c60051b801 |
| eval/physiq/results/physics-IQ-benchmark/results/ft2-op-run_01_metrics.json | 25b5255b8eab523cdc847e6e0fe79fcc0379404230caedb4c11b9b2ebd544fb1 |
| eval/physiq/results/physics-IQ-benchmark/results/full1-op-run_01.csv | 3894a17bcae60e6d8763a506fe967c4267f80b158e1704a2dfd982ab443e577c |
| eval/physiq/results/physics-IQ-benchmark/results/full1-op-run_01_metrics.json | ce126f573a00d81ad3f5fa39b816eae48392e494146e9fd028bbe0118276dec6 |
| eval/physiq/results/physics-IQ-benchmark/results/m_full1_a03-op-run_01.csv | 48947f551d5062e89253e16b3c8b13418de97271b2554e6609f8551bc2f12106 |
| eval/physiq/results/physics-IQ-benchmark/results/m_full1_a03-op-run_01_metrics.json | b0c8317241e94799986483b3c11d6f552c4672c7e333eaf9894b13dcc11521f1 |
| eval/physiq/results/physics-IQ-benchmark/results/m_full1_a05-op-run_01.csv | f838182daec1be0e3ea8afcd76b5ca3c3ee888a28c56a71116a93df8b73cecee |
| eval/physiq/results/physics-IQ-benchmark/results/m_full1_a05-op-run_01_metrics.json | 99e2de4890cc021d21219b20b7c621587a5a1fc36234bfe1804a968bb0be9fe9 |
| eval/physiq/results/physics-IQ-benchmark/results/m_full1_a07-op-run_01.csv | 59a74889db847ab6dbfaef7a142b0be6da07f76fb895637e640f7b7340929477 |
| eval/physiq/results/physics-IQ-benchmark/results/m_full1_a07-op-run_01_metrics.json | 0a46df57d6803da0cafd8bbb94f5f449c596781cb6da35502a0b428cb8c7e4e0 |
| eval/physiq/results/physics-IQ-benchmark/results/round4-op-run_01.csv | 590159b8f04f2818ee51063081c40deb148a62991f22a8bbf07e8ba5e0a6a73a |
| eval/physiq/results/physics-IQ-benchmark/results/round4-op-run_01_metrics.json | 4e11e225590810c2ac7ec0de3a82615e8dcb2893d5827686ef64b35de9b767a4 |
| eval/vbench_i2v/cases.jsonl | af64dd751d0d4400fef61b714983d86b4b4ce2613b2a6a64257091ddf9b89128 |
| eval/vbench_i2v/scores/base__aesthetic_quality_eval_results.json | 36333e6b60d41cdddc891d87556707cc5423fde639d0349448ac15d5a980a110 |
| eval/vbench_i2v/scores/base__background_consistency_eval_results.json | 4aff7d560f31296c832cb1206cadb841762b4ba7a5934e968be0bdc76bdc6c00 |
| eval/vbench_i2v/scores/base__dynamic_degree_eval_results.json | 33a7d4b57a8cb73eff019fb192a4a1c4bb255f56a2a962e23a6266643ebf3c2a |
| eval/vbench_i2v/scores/base__i2v_background_eval_results.json | fd1a3aceff820ad910e4f9b1f355224cd0d9f71c5e65c617450ca4bd09db21ef |
| eval/vbench_i2v/scores/base__i2v_subject_eval_results.json | cf0220b40d7a281a47cbecc5c2270bab55d4414a3c6956b156fd8d4318dfe171 |
| eval/vbench_i2v/scores/base__imaging_quality_eval_results.json | bb0cb29b5d6bd039bbf175c76f4ace2df90022ea2d045586c2b4c4a20819cdaf |
| eval/vbench_i2v/scores/base__motion_smoothness_eval_results.json | d839cf8915192713ca23eb764d4c18e6abfb0b5f24467d088e9d3443ee151db0 |
| eval/vbench_i2v/scores/base__subject_consistency_eval_results.json | 08d8cd04798dd87484e8aed52116bfc712d6aadafb61fce915892c854b76cb34 |
| eval/vbench_i2v/scores/ft2__aesthetic_quality_eval_results.json | 8a39b750532d852bc0acf2d2c7d14bc983160f56bd8f2d9a5b0261355f4632f7 |
| eval/vbench_i2v/scores/ft2__background_consistency_eval_results.json | 08cccbf32d2621fbbd02e55552a708bba22ef606c3e53961c121dbbd62032562 |
| eval/vbench_i2v/scores/ft2__dynamic_degree_eval_results.json | f359c36ae64a16df2f5e83bb606ac21faee2a1da739475d72c7edc6623d7429a |
| eval/vbench_i2v/scores/ft2__i2v_background_eval_results.json | c99f17c93290e52bb523f7a6ccf576004848b62872d3eaa6c4538b057ea0c6c9 |
| eval/vbench_i2v/scores/ft2__i2v_subject_eval_results.json | f8d9ddf1351895f223bd51693bd48dc38862a1d77e8bbf8b6a461d2ffccb8936 |
| eval/vbench_i2v/scores/ft2__imaging_quality_eval_results.json | aef2124f01cc9a99f686ddbd415929f1338ac0e267b946bc1d075ddb1057c0b7 |
| eval/vbench_i2v/scores/ft2__motion_smoothness_eval_results.json | c7812172e669dde68b220a44eafae85178542e960c7dce2dddbe6885465b6926 |
| eval/vbench_i2v/scores/ft2__subject_consistency_eval_results.json | 67771c2c807f774592825a22ae99a28aa256d863aadf707d04e4d622a0de3f2b |
| eval/vbench_i2v/scores/m_full1_a03__aesthetic_quality_eval_results.json | fc1e144f6f6e418f3d23f9e9a7efb346eb60a153c734ba59fe5babe1c4b5e1f6 |
| eval/vbench_i2v/scores/m_full1_a03__background_consistency_eval_results.json | a5b52787375df451d5a354e6f5c21c97afc7e8366cad9dca8de1dda3abb243f2 |
| eval/vbench_i2v/scores/m_full1_a03__dynamic_degree_eval_results.json | 907e6b029311653b4cb0225daf53ff61f3cca0139fd02ebfe11716338690bafd |
| eval/vbench_i2v/scores/m_full1_a03__i2v_background_eval_results.json | d3873cc34cae48a32c536f6245cc935778100a4c0c9c27beb443b6efdb77b147 |
| eval/vbench_i2v/scores/m_full1_a03__i2v_subject_eval_results.json | f5d8d154f45cf3552e3d55c45ba2c8a8474815dfecc77e6a0461730ecac7ebd2 |
| eval/vbench_i2v/scores/m_full1_a03__imaging_quality_eval_results.json | 3101cfc9476653ac4c67ef547dfb8ccbffa67fdf56fcb3dbb37e7e5669398ead |
| eval/vbench_i2v/scores/m_full1_a03__motion_smoothness_eval_results.json | 26cf5173da9e4877a58698d2fb62e7484a7c92be5510ad3fd223b180ec88fe81 |
| eval/vbench_i2v/scores/m_full1_a03__subject_consistency_eval_results.json | 8ecbec69489980425802b0e242f91a188ed0da93fbbb537599d3555478c4e5a2 |
| eval/vbench_i2v/scores/m_full1_a05__aesthetic_quality_eval_results.json | 0504f44b98a7ad446d383ee280588fdcb5564dfc981a9a8ff794d98d614f4592 |
| eval/vbench_i2v/scores/m_full1_a05__background_consistency_eval_results.json | a54b4448499b3c10d36c3ebdf45e7668295ffc18492ccf83e2ce28ce740f7d97 |
| eval/vbench_i2v/scores/m_full1_a05__dynamic_degree_eval_results.json | 2f9c52df644e16dc228b4b9f6ab0ef5c17bde630807b8596574bf3462a8042fc |
| eval/vbench_i2v/scores/m_full1_a05__i2v_background_eval_results.json | bdd4adff58788ebf9f5ca5ab7f246029ae865b0101bbd23cab09432f825adbce |
| eval/vbench_i2v/scores/m_full1_a05__i2v_subject_eval_results.json | c4af01381ec7a6caf3c6d1492a572082ad619f9231f421741a12b0adbac1b5a9 |
| eval/vbench_i2v/scores/m_full1_a05__imaging_quality_eval_results.json | 14b681ad5d53551a462cf96962a01800031dba9589f7c27e26c4df829421c7b8 |
| eval/vbench_i2v/scores/m_full1_a05__motion_smoothness_eval_results.json | 5fe57efb4762fe4baa918346022981fc0c310255fceaa258e18ad8cf7971c242 |
| eval/vbench_i2v/scores/m_full1_a05__subject_consistency_eval_results.json | a3645cd5617649eb9c630450be025f741854b2d523eeef05119fa9f2e7877d0c |
| eval/vbench_i2v/scores/m_full1_a07__aesthetic_quality_eval_results.json | d19112c53174ac644cf8146771baccb43010484b46c9f8dc7dd8b6f652c98475 |
| eval/vbench_i2v/scores/m_full1_a07__background_consistency_eval_results.json | 0a915184831c1126275a2cd90ab984ee853f93003b28839b5b5c64d1f63bc9de |
| eval/vbench_i2v/scores/m_full1_a07__dynamic_degree_eval_results.json | 08d9ce7b015c08fa31112bcb5c637400a898a36931769d2d4446481216f4f7ad |
| eval/vbench_i2v/scores/m_full1_a07__i2v_background_eval_results.json | f7d0a0348c9759baeff78122ee57c5551930bba1c01b4bd32598275e88f7dde2 |
| eval/vbench_i2v/scores/m_full1_a07__i2v_subject_eval_results.json | 1a58da2f0a1af220f132f49309f5681268f03a9165eaeb6f8ac9b6af1843bfc0 |
| eval/vbench_i2v/scores/m_full1_a07__imaging_quality_eval_results.json | e202ab57cd07859d4fa0192264d1b95201393011af77d2f9cf245b8f100c31bc |
| eval/vbench_i2v/scores/m_full1_a07__motion_smoothness_eval_results.json | 5665bbcfcb612d52e726b335fa7d9aafd529d58db555c00a8aff4ea9b0143e5e |
| eval/vbench_i2v/scores/m_full1_a07__subject_consistency_eval_results.json | 136fc11491e72012cfc051f588d9f730cb644fec7779494580eef35b9934be82 |
| eval/vbench_i2v/scores/round4__aesthetic_quality_eval_results.json | a77567533474f2fefc497cc4880beb125d510fbe4c719e73e57b74a498d73f0b |
| eval/vbench_i2v/scores/round4__background_consistency_eval_results.json | 8acdf0c7218d7bc7ff5bd368be374c5901cb735bb6fb9041c9e6933ce0f30898 |
| eval/vbench_i2v/scores/round4__dynamic_degree_eval_results.json | 70b3abb2cd7ec808805610424c568a3b7ee0e9d652edb93bb3468a3340eb9bb9 |
| eval/vbench_i2v/scores/round4__i2v_background_eval_results.json | 04ce2a22cb92d311d7a170f3f5ee8030c721e0319c034c26298bafe988d37694 |
| eval/vbench_i2v/scores/round4__i2v_subject_eval_results.json | ebe32eacdaebe4d170c84f0f7198cfdae880078bebf83b151c2d8b45101c253a |
| eval/vbench_i2v/scores/round4__imaging_quality_eval_results.json | 6a21b6edb233b386d610deed9d859dac95f722b1189631d3e312a4552ec47197 |
| eval/vbench_i2v/scores/round4__motion_smoothness_eval_results.json | 17b983357402c55a5d7a3b3855c71e41dc72756a254385dfe5c84ca37937d230 |
| eval/vbench_i2v/scores/round4__subject_consistency_eval_results.json | e1bf23b99447547dde095653798d0d84f112a8a244ca15b817fddcc7930008b5 |
| eval/vbench_i2v_leaderboard_20261011.csv | 13421761bdbe06040fa1be5ef633ec84a2e5a747e065ac73f1261318d1b08e57 |
| eval/wm_bench/cases.jsonl | 66dc6e46338987058593af761e3d3b7f23dfd40d26f8231d8a9dcf50fce56f8d |
| eval/wm_bench/gtref_base.json | a0172348ef35560618992f64e9c9353932c7646521e2c19ebf24f2450a394cbb |
| eval/wm_bench/gtref_cogvideox.json | 05902b51f32cb448a0c0024f529b70261daf2875ae7d7402e2cbff7762d34a00 |
| eval/wm_bench/gtref_ft2.json | 53b86b4404f36b013aef5dd7f64612fe626a857d378baee1b66667c472a7e9c4 |
| eval/wm_bench/gtref_full1.json | fd43a7f6ab024d59a1168aa82d7638944eb860831319dcc99fc960e144aa3016 |
| eval/wm_bench/gtref_hunyuan.json | 9b8fc43577d8ac929a9d6ff7489ca5f3a58419a533d8210d59ad8efa714bbbb4 |
| eval/wm_bench/gtref_m_full1_a03.json | d333d34161943fe770bc84346e71860661499a144c477d1e5cdce83e0492272b |
| eval/wm_bench/gtref_m_full1_a05.json | f25e398d633719dc53faf86104b6eb44d23824f523c255913775183d7193ceb6 |
| eval/wm_bench/gtref_m_full1_a07.json | 6fa2effda6d4794b6a8d85502e928b5d6f7a7365844913a3155d3f5fde044302 |
| eval/wm_bench/gtref_round4.json | 9731588d03073933da3e543dd1c89e5deea438c6fcd9af1d794243bba08be2ab |
| eval/wm_bench/gtref_wan21.json | b6514d36644025a19f42961d0e5258a15b3ea55551de97945c9b7fdfe1cf9041 |
| eval/wm_bench/scores/base__aesthetic_quality_eval_results.json | 4e5eae8351820922ad349adc5bf8e77e9c1d0e2f84e9703cdab3e0df51220e00 |
| eval/wm_bench/scores/base__background_consistency_eval_results.json | 1b474a4f3f2d6995d35afaccb6a3ed2560c12085f897f04e535c314716b4d326 |
| eval/wm_bench/scores/base__dynamic_degree_eval_results.json | 999affd8abef092272d8352eb08338d3f7e565328b14a192126d48a40b39f524 |
| eval/wm_bench/scores/base__i2v_background_eval_results.json | 928171d4ee19b1a5940a668b84b933ff18d32fef118393fe469687a0d52e1e8f |
| eval/wm_bench/scores/base__i2v_subject_eval_results.json | 4c6884c6186a23cc9e8bc04bbb5d5e0caffa8b6a3fc3e6e768b328be0f3773b2 |
| eval/wm_bench/scores/base__imaging_quality_eval_results.json | d30334c3ba69eef79f316634b6ed500c399850d89256db1d4c316db638202e43 |
| eval/wm_bench/scores/base__motion_smoothness_eval_results.json | 7e96dde192845d4b4b1d3004794eb2ed7890cc176d5f660b83b5bffc0fba839d |
| eval/wm_bench/scores/base__subject_consistency_eval_results.json | b33065d033416e8fe76a98d08a0bcc7183019732138549bef7b5c34f82f4e6ad |
| eval/wm_bench/scores/cogvideox__aesthetic_quality_eval_results.json | dff340aed7c315fa8340ea4fed3b919bcd6742e8d54da34cae45c34d29ac100d |
| eval/wm_bench/scores/cogvideox__background_consistency_eval_results.json | b95f6c26b22f5fb4725e6fde2c096e2c4544effb9915be54235e4237c45ab61b |
| eval/wm_bench/scores/cogvideox__dynamic_degree_eval_results.json | 396bb4a235e722df72da6221b1e5e7cd0912d87359c00baf1eda0b93434a02b4 |
| eval/wm_bench/scores/cogvideox__i2v_background_eval_results.json | 5d65f560b32b00d03ab04aad239d5128d1188231b4e74000ac7c9cfff39b1655 |
| eval/wm_bench/scores/cogvideox__i2v_subject_eval_results.json | 917eb193b43186bad6a501b5c3fe81b14956e7dcabc0f4cc8d3b3426f0da7bd4 |
| eval/wm_bench/scores/cogvideox__imaging_quality_eval_results.json | c71c56a0173bed7792ee5db8c6936b312e87beb2059c5d18fa6c3377569daf8c |
| eval/wm_bench/scores/cogvideox__motion_smoothness_eval_results.json | 99a6cfa48be49f6395cb99a36b674624f395a85925f6190c38dfeeaebce4bfec |
| eval/wm_bench/scores/cogvideox__subject_consistency_eval_results.json | ea294ba7d7166bc34835ec285ba6098dcd3c102ed76ff240d3712bbf1874ab91 |
| eval/wm_bench/scores/ft2__aesthetic_quality_eval_results.json | 2dc530cea33ccd712c1f8cf3fd9b9fb527881e9da2396e28eabe115a760937f7 |
| eval/wm_bench/scores/ft2__background_consistency_eval_results.json | cba2e44c081b92259caff801f1a1d582ec6c14d65edef464178f81db93425c68 |
| eval/wm_bench/scores/ft2__dynamic_degree_eval_results.json | 96c5253419424dd9e3ad1e7a8c35b66670be89a71ce4c95657d8272355d95131 |
| eval/wm_bench/scores/ft2__i2v_background_eval_results.json | c9356da11ff965cf8d0a4c6fe20a187c74a75590f289f7783f68d7b4da884dbc |
| eval/wm_bench/scores/ft2__i2v_subject_eval_results.json | 03a1aa286ba0f9fcfa107a5a4fbc90bb98977d0d158282dcb9e6350fce1bf76d |
| eval/wm_bench/scores/ft2__imaging_quality_eval_results.json | 48b7559a10920842cfc2ccde9626bb9fece411ff621277192e78ed615bc2dbfe |
| eval/wm_bench/scores/ft2__motion_smoothness_eval_results.json | 20f5ff79b5f3e5aee517c21fe0bfc985bb9c71e08948c87febe10c4f5e6165b2 |
| eval/wm_bench/scores/ft2__subject_consistency_eval_results.json | 92b1f8e1e7966d0b9ede5551aaacc770c097b02ad8f27f4b3636ab2a605d2c34 |
| eval/wm_bench/scores/full1__aesthetic_quality_eval_results.json | f557a3475ed80f49957ea43d0784947fba021c5d241865ec0d9a7e47e99bd47c |
| eval/wm_bench/scores/full1__background_consistency_eval_results.json | 8b269860414cd96ca450cc7e119eab8e816185946b3c77899dc899dce54633f3 |
| eval/wm_bench/scores/full1__dynamic_degree_eval_results.json | 1f2148b50d2cfb403fb61b3b221803426a268c9c24ea476f0c95f0d0bc242f16 |
| eval/wm_bench/scores/full1__i2v_background_eval_results.json | 5cf440217750c2a51fb87508ccc8b5dbf0cb6c0ba30c748dd947040a533d5929 |
| eval/wm_bench/scores/full1__i2v_subject_eval_results.json | f9e546d4d2071df7e0b34ac5ccc158b84a3a681100adb969381f1d0d54b1987c |
| eval/wm_bench/scores/full1__imaging_quality_eval_results.json | f84fd4f5d2578692b7eebd0283e91d39e7c24ae9ba974f9736145a73144a487f |
| eval/wm_bench/scores/full1__motion_smoothness_eval_results.json | 0241e227f23d991ae9f1430f3dda3999ec6064778c442a5546b77a1da48e96c4 |
| eval/wm_bench/scores/full1__subject_consistency_eval_results.json | 35fd033b9c3ca2f1593c36c9bee41d8894b41dbc1f1a665ece6a5d82fe975361 |
| eval/wm_bench/scores/hunyuan__aesthetic_quality_eval_results.json | 1b39be88df7dd94f78462e65616b695887bc894e36ecbcd1c122d3a568332f6d |
| eval/wm_bench/scores/hunyuan__background_consistency_eval_results.json | 29d51aac8004ca9ca3e7583031588a4d52cbfb3a49973d8829a54771118f502d |
| eval/wm_bench/scores/hunyuan__dynamic_degree_eval_results.json | 50023f6483b0430efd723092767a336b830b547b1fab3a70645c0ca47cfd562d |
| eval/wm_bench/scores/hunyuan__i2v_background_eval_results.json | 7b09c493fdbcec367e67eaabf7963ba27966bb1f96bf678c0bf54f8556c3dc1d |
| eval/wm_bench/scores/hunyuan__i2v_subject_eval_results.json | 212797d39d778c416a4d3cdaf5bf6ff3b4c50ace198f52d9a8726c042353cacf |
| eval/wm_bench/scores/hunyuan__imaging_quality_eval_results.json | 22dd19cc8366af551fab6c5fc163cae5aecadd8c637b99a4bfa89756ba794633 |
| eval/wm_bench/scores/hunyuan__motion_smoothness_eval_results.json | d03ea9f5d9c20b238c0d1c108bedaee10a45f9947233b7e36e660cc5d048d2b5 |
| eval/wm_bench/scores/hunyuan__subject_consistency_eval_results.json | eb948cd4ff2424cb562c052755c74c3d9fc09dcb686d752b07de34383caa744c |
| eval/wm_bench/scores/m_full1_a03__aesthetic_quality_eval_results.json | 6c3d884b509d528623f73b522524e1ba923f2377ed0442d8f9e14104882d2534 |
| eval/wm_bench/scores/m_full1_a03__background_consistency_eval_results.json | 44c7d2e4fa1a216546caddef3a107808d1df4236102682fecabc4305ea55c1a9 |
| eval/wm_bench/scores/m_full1_a03__dynamic_degree_eval_results.json | cedaba10ef436f48034ed3f1117c33cea7a1cf18a57bbf448804ccd207fe8075 |
| eval/wm_bench/scores/m_full1_a03__i2v_background_eval_results.json | 827884d6749edd800633d173db04f2c58e60ca247d93a4c683bda4388d016dae |
| eval/wm_bench/scores/m_full1_a03__i2v_subject_eval_results.json | 12a7b152963093816ce2615a45c1649d71388e68529b75cea07d848b61e55512 |
| eval/wm_bench/scores/m_full1_a03__imaging_quality_eval_results.json | 63cff8882afdd2565b7110813b64b59ffc4a2f209206daa70bb1314db2051138 |
| eval/wm_bench/scores/m_full1_a03__motion_smoothness_eval_results.json | 654c62957c06f51a2954bbb284b309c5578bd7b86e12361f8ffe05e622ce16c4 |
| eval/wm_bench/scores/m_full1_a03__subject_consistency_eval_results.json | 13d98d3d02e6bf7a10b8923a4cc8ba8ab999dd8a583b464b510e4ac669fed32c |
| eval/wm_bench/scores/m_full1_a05__aesthetic_quality_eval_results.json | 3c66fca896de6911819c00480e59351ef96f397a54c639c81a3aedc109d429a0 |
| eval/wm_bench/scores/m_full1_a05__background_consistency_eval_results.json | 9e23e25888bdc0a176f44ebfb59cdafc5926c4db1cba4ab60c5897f9ed09c874 |
| eval/wm_bench/scores/m_full1_a05__dynamic_degree_eval_results.json | 239f9bd89378cc8a5b483ac26e9d443f3272e9f458aaf4295915b683a5744b5e |
| eval/wm_bench/scores/m_full1_a05__i2v_background_eval_results.json | 768937ee76408b947bb5d7ce3f42baba14d3e2a2e5037ab6ca5427fdc0f78a1f |
| eval/wm_bench/scores/m_full1_a05__i2v_subject_eval_results.json | ef017898c877063fb57594ef232a13de90bc2375fba37d17522625605f51b1b6 |
| eval/wm_bench/scores/m_full1_a05__imaging_quality_eval_results.json | 6e4b766946dc8a8c00baa827d0ebc210eef7563549a4cf1db17ebf96afad2205 |
| eval/wm_bench/scores/m_full1_a05__motion_smoothness_eval_results.json | 1d7d1fcb622f121a5c62f5c4a2e2e4ee95265b48981535650c42664eba27a358 |
| eval/wm_bench/scores/m_full1_a05__subject_consistency_eval_results.json | 585059cf7ae9b8ff442d4891225a118ddc7957cbe11a72642ca45cc252709a52 |
| eval/wm_bench/scores/m_full1_a07__aesthetic_quality_eval_results.json | ec2147577a44d1abfdd7e68fb33a38b1eba83e45c0a49d321c61996f4cd9fc7d |
| eval/wm_bench/scores/m_full1_a07__background_consistency_eval_results.json | fa3a65b02edec35bb28578273259179b17cb2ef75b5d4e778660adbeed2d3d1d |
| eval/wm_bench/scores/m_full1_a07__dynamic_degree_eval_results.json | fd5c1da6a8e2804f85ac1d959c7c4f8ffb3d9dca243e7f560bb57a16b28971d0 |
| eval/wm_bench/scores/m_full1_a07__i2v_background_eval_results.json | 62228d65a51a04009a844b9c93e27c61796087f1c338e05672ed637413b055c9 |
| eval/wm_bench/scores/m_full1_a07__i2v_subject_eval_results.json | 0fa1260e8b96e13fc821f31236d5d93edfe62f5545903f55546c8e3bdd167d2f |
| eval/wm_bench/scores/m_full1_a07__imaging_quality_eval_results.json | 509b61d5c4dd00314114793ba09040d64e5f4835cfbc970226151ca615f428a2 |
| eval/wm_bench/scores/m_full1_a07__motion_smoothness_eval_results.json | 98f19fdf42f242f438f6a74569917865cf3f6e38531398f4c474cfb11d72f3b0 |
| eval/wm_bench/scores/m_full1_a07__subject_consistency_eval_results.json | 373112527b10279709f189c95f23d4478bae5d16380c05cd03b133f751082f6b |
| eval/wm_bench/scores/round4__aesthetic_quality_eval_results.json | c18bddfa2b55fcd65939bea29d43df8029415117f88de446a291ea16e06e745c |
| eval/wm_bench/scores/round4__background_consistency_eval_results.json | b798825ffa1e478f84bab805b9ef22709eb5fb66672cf15bfa4b5eb2be9befad |
| eval/wm_bench/scores/round4__dynamic_degree_eval_results.json | 940304408803c72f7e0d37d5211ef9f5f8e946903b9607e3188dce7d3ba09ab5 |
| eval/wm_bench/scores/round4__i2v_background_eval_results.json | a005efa9d15979d8c6ee01fafbd134a8e925e9f28be0ded855a10e1ea7516312 |
| eval/wm_bench/scores/round4__i2v_subject_eval_results.json | 96446b8f6285d9f4b0f3496832d75485bff0a3155dce1c867ddadfe0e3b5a26e |
| eval/wm_bench/scores/round4__imaging_quality_eval_results.json | ceb1836e432aba2d7a73eca8e900b748b818b520df120f5fc67f6b0ed828d4af |
| eval/wm_bench/scores/round4__motion_smoothness_eval_results.json | e17be14f90c039ba6f367b643fc827b6176b9c9ce43c68aebc17b5ac8fd847b3 |
| eval/wm_bench/scores/round4__subject_consistency_eval_results.json | 470db3aa73bc89cb000fdb0d45ff587c239ce587c2cd770fa51c0d7827aa976c |
| eval/wm_bench/scores/wan21__aesthetic_quality_eval_results.json | ab303ab953ec8e12c450eb486c595565b96cc776989d3760be8875343d485468 |
| eval/wm_bench/scores/wan21__background_consistency_eval_results.json | 0e5514d6ecb20979e7e9ee979c312ef5b02fe00b655d144dcf0c3a61d94a777a |
| eval/wm_bench/scores/wan21__dynamic_degree_eval_results.json | a1d1bd4cc10114c0c7c4935e4692b2c66079ca2bc08f80d9e928fb32eb5b8c50 |
| eval/wm_bench/scores/wan21__i2v_background_eval_results.json | cc768a8211ad3adac15a56769f3688457a72a3ca736c6764e02754313c1d081e |
| eval/wm_bench/scores/wan21__i2v_subject_eval_results.json | 6f836ceb2f1f66c064ab068e8b475235ebb5ecc4c506535c716156a3ab7a9711 |
| eval/wm_bench/scores/wan21__imaging_quality_eval_results.json | fabd82525c6076d9320aadf73509e951bbf00c8cbc3fffdcdc40451437fbe073 |
| eval/wm_bench/scores/wan21__motion_smoothness_eval_results.json | 5c4027cac9327188334cfb923bab6b1e4f3d19274c4e1b5c414c8909cbfabd51 |
| eval/wm_bench/scores/wan21__subject_consistency_eval_results.json | 35e9bc04933d35d972a3fb61663012302befb9a6e19dad73a40f10a08c4b766b |
| eval/wmb/results/base.json | 5305c557de9d1616ec47214f31a3c0ee597c959558df0764e6d93649ffeb6109 |
| eval/wmb/results/ft2.json | 119c19606fa7f9d34a0d5b64afa72ae369924e5396b89a9b67b3fd4824b0a6ac |
| eval/wmb/results/full1.json | d25fe44799c9852dcd0d4eece547884f281da57f2f11f30bb4151d3ebeeddd4b |
| eval/wmb/results/m_full1_a03.json | 48ec2de1325531950ed636ad06625d5b53b1daff739d76f751493442c6a7d3b7 |
| eval/wmb/results/m_full1_a05.json | 2ccb8ea92977642a8f42761b242ec46af055ac890d00759f0951f8a68fd79245 |
| eval/wmb/results/m_full1_a07.json | 0dd9b0916fb6182bb2bca61d4e6d13d0dc0049e6c9875854ce9b8123e50508f0 |
| eval/wmb/results/round4.json | ba88c13d46f5d8cc49d00d229cc3582455d50ae79ea9a7b97dc9a99e6b5e2f1e |
| models/train/round2_5s_lora_r32/eval/loss_base.json | 8f269b932218ed6813ab6242f7bdb774a8bec67aec68ec0e0fc5c6c752784898 |
| models/train/round2_5s_lora_r32/eval/loss_ft2.json | b6ef9e5b647363a04737715deaddce27eb9ce8a6b3031f56bba986777d390e6a |
| models/train/round2_5s_lora_r32/eval/loss_full1.json | 16761708b00710bab9c8df1627cd3f7ff6676136515558d90f52bf310d819679 |
| models/train/round2_5s_lora_r32/eval/loss_m_full1_a03.json | d18ee50e7e59fc015499b58a34e1e118da0c7b46a9a4a052f1d81c6fb0328003 |
| models/train/round2_5s_lora_r32/eval/loss_m_full1_a05.json | 9d54a6a345b481510a20d518d61dc373c6f3f1dd6730d3d49b348cb193eaa098 |
| models/train/round2_5s_lora_r32/eval/loss_m_full1_a07.json | 14975da8eb25cd204ca3da7e29e0e89f552dd9dfd09c926bd884a9070d67adf4 |
| models/train/round2_5s_lora_r32/eval/loss_round4.json | 3489582954d2b853a7c41fdf3cdc2f51e0679a9a2278c968fceabab75ae32064 |
| scripts/analyze_merge_metrics.py | f02ce4113d89a3fa5362b43d85a5bdcbcc68453cd37ec3fac8923a693835eabc |
| scripts/build_teacher_review.py | 648f87dda9ec21767e1c2857838d8758987f69a10d76787a6b90ce24fb46fa51 |
| scripts/merge_final_gate.py | a4a4fd943d0bed43714eabea4519748686da2d041b4b9e76e0ed959c76120b90 |
| scripts/report_merge_results.py | d271eb683e4826d425390649b84044e6c6d6f7ab2d51c595ae4f076e6d4389d5 |
| scripts/score_merge_cpu.py | 2ba690031bd82c994ad17d43e6e4a3a7fa945e9653b7b0620b1774378e29163d |
| scripts/summarize_eval.py | 01504506ed0b9f6f737b5dc57427459670a55777021746138d3a46445fa9c058 |
| scripts/val_merge_score.py | 398ad54248ae942f0ab5e4db913250f70d3f2dece3f43e95ce819524b8d70eaa |
| scripts/wmb_merge_score.py | b1cfe7cb2dd59ca015e0d8b6223c713cd6844c05d318a671b45286d172f25478 |
| training_metadata/round2/round2_val.csv | 211e1bc5d275a1055cdd2736fa7662e4ea0dee5e9bf7b7702e7bbf24bbcf0fc2 |

### model checkpoint 原migration hashes

| checkpoint相对路径 | 原SHA256 |
|---|---|
| models/train/ft2_5b/run1/step-2158.safetensors | 2424c610cbaec6db4ee98aacd01307dfa05c8210de4c0d0fbe64900fe4b0b3d6 |
| models/train/full1_5b/run1/step-2203.safetensors | 051da9cc1644298e7006c687656994d4a046cea24305cd904c20e72f79d96584 |
| models/train/merge/full1_a03.safetensors | c3d37a1f8ac0cbbb9a6264592b6da7247d1f5ffe3e90a17ec3a5c48ab1ce7cf8 |
| models/train/merge/full1_a05.safetensors | 23f820afd257e7ab0a8e9b4189a0c924c463375a0ca527f86ea51306fd7429b2 |
| models/train/merge/full1_a07.safetensors | 057d9a1e682e2f237594f5ef99f994f07699188c97b4196016fd371ab40402a6 |
| models/train/round4_aes_lora_r32/run1/step-1644.safetensors | 1e79b7cc0d9fd29db1170f6465cb840982676bae53870c4c3f35f414a266c6ae |
