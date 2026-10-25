# 52 四基准SOTA逐维对照与协议核验（2026-10-25）

## 1. 版本与定义

当前尚无新Teacher通过全面晋级。以固定base Wan2.2-TI2V-5B raw、shift5/CFG5、50steps、seed0为最可靠整体基线；不是所有单项最好。E3 C开发集87.295%低于base87.431%，dynamic差0，只是合格臂中最高；E2 shift12在WMB8.397高于base8.297但其他基准退化，未晋级。

先报告同输入base配对增量，再分别对照公布SOTA与同协议验证；不把超过base或过护栏叫SOTA。没有可比协议证据时gap留空，禁止强减不同比分。

## 2. VBench-I2V逐维官方公布最高

实际GET官方Space公开config，dataframe69，36模型，访问2026-10-25；不是搜索摘要。单位百分数，差为base−公布最高，pp。

| 维度 | base raw | 公布最高 | 对应模型 | 描述差pp | 状态 |
|---|---:|---:|---|---:|---|
| i2v_subject | 97.827 | 99.38 | LTX-2-I2V (w/o prompt optimized) | -1.553 | 公布值描述差 |
| i2v_background | 99.093 | 99.64 | ToMoviee-2.0 | -0.547 | 公布值描述差 |
| subject_consistency | 95.449 | 98.90 | Animate-Anything | -3.451 | 公布值描述差 |
| background_consistency | 97.541 | 98.79 | VideoCrafter-I2V | -1.249 | 公布值描述差 |
| motion_smoothness | 98.705 | 99.61 | STIV Apple | -0.905 | 公布值描述差 |
| dynamic_degree | 56.098 | 83.01 | DynamiCrafter-CIL-512 | -26.912 | 公布值描述差 |
| aesthetic_quality | 61.971 | 70.21 | CogVideoX1.5-5B-I2V | -8.239 | 疑似错列，待核验 |
| imaging_quality | 71.061 | 97.07 | CogVideoX1.5-5B-I2V | -26.009 | 疑似错列，待核验 |

CogVideoX1.5异常原行：smooth40.98、dynamic62.29、aes70.21、imaging97.07、Total71.58、Quality50.90；本地20261011CSV也存在。不能自行换列修正，也不能当可靠SOTA。排除此疑似异常行的美学最高Animate-Anything67.12（描述差−5.149pp），成像最高Pi72.86（−1.799pp），**仍不是已核定真正SOTA**。

总榜第一DreamX-World-1.0官方原Total90.49含camera；我们无camera的固定8维Total*重算90.787722，base89.860075，描述差−0.927647pp。此前dynamic差−16.012pp是对DreamX统一参照；逐维运动最高差−26.912pp，其他维度也有缺口。“差距79%来自dynamic”仅针对DreamX统一参照，不能外推全部逐维SOTA。

协议差异：本地core355图各1视频；官方每pair5视频（core1775个唯一视频），另有camera109图×7指令×5视频。各模型分辨率/时长不同；本地1280×704/121帧24fps，官方5B po样例1280×720/81帧16fps。DreamX榜单精确checkpoint与raw/po未明示，不能把另一个公开AR或camera模型当该row同款。官方完整9维Total未测，不产生排名或同预算显著性结论。

来源：[官方榜单](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)、[公开表](https://vchitect-vbench-leaderboard.hf.space/config)、[官方I2V代码](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v)。

## 3. Physics-IQ Original与Verified必须分开

base Original原算法final_score_orig=30.130379%。官方Original公布I2V：Cosmos3-Super无BoN43.8；加WMReward BoN48.9，Sora2+WMReward46.4，Wan2.2+WMReward44.4。最高跨模态Magi-1+GeoPhys64.5是多帧V2V，不是I2V对照。

新Verified更改prompt、mask、scoring。公布Cosmos3-Super+prompt upsampling+best8=55.10（1次selectionrun），noBoN48.62±0.46（4runs）；Odyssey3Pro I2V best8=54.69。候选8×198=1584生成，不与本地单候选198同预算。base文件里的final_score_view32.99来自旧Original数据，不能冒称新Verified成绩。

当前同协议SOTA gap留空：exact checkpoint、prompt、context、run-count与评分版本未全面对齐；后续同Original本地重评或另开明确Verified新协议，不混算。本次C子集14完整scene三视角仅为Original诊断，不是完整SOTA验收。

来源：[官方Original](https://physics-iq.github.io/)、[维护代码](https://github.com/google-deepmind/physics-IQ-benchmark)、[Verified提交证据](https://github.com/google-deepmind/physics-IQ-benchmark/pull/87)。

## 4. WorldModelBench350与WM108不是同一个基准

WMB原judge公式Inst＋5×physgood＋2×commgood，base8.297143，E2 s12 8.397143。官方论文judge KLING9.10，human表KLING8.82，不能混。judge子项Inst2.32、Physics4.81、Common1.95（各自四舍五入，不用子项加和反改原总分）。表历史2024-11-15、混T2V/I2V、KLING精确版/seed/分辨率未明示；只并列参考，同协议SOTA gap留空，不假称现代I2V排名。子集70条7域完整8问同judge诊断；不拿子集均值配对全集总分。

WM108是本项目自选领域留出，不是公开同名认知benchmark或WMB350；外部SOTA及gap留空。已有同108相关baseline不代表2026SOTA，后续对确定公开checkpoint在同108输入本地评测才能比较。base该集Total*约87.06、DreamSim约0.1746；运动误差约0.9942比某些已有baseline更差，不能凭GT距离较低说全面最好。

来源：[WMB官方项目](https://worldmodelbench-team.github.io/)、[judge代码](https://github.com/WorldModelBench-Team/WorldModelBench)、[原论文表](https://arxiv.org/html/2502.20694v1)。

## 5. 下一步与质量门禁

同时覆盖VBench raw/po、WM、Physics、WMB；现已冻结并发布C276子集，生成/评分/配对未完成就不写结果。ALG低通三branch、DyMoS参考attention再平衡按docs51独立有限方案推进，不叠LoRA/Zero*/prompt扩写刷分。

每次同报运动、图像保真、时序一致性、美学/成像、GT、物理与常识，不只dynamic。所有数据/模型/code/hash、失败计费和同协议证据先冻结。当前未证明各基准SOTA，AR/Student/蒸馏门禁继续关闭。无身份机器可读参考：eval/diag/teacher_sota_protocol_public_20261025.json；这是参考资料不是新增实验结果。
