# 51 免训练Teacher多维策略有限实验预注册（2026-10-25）

## 1. 目标与前置

用户要求继续不同策略改善各维度，并用四节点执行。先完成C276子集诊断、seed1复验及结果四关发布；本文件冻结下一有限候选，不把原文其他模型收益当本地结果。默认base shift5/CFG5/50步、121帧、1280×704、24fps，禁止AR/Student/蒸馏跨门禁。

运动仍为主要差距，需同时报告美学/成像、主体/背景一致性、平滑、GT与物理/WMB，不能只刷dynamic。公开完整基准不参与参数选择；276子集也不能根据结果回扫C或新方法。当前full1366/180GPUh未授权。

## 2. 两个独立推理候选（不组合、不扫参）

- **ALG**：按官方Wan2.1 I2V14B原三分支公式，迁移到5B fused-firstlatent：v=U_original+CFG×(C_lowpass−U_lowpass)。只对首latent参考z做bilinear down/up，factor=0.4，round后的H/W，align_corners=False、antialias=True；前10/50步active。future噪声与原z不改，scheduler仍在原x上step且首latent锁回原z。不滤RGB、不全轨迹模糊。
- **DyMoS**：在scaled-QK之后softmax之前，对非参考query→首latent参考key减gamma=0.6；仅conditional分支、前10/50步active，phi=1。首参考query和其他key不变，不修改cross-attention。首参考span按真实patch几何880tokens，总27280tokens；禁止显式S²矩阵。只接受实际可用融合score_mod内核，CUDA不支持或预算不足如实停止。
- 两者在未微调base上单因素执行，不叠C LoRA/CFG-Zero*/STG/prompt扩写/TeaCache/sequence_parallel/cfg_merge。每次原实现和状态都须finally恢复，不修改当前冻结DiffSynth源码；独立适配脚本和运行源码先hash冻结。

## 3. 有限预算与准入

每算法独立45 allocated GPUh上限：完整规格单视频机制smoke1、94例×seed0/1共188候选生成24、原指标评分16、失败余量4；两算法合计最多90。每个阶段额度不能随意挪用或事后扩大。加载/编译/等待/失败/自有预留全计费；占卡另记。

先CPU机制测试，再单卡完整规格1video smoke。以smoke实测可行性预算验算188候选；若生成/评分无法在其阶段上限完成，停止并发布准入不足，不擅降步数/分辨率或借旧E1/E2账本。base两seed只有输入、模型、生成配置、全解码及评分版本可核实匹配时才复用，否则需要新明确预算，不把未复核历史作为免费同协议证据。

闲卡占卡、真实任务先pause释放。GPU数自主选择，四节点均纳入批量调度；CPU研发/数据门禁不意味着有效实验32卡满载。

## 4. 独立开发集与结果规则

沿用94例冻结独立开发集及父来源排除，原图/原prompt/同seed0/1，各188候选。预先冻结job/权重/代码/评分hash，两个seed都生成完才看指标，不加第三seed拯救候选。

每seed独立四护栏：smooth差≥−0.003，subject差≥−0.010，DreamSim差≤+0.005，运动误差差≤+0.05。不能用跨seed平均掩盖失败；全8维、Total*、各dataset、排egoexo4d88敏感性、GT与全帧视觉诊断一起报告。

本新增候选资格规则（不是回改E3）：两seed四护栏都过、每seed dynamic差>0且Total*差≥0，并确认非抖动/闪烁/纯镜头漂移才具备申请进一步冻结完整验收的资格。多候选同过按两seed平均Total*最高、精确同分优先DyMoS；不得在公开基准据结果改参数。case两seed一起分组bootstrap2000次/seed20261025，优先可恢复父源分组；两seed不是总体方差充分估计。

任何missing/duplicate/nonfinite/hash漂移/坏片/预算超限拒绝完成声明。CPU测试不是CUDA内核实测，smoke也不是质量通过。失败如实发布，不启动AR。

## 5. 多维后续顺序

ALG/DyMoS完成后，以实测决定是否另预注册STG等画质策略或DenseDPO偏好训练。DenseDPO需先明确合规训练池、独立标签/偏好复核与新硬预算，不使用VBench RAFT/LAION评分器作为训练奖励；不能因“继续不同策略”绕过训练数据门禁。当前DenseDPO尚未启动。

## 6. 当前执行状态

仅ALG隔离适配和SOTA协议审计在CPU研发；无本地ALG/DyMoS质量或CUDA通过结论。每轮方案/代码/分析/否定结论仍按白名单、干净树、完整公开CPU回归、提交与远端字节核验发布，不公开媒体或逐case数据。
