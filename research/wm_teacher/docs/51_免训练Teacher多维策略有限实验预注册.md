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


## 7. 全部后续阶段：先免训练原型，后证据驱动优化（2026-11-09）

用户已明确选择全部后续阶段，要求先整理实现并推送GitHub，再继续实验。现在把系统原型与标签训练路线分开：**免训练系统验证无需等待监督caption/cache门；训练和SOTA宣称仍需各自证据。** 旧预算、失败和claim保留；本次不调用optimizer、不启动AR/Student/蒸馏、不借标签预算生成质量评测。

### 7.1 理论依据与不能推出的结论

现有Teacher给出窗口条件分布qθ(X下一窗 | 首图, 文本条件)。将末帧作为下一窗首图，配合外部状态记录和新指令，可构造窗口级接续：

qθ(X1…XK | X0, a1…aK) = ∏k qθ(Xk | last(Xk−1), prompt(ak, memory))。

这是**预训练模型的推理组合**，不需要新增参数训练；动作仅转为文本条件，raw设备输入不执行，外部观测状态与预测分层。分支fork复制同初态，分别条件采样，不能把多个分支拼接成已发生事实。

此分解不证明末帧是世界状态的充分统计量，不提供物理正确性、实体重现、长期一致性、动作可控性或实时延迟保证。双向Teacher内部仍联合生成一个窗口；窗口接续不是帧级因果AR/KV-cache。Student/一致性蒸馏不会仅靠接口获得低步数能力。**理论上可搭建原型，不等于理论上已达到SOTA。**

### 7.2 本次实现与全阶段对应

实现入口：scripts/interactive_world_model.py；测试：tests/test_interactive_world_model.py。

|阶段|本次可以运行的免训练部分|明确尚未实现/证明的部分|
|---|---|---|
|Teacher|TeacherWindowBackend适配已加载Wan管线；固定1280×704/121帧/24fps/50步/shift5/CFG5，单调用锁与失败计数|新适配器真实GPU视频尚未验收；已有Teacher质量不自动晋级|
|AR / 长程接续|InteractiveSession逐窗接续末帧；新观测校正参考，预测与观测分层|不是学习的帧级AR，无跨窗口原生attention/KV-cache，长期误差待测|
|记忆 / 多分支|泛型实体外部证据、时间、来源记录；fork深复制隔离；同状态不同动作条件|不自动跟踪身份、不推断不可见状态；entity ID由外部观测提供|
|动作交互|raw_action、semantic_action、expected_outcome分开；语义动作条件进入prompt|无键鼠控制、真实机器人执行或原生动作embedding；expected_outcome不写入真实state_after|
|Student / 蒸馏|后端接口可替换，端到端协议可以先用Teacher验证|**尚无Student权重或蒸馏实现/执行**；后续需独立数据、Teacher质量和预算准入|
|全双工|生成期间可接收文字、外部语音转写、动作、观测与打断；revision淘汰过时输出|不是原生音频生成/ASR/TTS或嘴型同步；无音视频实时流证明|

所有输入携带evidence字段仅为来源记录，调用方仍须核实内容安全、许可和真实来源；非空字符串本身不是证据审核通过。模块不打开网络、文件或设备，不申请GPU、不发消息。

### 7.3 实际行为与边界

- 输入按各流时间戳单调更新，跨流允许不同到达延迟；外部事件增加revision，运行中窗口只有revision不变才能提交。
- 打断会丢弃过时窗口，**不能即时停止同步CUDA推理或退款**。取消任务必须等线程结束后才释放管线锁；硬超时、进程清理与allocated GPUh仍由原外部监督器负责。
- 状态只由外部observation修改；预测末帧保存独立provenance。新动作/文字从已提交预测接续，不跳回旧观测；新真实观测才重置预测参考。
- 各session最多1024输入事件、128实体、32提交窗口；失败消耗max_calls，没有自动重试。预算callback在调用前/后执行，不替代外部硬deadline。
- 全双工仅指调度层允许输入与生成并发，不称低延迟模型；本原型无浏览器媒体发布、外部上传或人类判定代填。

### 7.4 可复现CPU实验

从源码根目录运行：

```sh
CUDA_VISIBLE_DEVICES= python scripts/interactive_world_model.py --cpu-demo
CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_interactive_world_model.py
```

演示覆盖：两窗接续、同初态分支、动作条件、生成中转写和打断、过时输出淘汰、观测与预测分离。结果明确synthetic_CPU_only，4次合成后端调用；未调用视频模型、GPU或音频模型，不能作为画质/延迟/动作控制证据。新适配器固定调用参数和完整121帧输出拒绝门通过CPU测试，但CPU假管线不是Wan实机证明。

### 7.5 发布后实验顺序

1. 原型完整CPU回归、固定白名单脱敏源码发布与独立远端字节核验。
2. 在独立有限推理计划中复用已核验Teacher资产，先2窗口连续性/分支机制验收；不降低帧数、分辨率或步数、不借旧费用、不自动重试。实机前绑定输入/源/模型/输出和硬cap。
3. 分开测窗口边界、实体回访、动作方向/次序、打断提交延迟、真实输入到输出延迟；同时报告同步推理不能即刻取消的代价。合成demo与真实结果分开。
4. 模型质量路线独立冻结完整集合，raw/po、I2V/V2V、单采样/BoN和camera/采样次数对齐，不能用原型系统实验替代官方SOTA评测。
5. 只有需要学习帧级因果结构或低延迟Student且满足质量/数据/预算准入，才实施训练；接口先行不宣称这些模型已经存在。

本节替代“标签训练是所有后续工作的必经路”的错误串行安排，但不撤销原训练/质量门和外部GPU授权边界。


### 7.6 完整媒体/评分harness与实机接口准入（2026-11-10）

world_model_harness.py为CPU验收CLI；交互状态复用interactive_world_model.py，不再增加重复会话实现。协议绑定输入/模型/scorer SHA、raw/po、I2V/V2V、samples_per_pair、single/all_samples/BoN、媒体规格和采样steps/CFG/shift。完整候选必须包含每pair全部样本及分维期待集合，评分缺失/重复/NaN拒绝。

审计重新校验视频文件SHA、ffmpeg -xerror全帧解码、ffprobe帧数/尺寸/fps；评分文件同时绑定协议与全部视频SHA，评分文件payload与磁盘字节一致。只是调用方声明集合的完整性，不会把complete声明当官方全集、更不会证明指标实现正确或SOTA。官方VBench仍用既有1118pair/5590视频/9维门；自定义合同不替代官方契约。

命令：python scripts/world_model_harness.py validate --contract PRIVATE_CONTRACT.json；或audit --contract PRIVATE_CONTRACT.json --videos PRIVATE_VIDEO_RECEIPTS.json --scores PRIVATE_SCORE_RECEIPTS.json。全部CPU，不自动启动GPU或评分器。

run_world_model_window_probe.py为独立的一次性**技术接口实验**，不是质量评测：本地合成红盘首图；原生Wan5B 50步、1280×704、121帧、24fps，先生成1窗，然后同末帧fork两条文本动作分支（固定seed0/1），共3窗。只验证首图链路/同父末帧/分支与观测不互写、真实FA2调用、完整可解码输出；不据此判断物理动作正确。复用原三DiT与六encoder/tokenizer pins，冻结框架/运行时/源码，完整测试及远端发布通过才run。

费用来自已有technical_admission剩余额度：单卡、独立1 GPUh上限/3600秒生命周期，计入原8 GPUh技术阶段，**不借标签/cache96或条件质量评测64**；原5 GPUh技术费用不退款。GPU4仅精确已知holder释放，未知进程/旧lease拒绝，finally释放自有子树并恢复模式，原claim和失败不可复用。CPU prepare与资产核验本身不分配GPU；run还须显式GPU开关及独立budget行。

此前281文件原型提交68356db已正常发布，1626测试通过/3skip，CPU demo4合成调用通过。新增媒体harness+probe的60项局部回归通过，包括真实CPU合成视频的正/反向解码与SHA漂移测试；这不是新增GPU视频或模型质量成绩。
