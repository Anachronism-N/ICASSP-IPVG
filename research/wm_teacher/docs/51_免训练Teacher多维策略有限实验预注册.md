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


### 7.7 新harness与真实Wan三窗口终态（2026-11-10）

- 原型提交68356dbb42a2338d9d105f66213cf407741a86cb已正常推送；完整媒体harness与一次性probe提交df3203896bca892f72ae99ce5449938ccd515f9a已正常推送。后者1660项完整CPU测试通过、3项环境skip，285个精确白名单文件独立远端克隆字节一致。私有模型、caption、媒体、日志与凭据均未发布。
- 原生Wan5B三窗口技术实验完成：1张A800，1个本地构造红盘首图生成首窗，再用其同一末帧fork左/右两分支。每窗1280×704、121帧、24fps、50步、shift5/CFG5；真实FA2累计18000调用。两个分支参考像素SHA相同，外部首图/父分支未被子分支写回。零优化更新、无Student或学习型AR训练。
- 独立CPU重新核验3个视频文件与363帧全解码、精确规格及源码字节；自有GPU进程已退出、模式恢复通过。后来的外部占卡不算泄漏，不停止未知进程。
- 实际allocated消耗0.2824495832314197 GPUh，生命周期1016.818499633111秒；按原独立上限记1，不退款。技术阶段累计6/8，标签/cache47/96，条件训练评测0/64；历史标签费用另保留。
- 每窗生成332.591186/331.962205/332.131554秒，峰allocated26.350689 GiB；生成5秒视频需约332秒。**不能称实时或低延迟**，调度层可收输入不等模型能实时返回。

固定红色像素诊断在全部原尺寸帧执行：R>100且R−G>40、R−B>40，至少64像素时计算质心。不是实体跟踪或物理/动作评价。三个窗口红像素121/121帧可检测：

|输出|首末质心x差（像素）|全窗x范围（像素）|
|---|---:|---:|
|首窗（请求向右）|−11.218641|34.044693|
|左分支|−83.615338|220.973379|
|右分支|+6.112096|127.744614|

首窗请求向右但红像素首末净位移向左，因此不能宣称指令准确；全窗轨迹/物体变化/镜头变化不能由净位移一项判定。左右分支用了不同seed，**不是同seed动作干预因果对照**，不把两者差异归因于动作。下一轮动作比较须同初态同seed并预注册完整规则。

子分支首帧相对父末帧RGB平均绝对差约0.060401/0.055698（0..255尺度）；虽参考输入SHA完全同源，实际VAE重建/压缩后输出并非像素严格相等，不能称无缝边界或长期身份一致。

真实GPU输出已接入world_model_harness：3视频及1个自定义红像素诊断维度，全部视频/score/protocol字节绑定和完整覆盖通过。此为**原型生成→输出解码→诊断评分→独立审计闭环**；不是VBench/Physics-IQ/WMB完整评分，quality/metric_correctness/SOTA仍false。

官方VBench1118pair、5590视频、9维、每pair5样本的新CPU合同也已对照原官方metadata与per-dimension集合核验；1775core＋3815camera。该合同没有启动官方全集生成/评分，外部对手精确checkpoint映射仍未证明。不能把三窗口或合同prepare当作全集完成。

**当前可用范围**：免训练窗口接续、外部证据记忆与fork、输入并发/过时输出淘汰、真实Wan后端、完整媒体/评分绑定harness已获得对应层次验证。**未完成范围**：学习型AR/Student、原生音视频全双工、真实动作可控性、长程物理/身份稳定、实时性和协议对齐SOTA质量；它们不由原型测试推导成立。

后续预算边界：本次1 GPUh属于既有技术阶段，不能直接拿剩余技术2 GPUh或标签额度进行大规模质量评测。原64 GPUh明确定义为门后Teacher训练评测且禁止阶段借用；转为免训练质量评测需要明确调整该阶段范围，历史费用/失败不重置。已有免训练开发集预算若仍有效也须按其原claim/失败/剩余额度核验，不自动复用旧分配。未经新范围绑定不派发正式全集。


### 7.8 正式质量验证的协议与资源准入复核（2026-11-10）

本轮只做公开源核查及CPU准入，不启动新GPU分配、不借用阶段额度，也不把预算待确认判为目标不可能实现。

**对照身份核查：** 已重新读取DreamX技术报告2606.16993v1第5节。其5秒基本评测使用WorldScore派生camera指标及Omni-WorldBench视觉指标，报告总体84.76、camera73.75；第5.2节30秒总体70.41。这些不是VBench-I2V官方总分90.49，也不能与本地VBench八维Total*直接相减。报告没有提供能把VBench90.49榜单行绑定到精确开源checkpoint SHA的证据。没有因为模型都叫DreamX-World-1.0就视为同一评分协议/权重。

当前公开仓库默认分支为master，revision仍a1f4c6e5e45600718e5236955f2e0702e53fc275；完整source tree中名称含eval/bench/report的文件仅configs/dreamx/eval.json。模型API重新确认：双向5B-Cam revision a4379c7723f6ebd02139e2e8fd62d6ef523e86e3，AR 5B revision67487c4a61466bb7166d30b7187dd465e0ac9f6c；两个model-index均未给出评分记录。远端LFS摘要不是本地权重已验证，未下载/运行这两个模型。不能用AR版本替代已冻结双向Teacher对照，原AR门禁保留。

**公开表刷新：** 2026-11-10重新读取VBench36行并复核原本地八维score文件SHA/聚合值。DreamX仍报原九维Total90.49；同八维诊断公式为90.7877216752，本地base89.8600750669，描述差−0.9276466083pp、dynamic差−16.0124390244pp。这是同公式复算，不是同生成协议SOTA差距；没有生成新模型输出或新增GPU评分，官方camera/每pair5样本的缺口不因此消失。

**完整集合不能塞进64 GPUh：** 一模型仅官方VBench原始提示完整集合就需5590视频（1775core＋3815camera）；另加Physics-IQ Original66场景×3视角198、WMB350、领域WM108，共6246视频。领域WM108不叫WorldModelBench；po分开、不含在上述raw计数内。

仅把既有三窗口中最长生成332.591186秒代入数量，得到下列**资源算术示例**，不作为真实数据或其他模型的吞吐保证，也不包含评分、加载、保存/清理和额外余量：

|范围|视频数|同实测速率线性代入GPUh|
|---|---:|---:|
|官方VBench单模型|5590|516.44|
|四集合单模型raw|6246|577.05|
|四集合两模型配对raw|12492|1154.09|
|94例×2seed×2模型开发验证|376|34.74|

先前提出但尚未批准的生成48＋评分16，按相同线性代入最多519视频（未加余量）。即使预算范围获准调整，也只能视为有限质量验证，**不能称完成5590官方全集**。增加节点可以并行执行，但不会消除总allocated GPUh。三窗合成吞吐不证明真实复杂视频与其他checkpoint具有相同成本，评分16也尚未获得本轮实测成本证明。

执行顺序仍为：预算范围明确后，先冻结不再据公开成绩调参的候选、独立开发/确认集合和同seed对照，实测生成/评分准入；后续申请能够覆盖完整集合及对照的独立分项预算。原标签/cache47/96、技术6/8、旧失败和checkpoint原样保留。开发集或某个完整子benchmark结果可报告对应范围，但不是整体SOTA验收。当前没有新的正式质量任务在运行。

机器可核验的私有凭证已留存：本轮论文评测章节源响应SHA、仓库默认分支/完整树、两个模型revision/API响应SHA、36行公开表与本地八维score摘要、6246完整范围和预算算术。仅无身份聚合结论进入本节，原媒体、caption、权重与运行凭据不公开。


### 7.9 正式任务与评分绑定修复（2026-11-10，前瞻CPU改动）

持续推进正式质量执行前置时发现两处需要修复的合同漏洞：

1. 原官方任务生成器只比较文件名/sample/prompt/image集合，没有逐项比较per-pair评分维度。新版本将维度精确绑定官方metadata中选定九维的交集，拒绝遗漏camera、增加错误维度、重复维度、布尔sample，以及同pair_key对应不同image/prompt或同输入对被拆成多个pair_key。任意sample0/1等数值相等不再允许bool冒充int。
2. 原world_model_harness凭证只对sampler/模型等protocol求SHA，没有把实际jobs（含seed/输出ID）与expected_scores一起绑定。新版本的protocol_sha256字段保存完整评测合同摘要：protocol＋jobs＋expected_scores；同一采样设置下改seed或维度，旧视频/评分凭证必然失效，不能静默复用。历史010合同/评分/失败文件不改写；新的摘要语义不会倒推旧实机结果成为新协议通过。

另解除CPU免训练任务对未使用监督cache的错误依赖：base NONE的CPU计划不需要监督cache，仍要求原模型资产、输入裁剪来源、backend/sampler/预算、远端源码准入；formal_training_allowed=false、execution_allowed=false。带训练checkpoint的候选保留其训练来源核验要求。该变更不授权GPU、不取消原训练/语义门，不借旧64阶段预算。

新增正反向回归后，builder＋媒体harness＋原型＋probe共83项局部CPU测试通过；真实5590任务合同的seed变动已验证会改变摘要，错误视频计数不得放行。原官方完整集合仍为1118输入对×5样本=5590，camera3815；新任务元数据验证不等新模型生成或官方评分完成。

输入图像和pair/维度的真实源复核采用新独立CPU探针，输出原始私有合同和审计凭证；未经过新的模型/backend/预算执行准入前，全部jobs保持不可派发。预算范围仍待之前提出的一次性调整问题确认，没有新增GPU费用。
