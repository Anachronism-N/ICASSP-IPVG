# 双赛道有限实验：身份专用模型对照与视频时间窗探针

## 2026-10-18 已读资料及 base 定位

当前已完成的工程与H1基线是预训练 Wan2.1-VACE-1.3B + DiffSynth固定版本；H1仅参考裁剪，没有加载训练LoRA。VACE是通用条件模型，不是专用人脸身份模型，不宜将小模型裁剪收益当最终参赛强基线。

已阅读仓库 `RELATED_WORK_AND_TRANSFER.md`、`resources/mm2026_track1_evidence.json`，并重新核查MM2026官方榜与论文：Track1 AESR/MIPL和Chronos/USTC并列第一；Track2 Chronos第一、KeyID第二。AESR是商业API编排，缺软件许可；Chronos代码/权重未核实；KeyID代码仓库当前仅README。借鉴身份锚定、局部时间条件与草稿/关键帧分工，不声称完整复现其冠军或亚军pipeline。

KeyID论文明确使用Prompt Relay式时间路由、身份关键帧编辑与LTX插值，涉及多种模型和额外计算；不能与单次生成不计成本比较。IPVG-STD属于MM2025，不能混成MM2026的名次。AnyID虽非这里已证实的MM2026获奖方案，是可公开执行的身份专用现代5B对照。

## Track1：AnyID / Wan2.2 TI2V-5B

固定AnyID代码commit `6c714f71af6e08d2af29aecd74a3a9fd9e8f94a3`，HF preview LoRA revision `d516b069c7ba6cfd487cfa432a4d44509b020fa2`；CC BY-NC-SA4.0上游许可分别注明。复用并校验官方原始Wan2.2 TI2V-5B与官方AnyID预训练preview LoRA，不复用本机其他项目的微调checkpoint，独立目录CPU合并LoRA后推理。上游没有公开可直接复现的SFT/RL训练入口，不编造trainer能力。

固定CREMA-D开发集最初8输入、4演员、单张已冻结H1裁剪；评价anchor仍原参考。和原VACE-H1对照保持sample_id/seed、480²、49输出帧、16fps、20steps、每输入1候选，不改prompt。AnyID入口参考prefix和输出latent分开（13输出latent+1参考），不是简单传49总帧后错误得到45帧；具体由实际输出shape验收。

较大底座/模型结构与训练方式不同，结果标为模型对照，不归因为单因素方法增益。首1pilot后7batch，最多4GPU小时（1+2+唯一修复1），不进行无限参数搜索；CPU模型合并与源权重SHA独立计账。T5 CPU/offload及A100注意力兼容采用独立环境，不改旧环境或基础权重。

## Track2：视频时间窗guidance受控原型

这是独立实现的推理时窗口guidance，不是ReactID训练复现，也不是将时间线拼到prompt就声称时间控制。固定global+两event条件，同去噪step、同latent计算global/event1/event2三个positive预测，用视频物理时间mask混合：

`prediction = global + mask1 * (event1 - global) + mask2 * (event2 - global)`

negative一次、CFG由原pipeline完成；参考prefix的mask为零持续走global。视频latent首位对应frame0，其后四帧一组，按frame_index/fps求半开事件占窗比例；视频时间不等于diffusion timestep。输出只由上游最终裁参考prefix一次。

等预算flatten对照也真实执行3个相同positive与1个negative调用，保持原预测；window和swapped同样每步4个去噪调用。关闭CFG合并/TeaCache/滑窗，统计实际调用数及墙钟成本。swapped仅交换video-time masks，不改变event语义，核对是否真使用时间信息。

自绘红球左右、蓝方块上下两主体，2seeds，3modes，共12视频；320×192、33帧、8fps、20steps，每video80个去噪调用。颜色连通域质心检测测逐窗位移方向：检出coverage至少.8、标准化方向位移至少.02才计event通过，未检出/无运动记失败而非假零身份分。

该受控探针不代表真实人类/动物复杂动作泛化，不用ArcFace测非人主体。最多4GPU小时（6有限run共3、唯一工程修复1），负结果同样记录，不因无改善而篡改指标或追加无限种子。

## 并行、复现与发布

两节点独立单GPU，各负责一轨，排除有历史故障的GPU；不用跨节点DDP、不抢占其他任务。所有作业有资源声明、超时、新run_id、源码/模型/输入hash、实际成本和输出完整解码。CPU模型导入/合并问题另记录，不能当生成质量结果。

公共仓库只发布薄adapter、CPU测试、来源版本锁、可复用配置及聚合结论；不复制完整上游、不上传媒体、embedding、真实身份清单、账号、私有节点地址或凭据。官方视频库403属于外部阻塞，不重复或绕过；2026公开旧测试将来可作历史回归，不能冒充2027独立holdout。

## 实测结果

### Track1：完成8条身份专用基线，但不自动晋级

两个有限run共8/8条输出均完整解码，480²、49帧、16fps；参考图像、prompt和实际seed与已有VACE-H1基线逐样本一致，未使用目标视频或训练/holdout数据。AnyID加载的是经SHA核验的原始Wan2.2 TI2V-5B及官方preview LoRA的独立合并DiT。

| 指标 | VACE-H1 | AnyID preview |
|---|---:|---:|
| 完整媒体输出 | 8/8 | 8/8 |
| 至少一个有效身份帧的视频数 | 8/8 | 7/8 |
| 全8条身份可评分帧coverage | 1.000 | 0.875 |
| 同7条可评分视频的身份余弦均值 | 0.472679 | 0.831946 |

共7条视频、4演员满足至少7/8共同有效帧，按演员配对提升+0.373556，描述性bootstrap95%区间[0.177438,0.506938]，配对P10提升+0.350826。AnyID另1条完全未检出可评分脸，不能把该视频删除后声称全量身份质量提升；缺失相似度未填0，完整媒体覆盖仍计8/8。全8条灰度运动proxy比1.295167、静态比例变化0，均不能替代动作/文本或人工质量评估。

AnyID运行总1.214463GPU小时，上限4，未使用GPU修复槽；单条生成计时中位525.956秒，包含CPU T5编码、模型迁移、去噪和解码，不是纯DiT时间。对应VACE-H1样本采样中位32.443秒，两者不是等实际GPU墙钟成本。AnyID使用原生UniPC/shift3与默认负prompt，VACE使用自身调度及空负prompt；模型容量、VAE和采样器不同，效果不归因于单一方法。

结论：AnyID是可运行且值得保留的身份专用候选，在可评分视频上身份相似度明显更高；但身份coverage下降12.5个百分点、只有4演员且成本增加，**本轮不将其直接升级为已验收参赛最终底座**。下一阶段需在更广动作、多身份与完整覆盖下评价，不因高均值忽略缺脸失败。

元数据更正：首pilot配置从VACE模板继承了旧`model_revisions`字段，实际加载的AnyID源码与模型路径/SHA已经独立核验；batch配置记录正确AnyID版本。原pilot证据不覆盖，公开锁文件及聚合结果明确此更正。

### Track2：完成12条，时间窗口假设未获支持

所有6个有限run、12/12条输出均完整解码，320×192、33帧、8fps；每条确认60次positive+20次negative去噪调用，同latent/timestep和参考prefix零mask。两主体×两种子，每种模式4条视频、8个事件。

| 模式 | 事件方向通过 | 两事件都通过的视频 | 颜色检出coverage |
|---|---:|---:|---:|
| flatten（重复positive等调用） | 2/8 | 0/4 | 1.000000 |
| window | 2/8 | 0/4 | 0.954545 |
| swapped（按交换后方向计分） | 0/8 | 0/4 | 0.954545 |

总0.072258GPU小时，上限4，未使用GPU修复槽。window并未优于flatten，没有任何案例成功完成两事件；不能把上下文/mask接通表述为任务已解决。交换窗口后事件得分变化仅作诊断，不证明稳定时间控制。**保留`window_not_supported`负结果，不继续盲目追加种子或训练。**

颜色质心proxy只看预设颜色连通域和窗内首末位移，物体形变、遮挡或连通域切换可干扰；即使通过也不能作为一般动作、主体身份或官方Track2分数。下一步应研究更合适的cross-attention时间路由或真正事件条件训练，并在允许的真实数据上单独验证。

### 复现与错误修复

- `scripts/merge_anyid_cpu.py` 对固定上游在T5函数签名中调用CUDA设备的import副作用做临时CPU上下文处理；模型合并不使用GPU，不修改原权重或上游checkout。
- `scripts/adapters/anyid.py` 支持上述单参考推理，源模型/LoRA的版本与SHA在 `resources/two_track_models.lock.json`。
- `scripts/prepare_window_probe.py --output-dir NEW_DIR` 生成固定4个受控输入；同清单分别用三mode，`scripts/evaluate_window_probe.py` 计算明确标注的方向proxy。
- 时间窗adapter的重复positive返回原预测，避免重复BF16平均带来数值变化。初版CPU恢复断言错误地比较bound method对象身份，已改为方法相等判断并复验，未改变模型算法。
- 完整视觉/模型评价环境CPU回归203项通过，包括51项时间窗/prefix/CFG/方法恢复测试；干净标准库控制环境84项通过、13项因可选依赖缺失跳过，CPU烟测通过。真实模型试跑证据不能由CPU mock替代。
- 已读取新增远端 `wm-teacher-experiments-20261018` 分支的聚合分析与完整性报告：它是另一条Wan5B Teacher研究线，108例GT参考诊断、66组三视图物理诊断和3913条媒体完整性检查，不是当前IPVG身份或动作分数。其文件明确说明单种子、相关样本和多重比较局限，不代表Teacher最终验收；本轮未合并该分支或用其结论证明IPVG收益。

公共聚合结果见 `resources/two_track_results.json`；原始人脸媒体、逐演员评分、临时凭据和内部路径不上传。两个赛道所有已启动作业已结束，无遗留GPU任务。

## 2026-10-22：原生强身份基线同规格对照完成

### 冻结方案与输入

先定版横向可比评测（08f4e1e/CI39641297184）后再验证基线。本轮采用未训练VACE1.3B、AnyIDpreview/Wan2.2-TI2V5B、Stand-In/Wan2.1-T2V14B三个现成原生系统，不新增训练、结构或超参搜索。固定CREMA-D辅助域16dev输入/8演员×2原prompt，train身份/source隔离、holdout未用；参考源独立于目标，身份评测始终原anchor。统一480×352、49帧、16fps、20steps、CFG5、每输入1候选；实际seed=(42+原sample_id哈希前8位)%2^31。不同原生sampler/shift/negativeprompt/参考预处理是完整系统差异，不声称只换参数规模、相同随机噪声或完全同预算算法。

每模型1单例pilot1800s/.5GPUh，各通过后每模型完整16新视频（重新生成pilot输入，不拼接历史输出），Stand-In7200s/2GPUh、AnyID/VACE2400s/.667各；共51新视频，方法比较仅48条完整main。core五指标Cur/Arc16、GME7B、AMT平滑、MUSIQ成像，辅助RAFT动态、DINO主体、CLIP背景、LAION美学/ViCLIP内容诊断。两节点各24视频，四类评分job各900s/.25GPUh，总8评分job。生成预留4.833GPUh+评价2+唯一工程修复.5≤8硬上限，历史51.5不增；未用修复槽，全部14作业正常结束。

### 固定模型/预处理与真实接线

Stand-In代码e351224366be169076e94af1454115d91d458313；WanT2V14B原六FP32分片a064a6c71f5be440641209c07bf2a5ce7a2ff5e4，不下载VACE14B控制模块。Stand-In原checkpointc6e9017606276ee1a01c63fdb8f90da789678647，SHA52f47630b4fefbdac841d3fd1ce146cca2a6a94ec0a6ab9407d16f2f635177e9；官方facexlibBiSeNet53,289,463B，下载一次后计算SHA468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567，不冒称作者发布摘要。新增权重57,836,726,941B≤60GB，复用同SHA原T5/VAE/umt5tokenizer，不重复下载完整antelope身份/属性包。

保留原生最大脸bbox、1.5裁剪、白底pad512、BiSeNet去背景，以同SHA原SCRFD检测器CPUORT4线程替代仅预处理所需FaceAnalysis检测部分；设备差异记录，不改预处理几何、不执行FaceSwap、不发送外部API。独立torch2.7.1/transformers4.46.2/facexlib0.3/ORT1.23.2环境，不动其他环境；A100不能用本机HopperFA3，导入时只隔离该可选模块，走原FA2fallback，原上游文件不改。

`scripts/adapters/standin.py` CPU weights_only实际核240完整IP q/k/v down/up tensor键/形状/有限值，原native set_stand_in后逐tensor精确匹配，非只看fileexists。每视频原生40CFG去噪call，仅首次有[1,16,1,64,64]IP latent、首次q_lora[1,1024,5120]；缓存按native重置，16例均核验。原nativeVRAMmanagement实际通过单A10040GB，main峰值allocated31.612GB/reserved39.345GB；后者接近40GB，不给更高分辨率/帧数显存保证。AnyID原单reference前缀inner_t=[1,13]/native_frame_num53/40call，输出49帧；VACE40条件forward。三模型生成源码commitdd60c820d7c8d5f76cec6d4527092b5397194687，sourceSHAe03405eeea8d54c5dad5e14f29feb43dc13d4cecdd63ec8087385a7fd169fcab。

### 五核心与覆盖率结果

| 指标 | VACE1.3B | AnyID5B | Stand-In14B |
|---|---:|---:|---:|
| Cur身份作者式全16视频均值 | .263468 | .662902 | .496866 |
| Arc身份作者式全16视频均值 | .239781 | .651611 | .488481 |
| 身份可评分帧coverage | 100% | 98.8281% | 75% |
| 有至少1可评分脸视频 | 16/16 | 16/16 | 12/16 |
| GME帧文本匹配 | .554678 | .442795 | .500700 |
| AMT运动平滑 | .992506 | .990752 | .995131 |
| MUSIQ成像质量 | .731272 | .697678 | .689096 |
| RAFT达到明显运动阈值的视频 | 3/16 | 2/16 | 4/16 |
| LAION美学 | .500643 | .522495 | .520139 |

Cur/Arc采用已固定largestface/pad/16frames作者式协议：单帧missing不补0、完全无有效脸视频才按作者规则0。因此Stand-In4/16全missing令视频均值显著受损；条件在12可评分视频上的Cur/Arc约.662488/.651307，与AnyID全16的.662902/.651611接近，**但这是不同分母，不能删4失败案例说打平或胜出**。本轮未有人工作为脸消失/检测误差真值，不将75%coverage损失简单归因纯检测器或生成方法，应独立诊断。旧strict8frame值不替换或混用。

相对VACE，同16输入等视频/等演员配对AnyID Cur+.399434/Arc+.411830，但GME−.111883、MUSIQ−.033594；Stand-In Cur+.233399/Arc+.248699，GME−.053978、MUSIQ−.042176。AMT接近高分天花板、RAFT对真实说话头部细微运动也可0，动态比例不是动作完成率/自然性或时间顺序验证。所有方法13列数值/coverage完整可读不代表身份帧全部有效，覆盖率必须和均值一起看。

### 结论、资源与复现

保留`report_native_system_tradeoffs_no_automatic_winner`：实际强基线已部署并比较，但不存在五核心+覆盖率均占优的统一胜者，更大14B不必然胜5B/1.3B。AnyID是身份诊断候选；Stand-In提供现成独立身份IP分支对照，但4全missing和高reserved显存未解决；VACE继续作文本/成像工程控制，不将其未训练人脸低分或高GME解释为广域最佳。不直接采用本轮任一模型作为最终base，不立即进行新的微调或结构搜索。

下一步应先对Stand-In零脸案例核清主体可见性和检测可用性，并在更广独立动作/场景/姿态输入上检验本轮取舍，再决定强基线；本轮小集单seed不宣称已近Top/SOTA。若做生成参数或模型修改，另冻结单因素对照，不在当前结果后调指标/删案例。官方视频403/人工未完成仍如实单列，不阻塞自动比较也不伪结案。

six生成controllerGPUh=.805647627，八评分完整wall=.187078033，总.992725660≤8；其中Stand-In16main实际1725.743秒/.479373GPUh，工具晚发4小时通知不是backend耗时，不计为4小时训练或占卡。main每条生成中位VACE22.911秒、AnyID21.239秒、Stand-In100.885秒，模型/预处理/加载/评分分开记录，速度不是质量。所有生成/评分作业已正常结束，两节点无遗留GPU任务。

完整与缺失分母由`scripts/analyze_native_baselines.py`复核：同input/ref/prompt/seed、规格/51video解码/hash、模型IP接线、score/manifest/protocol/weights/precision、预算与16配对视频/8actor，独立CPU复算三组×13维均值和五核心差值一致。17项基线针对性测试、完整558passed/7skip、干净220passed/37skip/烟测passed。`resources/native_baselines_result.json`为脱敏聚合，`resources/standin.lock.json`锁模型/源码与预处理权重；六份`configs/experiments/native_{vace,anyid,standin}_{pilot,main}.json`需要实际本地权重/授权manifest再运行，模板SHA不冒充原执行SHA。媒体、逐身份清单/embedding/checkpoint/私有环境路径与凭据不入Git。

## 2026-10-23：独立动作构图共同对照完成

### 预注册与执行

旧零脸全帧诊断已完整结束，见`resources/face_visibility_diagnostic_result.json`及[评价协议](METRICS.md)。320尺度原4条Stand-In全missing未因49全帧采样恢复；640仅1条恢复4帧，另外3条仍全missing。诊断不更新旧身份分数，不把检出/未检出当作人工可见性真值。

新cohort从原32dev演员排除旧8演员与所有train/holdout，用固定盐`native-action-v1`SHA排序取额外8演员，每人2输入，四固定模板各4：portrait转头后返回、halfbody抬右手后放下、fullbody横向行走、fullbody坐到椅子。仅使用原独立来源reference，移除继承targetvideo/media字段；不消费holdout或动作目标视频，不按生成结果挑人/改prompt。参考来自说话头部域，本轮是开发集prompt-framing泛化探查，不是官方动作GT。与旧说话头部cohort单列，不计算跨cohort配对因果delta。

模型/预处理/negative/shift与上一轮原生系统完全保持，统一480×352、49帧16fps20steps/CFG5，每input1候选，seed42+sampleIDhash派生。三新contentpilot各600秒通过原生条件/媒体/hash后，三个main各16重新生成（不拼pilot输出），共51新视频。生成executioncommit为`e11be0e`、sourceSHA`b3e5b245ec85433066a56a629fb928ea0acc72543a16237b0820490c74ff340a`；Stand-In240IPtensor/1024token/40call、AnyID单reference/inner_t[1,13]/native53frame/40call、VACE40条件forward全部通过。

48main同五core+aux，两个node各24视频、各四独立900秒评分job，权重/采样/精度与冻结协议一致。六生成及八评分均completed，0失败/重试，实际generation.778234419+score.176778686=.955013104GPUh≤7；原51.5GPUh硬上限不增加。CPU检测.0GPUh单列；占卡程序的利用率与时间不是本轮生成质量或GPUh结果，实验卡启动前暂停占卡，结束后恢复，故障workerGPU7继续排除。

### 全16输入结果（完全missing视频的作者式0保留）

| 指标 | VACE1.3B | AnyID5B | Stand-In14B |
|---|---:|---:|---:|
| Cur16 | .076055 | .471180 | .427261 |
| Arc16 | .076323 | .471088 | .416936 |
| 身份可评分帧coverage | 45.7031% | 60.5469% | 75.3906% |
| 完全无可评分脸视频 | 5/16 | 6/16 | 2/16 |
| GME帧文本匹配 | .548098 | .321755 | .445105 |
| AMT平滑 | .995012 | .989838 | .996092 |
| MUSIQ成像 | .530477 | .657986 | .669959 |
| RAFT明显运动阈值视频 | 4/16 | 2/16 | 1/16 |

不同可评分视频分母的条件Cur均值为VACE11条.110625、AnyID10条.753888、Stand-In14条.488298，只作诊断，不能用条件均值删除失败或选winner。作者式视频均值coverage=100%不等于身份帧都有效。

### 四动作分层（每格4视频，facecoverage / 全missing数）

| 模板 | VACE | AnyID | Stand-In |
|---|---:|---:|---:|
| portrait_turn | 100% / 0 | 92.1875% / 0 | 100% / 0 |
| halfbody_raise_hand | 76.5625% / 0 | 100% / 0 | 53.125% / 0 |
| fullbody_walk | 1.5625% / 3 | 50% / 2 | 75% / 1 |
| fullbody_sit | 4.6875% / 2 | 0% / 4 | 73.4375% / 1 |

AnyID半身举手Cur=.791215、GME=.310956；全身坐下四条身份全missing且GME=.342632，不能把该失败移出平均。Stand-Inportrait Cur=.705480、GME=.457156；全身坐下Cur=.143314，虽facecoverage较高且MUSIQ=.674634，RAFT阈值视频0/4。VACE全身行走MUSIQ=.285815、facecoverage1.5625%，高GME/平滑不意味着合格动作或身份。每动作只有4演员输入，不能据分层点估计保证普遍泛化。

### 结论、审计与复现

仍保留`report_native_system_tradeoffs_no_automatic_winner`。AnyID不能凭旧头部域身份最高晋级为广动作base；Stand-In是覆盖/成像方向的独立诊断候选，但高平滑/主体一致性可能伴随少运动，不能推断行走/坐下完成。VACE未训练身份与全身成像不足，继续作为工程控制，而非最终竞争base。GME顺序盲、RAFT非动作完成率，自动协议仍允许比较，不能声称自然动作已验证或官方SOTA。

`scripts/prepare_native_action_benchmark.py`固定cohort，可用已有六份native配置作为原生设置模板；新生成前冻结manifest与config SHA。评分完成后仅在`scripts/analyze_native_baselines.py`补分动作完全missing/条件均值，记录`analysis_script_sha256`，与实际生成sourceSHA分开；旧评分/输出/预算不改。完整原始记录仅本地，公开`resources/action_baselines_result.json`包含全组/四动作逐维、缺失分母、五核心等视频与等actor配对差、实际六生成/八评分hash和成本，未附媒体或逐身份信息。原八评分独立CPU重算736项全组/分动作均值、coverage/完全missing、条件均值、核心配对等视频/actor差及总预算，0 mismatch；历史原生基线重新CPU审计均值/配对差/预算完全不变。614passed/7skip、CPU烟测passed。后续应优先针对明确的全身构图/身份/运动取舍设计一个有限单因素方法，而不是继续裁剪/融合强度搜索或直接堆小域训练；官方403/真实人工仍为外部缺项，不伪造为已完成。

## 下一轮预注册：Stand-In固定等权分区注意力（2026-10-23）

### 唯一结构因素与假设

原生SelfAttention对video Q和拼接video/IP KV使用联合softmax。逐query/head原读出等于`Zv/(Zv+Zi)*Av + Zi/(Zv+Zi)*Ai`，其中Zv/Zi为两分区注意力指数和。候选改为`0.5*(Av+Ai)`，即分别softmax后的固定等权平均；既不训练LoRA、不搜条件缩放，也不改变视频查询、IP1024token/240tensors、RoPE、文本cross-attention、IPselfread、KVcache与正负CFG身份共享。平均避免简单相加，但不保证范数或运动保持；替换训练期门控分布可能有负效果。待证伪的是分区竞争是否造成内容/身份取舍，不是声称已证明IP阻碍动作。

主仓库适配器仅封装40block的AttentionModule读出。native直接原forward，不多执行attention；partition仅video查询读jointKV时两次原attention kernel，IPselfread与无IP路径不变。单视频native1600主读kernel+40首次IPkernel，candidate3200+40；调用数不等于FLOPs严格翻倍，真实时间/峰值另测。不修改上游文件或kernel选择；旧运行FA2原路复用。

### 固定输入、控制等价与有限预算

使用上一轮同16开发输入（8演员/四模板各4），按sampleID哈希预先各取一portrait和fullbody_sit为两个singlepilot，禁止按分选择prompt/人。每arm两个pilot各900秒/.25GPUh；native须历史同输入输出SHA精确复现，candidate仅验结构确实参与/有限输出/媒体及单40GB峰值，通过后两arm各16全新main（7200秒/2GPUh各），不拼pilot输出。native16全部精确复现旧同种子对照，否则不以容差替代、不扩候选。固定49帧480×35216fps20steps/CFG5、原参考、negative/shift/模型与评测anchor。

32main core5+aux沿用原scorer公式/权重/precision；two shards每片16、四类各900秒/.25GPUh共八job。预留四pilot1+两main4+评价2+一次明确工程修复.5=7.5≤新增8硬上限，历史总体51.5不增加。每个job实际GPU数×backend墙钟计账，失败/排除计入；科学失败不改权重/更多seed，工程修复必须新runID且至多一次。主单40GB依序生成以避免双加载新资源变量，worker与launcher并行共评；闲置健康卡holder保持，真实作业前让出对应卡，结束恢复，不用故障worker7。

### 预注册判据与完成条件

候选相对本轮native需GME至少+.02、Cur/Arc各不降超.02、facecoverage不降且完全missing≤2/16、MUSIQ不降超.02。原16完整分母与各动作4分母全部保留，条件均值不能单独选优。AMT/RAFT/一致性仅诊断，不据其称自然动作完成。达标仅下一轮验证候选、不是统一winner或SOTA；未达标公开负结果且有限结案。

CPU tiny/fake模型验证native逐位等价、两softmax解析读出、IPselfread/cache/negative不变、40closure独立、异常恢复与计数。GPU验原240tensors/40CFGcall/1024tokens/seed/source/config/input/output/hash、控制SHA与候选真实接线。版本化冻结/共评/审计入口复用现有四scorer与配对/coverage helpers，分析源与生成源分别记录。所有启动job跟进结束、独立CPU复算与完整回归烟测、脱敏结果/中文正负结论正常push并验远端CI；原媒体/参考/逐身份/attention maps/权重不公开。以上为生成前冻结的预注册，实际结果如下。

### 全部实测结束：固定等权分区不晋级

四单例pilot、两arm各16完整main共36新视频及八共同评分job全部完成，所有媒体满足49帧480×35216fps。native16/16编码视频SHA精确复现历史同输入，候选输出确有变化；两arm相同参考像素/seed/240IPtensor/1024token/40CFGcall与FA2。candidate每视频1600joint读变为3200主kernel，首次IP仍40kernel；原生1640/candidate3240总kernel证据通过。没有训练、更多候选、删样或GPU重试。

| 指标（各16视频完整分母） | 原生joint softmax | 固定等权partition |
|---|---:|---:|
| Cur16作者式 | .427261 | 0 |
| Arc16作者式 | .416936 | 0 |
| 身份检测/评分帧coverage | 75.3906% | 0% |
| 完全无可评分脸视频 | 2/16 | 16/16 |
| GME帧文本匹配 | .445105 | .409861 |
| AMT平滑 | .996092 | .994201 |
| MUSIQ成像 | .669959 | .241604 |
| RAFT明显运动阈值视频 | 1/16 | 0/16 |
| LAION美学 | .532151 | .233617 |

原生输出媒体16/16逐字节一致，身份与12/13指标均值精确复现；AMT重新评分均值由旧.9960915039650811变为.9960915037837526（−1.81e−10），如实保留新计算值而不覆盖旧分数，不宣称所有评分逐位一致。

等视频/等actor配对GME−.035244、Cur−.427261、Arc−.416936、MUSIQ−.428355；全16视频与8actor均参与，作者式完全missing视频0保留。所有八项预注册检查均false，不晋级`not_promoted`。候选四动作每类4/4全missing，条件Cur/Arc均值null；native对应portrait/halfbody/walk/sit全missing分别0/0/1/1。分层结果不支持把任何高分子集救回整体；未有人工作为可见性/自然性真值，不能从检测缺失说画面一定没有人物。

| 模板（候选各4视频） | GME | MUSIQ | 全missing |
|---|---:|---:|---:|
| portrait_turn | .430563 | .215434 | 4/4 |
| halfbody_raise_hand | .385918 | .272062 | 4/4 |
| fullbody_walk | .420234 | .233553 | 4/4 |
| fullbody_sit | .402728 | .245368 | 4/4 |

GME仍在.39–.43、ViCLIP overall .107426→.119582，而身份/成像失效；这进一步说明内容相似诊断不能独立判动作或综合质量。平滑/一致性高值不构成自然动作验证，RAFT0不能证明画面绝对静止。此次固定等权平均改变原联合softmax逐query/head的数据依赖门控，结果与预期收益相反；训练分布偏移可能解释退化，但没有测量原native分区质量或训练对照，不能称已证明唯一原因。否定的仅是这项零训练固定均分替换，不外推所有可训练分区/身份路径均无效。

实际generation1.082196291+score.139000479=1.221196770GPUh≤8；其中四pilot.192459212，native16main.438929492/candidate16main.450807588。主单40GBmain中位采样92.284/94.386秒、peakallocated31.611/31.612GB、peakreserved39.345/38.680GB，未OOM；效率不是质量晋级证据。两main实际executioncommit136efa2、四pilot e35e6c0，仅测试可移植性提交使HEAD变化，src/scripts指纹相同`a01f5117cbb9604b58aba4190c4ced4f18ed1b5db788ff8b82911621b6ec017b`。

初次prepare后发生最后审计源码校验编辑，preflight在GPU启动前拒绝旧指纹，保留未启动准备目录并新fresh目录重新冻结；没有GPU失败或重复生成。首两CI因测试导入仅本机存在的ignored日志路径失败，改为四scorer完整argv契约断言，不跳过检查、不改变实际scorer；136efa2/CI39947564485两Python版本成功。完整683passed/7skip、干净291passed/42skip与烟测passed。

从八rawscores独立CPU重算491项全组/分动作均值、count/coverage、全missing/条件均值、五core等视频/actor差和总预算，0 mismatch；136个私密ID/path/mediahash token扫描0泄露。`resources/standin_partition_result.json`保留全部13列、分动作缺失、六生成/八评分source/hash与真实成本，旧基线/评价权重/holdout不变。所有真实任务已结束，15健康GPU恢复占卡，故障worker7排除。科学负结果有限结案，不在当前权重上追加强度/seed搜索；继续研究需新预注册，而不是把功能通过当效果成功。
