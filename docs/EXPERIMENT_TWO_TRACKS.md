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
