# 项目状态与下一步

状态日期：2026-10-20。仅记录有产物和校验支持的事实；计划、媒体可解码和模型质量分别报告。

## 已完成

- 已确认2节点、每节点8×A10040GB，双节点独立生成可用。历史异常GPU未参与新批次；不发布内部地址、登录或账号信息。
- VACE1.3B固定上游训练/推理适配：单卡14步LoRA训练、单节点7卡各2步DDP、两类checkpoint回载通过。验证非零有限梯度、实际优化器步数、参数更新及跨rank初始/最终参数hash一致；仅功能验收，不证明身份质量。
- CREMA-D官方固定版本辅助集：64名演员、192短片，参考来自同演员另一utterance/source。16train/32dev/16holdout演员隔离，只有64dev对用于本地未微调VACE基线。
- 双节点64/64条基线输出均完整解码；480²、49帧、16fps、20steps，每输入1候选。声明GPU数乘实际backend墙钟累计0.602152GPU小时；单条采样中位32.474秒，最大allocated17.405GB/reserved22.758GB。没有身份评价前不称其保持身份。
- CPU数据/预算/失败路径测试87项通过；新增身份/crop与畸形配对防护后，完整评价环境142项测试通过，独立重算配对结果一致。真实脸部及其逐样本日志不发布。
- buffalo_l离线CPU检测/ArcFace识别已校准：真实源视频16演员校准与16演员校验，各32正对/480负对。AUC=1、EER=0，校验经验FAR=.008333、TPR=1；阈值.183504。权重非商业研究许可与源数据许可分别核对。仅域内校准，不是官方评分或低FAR认证。
- 已补充MM2026官方名次、身份专用基线与ITA至时间线迁移设计，见 [相关方案](RELATED_WORK_AND_TRANSFER.md)，保留其公开来源和边界。

## 双赛道实测完成（2026-10-18）

- Track1新增身份专用AnyID preview over Wan2.2 TI2V-5B对照：8/8视频有效，7/8视频有可评分脸，身份帧coverage为87.5%（同8例VACE-H1为100%）。同7个可评分视频的原锚点ArcFace均值VACE-H1为0.472679、AnyID为0.831946；共同帧按4演员聚合提升+0.373556，但样本小、缺脸与成本问题尚未解决，因此只是可用候选，不升级为最终参赛base。总1.214463GPU小时，生成含CPU T5/迁移计时中位525.956秒。
- Track2独立视频时间窗guidance完成12/12条受控非人视频，三mode每步均3positive+1negative、参考prefix mask零、输入同噪音接线验收。flatten和window各2/8事件方向通过，swapped为0/8；无案例完成两个事件，窗口假设未获支持，保留负结果且不扩大搜索。总0.072258GPU小时，颜色proxy不能替代一般动作/主体身份或官方评分。
- 双赛道完整视觉评价环境203项CPU回归通过；无可选视觉依赖的干净控制环境84项通过、13项跳过，CPU烟测通过。新增Teacher研究分支的聚合分析与完整性报告已读取，未混入IPVG分数或合并为本轮实现。
- 完整方案、MM2026借鉴边界、原生采样器/负prompt差异、覆盖与成本、错误修复详见 [EXPERIMENT_TWO_TRACKS.md](EXPERIMENT_TWO_TRACKS.md)，聚合证据 `resources/two_track_results.json`，模型来源及hash见 `resources/two_track_models.lock.json`。最新完整CPU回归203项通过，AnyID评分分母、原锚点和Track2实际去噪调用均独立复核；已启动作业全部正常结束，两节点无遗留GPU任务。

## AnyID原生GPU-T5执行优化完成（2026-10-19）

新增CPU控制1条及GPU候选1+7条均正常结束，9条输出完整解码。只将原生T5编码设备改为GPU，offload、模型、prompt、seed和输出规格不变。新同输入文本编码504.539秒降至0.383秒，完整生成525.787秒降至31.118秒；8输入生成中位525.956秒（历史CPU批次）降至25.257秒（本轮GPU批次），约20.824倍，非独立重复统计保证。GPU8输出backend成本0.105299GPU小时，含CPU控制本阶段0.272482GPU小时，最大allocated14.599GB/reserved17.507GB。

固定原anchor身份协议下可评分视频均7/8、coverage均87.5%，缺脸失败仍保留；共同帧4演员身份均值变化+0.015652，不能归因于训练/身份方法提升。GPU-T5通过本配置的执行效率门槛，不代表最终参赛质量晋级。代码默认CPU路径保持兼容，已验证后续配置显式设 `t5_cpu:false`。完整协议与结论见 [EXPERIMENT_ANYID_T5.md](EXPERIMENT_ANYID_T5.md)、`resources/anyid_t5_result.json`；真实媒体与原始身份清单不发布。

## AnyID上下文效果实验完成（2026-10-19，不晋级）

执行效率仅是工程手段。本轮单因素参考margin0.3→0.6，12dev演员×2原prompt×2seed×2condition共96/96输出、10/10run正常完成；未消费holdout。固定原anchor协议下身份帧coverage72.3958%→76.5625%，共同31视频/12演员身份delta+0.005326，描述性CI跨零，不证明稳定增益。新增8dev演员的至少7/8有效帧视频22→21，未通过预先规定的不下降门槛，保留`wide_context_not_promoted`。灰度motion比0.924679、静态比例均0，不能证明动作或感知质量。

原缺脸诊断pilot由0/8恢复8/8有效身份帧、余弦0.864936，但不能以一个修复案例覆盖新增失败。总backend0.910532GPU小时≤4，无工程重试、无遗留GPU任务。64源dev视频CLIP匹配prompt辨别率仅53.125%，生成全帧文本余弦0.256859→0.255941；该代理不足以验收说话/表情语义。完整方案、各组结果和统计边界见 [参考上下文实验](EXPERIMENT_ANYID_CONTEXT.md)。最终完整视觉环境246项CPU测试通过，干净环境102项通过/15项可选依赖跳过、CPU烟测通过；公共配对入口独立重算一致，聚合证据`resources/anyid_context_result.json`。

## 原生双尺度效果实验完成（2026-10-20，不晋级）

新增48/48条同图双尺度参考候选均结束，原单crop48条为历史对照，固定原anchor/seed/prompt/模型。真实两reference、inner_t=[2,13]/frame_num57、输出49帧及每视频40次DiT调用/timestep3375核验通过；新backend0.465039GPU小时≤2.25，无工程重试或遗留GPU任务。身份framecoverage72.3958%→73.1771%仅+0.78125个百分点，未达预设+2个百分点；33配对视频/12actor的identity delta=-0.001619、CI95[-0.047266,0.046942]不支持非劣。

新增局部mouth_motion_v1先在32dev演员源视频与freeze/rigid控制独立校准：阈值0.455497，val源activation=1、两control均0，不是说话/表情TPR。生成mouth可评分34/48两方相同，但全量above-control口动视频25→21，合格率52.0833%→43.75%；paircoverage71.4844%→72.2656%未达到75%，共同视频口部flow比0.801056。保留`dualscale_not_promoted`，不统一扩参考数量，不将身份高分或全局灰度motion当说话效果。双参考也改变输出RoPE起始位置，不能将效果单独归因全图语义。完整环境277项CPU测试通过，干净环境108项通过/17项跳过、烟测通过，独立复算一致。具体方案和重算入口见 [双尺度效果实验](EXPERIMENT_ANYID_MULTISCALE.md)，聚合证据`resources/anyid_multiscale_result.json`。

## MM2026相关共同评价与选锚消融完成（2026-10-20）

VBench原AMT运动平滑/MUSIQ成像质量已离线共评48视频：24条Track1历史输出及24条物体source/重建；16条新稀疏重建均33帧320×1928fps、K5/28call、无损锚像素通过。缺CurricularFace/GME/Story/人工及官方完整scorer，组合总分保持null，不以此声称官方MM2026成绩。

Track1同8输入上VACE-H1、AnyID-single、AnyID-dual的ImagingQuality分别.699044/.566525/.599807，MotionSmoothness分别.993428/.992380/.993344；同7条可评分video的strict ArcFace分别.472679/.849399/.872528。身份与质量存在取舍，小样本不是最终base比较或对获奖队伍胜出证据。

KeyID-inspired仅改固定事件分段内内容自适应选锚，不含身份编辑/IPKE、PromptRelay、LTX。4生成物体草稿共同非锚MAE1.905378→1.869175（约降低1.90%），事件方向通过均3/8、两事件均完成均1/4，不能声称补全缺失动作或KeyID整体提升。MUSIQ.421485→.421974、AMT平滑.996665→.996381，保留小型重建组件信号，不扩大稀疏重建当任务主线。4自绘正例另报，不混入生成效果。首pilot因NumPy本机lscpu被断网子进程guard误拦失败，一次初始化兼容修复后完成，未放松模型下载保护。

具体方案/开源状态/作者权重更新/资源和所有限制见[实验报告](EXPERIMENT_MM2026_KEYFRAMES.md)。KeyID最新仍README-only，没有完整KeyID复现或其改进结果；后续优先独立引入Stand-In14B或许可明确TPIGE强基线、补齐GME/Cur评测，不再只做参考预处理小搜索。

## 两节点真实dense-caption训练与效果评价完成（2026-10-20）

32train视频/16演员已生成完整dense-caption，49帧480×352、全时长取样，参考独立source，dev/holdout身份隔离。原内部180词上限两次失败保留；显式dense_caption_v2验收100–200词完成104–200词，不截句或拼模板，内容尚未人工核真。VACE1.3B条件rank16LoRA单卡2步、两节点各2卡DDP pilot16步及main64步均通过；实际有限/非零梯度、300tensor更新、跨rank初末hash同步、跨节点同初始化及checkpoint16/32/48/64SHA均核验，main不续pilot。

新同规格baseline+两LR共48/48dev视频、8独立dev演员、原prompt/reference/seed42派生样本seed全部固定。两LoRA真实回载各150pairs且改变150底座tensor。ArcFace均值0.238796→0.420010/0.672639，coverage均100%；等演员identity delta+0.181213/+0.433843，CI95分别[0.133131,0.227019]/[0.385491,0.482921]。MUSIQ0.731272→0.738241/0.741597，原CLIP0.266221→0.267244/0.270089，但灰度motion比例降至0.755913/0.518992，口部全量above-control16/16→15/16两方。较高AMT平滑不能独立证明动态更好，口部flow也不是动作/自然性真值。

结论`identity_gain_with_motion_tradeoff_not_promoted`：真实身份小域收益，整体不晋级或继续堆epoch；不是完整KeyID/AnyID训练、广动作或SOTA。五训练controller0.758255GPUh、三生成0.330425；quality profile0.020070，含启动预检命令取整上界0.025833；caption最终用预留上界保守计阶段≤1.944791GPUh，不用通知延迟时间当算力成本。已启动GPU作业均结束，两节点显存0。完整CPU回归314passed，强化分析复算一致；八份便携配置、聚合资源/真实正负结论见[真实训练报告](EXPERIMENT_DUALNODE_TRAINING.md)、`resources/real_vace_training_result.json`，无真实媒体/caption/节点信息发布。

## 固定LoRA融合强度实验完成（2026-10-21，不晋级）

alpha1/.5/.25五run（两pilot+三16视频arm）50新视频、原AMT/MUSIQ共评48+32记录均完成。alpha1新16/16媒体SHA精确复现旧结果；各alpha实际150LoRA pairs/150base tensor融合核验，原评测anchor/prompt/seed未变。身份均值.672639/.396907/.271487，motion相对未训练baseline比.518992/.722036/.893224，口部活动15/15/16（各16视频），coverage均100%。.5仅保留36.44%原身份收益且动态门槛未过；.25虽保动态却仅保留7.54%身份收益。`fusion_strength_not_promoted`，不扩alpha搜索。实际新增.394149799GPUh≤4，完整349CPUtests通过，独立复算一致，所有GPUrun结束。完整预注册/正负结论见[真实训练报告](EXPERIMENT_DUALNODE_TRAINING.md)及`resources/lora_alpha_result.json`；不是结构优化或全量训练胜负证据。下一阶段转参考条件/时序残差结构分离，未完成不称成功。

## 参考差分时序路由结构实测完成（2026-10-21，不晋级）

新增结构不是LoRA缩放：同权重两原生VACE分支，有参考减null参考，只在视频body调整参考差分的时变成分，保留prefix/RoPE及null分支动态。g1控制40outer80nativecalls，16/16媒体SHA精确复现旧alpha1；g0真实改15层hint，prefixexact、BF16时间均值误差最大.001803如实记录。两pilot+两16视频arm共34新视频、64原AMT/MUSIQ共评及新ArcFace/CLIP/mouth均完成。身份.672639→.666769，保留98.65%原身份增益，motion对原结构仅+2.66%、相对未训练baseline仅.532801，mouth活动仍15/16；失败整体motion/口部保护。`reference_routing_not_promoted`，不继续常数g盲搜或称已训练门控。真实结构功能验收不等于效果优化成功。新增.348596049GPUh≤4，完整374CPUtests与独立复算通过，所有GPUrun结束；完整方案正负结果见[结构报告](EXPERIMENT_DUALNODE_TRAINING.md)、`resources/reference_structure_result.json`。后续需更针对性的训练目标/时空身份路径与更广动作数据，当前不宣称泛化或完整MM2026成绩。

## 评测优先阶段（2026-10-21，自动面板完成）

按用户纠正，顺序调整为先补评测→同集强基线→方法优化。固定评分GME7B/Cur单权重/KaLM0.5B已获得并SHA验证，独立env部署；新增GME原协议、ConsisID-style Cur/Arc16、本地caption+KaLMStoryproxy与原VBench扩展五维均完成真实控制。Cur64源校准AUC1/EER0、valFAR.008333/TPR1、阈值.20226049；GME首次懒加载CPU探测误拦失败，唯一修复r002完成。控制揭示GMEframe平均顺序盲、冻结可高分，本地Story和ViCLIP对动作顺序也不可靠，不能据其做自然动作晋级。原GeminiStory/人工/官方完整总分仍null，不能把local替代塞入作者组合分。

120/120已有视频共评已结束（96同规格六arm+24历史异规格单列），没有新增模型生成/训练/结构。Cur/Arc16支持身份增益，但GME baseline .554678→高LR .527720、结构 .531047，原RAFT明显运动比例baseline3/16→高LR/结构0/16；身份高分不证明动作或文本收益。13控制揭示GME顺序盲，本地Story/ViCLIP对冻结/倒序不可靠，`stable_task_action_evaluator_verified=false`。19评分GPU作业17完成2失败均已结束，GME懒导入/worker源目录裁剪问题修复且旧结果保留/排除计账，总.667025325GPUh≤12。完整525passed/7skip、干净187passed/37skip/烟测通过，独立数值复算一致。权重锁与完整脱敏结果见[评价协议](METRICS.md)、`resources/evaluation_panel.lock.json`/`evaluation_panel_result.json`。

本地80对neutral-name盲评包已准备，但没有真实人工分数；官方GeminiStory/人工/总分仍缺，不能声称官方全套评测或与冠军分差。用户随后明确“选择一个可以横向对比的方案即可”，因此采用`ipvg_cross_model_comparison_v1`：Cur/Arc16、GME、AMT、MUSIQ五项核心逐维比较，coverage/RAFT/美学等辅助，localStory/ViCLIP仅diagnostic、不参与核心选型，无合成总分。已用CPU重新核120媒体、prompt/ref/actor、规格、评分权重/精度，96主集与24历史集独立配对差值；原评分数值未改变。`comparable_evaluation_ready=true`只表示允许同协议强基线benchmark，不表示动作自然性或官方评分通过。人工未评分不再成为所有基线自动比较的唯一阻塞；下一步同集强基线，然后才模型方法优化。未获批合成方向探针不执行。新版配对协议CPU实际运行与19项针对性测试通过；完整541passed/7skip、干净203passed/37skip/烟测通过。配对结果见`resources/cross_model_comparison_result.json`，五核心逐维与配对视频/actor分母均报告，没有新GPU评分。

## 原生强身份基线共同对照完成（2026-10-22）

用户批准三个单例pilot通过后各16原输入完整对照：Stand-In+Wan2.1-T2V14B、AnyIDpreview5B、未训练VACE1.3B，49frame480×35216fps20steps/CFG5，同原prompt/ref/ID与seed42派生噪声种子。只比较现成原生系统，不训练或加结构；各自原生预处理、sampler/shift/negativeprompt和成本差异显式记录，同seed不是同噪声。先验40GB可运行/真实身份分支，再生成48新视频共评固定五core及aux；不依据单例或模型规模宣称质量优势，未完成不写成功。

Stand-In代码e351224、T2V14B权重a064a6c7六分片/adapterc6e90176单checkpoint/BiSeNet官方53MB，新增57.837GB≤60；复用已核T5/VAE/tokenizer与同SHA检测器，原生crop1.5/whitepad512/BiSeNet保留，只改检测CPUORT和离线权重注入，未运行FaceSwap。三pilot+三个16video mains共51新视频均正常结束，48共同main原五core/aux共评完成。Stand-In实际240IPtensor精确回载、IP1024token及40CFGcall参与，单40GB卡main峰值allocated31.612GB/reserved39.345GB通过，不保证更大规格。

同16输入Cur/Arc16均值：VACE.263468/.239781、AnyID.662902/.651611、Stand-In.496866/.488481；身份帧coverage分别100%/98.8281%/75%，Stand-In4/16视频全无有效脸，条件12视频身份均值只作诊断不能删失败刷分。GME分别.554678/.442795/.500700，MUSIQ.731272/.697678/.689096；高身份专用基线在本域仍有内容/成像/coverage取舍，不宣布统一胜者或最终base。six生成controller.805647627GPUh+八评分wall.187078033，总.992725660≤8，无失败/重试或遗留任务。完整558CPUtests/干净220+37skip/烟测通过，独立复算三组13维与五核心配对差一致。完整方案、资源/原生negative/shift差异、真实正负结论见[原生基线报告](EXPERIMENT_TWO_TRACKS.md)末节、`resources/native_baselines_result.json`与`standin.lock.json`。

下一步在更广独立动作/场景输入核验这些取舍，并检查Stand-In零脸案例；不是立刻给小域最高身份模型加训练。官方403/真实人工未完成单列，不阻塞固定自动benchmark，也不被作业全部结束伪造为已解决。

## 独立动作构图与可见性诊断冻结（2026-10-23）

在已完成三原生基线对照上继续验证域偏倚，不训练或修改生成结构。旧48主视频及8去重原参考只做CPU全帧检测：同det_10g权重/.5阈值、320/640两尺度、无脸才灰128边25%pad恢复，全49帧与原16floor采样比较，不重新计算或回填身份分。时间上限1800s，未完成须明确partial，双尺度无检出不当作人工确认无人脸。

随后从原dev32演员排除旧8及train/holdout，固定盐native-action-v1哈希取8额外dev演员，2prompt/演员、四固定动作模板各4例（portrait转头、halfbody举手、fullbody行走、fullbody坐下），仅复用独立来源参考，清除继承target视频信息。新16共同输入不是官方测试或动作GT，与旧16域单列，不做跨域配对delta。三contentpilot各600s通过后3main各16新生成/同规格原五core+aux共评，总51新输出，生成reserve3.833+评分2+唯一工程修复.5≤新增7GPUh，原51.5不增。prompt/身份/媒体/模型参数先冻结，不按pilot分数挑样或临时降规格。所有已启动任务跟进结束，分动作coverage/missing/内容/成像与实际预算公开，不凭参数大或整体身份均值选择winner。

### 全帧诊断完成与正式对照运行中（2026-10-23）

CPU双尺度诊断完整处理48/48旧主视频、2352个实际帧和8去重参考，输入/模型/媒体pre/post SHA均通过；5218次检测≤9440，实际359.339秒≤1785秒内层限额（外层1800秒），GPU成本0。运行中写出的`incomplete`仅为进度快照，最终`complete`，没有超时失败或加时重跑。320尺度全帧coverage：VACE100%、AnyID98.8520%、Stand-In75%；Stand-In原4条16采样全missing在320全49帧均未恢复，640仅1条恢复4/49帧，仍有3条全missing。Stand-In两尺度均有98/784多脸帧，最大脸评分不能证明匹配到参考身份；8参考在两尺度均单脸检出。诊断只说明固定检测器的敏感性，不是人工可见性真值，也不回填旧Cur/Arc分数。

新动作构图三pilot均通过49帧媒体、原生条件分支、配置/源码/输入/输出哈希验收，实际合计0.093374GPUh。正式三模型各16输入已启动；48主输出五core+aux评分和分动作分析尚未完成，不提前报告结果或选winner。完整CPU回归611passed/7skip、CPU烟测passed；独立CPU从逐帧记录复算50,166项检查0 mismatch，公开聚合映射/隐私扫描0差异或泄露。诊断完整JSON及新日志权限收紧至600，逐视频/逐帧/框与原参考不公开。此审计验证记录一致性，不代替视觉真值；1785秒为协作式检查，1800秒为外层进程组终止限额，不能宣称硬实时。

### 独立动作构图对照完成（2026-10-23）

三个main各16新视频及八共同评分job全部completed，3pilot+48main共51新输出，0失败/重试；generation.778234419+score.176778686=.955013104GPUh≤新增7硬上限。相同原reference/prompt/seed/规格、原生条件分支、输入/output/hash及评分权重/精度均验收，未训练/改生成结构。两节点真实实验卡结束后恢复占卡，worker故障GPU7不使用；占卡利用率不当作实验效果或混入controller预算。

全16输入Cur/Arc：VACE.076055/.076323、AnyID.471180/.471088、Stand-In.427261/.416936；facecoverage45.7031%/60.5469%/75.3906%，全missing5/6/2条。GME.548098/.321755/.445105，MUSIQ.530477/.657986/.669959；没有统一winner。AnyID全身坐下4/4视频全missing、全身行走2/4全missing；Stand-In两全身模板各1/4missing，coverage75%/73.4375%，但RAFT明显运动全身两组均0/4，不能将高coverage/平滑说成完成动作。portrait/halfbody/fullbody取舍显著，旧头部域最高身份不足以晋级广动作base。

分动作全部缺失/条件均值在评分结束后仅补分析脚本，原生成executioncommit仍e11be0e/sourceSHA不变，分析SHA另记。新增四视频全缺失/部分有效/全有效测试后完整CPU回归614passed/7skip、CPU烟测passed。从原八评分独立CPU复算736项group/动作均值、coverage/完全missing、条件均值、五core等视频/actor差和总预算0 mismatch；历史基线重新CPU审计原分数/配对差/预算完全不变。完整三组/四动作逐维及配对差/成本见`resources/action_baselines_result.json`和[实验报告](EXPERIMENT_TWO_TRACKS.md)末节；GME顺序盲、RAFT非动作完成率、没有真实人工/官方分数，结论保持开发系统取舍、不宣布最终base或SOTA。

## Stand-In分区注意力结构实验预注册（2026-10-23，尚未实测）

已完成评测和强基线后，仅改变独立身份IP与视频KV的注意力归一化：原生video query对video+IP联合softmax，候选分别对两分区softmax后固定等权平均，保持IP240tensor、1024token、KVcache、正负CFG共享、文本cross-attention、原预处理/权重/seed/20steps和输出规格。不是LoRA缩放、裁剪或旧VACE时间均值路由。统一softmax的竞争可能影响时空内容是待证伪假设；训练分布偏移/输出范数变化可能令候选变差，不能提前宣布优化。

复用同一已冻结8额外dev演员/16输入/四模板，不新增人或调prompt。原生控制与partition各两个单样本pilot（portrait、fullbody_sit按预定哈希选择），每pilot900秒/.25GPUh；控制须历史同输入媒体SHA精确一致、候选先验接线/有限变化/媒体/显存。通过后两arm各16完整新main，每main7200秒/2GPUh，不拼pilot或旧输出；native全部16须历史SHA一致，否则有限结案。32main原core5+aux两node各16、八评分job各900秒/.25GPUh；预留5生成+2评分+唯一工程修复.5=7.5≤新8硬上限，原51.5不增。主节点单40GB依序生成，其他健康闲卡继续占卡，评分两节点并行；用卡先暂停holder、结束恢复，worker故障7排除。

先验开发门槛：GME≥native+.02，Cur/Arc分别≥native−.02，facecoverage不降且全missing≤2/16，MUSIQ≥native−.02；各动作4视频缺失分母与条件均值单列，不删失败。AMT/RAFT/一致性只作诊断，不能证明完成动作/自然性。达标也仅进入下一轮验证，不宣称SOTA；未过门槛保留负结果，不追加强度或seed搜索。实现的统一结构证据契约为每视频1600次joint读、40次IPselfread、native1640/partition3240次kernel；partition_reads记录被分区的joint次数1600而非kernel次数。36项适配器定向CPU测试通过，完整671passed/7skip、干净279passed/42skip与两环境烟测passed。代码commit后冻结source/config/manifest再启动，当前尚无本轮GPU结果。

## 当前实验与限制

唯一H1对照是参考人脸bbox扩边.3的正方形裁剪，原参考仍作身份评价anchor。模型、prompt、sample_id/seed、采样步和候选数固定，不训练新LoRA。实验预注册及聚合结果见 [CREMA-D H1](EXPERIMENT_CREMA_D_H1.md)。crop64/64条完整解码；63样本/32演员共同帧配对，原基线身份均值0.143337、crop0.349797，配对提升+0.205661，演员bootstrap95%区间[0.168833,0.245416]。身份P10提升+0.192157、coverage+0.007813、灰度运动proxy比1.029227、静态比例无变化。新生成耗费0.602265GPU小时，所有任务正常退出。只在该单种子说话头部dev上支持H1，不是完整动作/人工/holdout/SOTA结论。

已认证的官方VIP视频与ReactID文件请求返回403并明确提示作者审核中；公开VIP注释首分片已校验，含10,001身份记录/78,786segments。公开注释不附带视频或参考图，不能单靠注释产生身份质量结果。记录实际访问状态，不重复被拒请求或代签协议；账号明文凭据不进Git。

当前不声称：官方成绩、SOTA、广动作泛化、论文录用、长期训练稳定性、optimizer/RNG断点恢复或人工身份盲评已完成。灰度运动proxy不能替代文本/动作符合度。

## 下一步

1. 不统一采用margin0.6或双尺度附加参考：两轮都未通过多维质量门槛。局部mouth控制已可区分freeze/刚体负例，但不能证明说话、微笑或自然性。后续优先固定独立68/106点几何诊断或本地人工盲评，核实真实嘴部变化与伪影；不继续盲搜裁剪/参考数、也不将身份子集均值和速度当最终效果。
2. Track2输出预测窗口混合的受控结果为负，优先研究更合适的时间cross-attention路由/事件条件；任何新方法另立有限方案和不混改的评价协议，不继续盲目加种子。
3. 官方视频可访问后再获取最小白名单分片，构建可追踪参考/身份/源视频匹配和广动作评价；访问拒绝未解除前不重复或绕过。
4. 人工盲查、独立种子及holdout验收按预定义阶段进行；holdout不参与当前开发选型。
5. 正式提交允许操作、输出细则和出版类别仍由研究者本人核准。

## 证据与复现

源码commit、固定模型revision、数据/split/hash、seed、每run超时与GPU小时、产物校验、代理指标协议与失败原因均在本地台账留存。公共仓库只发布代码、方案、许可说明及聚合结论，不含真实媒体、embedding、私有路径或原始身份清单。

会议/赛事时间及已确认与未确认事项见 [RULES_AND_DATA.md](RULES_AND_DATA.md)。参赛报告不自动等于CCF B Full/Regular论文；不承诺录用。
