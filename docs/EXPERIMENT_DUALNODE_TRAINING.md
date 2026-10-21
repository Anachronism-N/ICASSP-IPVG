# 真实dense-caption LoRA训练与同规格效果对照

## 2026-10-20已批准方案及显式修订

CREMA-D32train视频/16身份，独立参考source、dev/holdout身份隔离，仅说话头部辅助域，不是VIP/ReactID或足够SOTA训练规模。只训练VACE1.3B条件模块rank16LoRA，不声称AnyID/KeyID公开训练入口可用。完整MM2026官方总分/Cur/GME/Story/人工缺项保持null。

### 数据准备

原480×360、29.97fps、51–86帧视频中心裁8px为480×352，49帧按全时长均匀抽样，准备fps=49/原duration。上游训练读取前49帧且不使用fps，故预先重采样覆盖整个源；源1.7–2.9秒不冒称原49帧16fps3秒。视频实际解码像素8时点由本地公开Qwen2.5VL7B描述，原请求100–180词、仅可见场景/外观/beginning-middle-end动作，不猜姓名/音频/台词。自动caption并未人工核真。

初准备r001第10条188词超过内部180上限停止，合格9条保留。r002对同图至多两次长度修正仍188词，无拒答；保存失败/所有尝试。之后显式批准验收协议dense_caption_v2=100–200词，与完整dense目标一致；不截断、不拼模板、不改模型效果门槛。最终r003复用hash/来源确认的9条合格数据、新23条各一次生成，完成32/32；实际104–200词，未用dev/holdout媒体。模型原始五分片hash一致，复制到新本地目录避免共享读盘，不改其他项目权重。旧失败不覆盖。

### 实际训练因素及资源选择

一个节点足够，保留两个节点只并行两学习率（用户曾授权两节点，未实际要求回收），每节点2健康A100，不再每节点7卡占用。排历史故障卡，不跨节点DDP、不抢占他人。两方same seed42/rank16/32dense/gradaccum1/workers1/BF16/gradient-checkpoint-offload、official base初始化，唯一因素标称LR1e-5 vs5e-5。上游ConstantLR初factor1/3及Accelerate多进程scheduler推进保留，记录每次actualLR，不称全程恒定标称LR。

单卡真实49frame480352/rank16、2训练样本pilot1800s/.5GPUh；失败停止不自动降规格重试。通过后两节点各2GPU1epochDDP pilot（期望16实际optimizer updates/rank）1800s/1GPUh。通过后main各4epochs（期望64updates/rank）、save16、7200s/4GPUh；从同official base重启，pilot不续训。每rankfinite loss/grad、实际step、参数非零改变、跨rank初/末hash一致、两node初LoRAhash一致、导出/实际回载验证。DDP复制底座不分片，一张40GB必须先实际通过。

### 同规格效果验证

baseline未训练+lr1/lr5各16原dev输入（8actors×2原prompt），各49frame16fps48035220stepsseed42，同一独立原reference；必须新同规格baseline，不与旧480平方直接比较。train/dev/holdout身份隔离，评测原prompt不优化刷分。原ArcFacecoverage/P10/共同帧actor配对、CLIPB32辅助、MM2026相关AMT MotionSmoothness/MUSIQImagingQuality一起报告，缺项总分null。

loss下降仅数值健康，不算效果改善。8devactors的CI只描述，身份/coverage改善且文本/成像/动态不明显退化才开发候选；不承诺SOTA/最终base。两LR皆无效不堆epoch，转授权更多真实数据与强基线。Track2真实event训练数据目前没有，不伪造双赛道训练。

### 有界预算和交付

原总硬上限51.5GPUh不增加。已caption失败r001约1144s/.318GPUh、r00245s/.013；最终caption最多.5；单卡pilot.5、两2GPU-DDPpilot2、两main8、同规格生成/评价预留4，最坏约15.331GPUh，剩余不为占卡使用。每run自身timeout×卡数匹配预算。最终captionr003是从未用training预留重分配的最后一次修订，任何后续数据/数值失败不追加工程retry，保留失败证据、明确未执行任务。

所有已启动作业跟进真实退出，保存代码/model/reference/split/manifest/seed/hash/成本与实际step/checkpoint。导出LoRA不表示optimizer/RNG恢复验收。代码/计划/真实正负结论脱敏正常push并确认CI；media、caption逐条、embedding、checkpoint、内部节点/凭据不Git。

## 当前阶段

本轮已结束：32/32dense数据，单GPU2步、两节点各2GPU的16步DDP pilot及64步正式训练均通过。两main分别517.398/532.311秒；所有rank的loss/grad有限且非零、300trainable tensors更新、初末hash跨rank同步，两方官方底座重新初始化的LoRA初hash一致。训练含pilot总controller成本0.758255GPUh，实际peak allocated最大18.815GB。上游scheduler在两卡运行中前3次optimizer update使用标称LR/3、第4次起标称LR；保存16/32/48/64 checkpoint并核SHA。不是optimizer/RNG恢复验收。

### 同16dev输入实测

新增48/48视频完整解码，三个arm均480×352、49帧、16fps、20steps，原anchor/prompt/sample_id不变；seed42加原sample_id的固定hash为实际样本seed。两checkpoint均完整150LoRA pairs回载，真实改变150底座tensor；不是只检查文件存在。

| 指标 | 未训练baseline | LR1e-5 | LR5e-5 |
|---|---:|---:|---:|
| 原anchor ArcFace视频均值 | 0.238796 | 0.420010 | 0.672639 |
| 身份frame coverage | 100% | 100% | 100% |
| MUSIQ ImagingQuality | 0.731272 | 0.738241 | 0.741597 |
| AMT MotionSmoothness | 0.992506 | 0.994008 | 0.995406 |
| 原prompt CLIP全帧余弦 | 0.266221 | 0.267244 | 0.270089 |
| 灰度相邻帧motion均值 | 6.576113 | 4.970970 | 3.412948 |
| 全量above-control口部活动 | 16/16 | 15/16 | 15/16 |

身份共同16视频/8actor的等演员配对delta为+0.181213，CI95[0.133131,0.227019]，P10delta+0.168511；高LR为+0.433843，CI95[0.385491,0.482921]，P10delta+0.430912。全部身份有效，因此此轮有效视频均值与配对均值的分母可核对一致；仅小域单seed描述性bootstrap，不是官方FAR/整体效能显著性。

### 结论：身份收益真实，但整体不晋级

两LR在未见训练演员的本域dev上提高身份代理和成像均值，高LR身份更好；同时整体灰度motion相对baseline降至0.755913/0.518992。局部mouth残差flow比0.931782/1.070636、pair coverage均100%，但两者均有1条从above-control活动跌到阈值以下。全局motion下降不必然证明坏动作，嘴部flow也不能证明说话/表情自然：两者必须一起报告，不能把更高AMT平滑分当动态提升。没有固定人工或动作真值解释减少的运动是否去除了抖动、是否损失必要头部动作。因此保留`identity_gain_with_motion_tradeoff_not_promoted`，高LR只作为身份诊断候选，不替换最终base，不继续给32视频堆epoch。

训练/生成已耗controller GPUh分别0.758255/0.330425；两quality profile合计0.020070GPUh，含预检/导入的受管理命令按秒取整上界(56+37)/3600=0.025833。caption旧失败约0.330278，最终r003用0.5GPUh预留上界保守计，阶段合计≤1.944791GPUh，远低51.5hardcap；不把工具延迟通知时长当GPU训练成本。所有已启动GPU作业已结束，两节点显存占用0；未为占满卡追加重复实验。

### 复现与边界

公开聚合证据`resources/real_vace_training_result.json`；八份便携配置在`configs/experiments/real_vace_{single_pilot,ddp_pilot_lr1,ddp_pilot_lr5,main_lr1,main_lr5,eval_baseline,eval_lr1,eval_lr5}.json`。按单卡→DDPpilot→main顺序验收，main不续pilot；LR5配置在另一节点独立运行，`execution_host`不是远程调度。自行配置授权数据和本地权重，真实媒体/caption不发布。

`scripts/analyze_real_training.py`复核五run的实际步数/LR/初末hash/checkpoint覆盖、三方实际LoRA回载、同seed/reference像素、49帧媒体和所有评分output hash，复用冻结身份/口部协议，不改评测门槛。完整CPU回归314passed；无视觉可选依赖的干净环境119passed/21skipped，CPU烟测passed。强化分析复算结果与初版相同，独立CPU重算身份/P10/coverage/CLIP/AMT/MUSIQ/mouth全部一致至1e-12；人工动作真值未核。命令如下（路径按实际本地替换）：

```sh
PYTHONPATH=src python scripts/analyze_real_training.py --runs-dir runs --scores-dir local_scores --train-manifest data/dense_train/train.jsonl --eval-manifest data/dense_train/eval16_original.jsonl --output runs/real_training_audit.json
```

数据许可与InsightFace非商业研究权重分开；caption视觉内容未人工核真，train/dev同为CREMA-D辅助域，holdout未使用。Cur/GME/Story/人工与完整官方MM2026总分仍缺，Track2真实event训练尚未开展；没有KeyID整体优化、广动作泛化或SOTA声明。下一步需独立动作/伪影人工诊断与更广许可兼容数据，不能仅靠身份分选择更高LR。

## 2026-10-21预注册：固定checkpoint融合强度

只改变LR5e-5训练step64的原生融合alpha=1/.5/.25，不新增训练，不声称LoRA优于全量。原生W+=alphaBA，每arm新进程、新官方底座；同16dev/8演员、480×35249帧16fps20steps/CFG5、原prompt/reference/seed42派生ID种子。先alpha1/.5各1视频600s pilot，再freshalpha1完整16条及历史baseline/alpha1共同评分48；重复一致性通过才两节点各1GPU跑.5/.25各16条并评分32。共50新视频，pilot不混方法统计，推理预留2.333GPUh、质量1GPUh，新增硬上限4，历史51.5不增加，无retry。

repeat优先报告媒体hash精确复现率；非逐位一致时身份均值差≤.02、MUSIQ差≤.01、CLIP差≤.005、灰度motion比.9–1.1，coverage/≥7有效身份帧视频数不下降。repeat失败则停止候选与因果claim。freshalpha1对baseline等演员身份增益须>0。

.5/.25候选同时满足：16完整输出/100%身份coverage/≥7帧视频无下降、等演员身份增益≥freshalpha1的60%且描述性bootstrap下界>0/P10delta≥0、灰度motionratio≥.8、口部全量above-control视频率下降≤.05（baseline16/16故须16/16）、口部paircoverage≥.95及共同flowratio≥.8、MUSIQ下降≤.01、CLIP下降≤.005。两方均通过取身份收益较高者；全是本轮开发参考门槛，不是官方或人工动作真值。未达则结束假设不扩搜索。原ArcFace/CLIP/mouth/VBench协议与阈值不改，保留缺脸和所有失败。

全量训练另区分VACE模块全量（已有mode=adapter，仅--trainable_models vace）与生成主干+VACE联合全量（当前导出/回载和显存未验）。32train小域不是全量优劣证据；后续同曝光真实对照及更广数据独立冻结，不与本轮融合因素混改。本轮alpha接线、CPU真实固定GeneralLoRALoader及GPU实际150pairs融合均已验。

### 2026-10-21结果：融合强度不能解决身份与动态取舍

五run全部completed，50新视频完整，无工程retry。freshalpha1的16/16媒体SHA与历史alpha1逐位一致，新增原AMT/MUSIQ共评stage1 48+stage2 32=80记录（含历史对照/重复，不等于80新生成），ArcFace/CLIP/mouth原协议不变。微小AMT重复分差约4e-9属数值漂移，不改结论。

| 指标 | baseline | alpha1 | alpha.5 | alpha.25 |
|---|---:|---:|---:|---:|
| 身份余弦均值 | .238796 | .672639 | .396907 | .271487 |
| 相对baseline等演员身份增益 | — | .433843 | .158111 | .032691 |
| 保留alpha1身份增益 | — | 100% | 36.44% | 7.54% |
| 灰度motion比baseline | 1 | .518992 | .722036 | .893224 |
| 口部above-control活动 | 16/16 | 15/16 | 15/16 | 16/16 |
| MUSIQ ImagingQuality | .731272 | .741597 | .736255 | .735986 |
| 原prompt CLIP余弦 | .266221 | .270089 | .268197 | .266212 |

所有身份帧coverage均100%、16视频/8actor配对完整。.5身份delta CI95[.115985,.199216]，.25为[.017129,.049154]；仅本域单seed描述性bootstrap。.5同时失败60%收益保留、motion>=.8与口部活动保护；.25通过动态参考门槛，却仅保留7.54%身份收益。因此`fusion_strength_not_promoted`，不追加更密alpha搜索，不能称结构优化或全量训练比较。

实际五生成controller0.359535639GPUh，原质量共评含预检墙钟(78.502859+46.108116)/3600=.034614160，总.394149799≤4GPUh；原总51.5hardcap不变，不按工具延迟通知时长计费。实际checkpoint/media SHA、全49帧解码、sample seed/reference像素、scorer样本/actor/hash与固定阈值均由公共分析脚本复核；独立CPU复算数值一致。完整环境349passed，纯控制环境134passed/29skipped，CPU烟测passed。所有已启动GPU作业已结束。

公共证据`resources/lora_alpha_result.json`、五份`configs/experiments/lora_{pilot1,pilot05,alpha1,alpha05,alpha025}.json`。`scripts/analyze_lora_alpha.py`接受冻结本地ledger、runs-dir、scores-dir与stage=repeat/final，拒错误alpha/checkpoint、输入漂移、重复/非有限分数、缺视频、actor错配或预算不自洽。ledger含真实路径只留本地，换路径需重新冻结，不用便携模板的SHA冒充原执行SHA。

下一步改参考条件残差的结构路由，而不是继续缩放整套LoRA：先区分参考条件带来的时间共享成分与时变成分，保留主干动态路径及参考prefix/RoPE；结构假设、固定门控消融与后续可学习门控训练独立执行，未验证前不声称纯身份解耦或效果成功。

## 2026-10-21预注册：参考差分条件的时序路由结构

现VACE在base第0/2/.../28层注入15个[B,N,1536]hint，参考/视频token混合，不是纯身份通道。本轮不训练或缩放LoRA，保持LR5step64/alpha1权重；原生VACE同x/text/timestep/RoPE分别算有参考H_ref和null参考H_null（仅96ch条件的首个参考latent前16ch置零，其他不变）。输出视频段Q=H_ref−H_null、D=Q−mean_time(Q)，新hint=H_ref+(g−1)D，参考prefix保持H_ref原样。g1返回原hint、g0将参考差分时间共享，同时保留H_null的逐时路径。null参考可能OOD、共享可能伤动作，因此只是结构假设，不能称纯身份编码或学习门控。

两arm每视频均40outer/80native VACE calls，固定49帧480×35216fps20steps/CFG5/原prompt/ref/seed，1参考latent+13视频latent、grid14×22×30/N9240、15hints。严格拒其他shape、control/分片/不同参考数，wrapper异常还原forward；不改底座源码、referenceprefix噪音或RoPE。g0要有实际body变化，g1零变化，FP32分解后BF16注入的mean误差如实记录。

两节点各1健康GPU：g1/g0单视频pilot各600s，通过后各16完整dev（3000s/arm），总34新视频；旧未训练baseline/旧alpha1与新g1/g0共同64视频原AMT/MUSIQ共评，两节点32各1800s。推理预留2GPUh+评价1，新增硬上限4，旧51.5总上限不加，无retry/额外g搜索。g1须复现旧alpha1媒体；否则使用已冻结identity差≤.02、MUSIQ≤.01、CLIP≤.005、graymotion比.9–1.1/coverage不下降，超界停止归因。

g0候选要求完整16video/8actor身份配对与100%coverage、保留g1身份增益≥60%/actorCI lower>0/P10≥0，graymotion相对未训练baseline≥.8、mouth活动16/16/paircoverage≥.95/共同flow≥.8，MUSIQ下降≤.01、CLIP下降≤.005。全部为开发proxy，非官方/自然动作真值；失败则终止此假设。若有效，下一阶段才注册15层可学习gate、初始化g1并实际训练/保存/回载；本轮固定gate不冒称训练策略收益。全VACE与主干联合微调独立冻结，不与结构门控同时改变。

结构数学/参考prefix/时间均值/批次隔离/autograd/两原生calls及finally恢复43项相关CPU测试通过；新增BF16均值舍入界回归，不把cast后的均值称精确保持。

### 结构实测完成：不是参数缩放，但未解决动态取舍

四个结构run均completed：两pilot+两完整16dev共34新视频，64视频原AMT/MUSIQ共评及新两arm ArcFace/CLIP/mouth已结束。g1控制16/16媒体SHA精确复现原alpha1，说明增加等调用null分支未改变控制输出。g0每视频真实40outer/80native/15hint，参考prefix逐位不变，实际body变化最大.265625；输出视频hint时间均值误差最大.001803来自FP32分解后BF16舍入，数学共享不等于编码数值精确共享。原checkpoint/source/input/hash/seed均核，未训练新gate。

| 指标 | 未训练baseline | 原结构g1 | 共享参考差分g0 |
|---|---:|---:|---:|
| 身份余弦均值 | .238796 | .672639 | .666769 |
| 相对baseline等演员身份增益 | — | .433843 | .427973 |
| 灰度motion比baseline | 1 | .518992 | .532801 |
| above-control口部活动 | 16/16 | 15/16 | 15/16 |
| MUSIQ ImagingQuality | .731272 | .741597 | .740859 |
| AMT MotionSmoothness | .992506 | .995406 | .995375 |
| 原prompt CLIP余弦 | .266221 | .270089 | .270675 |

身份coverage和口部paircoverage均100%，16视频/8actor配对完整；g0保留98.65%g1身份收益，baseline配对delta+.427973、CI95[.378121,.476687]，相对g1identitydelta−.005870、CI95[−.010377,−.001227]、P10delta−.011307。g0整体motion对g1仅恢复2.66%，仍只有未训练baseline53.28%，口部活动仍15/16；失败gray>=.8与口部全量保护门槛。其余门槛通过，不以略高CLIP或身份近似保留宣称整体优化。

结论`reference_routing_not_promoted`：完成真实结构实现和可证伪消融，但没有有效多维改善；简单把参考差分时间共享不足以修复主要运动收缩，不追加中间g搜索或把常数门控称训练成果。null参考可能OOD，差分非纯身份；更广动作/人工自然性未验证。后续训练可学习门控需新假设和真实训练/保存回载，并非本轮结构获胜后自然晋级。

四生成controllerGPUh合计.314951662，两个质量命令完整墙钟(74.497313+46.622482)/3600=.033644388，总.348596049≤4，原51.5硬上限未增加，无retry/遗留GPU任务。独立CPU重算数值一致；完整环境374passed，干净控制134passed/31skipped，CPU烟测passed。公共分析`scripts/analyze_reference_structure.py`复核结构calls/prefix/实际body变化、原checkpoint回载、全视频解码及scores/mediahash/actor映射、冻结原VBench权重和开发门槛。公共证据`resources/reference_structure_result.json`、四份`configs/experiments/reference_{pilot1,pilot0,gate1,gate0}.json`。原图/视频/个体caption/embedding/checkpoint/内部地址不发布。
