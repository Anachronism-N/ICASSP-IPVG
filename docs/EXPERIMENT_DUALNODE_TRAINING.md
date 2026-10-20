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

数据32/32完成，训练尚待真实单卡pilot和两节点验收；没有效果提升证据。实际结果随后追加，不用上一轮ad38099满足训练目标。
