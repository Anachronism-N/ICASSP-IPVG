# 真实dense-caption LoRA训练与同规格效果对照

## 2026-10-20预注册

当前32train视频/16身份来自许可兼容CREMA-D，参考与目标独立source，dev/holdout身份隔离。只有说话头部辅助域，不是VIP/ReactID或足够SOTA的训练规模。原短标签prompt不用于训练：本地公开Qwen2.5-VL7B从实际全时长重采样训练像素生成100–180词可见外观/场景/beginning-middle-end细节，禁止推断姓名/台词/不可见动作。自动caption不等于人工核真，不凭字数认定正确。

原视频480×360、29.97fps/51–86帧，中心裁为480×352、全时长均匀抽49帧，准备fps=49/原duration；不把1.7–2.9秒冒称49帧16fps原始3秒。上游训练读取前49帧、不读取fps，所以先重采样整个时间跨度而不是默认截前49帧。生成开发规格仍49帧16fps/480×352，三方相同。

## 资源决策与训练因素

一个节点能完成本轮，两个节点的用途是并行两学习率而不是效果必需条件；用户未要求实际释放节点，继续沿用此前授权双节点对照，若节点被收回则不启动该节点作业。每节点7健康A100独立DDP（历史异常卡排除），不跨节点DDP，不以占卡证明资源利用率或质量。小数据尾部padding造成每epoch35项中3重复，报告实际训练样本曝光，不扩大重复来伪造数据规模。

唯一因素学习率：launcher标称1e-5、worker5e-5；两方seed42、VACE1.3B官方底座、VACE条件模块LoRA rank16、q/k/v/o/FFN、同dense32、gradaccum1、workers1、BF16/gradient-checkpoint-offload相同。AnyID未有公开作者训练入口、KeyID仅README，不声称训练了它们；VACE本轮是可训练域内对照，不提前定为最终base。

先单卡真实49帧/rank16两样本显存与梯度pilot；随后两节点各1epochDDP pilot5实际optimizer steps/rank；数值finite、梯度非零、参数真实更新、跨rank初末hash同步及导出文件验收。正式两main各14epochs预期70updates/rank，save_steps5尽早保留checkpoint。pilot不续训，main从同官方底座初始化，比较两节点初始LoRAhash。上游默认ConstantLR开始factor1/3、Accelerate多进程推进scheduler，记录每次真实optimizer的actualLR，不称全程恒定标称LR。

单卡OOM唯一固定降级为17帧全时长重采样，必须两组同时同规格、新runID、保存失败；不能靠增加DDP卡数解决单卡OOM，禁止无限重试。保存仅LoRA，不宣称已验证optimizer/RNG恢复。loss下降只数值健康，必须看dev视频。

## 同规格效果验证

从原dev确定性选8actors×2原prompt16案例；不读取holdout媒体，不用dev调caption/训练。未训练baseline、lr1e-5、lr5e-5各16输出；480×352/49帧16fps/20steps/同seed42/相同独立原reference，不能直接比较旧480²结果。身份ArcFace/覆盖率/P10与共同有效帧actor配对、CLIPB32原prompt、MM2026相关AMT MotionSmoothness/MUSIQImagingQuality一起报告。缺Cur/GME/Story/官方人工则总分null，不称完全MM2026复现。caption只用于训练，评测原prompt不改写刷分。

若dev身份和覆盖率不升、图像/文本/动态明显退化，即使loss下降也不晋级；8actors小样本CI仅描述，不承诺稳定增益/正式参赛验收。两lr都无效则停止增加epoch，优先授权广动作真实数据和更强身份基线，不在小辅助域无限训练。Track2仍无真实event训练数据，单列未开展，不伪造双赛道训练。

## 冻结预算和交付

最坏51.5GPUh：caption单GPU1800秒/.5；单卡训练pilot1800秒/.5；两7GPU DDPpilot各1800秒/3.5共7；两main每7GPU10800秒/21共42；唯一单GPU修复1800秒/.5；两节点单GPU推理/评价阶段各1800秒/.5共1。启动前按预算核验；controller每runtimeout、保存hash/代码/model/split/seed证据，失败不覆盖。共享模型hash/copy属CPU预处理，但caption阶段GPUallocation等待也计成本。

所有已启动任务跟进到明确退出/失败，必要修复后新ID，限制预算内停止。方案/代码/聚合真实正负结论脱敏正常commit/push并确认CI；原始身份媒体/caption逐条/embedding/checkpoints/凭据不进入公共Git。

## 当前阶段

初次dense准备因第10条caption未满足100–180词检查而停止，保留9条合格caption与失败目录；共享权重hash/加载等待仍计caption allocation成本。修复版仅在新目录复用已校验hash/来源的9条合格结果，每个新caption最多两次看同一图序列的长度修正，不放宽字数标准、不凭空凑字数；保存所有尝试。公开权重复制到新本地目录并对比原hash，不改其他项目权重。修复阶段单GPU1800s/.5GPUh，使用原唯一工程修复预留，后续不无限重试；若显存/训练仍失败则如实结束并保存诊断，不超原总预算。
