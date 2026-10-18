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
