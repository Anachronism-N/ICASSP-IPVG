# 原生双尺度参考与局部嘴部运动诊断

## 2026-10-20预注册

效果优先。上一轮扩大margin不能通过新增dev完整覆盖门槛，不继续margin搜索。本轮唯一生成因素为原生参考列表从[crop0.3]改为[crop0.3,同一原始reference全图]。没有额外identity/source图像、视频帧或target条件；主图和原评价anchor保持。相同sample_id/seed42或43、12dev演员两原prompt、480²49帧16fps、UniPC20steps/CFG5/shift3/GPU-T5/offload及固定AnyID LoRA/5B版本不变。

原生多图输入经逐图VAE编码后拼接reference latent。两图inner_t=[2,13]、frame_num57，一图[1,13]/53；实际生成均49帧，不硬贴原图首帧。每step都是2DiT调用，但双参考序列3375而单参考3150tokens、多一次图VAE，实际成本不同必须报告。新增adapter只扩原生有序1–5参考数量，不改上游CFG/attention/去噪算法或权重。新增prefix会使输出RoPE起始位置1→2，因此只能报告原生双尺度条件整体效应，不能将效果单独归因全图语义；高度相关的两图不提供新视角/表情。

复用上一轮48条单crop历史对照，新增双尺度48条。历史对照不是本轮独立重跑。双方使用同一评价协议重评，4旧诊断演员和8新增dev演员分开；都不是最终holdout。1条工程pilot900秒，剩余47条四batch各1350秒，新生成预留1.75GPUh（pilot.25＋四batch共1.5）；最多一次1800秒工程修复.5，总硬上限2.25。先前计划的2.75为预算加法错误，launch前修正，无运行超额。两节点按actor分片，同sample固定物理GPU，排除历史异常卡。任何坏run保留，有限重试不因质量负结果追加候选。

## 独立局部动态协议

`local_mouth_motion_v1`固定17个均匀帧，SCRFD(.5、640)恰好一脸才对齐；仅双眼相似变换至112平方，不用嘴角拟合。Farneback参数(.5,3,15,3,5,1.2,0)，从参考ROI[28,58,84,78]取flow二维median平移向量，再从嘴ROI[28,80,84,106]减该向量、计算残差幅度均值；视频聚合只含两端有效的相邻采样pair，不跨missing拼接，至少12/16pair。mouth灰度差独立描述，不作混合判据。warp插值、2D/3D转头及检测点抖动仍可干扰，不能称动作自然度。

先对真实32dev演员两source目标按固定hash拆16cal/16val；两个负对照为首采样帧freeze与同一帧整图sin/cos平移(4px/3px)，每个负对照重新检测/对齐，覆盖不足不计分。阈值为cal刚体control视频分最大+1e-6，仅source/controls定阈值。要求两组各>=28scoreable视频、>=14actors，freeze激活0、val刚体激活<=.10、val源激活率>=.75。源标签只表示acted utterance，没有独立口动标注，所以不是TPR/灵敏度，不宣称说话/微笑/唇同步验收。dev的neutral/happy分别来自IEO/IOM两台词，情绪与台词完全混杂，口部活动差也不能解释为表情语义。校准失败保留负结论，不调ROI、阈值或采样刷通过；仍完成固定生成，但动态结果只诊断。

## 固定质量门槛

48paired inputs（两方法总96视频），候选48媒体完整。全量身份framecoverage至少提高.02，两seed及8新增dev演员至少7/8frame可评分video数均不降；共同7帧identity配对至少10actors，actor均值delta CI95下界>=-.02、P10 delta>=-.02；graymotion比>=.8/static fraction增量<=.10。身份missing不填0，恢复和丢失都报告，不能看容易子集均值。

若局部motion校准通过，再要求mouth paircoverage>=.75，合格口动video率不能下降>.05、共同有效video口部flow比>=.8；否则明确动作/语义质量未验收，不能认定最终base。mouth局部活动仍非语义/自然性证据。原ArcFace协议与CLIP诊断不变，所有条件同协议。CPU校准/评分各7200秒上限，独立记账，无新视觉权重下载。

## 工程边界与交付

固定源码commit/hash、模型与reference/split/manifest/hash、实际seed、prefix长度、output49shape/finite/完整解码、实际GPU墙钟账本。只发布adapter/测试/协议/聚合结论，不发布media、embedding、逐actor清单、内部节点或凭据。正式竞赛参考衍生操作是否允许仍待本人核准。

只读代理验证ViCLIP导入时触发上游自动下载公开词表约1.3MB，违反不下载约束；停止该路径，未读取或上传人脸media、未修改项目。原文件保留，本轮评价只用既有SCRFD/OpenCV，不以此次导入表示ViCLIP质量能力已验收。

## 实测完成：双尺度不晋级

5/5run、新候选48/48视频均正常结束，480²49帧16fps完整解码；原48条对照hash复核通过。各候选确认2reference、inner_t=[2,13]、native_frame_num57、40DiT调用、reference latent2且timestep/seq_len均3375。原单图仍3150tokens；1–5参考数量CPU校验通过，真实GPU证据限定为此双参考配置，不声称其他3–5参考或长分辨率验收。

| 全量48输入/condition | 单crop历史对照 | 同图双尺度 |
|---|---:|---:|
| 至少一身份帧可评分视频 | 36/48 | 36/48 |
| 至少7/8身份帧可评分视频 | 34/48 | 35/48 |
| 身份framecoverage | 72.3958% | 73.1771% |
| 不同可评分子集身份均值（非配对提升） | 0.827103 | 0.825869 |
| mouth达到12/16有效pair的视频 | 34/48 | 34/48 |
| 全量mouth有效paircoverage | 71.4844% | 72.2656% |
| 合格且口部残差flow超过控制阈值的视频 | 25/48 | 21/48 |

33共同身份视频/12演员配对delta=-0.001619，actor-bootstrap CI95[-0.047266,0.046942]，未达到非劣下界-0.02；P10 delta+0.005604。身份framecoverage仅+0.0078125，未达到+0.02。新增8dev演员≥7帧视频22→23（恢复2、丢失1），但旧诊断组framecoverage79.6875%→75%；两个seed的framecoverage增量分别0和+0.015625，不是稳定全量收益。原缺脸neutral诊断pilot双尺度仍0/8可评分，不能认为辅助全图解决该失败。

全局graymotion比0.890339、static比例均0，但mouth全量合格活动率52.0833%→43.75%（下降8.3333个百分点，超过5个百分点门槛），全量paircoverage未达到75%。双方34视频可测mouth均值，有32共同合格video，口部flow均值比0.801056仅接近0.8保护阈值；由于两条件有效时点可能不同，这一比例不是完全同时间pair因果估计，不能掩盖全量活动率下降。结论`dualscale_not_promoted`，不追加参考数/裁剪搜索，不将原生多参考接线成功或身份均值当最终质量成功。

### 独立局部动态控制结果

64dev视频/32演员全部source/freeze/rigid可评分；cal16演员源activation31/32=96.875%、freeze0、rigid0，val16演员源32/32=100%、freeze0、rigid0。阈值0.4554968884来自cal rigid-control最大+1e-6，一次冻结、不用生成调阈值。运行OpenCV5.0.0、NumPy2.5.4、ONNX Runtime1.23.2、InsightFace0.7.3、CPU provider；该验证只支持这里的小刚体对照与非静态口部活动代理，不能证明说话、微笑、真假脸或感知质量。

### 资源与复算

新候选backend0.465039GPU小时，上限2.25，无工程修复槽使用；历史单crop成本0.456580GPU小时仅作成本背景、不计新运行。新候选生成中位25.374590秒、历史25.081058秒，最大allocated14.600330GB。不是效率目标，也不声称新独立吞吐trial；两节点作业结束后无遗留GPU占用。CPU校准/评价受独立超时约束，不混入GPU生成allocation。

`scripts/mouth_motion_eval.py calibrate/evaluate`复现局部动态协议；`scripts/compare_anyid_multiscale.py`读取冻结ledger、run、identity/motion评分与calibration，重新完整解码、核验输出hash/协议/权重、复算五actor/seed分组及身份/动态门槛。独立复算与本地审计一致。公开配置`configs/experiments/anyid_multiscale_pilot.json`的manifest应提供有序[crop,原图]路径，实际prefix数量由adapter逐样本计算，不改sample_id。

实际生成源码commit750068b、source_sha256 `f3d9fd3baa503e9614b99de126a8eb95e10b0706f7bdf785f4e301d2965cd1a2`，上游/模型沿用固定锁文件。初计划台账的source hash在添加去噪观测hook之前取得，原台账不覆盖；所有正式run的实际source hash一致，实际run指纹是执行证据。公共聚合结果见`resources/anyid_multiscale_result.json`，不含媒体或逐身份数据。完整环境277项CPU回归通过，干净环境108项通过/17项可选依赖跳过、烟测通过；独立复算五组身份统计、口部coverage/活动率及固定质量门槛与本地审计一致。

最终建议保留单crop为当前开发对照。后续质量研究优先独立验证稠密面部关键点/唇部形变与本地盲评，排除头动或插值伪动态；本轮不加入新评分模型追溯修改门槛，也不声称最终比赛base、官方成绩或广动作效果已经验收。
