# 原生双尺度参考与局部嘴部运动诊断

## 2026-10-20预注册

效果优先。上一轮扩大margin不能通过新增dev完整覆盖门槛，不继续margin搜索。本轮唯一生成因素为原生参考列表从[crop0.3]改为[crop0.3,同一原始reference全图]。没有额外identity/source图像、视频帧或target条件；主图和原评价anchor保持。相同sample_id/seed42或43、12dev演员两原prompt、480²49帧16fps、UniPC20steps/CFG5/shift3/GPU-T5/offload及固定AnyID LoRA/5B版本不变。

原生多图输入经逐图VAE编码后拼接reference latent。两图inner_t=[2,13]、frame_num57，一图[1,13]/53；实际生成均49帧，不硬贴原图首帧。每step都是2DiT调用，但双参考序列3375而单参考3150tokens、多一次图VAE，实际成本不同必须报告。新增adapter只扩原生有序1–5参考数量，不改上游CFG/attention/去噪算法或权重。

复用上一轮48条单crop历史对照，新增双尺度48条。历史对照不是本轮独立重跑。双方使用同一评价协议重评，4旧诊断演员和8新增dev演员分开；都不是最终holdout。1条工程pilot900秒，剩余47条四batch各1350秒，新生成预留1.75GPUh（pilot.25＋四batch共1.5）；最多一次1800秒工程修复.5，总硬上限2.25。先前计划的2.75为预算加法错误，launch前修正，无运行超额。两节点按actor分片，同sample固定物理GPU，排除历史异常卡。任何坏run保留，有限重试不因质量负结果追加候选。

## 独立局部动态协议

`local_mouth_motion_v1`固定17个均匀帧，SCRFD(.5、640)恰好一脸才对齐；仅双眼相似变换至112平方，不用嘴角拟合。Farneback参数(.5,3,15,3,5,1.2,0)，从参考ROI[28,58,84,78]取flow二维median平移向量，再从嘴ROI[28,80,84,106]减该向量、计算残差幅度均值；视频聚合只含两端有效的相邻采样pair，不跨missing拼接，至少12/16pair。mouth灰度差独立描述，不作混合判据。warp插值、2D/3D转头及检测点抖动仍可干扰，不能称动作自然度。

先对真实32dev演员两source目标按固定hash拆16cal/16val；两个负对照为首采样帧freeze与同一帧整图sin/cos平移(4px/3px)，每个负对照重新检测/对齐，覆盖不足不计分。阈值为cal刚体control视频分最大+1e-6，仅source/controls定阈值。要求两组各>=28scoreable视频、>=14actors，freeze激活0、val刚体激活<=.10、val源激活率>=.75。源标签只表示acted utterance，没有独立口动标注，所以不是TPR/灵敏度，不宣称说话/微笑/唇同步验收。校准失败保留负结论，不调ROI、阈值或采样刷通过；仍完成固定生成，但动态结果只诊断。

## 固定质量门槛

48paired inputs（两方法总96视频），候选48媒体完整。全量身份framecoverage至少提高.02，两seed及8新增dev演员至少7/8frame可评分video数均不降；共同7帧identity配对至少10actors，actor均值delta CI95下界>=-.02、P10 delta>=-.02；graymotion比>=.8/static fraction增量<=.10。身份missing不填0，恢复和丢失都报告，不能看容易子集均值。

若局部motion校准通过，再要求mouth paircoverage>=.75，合格口动video率不能下降>.05、共同有效video口部flow比>=.8；否则明确动作/语义质量未验收，不能认定最终base。mouth局部活动仍非语义/自然性证据。原ArcFace协议与CLIP诊断不变，所有条件同协议。CPU校准/评分各7200秒上限，独立记账，无新视觉权重下载。

## 工程边界与交付

固定源码commit/hash、模型与reference/split/manifest/hash、实际seed、prefix长度、output49shape/finite/完整解码、实际GPU墙钟账本。只发布adapter/测试/协议/聚合结论，不发布media、embedding、逐actor清单、内部节点或凭据。正式竞赛参考衍生操作是否允许仍待本人核准。

只读代理验证ViCLIP导入时触发上游自动下载公开词表约1.3MB，违反不下载约束；停止该路径，未读取或上传人脸media、未修改项目。原文件保留，本轮评价只用既有SCRFD/OpenCV，不以此次导入表示ViCLIP质量能力已验收。

运行结果完成后追加，不能用上一轮已发布结果满足本次新目标。
