# 本地评价与外部评分器接入

核查日期：2026-10-17。本仓库实现视频有效性检查和外部数值汇总；身份、主体、文本、动作及时间段评分器需要在服务器另行部署。当前没有内置的IPVG官方总分，也没有用占位数值冒充模型实测。官方后续公布评分器与权重后，应建立独立协议版本，并保留本地代理结果。

## 先区分有效视频与生成质量

`ipvg evaluate`通过`ffprobe -count_frames`读取并解码计数视频流，检查缺失文件、解码错误、正的尺寸、帧率、时长及帧数；可按显式传入的规格检查宽、高、帧率和时长。默认容差为时长0.25秒、帧率0.01；它们是本地工程默认值，不是官方允差。缺少规格时，程序不会猜测官方输出要求。

一段完全静止但可解码的视频也可能通过这些检查。`report.ok`表示媒体完整性及指定规格通过，不表示身份、动作、时间对齐、审美或正式提交验收通过。外部低分不会改变这个布尔值，实验晋级策略必须另外检查质量、覆盖率与人工抽样。

```bash
ipvg evaluate data/dev.jsonl runs/baseline/videos --output runs/baseline/media.json
```

如果已经确定实验要求，可显式附加`--expected-width`、`--expected-height`、`--expected-fps`、`--expected-duration`及容差。开发集实验规格不代表比赛正式规格。`submission`命令只制作本地草稿包，不接收评分器参数，也不会联网提交。

## Proxy JSONL契约

每个样本最多一行，使用与输入manifest相同的`sample_id`。以下数字仅示范文件格式，不能作为任何模型的实验结果：

```jsonl
{"sample_id":"dev_0001","score_kind":"local_proxy","scores":{"face_identity_cosine_mean":0.82,"face_detection_coverage":0.75,"text_alignment_cosine_mean":0.27},"failure_classes":["identity_drift"]}
{"sample_id":"dev_0002","score_kind":"local_proxy","scores":{"subject_similarity_cosine_mean":0.68,"action_window_support":0.5,"action_start_mae_s":0.4},"failure_classes":["action_missing","timing_mismatch"]}
```

- `sample_id`必须唯一、存在于本次manifest，并满足安全文件名规则；禁止重复及未知ID。
- `scores`是指标名到有限数值的对象；接受整数或小数，拒绝字符串、布尔值、`NaN`和无穷值。不同指标可有不同量纲、方向和有效样本数；程序不会默认它们都在0至1之间。
- `score_kind`可以省略；若填写，只能是`local_proxy`。报告始终标明`local_proxy`，`official_score`保持`null`。
- `failure_classes`可省略，默认空列表；填写时必须是非空字符串列表，例如`identity_drift`、`action_missing`、`timing_mismatch`、`subject_not_detected`。它们是本地诊断标签，不是官方分类。同一行重复标签仅计一次。
- 某项指标无法可靠计算时，省略该指标并记录失败原因；不能把“未检测到”伪装成真实的相似度0或高分。若方法预先规定缺失惩罚，应另命名并明确报告协议及检测覆盖率。
- 同一样本的各评分器结果先合并到一行再导入。导入器不负责合并重复行；额外字段目前不进入汇总，模型版本等必须保存在旁边的协议文件中。

```bash
ipvg evaluate data/dev.jsonl runs/baseline/videos --proxy-scores runs/baseline/proxy.jsonl --output runs/baseline/evaluation.json --fail-on-extra
```

报告包含每项指标的`count`、`coverage`、`mean`、`min`和`max`，以及输出覆盖率、有效视频覆盖率、代理评分覆盖率和两类失败计数。**缺失分数不会补0，指标不会自动加权为总分。**指标均值对实际导入的数值计算；程序不会自动剔除媒体无效样本的已导入分数。适配器应只对可读输出评分，分析时同时核对样本级媒体状态，避免陈旧评分抬高结果。

## 在服务器接入评分器

采用“独立评分环境 → 按样本导出JSONL → 标准库汇总”的边界。把PyTorch、ONNX Runtime、图像/视频解码和评分模型依赖放在评分器专用环境；基础`ipvg`包不自动安装或下载它们。评分器读取固定manifest和`<sample_id>.mp4`，先验证视频，再按冻结的时间采样方案取帧或片段，产生数值和失败标签。

若复用[ConsisID原作者评价代码](https://github.com/PKU-YuanGroup/ConsisID/blob/main/eval/README.md)，先在固定commit下检查`get_facesim_fid.py`、`get_clipscore.py`及权重路径。原说明的输出格式尚未标准化，不能假定它们直接输出本仓库JSONL；需要编写薄适配器，把实际样本结果对应回`sample_id`。FID属于集合指标，不应复制到每条样本后伪装成逐视频分数；这类结果单独保存在实验级报告中。

以下为本项目建议的本地协议，不是IPVG官方指标定义：

| 维度 | 接入方法与建议输出 | 需要一同报告的限制 |
|---|---|---|
| 人类身份 | 用固定人脸检测、对齐和识别模型提取参考图与视频有效帧的归一化embedding，计算余弦；输出均值、低分位数及检测覆盖率 | 多人时预先定义目标跟踪/匹配规则；禁止逐帧挑最相似的人脸。遮挡、侧脸或检测失败必须记录，均值不能只剩容易的帧 |
| 通用主体 | 对动物、物体或非人主体使用固定crop/跟踪器及[DINOv2](https://github.com/facebookresearch/dinov2)等视觉embedding，输出参考主体相似度、跟踪覆盖率 | 背景可抬高全帧相似度；参考匹配与视频内主体一致性应分开，不能用人脸特征代替通用主体身份 |
| 文本符合度 | 用固定图文或视频文本编码器比较prompt与帧/片段；保留模型、文本截断、采样时点、归一化方式 | 单帧相似度不证明动作发生，更不证明先后顺序；不能把不同编码器的分数直接横向比较 |
| Track 2动作与时间 | 按manifest的`timeline`逐个检查动作在指定时间窗中的证据；动作识别器、视频VLM或人工标注可以提供预测区间 | 整段检测到动作不等于时间对齐；要记录动作缺失、顺序颠倒、起止偏移。VLM判断是代理而非真实标注 |
| 动态与视觉质量 | 可接入[VBench自定义视频支持](https://github.com/Vchitect/VBench#new-evaluate-your-own-videos)的主体一致性、动态程度、运动平滑等适用维度 | 遵循对应版本支持的custom-input维度；不要把自定义prompt测量直接称作官方VBench排行榜结果。高身份相似度须检查是否退化为静止画面 |

人脸embedding余弦为`dot(normalize(reference), normalize(frame))`，多个参考图、多个帧的聚合方法要预先冻结。比如固定参考模板、固定时间采样，分别报告有效帧均值和检测率；不要随模型切换聚合方法。人脸识别不能证明每个身体或发型细节相同。[InsightFace代码与模型许可不同](https://github.com/deepinsight/insightface#license)：代码MIT不代表预训练识别权重可以任意商用；受限数据和人脸帧应保持在获准环境内。

有可靠动作起止预测时，可自定义`action_temporal_iou_mean`与`action_start_mae_s`；先明确预测区间与目标时间窗的配对规则，缺失动作另记。若只有VLM是否支持某窗动作的判断，用`action_window_support`等明确的代理名称，不声称已获得真实时间定位。按视频真实时间戳取片段，尤其不能把可变帧率视频一律当作固定24fps。

## 已实现的离线身份评价入口

`scripts/identity_eval.py` 提供 `calibrate/evaluate/compare`，`scripts/prepare_reference_crop.py` 仅修改生成条件并保留原身份anchor。运行依赖在独立CPU ONNX环境安装，基础包不强制安装视觉库；没有本地ONNX文件即失败，不自动下载。权重hash及非商业研究许可见 `resources/identity_encoder.lock.json`。

固定8帧且恰好一脸才能取embedding；多脸/缺脸省略相似度并记录失败，识别coverage与检测coverage分开。真实源视频校准的视频均值阈值仅用于至少7有效帧的视频均值判断，不将其解释为帧级FAR。配对比较使用共同有效帧和原始参考，不按生成分数调阈值。

身份余弦、演员级bootstrap、P10、覆盖率、灰度运动与静态比例同时报告。灰度运动不是动作符合度，单种子说话头部域改善不能称SOTA或广动作泛化。完整预注册与本地校准/对照结论见 [EXPERIMENT_CREMA_D_H1.md](EXPERIMENT_CREMA_D_H1.md)。

## 已实现的离线文本辅助代理

`scripts/clip_text_eval.py`使用显式本地OpenAI CLIP ViT-B/32权重及严格SHA校验，不下载；CPU FP32、4线程、官方全帧预处理、8帧均匀采样、原prompt且拒绝截断、单位化余弦不乘logit_scale。协议`clip_b32_frame_text_v1`与ArcFace独立，对所有条件一致重评。

实际64条CREMA-D源dev视频上，匹配prompt优于另一neutral/happy模板仅53.125%，prompt contrast均值0.007626。该域的模板接近且单帧图文模型无法证明说话发生；本代理不能承担动作/表情晋级门槛，仅报告辅助诊断，不能将高余弦称为文本遵循已验收。完整方案见 [上下文效果实验](EXPERIMENT_ANYID_CONTEXT.md)。

## 独立口部局部动态代理

`scripts/mouth_motion_eval.py`以固定SCRFD五点中的双眼对齐112平方、17帧/16相邻采样pair，Farneback flow扣除鼻侧参照ROI的中位平移向量，报告嘴ROI残差幅度；无脸/多人不计分，不跨缺失帧连线。它与ArcFace身份协议独立，不调整人脸阈值/身份anchor，不加载新表情模型。源/控制校准使用16演员calibration、16validation，阈值为校准rigid-control视频最大+1e-6；此域val的freeze/rigid activation均0、源activation1，不是动作真值TPR。

原dev的neutral/happy同时与两种台词混杂，未有人工作为嘴动/说话阳性标签；2D对齐仍受3D转头、眼点抖动和插值影响。因此只能标`local_mouth_motion_v1`辅助非静态活动诊断，不能称说话、微笑、唇同步、自然性或官方得分。各生成条件同新协议重评，低覆盖省略motion均值、coverage另报；全量above-control视频率包含缺失造成的可用性损失，不把缺失身份余弦填0。共同motion比基于双方合格的同一视频，但有效时点可能不完全相同，仍需报告配对数量与掉帧局限。

`scripts/compare_anyid_multiscale.py`复核两方媒体/输出hash、评分协议/权重、身份/seed配对、口部16pair覆盖和均值自洽，重算身份与动态质量门槛。完整冻结设置、结果和副作用说明见 [双尺度实验](EXPERIMENT_ANYID_MULTISCALE.md)。

## MM2026相关共同质量面板

新增`scripts/mm2026_quality_eval.py`直接调用固定VBench fd18b3d原AMT-S MotionSmoothness与MUSIQ-SPAQ ImagingQuality，全帧、FP32、显式本地权重SHA和禁联网保护。每视频ImagingQuality=原MUSIQ MOS/100，MotionSmoothness=1-奇数帧重建MAE/255。已完成48视频共同评价；既有人脸8输入×3方法24视频与受控/生成物体24视频分开，不是MM2026官方test。旧ArcFace strict8frame/单脸/missing与CLIPB32cos辅助不变，名称不冒充作者checkpoint与protocol。

作者AESR选片式为`.15*(GME+Story)+.20*(Cur+Arc)+.15*(Motion+Imaging)`，不是含人工的官方FinalScore；目前缺Cur/GME/Story/人工，因此自动合成分保持null，不使用partial sum。MUSIQ是技术图像质量而非身份或美学充分指标；AMT既是本轮稀疏插值器也是smoothness评价器，有循环偏好，静止也可高分。完整设置、实例结果、缺组件和非完整KeyID限制见[MM2026面板实验](EXPERIMENT_MM2026_KEYFRAMES.md)、`resources/mm2026_eval.lock.json`。

## 真实LoRA训练同规格效果共评

`scripts/analyze_real_training.py`对五训练run及baseline/LR1e-5/LR5e-5共48同输入输出复核真实optimizer步数/LR、rank初末hash、checkpoint保存覆盖、实际LoRA回载、参考像素/派生seed、49帧480×352解码、所有评分output hash，并复用原strict ArcFace/CLIP/mouth和原AMT/MUSIQ；训练loss不加入质量分数。八dev演员等演员共同有效帧delta与CI是描述性小域统计，不套用旧H1的28演员/56pair晋级门槛，也不把未达到旧样本量解释为没有身份信号。

此轮身份收益但全局motion减少，口部活动16→15/16，保留不晋级结论。完整结果见[真实训练报告](EXPERIMENT_DUALNODE_TRAINING.md)。灰度motion不等价动作质量，AMT高平滑分可能伴随较小运动；必须独立补动作/伪影人工诊断，缺项官方总分仍null。

## 固定融合强度评价

`scripts/analyze_lora_alpha.py`按冻结ledger复核alpha1重复一致性与alpha.5/.25候选、实际checkpoint融合和50新输出，stage1/stage2共同原AMT/MUSIQ80评分记录（含历史与重复）分开计生成成本。身份要求完整16视频/8演员共同帧与原anchor，mouth全量视频分母不删失败；预注册60%身份收益保留与动态保护均保留，两个候选未整体晋级。纯融合强度不是结构/训练策略改善，结果见[训练与融合报告](EXPERIMENT_DUALNODE_TRAINING.md)。

## 参考差分结构评价

`scripts/analyze_reference_structure.py`独立核验同checkpoint、原reference/seed、两原生VACE分支/每视频80calls、15层hint/prefixexact/实际body变化与BF16均值误差，64共同质量records不当作64新生成。身份16视频/8actor全部配对、原scorer权重阈值固定，口部16完整可评分/共同pair视频是保护边界。g1控制需精确复现或通过预定义漂移门槛，g0仅保留身份而无充分动态恢复，未晋级；灰度/mouth不是动作自然性真值。完整结果见[结构报告](EXPERIMENT_DUALNODE_TRAINING.md)。

## 冻结协议与复现记录

每个run旁保留`metric_protocol.json`或等价文档，至少记录：

| 类别 | 必需信息 |
|---|---|
| 输入与产物 | manifest SHA-256、数据划分版本、生成视频哈希、生成模型revision、生成seed、分辨率/帧数/采样步数 |
| 评分器 | 代码commit、每项权重revision及SHA-256、依赖锁定、设备/精度、embedding层与归一化、crop/跟踪/匹配规则 |
| 时间与聚合 | 取帧/片段时点、目标动作清单、阈值来源、缺失处理、逐视频聚合、每个指标的量纲及高低方向 |
| 预算与人工 | 生成与修正候选数、训练/生成/评价GPU小时、评分器耗时、人工抽样清单与盲评结果 |

固定权重和依赖后在服务器离线评分，避免`latest`或自动下载悄悄改变评分器。新增评分器或改变采样/阈值时提升协议版本，并用同一新协议重算所有对照；不能只重算改进模型。评价代码、权重、受限数据与衍生素材分别保留自己的许可，不因为汇总器开源就获得额外权利。

## 2026-10-21评测优先补齐（先验收，再选强基线）

后续顺序严格固定：评分器/控制校准与冻结面板完成→同集强基线→基线上的方法优化。现阶段暂停生成模型下载、训练和结构搜索。保留原strict8frame ArcFace/CLIP/mouth/原AMT/MUSIQ；不能用我们小CREMA-D身份均值直接减KeyID VIP-200K论文分数或官方排行榜总分。

新增`scripts/gme_text_eval.py`调用固定GME7B原生embedding，GPUFP16/SDPA、最多81floorlinspace帧（现49帧全部）、原prompt instruction `Find frames that match the given text description.`、原归一化embedding dot均值无×100。GME原生processor截断max1800tokens明确记录，模型源码/config/tokenizer/九shards全SHA；仅本地PIL，禁联网下载。原类不用`lm_head.weight`输出投影为正常embedding路径，不表示识别encoder缺权重。第一次pilot因transformers懒加载DeepSpeed→cpuinfo本机file命令被Popen守卫误拦失败；预初始化本机依赖后唯一r002修复通过，没有放开模型联网。

新增`scripts/curricularface_eval.py`用固定ConsisID原生IR_101（实际100层）和公开glint360k权重，CurFP32，det/Arc CPUORT3204/1线程。作者式16floor采样/最大bbox（平面积last tie）/灰边25%pad重检/Cur224RGB再112归一、Arc112crop、cos裁0/validframe均值；完全missing视频0是显式作者约定，缺reference始终null。它与strict8frame missing协议并列，不能替换刷分。`scripts/calibrate_evaluation_panel.py`在64dev原source/32演员独立参考上hash-split16cal16validation，各32正480负，AUC1/EER0，阈值.20226049仅cal冻结，valFAR.008333/TPR1；self误差1.19e-7、重复embedding误差0。小域源控制不是官方FAR保证。

新增`scripts/vbench_extended_eval.py`直调固定原RAFT dynamic_degree、DINO主体一致性、CLIP背景一致性、LAION美学及ViCLIP overall；本地7资产和DINO源码SHA，禁自动init_submodules/URLloader，逐视频equalweight与原帧加权aggregate分开。ViCLIP原promptcontext32的截断逐项记录，不评测时改写prompt；移动视频不套static-only temporal_flickering。

新增`scripts/local_story_eval.py`用已核原Qwen2.5VL7B看8实际时点生成caption、KaLM0.5B原embedding，独立命名`local_story_similarity_v1`，不是作者Gemini StoryScore。13控制证明RAFT能区分明显位移/冻结，GME能区分红球与错蓝方块描述（.673484 vs .404795），但GME倒序完全同分、冻结反而.685463；localStory/ViCLIP也不能稳定分辨倒序/冻结。此能力必须降级为内容诊断，不是动作/时间完成指标；不能用高平滑/高一致性给静止片晋级。没有调用Gemini或外发人脸，没有人工分数造假。

将120已有视频同协议重评：96同480×352同16dev的六方法，24历史480平方三方法单列；不再生成新视频，不用holdout。新计算硬上限12GPUh/白名单下载36GB，原51.5硬上限不增加。`resources/evaluation_panel.lock.json`锁权重/代码/许可与源校准；GME README带`Built with GME`及使用outputs改善公开模型的命名要求，不能仅凭Apache YAML忽略。官方完整scorer、原GeminiStory与人工仍缺，合成总分保持null；全部自动实现通过也不能声称完整官方评测。

### 完整已有视频共评结束与结论

120/120现有视频在新11维自动面板完成（另3coverage列），每项输入/score/video/prompt/reference SHA核验，96同规格六组与24历史三组分层。主集Cur/Arc16各均值：baseline .263468/.239781，lr1 .436314/.422514，lr5 .681828/.671621，alpha.5 .416965/.396216，alpha.25 .286567/.271397，structure0 .677477/.664232；与旧strictArc8的数值不混。GME baseline .554678、lr1 .554295、lr5 .527720、structure0 .531047，揭示高身份训练并未同步改善文本内容匹配。原RAFT dynamic_degree视频比例baseline3/16、lr1 1/16、lr5/alpha.5/structure0均0/16、alpha.25 2/16；这只是明显运动阈值，真实说话源也可判0，不能叫动作完成率。

原MUSIQ/.AMT、DINO主体、CLIP背景、LAION美学与ViCLIP原视频文本cos均另报；身份/成像较高不等于自然动作改善。localStory六组仅约.853–.864、控制中冻结和倒序可更高（motion-vs-freeze−.011056、order−.002358、ViCLIPorder−.007038，GMEorder差0），因此`stable_task_action_evaluator_verified=false`。这部分没有补齐为可靠任务动作评分，禁止以其自动晋级或声称接近top。

19GPU评分作业17completed/2failed均结束，原GME32s失败与worker旧上游删减目录拒绝2s保留，旧workerquality已完成但从最终结果排除并在独立clean fd18源码r002重评；原权重/公式未变。总实际allocatedGPUh=.667025325≤12，含所有失败/排除成本，不按工具延迟通知计时，无新模型生成/训练。完整525passed/7skipped，干净187passed/37skipped、CPU烟测passed，独立CPU复算126groupcells完全一致。结果`resources/evaluation_panel_result.json`。

`scripts/prepare_blind_review.py`已准备本地80对neutral-name比较，身份/动作完成/自然运动/伪影/偏好与置信度分列。所有人工值未填，媒体与method key只在本地，不进Git/外部API；需要实际独立人类审核，agent不能代称已完成人工评测。原GeminiStory/官方human/完整finalscore仍null。下一步先补动作判定与人工边界，再同一冻结panel验证强基线，然后才研究结构/损失；本轮完成的是可复现自动面板和缺陷诊断，不是完整官方评测。

## 人工抽样与同预算对照

先固定按身份及源视频隔离的dev与holdout；holdout不参与agent逐轮选型。人工审核预先抽定的正常样本，再加入固定规则选择的高分、低分和失败样本；后者属于诊断集，应与随机样本结论分开。比较视频时隐藏模型名、随机左右顺序，分别记录身份、动作完成、时间对齐、可见伪影及整体偏好。

至少比较单次生成、固定预算重采样和选择性修正三类策略。固定总GPU时间或明确报告实际时间，计入失败重试、关键帧/候选生成和评分开销；统一分辨率、时长、种子集与样本集合。比较均值时同时报告覆盖率、失败数及逐身份结果；需要置信区间时优先按身份/源视频组重采样，避免把同一身份的大量帧当独立样本。

不要让agent同时调整评分器和被评模型来证明进步；不要只优化一个身份均值而忽略运动退化。候选筛选、重采样和测试参考图适配能否用于正式提交，仍应按主办最终规则核准。本地代理分数、人工开发集偏好和正式比赛成绩分别记录。
