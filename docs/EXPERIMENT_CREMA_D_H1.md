# CREMA-D 本地身份评价与 H1 参考裁剪实验

## 范围与预注册

本实验是非商业、本地研究的辅助说话头部域对照，不是官方 IPVG score，不承诺 SOTA 或广动作泛化。真实媒体、参考图、embedding、逐身份日志及账号凭据不发布。源数据 CREMA-D 官方固定 commit `1658cd342dff90010aa843eaeebd53610a08b1dc`，数据库 ODbL1.0、内容 DbCL1.0；版权、人格权和对外公开义务分别审查。

已完成数据基线：64名编号演员，每人独立参考视频和两个目标视频，目标 neutral/happy；16train/32dev/16holdout演员。仅32dev演员64对输入用于本实验，train及holdout不用于评分或调参。参考来自另一utterance/source clip，目标视频绝不传给生成器。提示词为保守标签模板，不是dense caption或音频字幕。

## 冻结身份协议

- 身份编码器是官方 buffalo_l 的检测/识别两ONNX，版本/hash及权重非商业研究许可见 `resources/identity_encoder.lock.json`；离线CPU，不自动下载，不使用属性或姓名识别。
- 每视频固定均匀8帧；恰好一脸才评分，多脸/缺脸记录失败而非选择最相似脸。五点对齐到112像素，单位化ArcFace余弦可以为负；缺失相似度省略，覆盖率另报。
- 原始参考图始终是评价anchor；裁剪仅用于生成条件，不能改评价器输入获得虚假收益。
- dev演员固定hash分16校准/16校验。只真实源视频选择阈值；每组32个正对、最多480个跨演员负对，共享演员的pair不能视为独立样本。校准经验FAR≤1%的阈值原样用于校验组，需正对≥28、演员≥14、每视频有效帧≥7/8、AUC演员bootstrap95%下界>.90、EER≤.10、校验TPR≥.80且经验FAR≤.01。失败停止H1，不降低门槛；低FAR小样本不能视为认证。

## 唯一 H1 变化与资源

基线为 VACE1.3B、480×480、49帧、16fps、20steps、固定sample_id/seed/prompt，每输入一个候选，无训练LoRA。唯一变化是原参考脸bbox各边扩0.3后的正方形裁剪；保持来源和其余参数不变，不改写prompt、不增加候选。

先8对pilot，通过媒体/接线后两节点各28对。新生成预算独立上限5GPU小时：pilot1、两批各1.5、最多一次诊断修复1；运行超时与失败留存，不无限重试。CPU评价累计上限6小时，原基线重评分而不重复生成。

## 统计与晋级

配对共同有效帧≥7/8，至少56样本/28演员，coverage变化不低于-.05；固定种子2000次演员bootstrap。只有身份均值delta置信区间下界>0且P10不退化才能称该开发集支持H1。独立报告灰度运动proxy；运动均值比<.8或静态比例（采样灰度差<.5）上升>.10则不晋级。灰度proxy不等价动作/文本符合，单种子结果不算完整论文证据。

## 已冻结校准结果

两组均32正对、480负对、16演员；AUC=1.0，EER=0，演员bootstrap95%区间约[1,1]。校准阈值0.183504，校验经验FAR=0.008333、TPR=1.0；自相似最大误差1.19e-7。仅说明此域源视频身份分离可用，不证明生成结果保持身份。

## H1 执行结果

待生成、身份重评、配对统计和资源审查完成后填入聚合结果；不将计划表述为已完成，不发布逐身份媒体。

## 复现入口

在独立评价环境安装固定runtime和本项目包，手动下载并验证锁定权重，运行 `python scripts/identity_eval.py calibrate --manifest ORIGINAL_MANIFEST --model-dir LOCAL_MODELS --output CALIBRATION_JSON`。校准通过后执行 `scripts/prepare_reference_crop.py` 创建新条件清单；评价时仍传原始anchor清单。

使用已有 `ipvg experiment plan/run` 配置本地后端；每次新run_id，记录模型revision、源码/data/split/hash、seed、GPU卡数、超时与实际成本。逐视频 `local_proxy` JSONL通过现有 `ipvg evaluate --proxy-scores` 汇总。完整本地记录不进入Git，发布仅代码、协议和聚合结论。
