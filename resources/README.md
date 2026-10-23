# 公开来源与版本

`upstream.lock.json`固定代码commit与已核查模型revision；`research_evidence.json`保存2026年10月17日的公开调研证据。仓库不内置受限数据、权重或第三方代码副本。

模型及数据revision来自只读元数据核查，不代表账号已获访问权。下载规划须显式给文件白名单与字节上限。`tokenizer`也必须单独下载并记录revision，不能认为VACE权重仓库包含全部tokenizer文件。

入口适配依据：

- [DiffSynth VACE参考图推理](https://github.com/modelscope/DiffSynth-Studio/blob/4e3496991455b300fd60ab023113422cd16899f5/examples/wanvideo/model_inference/Wan2.1-VACE-1.3B.py)
- [固定版本训练参数](https://github.com/modelscope/DiffSynth-Studio/blob/4e3496991455b300fd60ab023113422cd16899f5/diffsynth/diffusion/parsers.py)
- [固定版本训练入口](https://github.com/modelscope/DiffSynth-Studio/blob/4e3496991455b300fd60ab023113422cd16899f5/examples/wanvideo/model_training/train.py)
- [VACE LoRA回载示例](https://github.com/modelscope/DiffSynth-Studio/blob/4e3496991455b300fd60ab023113422cd16899f5/examples/wanvideo/model_training/validate_lora/Wan2.1-VACE-14B.py)
- [IPVG官方规则](https://hidream-ai.github.io/ipvg-challenge-2027.github.io/)

`real_vace_training_result.json`记录2026-10-20两节点真实dense-caption LoRA训练及同规格48dev输出聚合结果，包含checkpoint/source/split哈希、成本、身份收益与运动取舍，不含真实媒体、逐条caption或内部节点。复现设置见八份`configs/experiments/real_vace_*.json`及`docs/EXPERIMENT_DUALNODE_TRAINING.md`。

`standin.lock.json`固定原生Stand-In/T2V14B六分片、IP分支checkpoint、复用T5/VAE/tokenizer、同SHA检测器与官方BiSeNet预处理来源；`native_baselines_result.json`记录2026-10-22三原生模型共同16输入/48新主输出及3pilot的完整核心/辅助对照、脸部缺失与实际成本，不预设参数更多就效果更好。复现设置为六份`configs/experiments/native_{vace,anyid,standin}_{pilot,main}.json`，具体方案/结果见`docs/EXPERIMENT_TWO_TRACKS.md`末节。

`face_visibility_diagnostic_result.json`记录2026-10-23旧48主输出的全49帧/320与640双尺度CPU检测聚合与成本；不含逐视频标识、框、媒体路径或原参考。检测敏感性不是身份/人工可见性评分，旧Cur/Arc分数不变；`action_baselines_result.json`记录额外8dev演员/四固定动作模板各4输入的三原生系统48main+3pilot及八共同评分完整结果；含分动作coverage/完全missing、核心配对差、原生条件证据和实际.955013104GPUh预算，不含媒体或逐身份行。复现冻结入口`scripts/prepare_native_action_benchmark.py`，原生配置复用六份native模板，具体方案与真实正负结论见`docs/EXPERIMENT_TWO_TRACKS.md`末节，未宣布统一winner或动作真值。

VACE/Wan/DiffSynth的许可不覆盖数据访问协议或额外人脸权重。AnyID、Hunyuan等候选另有不同限制，见证据文件和对应上游许可。原创工程未替用户选择开源许可证。
