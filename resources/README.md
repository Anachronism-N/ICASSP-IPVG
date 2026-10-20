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

VACE/Wan/DiffSynth的许可不覆盖数据访问协议或额外人脸权重。AnyID、Hunyuan等候选另有不同限制，见证据文件和对应上游许可。原创工程未替用户选择开源许可证。
