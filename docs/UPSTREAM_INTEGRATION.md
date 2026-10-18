# 上游模型接入与验收

本项目将实验控制与真实模型分开。CPU synthetic smoke用于检验控制流程，不能代替GPU backend验收。VACE1.3B小尺寸单卡/7卡LoRA训练、导出和回载，以及64条真实参考图生成已在A10040GB验收；AnyID over Wan2.2 TI2V-5B已完成8条单参考推理。Track2视频latent窗口原型已跑12条，但未显示事件方向增益。具体参数、成本及未验证项见 [双赛道报告](EXPERIMENT_TWO_TRACKS.md)，不能据此推断长训练、14B、默认长视频或正式赛道已通过。

本文依据 2026-10-17 的已有源码/模型卡核查，实际服务器必须使用固定 commit 与 revision。完整参数应以该固定版本的源码和 `--help` 为准；未知项写为 TODO，不发明 upstream CLI。

## 推荐顺序与公开程度

| 模型/框架 | 用途 | 已公开部分 | 尚需本项目实现/验收 |
|---|---|---|---|
| Wan2.1 VACE 1.3B + DiffSynth | 首个训练与推理闭环 | 基座/条件权重；DiffSynth 条件模块训练、推理 | manifest/参考图接线，子集训练、恢复、导出回载与真实 proxy |
| ConsisID | Track 1 直接身份训练对照 | 预处理、train.py、单/多节点脚本、推理/评测、权重 | 适配本项目数据、清理作者路径、40GB峰值和许可检查 |
| Stand-In | 强身份推理与 Wan2.1 VACE 对照 | 身份 adapter、推理、Wan2.1 VACE 集成 | 训练仍 TODO；Wan2.2+VACE 不能当作者已验证组合 |
| AnyID | Wan2.2 5B 身份推理对照 | preview LoRA、CPU合并及8条480²/49帧单参考推理已验收 | 未核到官方训练/RL recipe；缺脸覆盖、广动作、默认长视频与原生采样器成本仍需确认 |
| Phantom 1.3B/14B | 人/动物/物体主体泛化对照 | 权重与推理 | 训练仍 TODO；时间线不是原生输入 |
| Wan2.2 I2V A14B + DiffSynth | 后续质量/容量升级 | 基座推理与框架训练能力 | 高/低噪声 LoRA、身份条件、分片和40GB适配 |
| ReactID | Track 2 研究参考 | 论文与门禁数据 | 当前未核到可直接下载的官方代码/权重，不作为唯一运行前提 |

首轮只维护一套训练主栈和一个身份对照。真实 GPU adapter 的当前能力/测试证据以 [STATUS.md](STATUS.md) 和脚本说明为准。

## 固定版本

| 项目 | commit / revision |
|---|---|
| [VACE](https://github.com/ali-vilab/VACE) | `48eb44f1c4be87cc65a98bff985a26976841e9f3` |
| [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) | `4e3496991455b300fd60ab023113422cd16899f5` |
| [ConsisID](https://github.com/PKU-YuanGroup/ConsisID) | `60429e4b0d24d36ce8df7fb58f7ab42b86e2b70e` |
| [Stand-In](https://github.com/WeChatCV/Stand-In) | `e351224366be169076e94af1454115d91d458313` |
| [AnyID](https://github.com/JoHnneyWang/AnyID) | 当前8条试跑使用 `6c714f71af6e08d2af29aecd74a3a9fd9e8f94a3`；初始调研版本 `239fa15a401ff4f1abcbd90ead4a57d5505e87e2` |
| [Phantom](https://github.com/Phantom-video/Phantom) | `bd84b602dcc949e23c89cbbf266b6f5975f2f025` |
| [Wan2.2](https://github.com/Wan-Video/Wan2.2) | `1ea34ff48f87168174e12956e200b1d908b1c5ff` |
| [Wan2.1 VACE 1.3B 权重](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B) | `574e6a744642ce3bee319afc31496b88bde8aac4` |
| [Wan2.1 VACE 14B 权重](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B) | `539c162b1387eac9dc4c20bd3f74671309e76a4c` |
| [ConsisID-preview](https://huggingface.co/BestWishYsh/ConsisID-preview) | `950bc3f0902db44799e223a12ad972f9c52b341d` |

更新版本时创建新实验，不静默改动原台账。依赖应按各 upstream 建独立 venv/容器并保存 `pip freeze`/镜像 digest；控制器可以调用对应环境中的 Python，不必和 PyTorch 共用环境。

仅代码准备示意（在 Linux 仓库根目录；路径处于 ignored upstream 中）：

```bash
git clone --filter=blob:none https://github.com/modelscope/DiffSynth-Studio.git upstream/DiffSynth-Studio
git -C upstream/DiffSynth-Studio checkout 4e3496991455b300fd60ab023113422cd16899f5
```

模型下载应单独用固定 revision 和白名单 plan。不要在服务器上直接执行 upstream 的示例数据下载脚本后假定已符合本届数据许可与版本要求。

## 本仓库 VACE adapter 的精确入口

先获取固定代码（不安装 PyTorch，不下载权重）：

```bash
python scripts/bootstrap_upstream.py --name diffsynth --target upstream/DiffSynth-Studio
```

配置 `configs/experiments/vace_reference.json` 接到：

```text
python scripts/adapters/diffsynth_vace.py --request REQUEST_JSON
```

command backend中将 `REQUEST_JSON` 写为 `{request}`。实际Python应为装好GPU依赖的环境；路径含空格时作为argv单项传入，不拼shell字符串。

`parameters` 需要填写 `upstream_dir`、`model_files`、`tokenizer_dir`。`model_files` 是本地模型文件列表，transformer多分片可作为其中一个nested list；1.3B单文件、T5和VAE应与固定revision一致。示意（绝对路径须替换）：

```json
{
  "upstream_dir": "/project/upstream/DiffSynth-Studio",
  "model_files": [
    "/models/vace13/diffusion_pytorch_model.safetensors",
    "/models/vace13/models_t5_umt5-xxl-enc-bf16.pth",
    "/models/vace13/Wan2.1_VAE.pth"
  ],
  "tokenizer_dir": "/models/verified_tokenizer",
  "width": 832,
  "height": 480,
  "num_frames": 49,
  "fps": 16,
  "num_inference_steps": 30,
  "cfg_scale": 5,
  "seed": 42,
  "reference_index": 0
}
```

tokenizer须按固定上游需要单独准备，不假设三个大权重文件已经包含它。可选 `lora_path` 或 `adapter_checkpoint` 按实际导出类型填写。

**当前参考推理是单进程、单GPU参考图生成，声明 `resources.gpus:1`。** 不支持已验证的多GPU参数分片。Track 2只有显式 `timeline_mode:"flatten"` 才将事件转为普通文本；这不提供结构化时间控制，不能宣称H3已实现。参考图能进入pipeline不等于专用face-ID编码器或身份loss已经接通。

训练配置 `configs/experiments/vace_lora_pilot.json` 接到：

```text
python scripts/adapters/diffsynth_train.py --request REQUEST_JSON
```

它将选定train样本导为run目录内CSV，再调用固定上游 `train.py`，checkpoint写入本run的output目录。核心参数为同样的 `upstream_dir/model_files/tokenizer_dir`，以及 `mode:"lora"` 或 `mode:"adapter"`、`num_processes:8`、`num_epochs:1`、`dataset_repeat:1`、`learning_rate:5e-5`、`lora_rank:16`、`save_steps:50`、`gradient_accumulation_steps:1`、`seed:42`。这些是起跑参数，不是40GB可运行或收敛保证。

首次只用极小集和小epoch/repeat，核对真实优化步数。8进程CUDA依赖、梯度同步、恢复/导出与真实生成尚未在目标GPU实测。按下面验收表产生证据后才扩大。

## VACE 1.3B：已知训练入口与改造点

已核上游文件：

- [`examples/wanvideo/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/4e3496991455b300fd60ab023113422cd16899f5/examples/wanvideo/model_training/train.py)
- [`full/Wan2.1-VACE-1.3B.sh`](https://github.com/modelscope/DiffSynth-Studio/blob/4e3496991455b300fd60ab023113422cd16899f5/examples/wanvideo/model_training/full/Wan2.1-VACE-1.3B.sh)
- [`lora/Wan2.1-VACE-14B.sh`](https://github.com/modelscope/DiffSynth-Studio/blob/4e3496991455b300fd60ab023113422cd16899f5/examples/wanvideo/model_training/lora/Wan2.1-VACE-14B.sh)

1.3B示例使用 `accelerate launch .../train.py`，含 `--dataset_base_path`、`--dataset_metadata_path`、`--data_file_keys "video,vace_video,vace_reference_image"`、480×832、49帧、`--trainable_models "vace"` 和 `--extra_inputs "vace_video,vace_reference_image"`。本仓库已实现JSONL到上游CSV的转换与训练adapter；参考图/控制视频的语义接线、梯度同步、恢复和实际训练仍需GPU验收。

接入时逐项完成：

1. 检查已实现的CSV导出结果是否符合该commit的dataset loader；验证 `video`、`vace_video`、`vace_reference_image` 在训练阶段的语义，不把验证目标视频当推理控制输入。
2. 先按原始公开例子验证条件模块与本地模型加载，再换成已授权的极小子集。
3. 检查模型 ID 自动下载逻辑，改成固定本地路径/已核 revision，避免训练过程中从 main 意外拉取新权重。
4. 单节点 50 步检查梯度，100—200 步 profiling；确认 trainable 参数和 checkpoint prefix 与加载接口匹配。
5. 保存、恢复、导出、回载、生成同一输入；对照未训练输出确认模型确实发生受控变化。
6. 才扩大样本、时空 token 和训练步数；单次改动保留 parent run。

14B LoRA 示例是 480×832、17帧、rank32；不能直接当49/81帧的40GB训练保证。1.3B full 路径训练的是 `vace` 控制模块，不自动等于整个模型全参。

TODO：GPU 服务器实际峰值、稳定 per-step 时间、有效 batch、可恢复 checkpoint、训练后权重回载、身份分数与运动权衡；这些必须由 pilot 产生证据。

当前reference-only推理遇到`control_video_path`会直接拒绝，不会静默忽略控制视频；训练CSV可以包含该列，但控制视频推理需另接并验收。训练上游默认按源视频帧率读取前N帧，当前wrapper没有统一重采样；`fps`只影响推理视频保存。应先在受控预处理阶段统一训练片段帧率并记录转换，避免动作速度变化。49帧保存为16fps的视频容器时长约3.06秒；时间戳控制不能由这个帧数设置推断为已实现。

## 分片与 offload

DiffSynth 提供 DeepSpeed 等训练集成，但不等于所有 Wan 模型都已在 8×40GB 上验证。特别是：

- 单 GPU `enable_model_cpu_offload` 的 offload training 路径不能直接叠 DDP/DeepSpeed。
- ZeRO-2/DDP 不会自动分片冻结基础权重；显存问题不能靠仅增加 DDP 进程解决。
- ZeRO-3/offload 与 activation checkpointing 的组合要用维护者支持的配置，先检查梯度同步和恢复。
- 8卡推理能装下不等于反向传播能装下；还要计激活、编码器、VAE、优化器状态与通信缓冲。

参见固定版本的 [DeepSpeed说明](https://github.com/modelscope/DiffSynth-Studio/blob/4e3496991455b300fd60ab023113422cd16899f5/docs/en/Training/DeepSpeed.md)与 [Offload Training说明](https://github.com/modelscope/DiffSynth-Studio/blob/4e3496991455b300fd60ab023113422cd16899f5/docs/en/Training/Offload_Training.md)。网络较弱时优先不同节点跑独立实验；多节点模板不是已经验证的训练配方。

## ConsisID 对照

公开 `train.py`、`train_single_rank.sh`、`train_multi_rank.sh` 和 Accelerate/DeepSpeed 配置。多节点样例是2节点16进程、ZeRO2、bf16、gradient checkpointing、720×480×49帧、每卡batch1，但包含作者环境路径/网络配置，必须修改后才能用于自己的集群。

先使用本地 checkpoint 和小数据，核对人脸预处理、关键点/脸区条件、训练目标、模型 license 与导出格式。上游默认推理的37GB allocated/44GB reserved或offload推理数字，不是训练显存预算。

## 自定义外部命令 backend 验收表

| 环节 | 必须通过 |
|---|---|
| 输入 | 解析 schema_version/request；参考图、prompt、timeline、模型路径真正进入正确条件分支 |
| 输出 | 每 sample_id 独立 MP4；没有隐式改变输入信息、偷偷使用目标视频或丢弃难例 |
| 状态 | 子进程错误返回非零；checkpoint/恢复完整；无未声明远程作业 |
| 资源 | GPU 数、timeout 与实际作业一致；真实峰值/速度另记 |
| 评价 | 解码/覆盖通过；proxy来自实际评价器且版本锁定 |
| 重现 | 固定种子/配置/权重，复跑差异有解释；并行不会覆盖文件 |
| 合规 | 仅已授权素材；数据、基础权重、face encoder及附属模型分别符合条款 |

不要给仅推理公开的 Stand-In、AnyID、Phantom 编造训练 CLI，也不要把模型输出的“好看”当身份约束接线正确。实际接入后在 STATUS 中附证据再变更完成状态。
