# 自动迭代的实际能力与接入契约

本仓库提供有预算和超时的实验控制框架。`campaign run` 按配置中的有限实验列表执行；**没有内置自主调用 LLM、生成科研假设或无限自我改进的 agent**。外部 coding agent 可以读取这些文档、提出改动、调用 CLI、分析日志并形成下一份配置，但仍受同样的数据、评价和预算约束。

synthetic backend 用 FFmpeg 生成测试花纹 MP4，只用于覆盖控制流程与媒体校验。它不加载视频生成模型，不提供模型质量分数，不参与真实方法晋级。

## 一轮闭环

1. 写一个可被证伪的假设，指定 parent run 和唯一改动。
2. 固定代码、模型、数据清单/划分、输入种子与 proxy 版本。
3. `experiment plan` 检查配置与最大资源占用；不启动 backend。
4. 启动小规模 run；收集真实产物、退出码、峰值/吞吐与模型指标。
5. 先检查文件覆盖、解码和资源，再看质量，不把坏片静默删除。
6. 同预算对照；保留、回退或诊断；写下结论和失败原因。
7. 有效方法通过阶段验收后，再生成下一份有限 campaign。

控制器不是科学结论的裁判。它能保证一些运行约束，但不能自动证明创新、因果归因或官方排名提升。promotion 仍需研究计划中的多维同预算证据和人工抽查。

## 可修改与不可混改

可在独立版本中改变：学习率、LoRA rank、采样步、身份条件强度、数据质量阈值、有限候选数、失败修正动作和局部算法。每次只改一个研究因素，工程必要修复另做 run 记录。

评价器、holdout、官方规则状态、许可条件、资源上限和结论不能随模型一起静默改变。改评价协议时使用新版本，对已冻结对照一起重评；不要覆盖旧结果。最终比赛 test 不用于调参，holdout 不进入普通 campaign 的训练/选型循环。

实验配置的 `purpose` 默认是 `development`；开发实验拒绝holdout，训练只能消费train。`purpose:"final_generation"` 是显式冻结推理入口，仅允许 `task:"inference"`，不能用于训练或放进campaign列表。阶段holdout验收或官方最终生成应使用独立manifest，移除配置中的 `split` 过滤字段，再单独 `experiment run`。配置的split只支持train/dev；不能写 `split:"holdout"` 绕过。记录层schema支持train/dev/holdout，官方测试可用专门无split清单，不能发明 `split:"test"` 值。该入口不授予测试调参权限，结果不回流自动选型。

## 外部 backend 的输入与输出

实验配置的 `backend.kind` 为 `command` 时，`backend.command` 是 **argv 字符串列表**。控制器通过 `shell=False` 启动，不解析管道、重定向、`&&` 或 shell 变量。把环境准备放到已检查的启动脚本或虚拟环境里，参数作为独立数组项传入。

占位符包括 `{request}`、`{output_dir}`、`{repo_root}`、`{run_dir}`，以及当前控制器的Python解释器 `{python}`。命令示意：

```json
{
  "backend": {
    "kind": "command",
    "command": ["python", "/srv/ipvg-backends/my_backend.py", "--request", "{request}"]
  }
}
```

`my_backend.py` 是需要实现或接入的后端，不是本仓库对任意上游模型都已提供的通用脚本。项目自带 VACE adapter 的实际入口、支持模式和限制见 [UPSTREAM_INTEGRATION.md](UPSTREAM_INTEGRATION.md)。

控制器写入 `request.json`，主要契约为：

```json
{
  "schema_version": 1,
  "run_id": "track1_pilot_r001",
  "task": "inference",
  "samples": [{
    "sample_id": "subject001_prompt01",
    "identity_id": "subject001",
    "source_video_id": "source001",
    "reference_images": ["/approved/data/ref001.png"],
    "prompt": "A person turns their head slowly.",
    "track": 1,
    "split": "dev",
    "timeline": []
  }],
  "parameters": {},
  "output_dir": "/project/runs/track1_pilot_r001/outputs"
}
```

媒体路径由控制器按项目根目录规范化。参考图、视频和 prompt 仅用于已获授权的本地运行；request 本身包含受限元信息，留在 ignored `runs/`，不要推入 GitHub。

推理后端为每个 `sample_id` 生成 `output_dir/<sample_id>.mp4`。不得只返回一张缩略图、JSON 指标或空文件充当成功。参数应保留底座本地路径、模型 revision、分辨率、帧数、种子、steps、精度与 offload/分片设置。训练后端要保存 checkpoint 和恢复信息；具体训练产物验收另按上游接入协议执行，不能以推理 MP4 代替训练成功证明。

可另输出 `backend_metrics.json`，用于记录吞吐、峰值、训练步和补充信息；它不自动成为官方评分器，也不作为 controller 的预算账本。真实 proxy 逐条记录使用：

```json
{"sample_id":"subject001_prompt01","scores":{"identity":0.8,"text":0.7},"failure_classes":[],"score_kind":"local_proxy"}
```

此处数字仅解释文件结构，不是实测结果。评分由接入的本地评价器计算，CLI 不会凭视频自动补造 identity/text 数值。`evaluate` 导入 proxy 后汇总，未提供的指标保持缺失。

## 预算、超时与失败

每个实验必须声明 `resources.gpus`、`resources.timeout_s` 与 `resources.max_gpu_hours`。预留最坏占用满足：

```text
max_backend_gpu_hours = gpus × timeout_s / 3600
max_backend_gpu_hours <= max_gpu_hours
```

campaign 有 `max_trials` 和总 `max_gpu_hours`；控制器启动前预留全部实验的 timeout 最大占用之和，超限拒绝启动。列表是有限的，配置 `stop_on_failure` 决定失败后是否停止；不做无限重试。

实际记录 `estimated_backend_gpu_hours = 声明GPU数 × backend墙钟小时`。这不是 NVML 利用率积分，也不是调度器账单；模型加载、同步等待和空闲都占 allocation。评价和调度器外运行的预处理开销要单独计账。若 backend 私自启动未声明 GPU 或远程作业，controller 无法据此推算真实用量，必须禁止这种接法。

超时终止子进程组；Slurm 任务再设置墙钟上限。资源上限应同时覆盖 checkpoint 所需时间。OOM、NaN、缺文件、坏片、评分器失败、清单漂移、容量不足都先诊断；原 run 保留，修复后用新的 `run_id`。重复运行同一 ID 不应覆盖历史结果。

## 留存与审查材料

`runs/<run_id>/` 保存配置、request、选定清单、fingerprint、stdout/stderr、result、输出 hash 与校验结果。外部训练器还要保存自身模型/优化器/RNG状态、步数、实际有效 batch 和导出配置。`report` 汇总实验台账，不自动把一次完成标为研究提升。

fingerprint包含实验配置、原manifest文件hash和 `source_sha256`。后者覆盖本仓库src/scripts下的 `.py/.sh/.sbatch`，因此未提交的这些源码改动也会改变指纹；它不自动覆盖外部upstream代码、模型文件、输入媒体、依赖环境或评价权重，需要另行固定版本和校验和。

`experiment run CONFIG --resume` / `campaign run CONFIG --resume` 只复用同指纹、状态成功的已有实验，并复核冻结配置/选定清单、视频解码/输出hash或声明训练产物hash。它不会继续一个失败的训练，也不自动恢复optimizer/RNG步数。当前VACE训练adapter没有通用的断点续训参数；checkpoint续训需单独接入上游实际支持方式并做GPU验收。失败、配置/源码变更或产物改变后使用新run_id，保留旧证据。

所有真实运行都需要：代码 commit、upstream commit、模型 revision、数据与 split hash、seed、GPU 型号/卡数、帧数/fps/尺寸、训练与采样预算、各维度 proxy、人工检查、失败原因、最终决策。

## API 与数据边界

训练、生成和本地视觉评价可以全部使用服务器 GPU。只有选择外部 coding controller 时才产生该服务的调用；本框架不内置付费模型调用。若使用，记录每轮模型、输入/输出 tokens、成本与总上限，避免把视频 API 成本与 coding API 成本混算。

[VIP 协议](https://github.com/HiDream-ai/ipvg-challenge-2027.github.io/blob/main/Agreement-VIP-200K.pdf)禁止向第三方提供数据。未经兼容授权，原始/可识别人脸、训练视频和相关受限衍生素材保留在获准环境；外部 coding controller 只接代码、脱敏日志和聚合指标。协议签名、账号接受和最终提交均由本人完成。研究范围内可逆的代码/配置改动可继续执行，但不能代签、绕过门禁或把自动生成的结论当成已核事实。
