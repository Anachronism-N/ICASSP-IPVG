# Linux GPU 服务器运行手册

先读 [STATUS.md](STATUS.md) 和 [AUTOMATION.md](AUTOMATION.md)。本手册区分 CPU 控制流程验收与真实 GPU 模型接入；没有目标服务器的实测记录前，不宣称 40GB 卡可以跑完任何给定训练配方。

## 安装与目录

在自己的服务器工作目录 clone 仓库，进入项目根目录。以下命令假设 Bash、Python 3.10/3.11 可用；控制框架最低版本见 pyproject，上游模型环境以其固定 commit 的要求另建，不强行把所有模型塞进同一个环境。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,download]'
mkdir -p runs logs/slurm artifacts submissions
ipvg --help
ipvg doctor --output runs/doctor.json
python -m pytest
```

FFmpeg 与 ffprobe 必须在 PATH 中；synthetic smoke 也生成并验证真实 MP4，需要它们。GPU backend 另需 PyTorch/CUDA、上游依赖与本地模型路径。系统包安装按服务器管理员配置，不把本地控制框架安装成功当作 CUDA 栈可用。

目录建议：`data/raw`、`data/processed`、`cache`、`models`、`runs`、`submissions` 在 Git 忽略范围内；容量较大时使用获准存储上的绝对路径。不要把私有文件放进 docs/examples。不要把 HF Token 写进配置或 shell 脚本；使用 HF 官方登录或当前进程环境，控制日志不要打印完整环境。

## 第一轮：只验收控制流程

```bash
ipvg experiment plan configs/experiments/synthetic_smoke.json --root .
ipvg experiment run configs/experiments/synthetic_smoke.json --root .
ipvg report runs --output artifacts/experiment_report.md
```

synthetic 输出测试花纹，不代表模型生成效果。配置中的 `run_id` 已有结果时，用新 ID 执行新的实验；不要删掉已完成目录来伪装首次运行。smoke 的具体 run 路径以配置/CLI 返回为准。

确认输出可解码、损坏/漏项能够被校验发现、日志与预算写入结果，然后再接真实模型。仓库测试结果必须来自实际执行，不照抄本手册当作通过记录。

## 数据资源：先计划再下载

例如只计划一个 VIP 视频分片：

```bash
ipvg resources plan \
  --repo-id HiDream-ai/VIP-200K-Video \
  --repo-type dataset \
  --revision e5909911c901960e8907919a3f27f789a9620380 \
  --allow-pattern 'video/vip200k_train_0001_of_0100.tar' \
  --max-bytes 4000000000 \
  --output artifacts/vip_shard01_plan.json
ipvg resources download artifacts/vip_shard01_plan.json --target data/raw/vip
```

第二条默认是计划/检查模式，不会实际下载。本人已取得授权、检查容量与清单后，才运行：

```bash
ipvg resources download artifacts/vip_shard01_plan.json \
  --target data/raw/vip --execute --access-approved
```

`--access-approved` 记录你已具备访问许可，不绕过 HF 门禁、不代替条款接受。`--max-bytes` 是所选传输量上限，不包括解压、缓存、模型和 checkpoint。大文件下载应做断点与完整性核查；不把全量 8.86TB ReactID 设成首次资源计划。

## Manifest 与隔离划分

项目标准格式是每行一个 JSON。下例仅示意字段，路径须换成真实且已授权的本地文件：

```json
{"sample_id":"id001_p01","identity_id":"id001","source_video_id":"src001","reference_images":["data/processed/id001_ref.png"],"prompt":"A person turns their head.","track":1,"timeline":[]}
```

Track 2 使用整数 `track:2` 和非空 `timeline`，事件为 `{"start_s":0,"end_s":2,"action":"turns around"}`；`video_path` 为训练目标，可选 `control_video_path` 为控制素材。两者许可与实际输入使用要核准。

```bash
ipvg data validate data/manifests/all.jsonl --root . --check-paths
ipvg data split data/manifests/all.jsonl --output data/manifests/split.jsonl --seed 42
ipvg data validate data/manifests/split.jsonl --root . --check-paths
ipvg data export-vace data/manifests/split.jsonl \
  --output runs/vace_train_metadata.csv --root . --split train --reference-index 0
```

查看校验报告中的 split 泄漏信息；不能只因命令有输出就认定零泄漏。按身份/源视频连通组隔离，核对 train/dev/holdout 的实际规模与场景分布。缓存按清单 hash、预处理版本、VAE/text encoder revision 分目录，避免读取不兼容缓存。

## 真实模型接入与 profile

先运行 `python scripts/bootstrap_upstream.py --name diffsynth --target upstream/DiffSynth-Studio` 获取固定代码；它不安装GPU环境、不下载权重。复制 `configs/experiments/vace_reference.json` 为新配置，填写本地模型/tokenizer、真实manifest、新run_id和资源预算。当前参考推理adapter是单GPU，声明1GPU；不把Slurm分到8卡当作adapter已使用8卡。先plan，再在服务器执行。训练另用 `configs/experiments/vace_lora_pilot.json`。参数见 [UPSTREAM_INTEGRATION.md](UPSTREAM_INTEGRATION.md)。

```bash
ipvg experiment plan configs/experiments/vace_reference.json --root .
```

示例配置不能在未填路径/未安装模型环境时当成已验收的 GPU 命令。VACE adapter 接入上游 API 后仍要在服务器验证；没有真实数据/权重时只做 dry run。

先单节点：较低时空 token、microbatch=1、冻结 base、训练条件模块或 LoRA；预缓存 VAE/text 的前提是缓存代码与训练器兼容。运行 50 步确认梯度，再扩 100—200 步记录峰值与速度。必须记录 warmup 规则、optimizer steps、有效 batch、梯度累积、数据并行/序列并行和具体分片策略。

至少检查：loss 有限、目标参数梯度非零、只有指定层更新、保存/恢复步数与 RNG 一致、导出回载可生成、每卡峰值有余量、CPU/IO 未持续阻塞。先用小批数据验证恢复，不等大训练中断后才测试。

GPU 外部探测示例（Linux，按集群权限使用）：

```bash
nvidia-smi
nvidia-smi topo -m
df -h
ipvg doctor --output runs/doctor_gpu_node.json
```

跨节点 NCCL 测试、IB/RoCE/网络与共享存储测试需采用管理员提供的运行方式。未测前节点之间跑独立实验。DiffSynth 的单 GPU model CPU offload 不能直接与 DDP/DeepSpeed 叠加；按固定版本文档核验。

## 有限 campaign 与 Slurm

```bash
ipvg campaign run configs/campaigns/pilot.json --root .
ipvg report runs --output artifacts/experiment_report.md
```

仓库 `pilot.json` 是CPU合成流程；真实实验应自建有限配置列表和新的run_id，不因Slurm分到了8卡就声称配置在使用8卡。controller声明的GPU数必须与实际backend一致。

重复运行已成功的相同配置可显式加 `--resume` 复用并重新校验产物；源码/配置/manifest改变或失败后要新run_id。这里的resume不是上游训练checkpoint续跑，当前训练adapter没有通用optimizer/RNG恢复接口。

Slurm 模板见 [scripts/slurm/README.md](../scripts/slurm/README.md)。两节点优先一个训练节点、一个生成/评价节点；四节点优先两个独立候选、一个生成、一个评价/机动。数组任务使用不相交样本与独立 run_id，不能让各节点重复生成全部 1000 对。

## 预算校准

```bash
ipvg budget --pairs 1000 --runs 1 --candidates 1 \
  --seconds-per-video 121.6 --gpus-per-job 8 \
  --nodes 2 --gpus-per-node 8 --efficiency .7
```

121.6 秒是上游普通 Wan2.2 I2V 的参考值，不是本项目身份模型实测；其基准帧数/steps 未完整说明。拿到服务器后替换为实际配置的耗时。有效效率 .7 是规划假设。GPU 小时约为 `输入对数×候选数×run数×每条秒数×每任务卡数/3600`；墙钟再除并行 GPU 和有效效率。Track 2 数量未知，不能套 1000。

## 评价与本地提交草稿

使用 run 中冻结的选择清单与 outputs 路径；不要拿原始全量 manifest 去评价只生成 dev 的目录。

```bash
ipvg evaluate runs/RUN_ID/manifest.jsonl runs/RUN_ID/outputs \
  --output artifacts/RUN_ID_media_validation.json
ipvg evaluate runs/RUN_ID/manifest.jsonl runs/RUN_ID/outputs \
  --proxy-scores runs/RUN_ID/proxy_scores.jsonl \
  --output artifacts/RUN_ID_proxy_report.json
ipvg submission runs/RUN_ID/manifest.jsonl runs/RUN_ID/outputs \
  --archive submissions/RUN_ID_draft.zip
```

`RUN_ID` 是需要替换的实际ID，不是环境变量自动展开。没有proxy文件就只做媒体检查；CLI不会自动计算未接入的人脸/动作模型。打包生成本地draft，不执行比赛上传，也不承诺ZIP就是官方最终格式。

冻结的holdout/官方测试生成需另建独立manifest和配置，设置 `purpose:"final_generation"`、`task:"inference"`，移除配置的 `split` 字段，并单独运行 `ipvg experiment run YOUR_FROZEN_CONFIG --root .`。不能放进campaign，也不能写 `split:"holdout"` 或记录层 `split:"test"`；当前记录schema支持holdout或无split的专门测试清单。正式测试的生成/筛选规则须先核准，输出不回流开发选型。确认官方最终规格后，primary完整优先。

## 常见故障

| 现象 | 先检查 | 处理 |
|---|---|---|
| GPU OOM | allocated/reserved、具体帧数/尺寸、双专家/编码器是否常驻 | 降时空 token/冻结层，核对分片；不把 DDP 当参数分片 |
| 很慢/利用率低 | 加载、磁盘解压、CPU offload、数据读取和同步时间 | 缓存/本地 NVMe、单节点测基准；不盲目增节点 |
| 有 loss 但模型没变化 | trainable 参数、梯度、optimizer、导出/加载路径 | 小样例验证目标层权重改变与回载效果 |
| backend 成功但缺片 | sample ID、目录、异常捕获、输出规格 | result 标失败，留日志；修复后新 run_id |
| proxy 提升但更静止 | 运动、动作和身份多维表、盲查 | 回退偏置配置，不单独优化身份相似度 |
| 下载401/403 | 当前账号授权、Token作用域、条款状态 | 本人完成正常授权，不绕过门禁 |
| 同一 run_id 被拒绝 | 已存在结果/fingerprint | 用新 ID；不要覆盖历史 |

实际服务器环境、测量和完成步骤逐项更新 [STATUS.md](STATUS.md)。
