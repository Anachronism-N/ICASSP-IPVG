# ICASSP IPVG

ICASSP 2027 身份保持视频生成比赛工程。优先完成 Track 1 的可靠提交，并验证 Track 2 的动作时间线研究。已确认资源为2节点、每节点8张A100 40GB，共16张；互联与可用时长仍需服务器确认。

当前版本提供**实验管理工具、CPU烟测、已小规模GPU验收的VACE训练/推理适配器，以及AnyID身份专用推理与视频时间窗原型**。功能验收与生成质量分开记录，见实验报告。不含比赛数据、模型权重、完整官方评分器或已经训练好的参赛模型。`campaign`执行有限的配置列表；可供coding agent调用，目前没有内置付费LLM调用或自主科研模型。

## 从这里开始

| 目的 | 文档 |
|---|---|
| 阶段目标、算力安排、验收门槛 | [项目方案](docs/PROJECT_PLAN.md) |
| 方法假设、同预算实验、论文路线 | [研究与消融](docs/RESEARCH_PLAN.md) |
| MM2026头部方案、身份基线与ITA视频模型迁移 | [相关工作与迁移方案](docs/RELATED_WORK_AND_TRANSFER.md) |
| AnyID5B身份对照、视频时间窗探针、成本及真实负结果 | [双赛道实验报告](docs/EXPERIMENT_TWO_TRACKS.md) |
| 本地身份评分校准与参考裁剪消融 | [CREMA-D H1实验](docs/EXPERIMENT_CREMA_D_H1.md) |
| 服务器安装、数据准备、运行命令 | [服务器手册](docs/SERVER_RUNBOOK.md) |
| 模型安装、输入契约、训练回载 | [上游模型接入](docs/UPSTREAM_INTEGRATION.md) |
| AI迭代职责、预算和实验决策 | [自动化协议](docs/AUTOMATION.md) |
| 本地评价与人工抽查 | [评价协议](docs/METRICS.md) |
| 规则、授权、数据规模、未确认事项 | [比赛与数据](docs/RULES_AND_DATA.md) |
| 已验证与待完成事项 | [项目状态](docs/STATUS.md) |

## CPU快速验证

基础工具仅依赖Python标准库，推荐服务器使用Python 3.10或3.11。FFmpeg与ffprobe须在PATH中；GPU依赖在独立环境安装。

```bash
git clone https://github.com/Anachronism-N/ICASSP-IPVG.git
cd ICASSP-IPVG
python -m venv .venv
# Linux
source .venv/bin/activate
# Windows使用 .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/smoke.py
```

烟测生成3条真实测试花纹MP4，经过解码检查、配置/源代码/输出哈希记录，打包本地草稿。它不读取示例路径中的人脸，也不产生身份或文本质量分数。所有运行产物保存在被Git忽略的`runs/`。

```bash
ipvg doctor --output runs/doctor.json
ipvg data validate examples/synthetic_manifest.jsonl
ipvg experiment plan configs/experiments/synthetic_smoke.json
ipvg campaign run configs/campaigns/pilot.json
ipvg report runs --output runs/REPORT.md
```

相同`run_id`不覆盖旧实验。仅未改配置、源代码和manifest且完整产物校验通过的成功实验可`--resume`；失败或改动后使用新的`run_id`。视频打包只创建本地草稿，不提交到比赛。

## 真实模型起跑

1. 本人完成比赛报名、数据访问授权，确认本届规则。训练数据和参考图存放在`data/`或获准挂载盘。
2. `ipvg data split`按身份与源视频连通关系隔离train/dev/holdout。先准备小开发集和少量训练片段。
3. `python scripts/bootstrap_upstream.py --name diffsynth --target upstream/DiffSynth-Studio`获取固定代码。在独立GPU环境安装其依赖；不从本机CUDA猜服务器版本。
4. 使用`resources/`的revision与文件清单获取本地权重和tokenizer；下载工具先规划、再显式执行。访问门禁由HF本人账号验证。
5. 修改`configs/experiments/vace_reference.json`与`vace_lora_pilot.json`中的实际路径和预算。先`experiment plan`，然后在分配的GPU作业中`experiment run`。
6. 在小规模试跑后记录峰值显存、秒每视频、训练步速、checkpoint回载和开发集结果，再扩展模型与数据。

当前VACE推理适配器是**单进程单GPU参考图基线**，没有实现8卡分片推理。训练适配器使用上游Accelerate DDP；1.3B小尺寸单卡及7卡训练/导出/回载已在A10040GB试跑，不能当作14B或长训练显存保证。AnyID preview使用Wan2.2 TI2V-5B与独立合并LoRA，已完成8条身份基线。Track2新增视频latent窗口guidance及等调用flatten/swapped对照，12条探针未显示增益；时间/动作问题仍未解决。

## 目录

```text
configs/       可版本化的实验与有限campaign配置
docs/          中文方案、运行与验收文档
examples/      无真实身份数据的manifest/评分格式样例
resources/     上游commit、模型revision、公开来源证据
scripts/       烟测、固定上游安装、GPU适配器、Slurm模板
src/ipvg/      数据、预算、实验、评价与草稿打包CLI
tests/         CPU逻辑、失败路径与集成测试
```

原始数据、权重、受限生成内容、token、日志和提交包不进入Git。外部coding控制器只接收代码及脱敏/聚合结果。GitHub为公开仓库，不应上传受限素材。

## 比赛时间与论文

官网列出测试集2026-12-09发布、结果2026-12-16截止、论文2027-01-07截止；具体时刻/时区和论文出版类别仍待明确。ICASSP的CCF B主会等级不自动赋予所有challenge论文Full/Regular资格。[比赛官网](https://hidream-ai.github.io/ipvg-challenge-2027.github.io/)

原创工程尚未指定开源许可证；各上游代码、模型和数据保留各自许可，见[来源与许可](resources/README.md)。
