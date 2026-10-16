# Slurm 作业模板

这些文件是 Linux 集群的调度起点，不是已在目标集群验证过的 GPU 配方。按实际分区、账号、GPU GRES 语法、CPU/内存和墙钟限制调整。`#SBATCH` 行不展开 shell 变量；可用 `sbatch` 参数覆盖。不要同时请求 `--gpus-per-node=8` 和等价 `--gres=gpu:8`，按集群支持方式选一套。

先在仓库根目录创建 `logs/slurm`，再提交；Slurm 在脚本启动前打开日志，脚本内 mkdir 无法补救不存在的日志目录。示例环境：

```bash
mkdir -p logs/slurm
export IPVG_ROOT="$PWD"
export IPVG_PYTHON="$PWD/.venv/bin/python"
sbatch scripts/slurm/00_control_smoke.sbatch
```

`00_control_smoke` 只用 CPU，生成测试花纹并验收控制流程。默认配置的 run_id 已存在时会拒绝覆盖；为新的实验复制配置并设置新 ID，再通过 `IPVG_CONFIG` 指定。

## 单节点开发 campaign

准备自己的有限真实backend campaign；其中每个实验声明GPU数、timeout和GPU小时上限。仓库 `configs/campaigns/pilot.json` 是CPU合成smoke。最终冻结生成使用 `purpose:"final_generation"` 的独立experiment，不放入campaign。

```bash
export IPVG_CAMPAIGN=/absolute/path/to/real_campaign.json
sbatch --partition=YOUR_GPU_PARTITION scripts/slurm/10_single_node_campaign.sbatch
```

将 `YOUR_GPU_PARTITION` 换成真实分区。模板为1节点8卡、6小时；若controller内所有实验timeout之和可能超过6小时，先缩减列表或显式调整Slurm上限与预算。一个controller调用一个backend；backend若用torchrun/Accelerate自行启动8进程，不要再用 `srun -n8` 复制controller。

2节点时先分别调度一个训练campaign与一个生成/评价作业；4节点时再加独立单因素训练和机动评价。controller 记录“声明GPU×backend墙钟”，调度器可能计整个8卡allocation及加载/等待；两套账要保留。

## 独立推理数组

创建纯文本列表，每行一个实验配置路径，不含空行/注释。每个配置有独立run_id，manifest样本互不重叠，模型与最终生成参数相同。将列表放在ignored `runs/` 下；不要把受限prompt/参考图放入公开配置。

```bash
export IPVG_CONFIG_LIST=/absolute/path/to/inference_configs.txt
sbatch --array=0-1%2 scripts/slurm/30_inference_array.sbatch
# 四个独立分片、最多四个作业并行：
sbatch --array=0-3%4 scripts/slurm/30_inference_array.sbatch
```

当前VACE参考推理是单GPU，因此模板每个array任务申请1GPU、64GB主存；Slurm可以把多个array任务放在同一物理节点，并不保证每任务独占不同节点。根据真实CPU/offload内存测量调整并发；不要为“用满16/32卡”超额申请主存。

未来若某外部backend已通过8GPU分片推理验收，可显式覆盖 `--gpus-per-node=8 --mem=192G` 并同步修改实验声明；目前自带参考adapter不具备这个能力。数组并行不是多节点同步训练，不把重复生成同一输入对误算吞吐。最终合并须验证ID唯一、覆盖完整、hash和配置一致；不能只平均多个分片分数。

## 多节点上游 profile 骨架

`20_multinode_upstream_profile.sbatch` 默认2节点×8卡、30分钟，最多8个allocation GPU小时；不带可直接运行的通用训练命令。先测网络/NCCL、共享存储、trainer分片与恢复，再指定部署者编写并验证的可执行launcher：

```bash
export IPVG_UPSTREAM_LAUNCHER=/absolute/path/to/verified_upstream_launcher.sh
sbatch scripts/slurm/20_multinode_upstream_profile.sbatch
```

该launcher须读取 `SLURM_NODEID`、`MASTER_ADDR`、`MASTER_PORT`、`IPVG_NNODES` 和 `IPVG_GPUS_PER_NODE`，每节点只启动一次8个本地worker，并使用真实upstream支持的训练参数/分布式配置。模板未给 `train.py --invented-flag` 一类占位命令，也未验证任何多节点训练兼容性。

本骨架绕过本仓库controller，仅用于人工受控profile；因此必须自行记录scheduler GPU小时、训练日志、checkpoint与退出状态，不能误认为已进入campaign预算或台账。正式自动campaign应把已验收的训练launcher接入command backend并明确资源计账。

## 故障与证据

`squeue`/`sacct` 的用法、GPU账户计费以本集群为准。保留job ID、节点名、具体卡型、allocation、峰值显存、吞吐、stderr和checkpoint。OOM或NCCL错误先缩到单节点诊断，不自动重复提交到成功为止。

所有模板尚未在用户GPU服务器提交执行。语法检查不等于调度器参数兼容，也不等于模型运行通过。
