# 首版软件验证记录

验证日期：2026-10-17，北京时间。开发环境：Windows、Python 3.9.18、FFmpeg与ffprobe；没有可用的PyTorch/GPU训练环境。

| 检查 | 结果 | 范围 |
|---|---|---|
| Python测试 | 71通过、1跳过 | 数据与连通隔离、下载预算/门禁、训练/开发集保护、预算、失败与resume、媒体检查、proxy导入、打包、上游argv |
| 跳过项 | Windows不能创建测试所需symlink | 不是GPU测试；Linux CI可执行该路径 |
| CPU端到端烟测 | 通过 | 生成3条64×64、8fps、1秒测试花纹MP4；解码校验、哈希、日志、台账、本地draft ZIP |
| GPU适配器帮助与语法 | 通过 | argparse入口与Python编译；不代表依赖安装、模型加载或GPU运行 |
| Slurm模板 | Bash语法通过 | 没有向真实Slurm提交作业 |
| 文档CLI与链接 | 26条示例参数解析、本地链接和围栏检查通过 | 路径占位符仍须在服务器填写 |

本机第三方pytest插件可能拖慢解释器退出；隔离插件检查使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`。仓库测试不依赖第三方pytest插件。干净CI环境使用普通pytest命令。

已回归覆盖一个GPU入口缺陷：训练子进程在上游Accelerator初始化前，用 `seed + RANK` 和 `device_specific=False` 设种子，避免提前访问未初始化的AcceleratorState。此处测试使用受控模块替身；真实多进程训练仍待验收。

没有验证：CUDA/PyTorch组合、40GB峰值、多卡梯度同步、完整训练断点状态恢复、真实身份生成质量、外部proxy模型、本届官方分数或最终提交格式。

CPU测试不下载大权重、受限数据，不调用付费模型API。生成素材与机器路径保存在ignored `runs/`，不随公开仓库上传。
