# 60 Qwen3-VL-32B 标签 pilot 新集群一次恢复

> 2026-11-03。仅标签重建阶段（label_rebuild_and_cache）的一次获批恢复；不是正式训练、不是 Teacher 质量成绩。

## 1. 第一次 32B pilot 的真实结果（旧集群，保留不改）

- 32 例固定病例（16 旧诊断 + 16 盲抽，seed 20261102），每例 16 帧首尾均匀源帧，BF16，4 卡，cap 1800s。
- 实际：第 0 例产出结构合格标签；第 1、2 例单例校验失败（当时错误信息统一隐去，**无法事后判断具体原因，不猜测**）；
  第 3 例前 "最坏单例耗时 × 剩余例数" 预测超出 1800s 剩余窗口，worker 主动停止，其余 29 例未执行。
- 费用：实测 0.318432 GPUh，按 cap 保守计 2 GPUh；标签阶段累计上界 2.165898 GPUh。不退款、不改写。

## 2. 恢复授权与新集群冻结

- 用户批准一次 6 GPUh 恢复：同 32 例、同 16 帧、1 节点 4 卡 BF16、墙钟 5400s；累计标签上界 8.165898 GPUh ≤ 阶段 24 GPUh。
- 迁移不重置恢复次数；旧 host / GPU UUID / 路径不作为新集群执行凭证。新集群重新冻结：
  - 模型：新集群本地副本，24 个被加载文件逐个 SHA256 与旧集群冻结记录一致（内容摘要相同）；隐藏的非加载条目单独记录、不加载。
  - 运行时：新 venv，torch 2.7.1 / transformers 4.57.1 / accelerate 1.10.1 等与旧集群同版本；新增 torchvision 0.22.1（处理器依赖），已纳入运行时冻结。
  - 病例、源视频字节、prompt、解码参数逐项与第一次 pilot 冻结计划比对，不一致即拒绝。
- 不自动重试、不换样本、不减帧；失败照样按 cap 计费。

## 3. 最小代码修改（新脚本，原 pilot 脚本不变）

`scripts/run_teacher_qwen3vl32b_recovery.py`：

- 每例记录安全阶段码（local_copy_verify / frame_decode / facts_generate / facts_parse / caption_generate / caption_validate）、
  固定白名单内的失败原因、命中的固定拒绝词（规范化后）、每例耗时、各阶段新 token 数；不保存模型原文。
- worker 记录停止原因（completed / projected_budget_insufficient / allocated_window_exhausted / model_load_failed / worker_exception）。
- 共享存储上的一次性 attempt claim，任何节点都不能第二次启动恢复。
- cap 5400s、6 GPUh、先前累计 2.165898 GPUh 写入成本账本与终态 closure。

CPU 协议测试：`tests/test_teacher_qwen3vl32b_recovery.py`。CPU 测试通过不代表模型或标签语义通过。

## 4. 门禁

结构覆盖 32/32 也不等于语义 review 通过；标签语义、训练技术、正式训练预算未过门前不启动正式训练，AR / Student / 蒸馏保持关闭。
结果与费用在执行结束后另行补充，部分产物不算完成。
