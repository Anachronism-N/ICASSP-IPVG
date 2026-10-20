# 42 LoRA有限训练技术smoke：预注册与实测结论

> 最新：2026-10-20，真实5B四更新技术验收PASS；不是Teacher质量或泛化PASS。历史“未测/待启动”仅按下文启动时间解释。

> 2026-10-20。用户重新授权四节点全部可用，GPU数量由助手自主安排、不再逐项询问。本实验自限单张A100、最多2 allocated GPUh、最多4次optimizer更新；不是全量FT3、Student/AR/蒸馏或质量升级。独立语义复核故障未通过，不以smoke绕过正式训练数据门禁。

## 1. 此次只验证训练链路

C1完整推理诊断已结束，修订输入带来平均匹配收益但操作失败仍在。下一质量候选仍需独立train/dev父来源与语义审核。本smoke仅确认基座LoRA梯度、优化器和保存恢复能在真实121帧条件下正确运行，没有生成新视频或SOTA评分。

输入复用M1已验收的6个训练可见真实GT缓存：x0形状[1,48,31,44,80]、首帧z、固定agent校订positive context[1,512,4096]。仅一次已存在VAE/text编码的结果；不重新校订原CSV、不混原/校订效果比较。技术用前4个固定case一次各1更新，不根据loss选样。

prepared_index、common tensor文件与各tensor内容、原GT和caption/source/framework/model hash必须核当前版本。既有noise/target不直接用于不同新噪声训练；每步native training loss自行构造当前noise和target。

## 2. 固定设置

- Base：Wan2.2-TI2V5B原DiT，只加载本地3shard明确路径，skip_download；不融R4/FT1/A07。
- 训练：LoRA rank32，alpha32，q/k/v/o/ffn.0/ffn.2，冻结所有基座参数。
- 精度：bf16基座，FP32 adapter，在bf16 autocast下前向；需实际验证此组合，不改原framework。
- Optimizer：仅adapter参数AdamW，lr1e-5，max grad norm1；无基座预测保真项，此次lr不是质量搜索结果。
- Native FlowMatchSFTLoss：1000步training scheduler，正caption、randt与noise，target=noise−x0，首frame覆z，预测/target去首frame后MSE乘training_weight。训练seed20261020固定，最多4更新。
- 激活：gradient checkpointing及offload开启。T5/VAE不更新也不驻GPU，使用已绑定缓存。这与历史在线encoder/VAE训练显存不是同一测量配置。

基座safetensors头部CPU核查目标300个线性层，adapter参数80,609,280；FP32参数307.5MiB，AdamW两moment615MiB，FP32梯度307.5MiB。**这些只含参数/优化器，不是峰值训练显存**；bf16DiT、激活、workspace、offload和PyTorchallocator需要实际smoke测量，不能据此宣称单卡容量已保证。

## 3. 验收和失败条件

1. 每步loss及至少一组adapter梯度有限且非零，记录clip前grad norm、优化器step及峰显存；初始LoRA A梯度为零可能正常，不要求每tensor非零。
2. 基座/T5/VAE冻结，optimizer参数组只能包含adapter；更新后adapter确实改变，不仅loss文件存在。
3. 私有保存adapter、optimizer与CPU/CUDA RNG状态；strict keys/shape/finite，先扰动adapter/optimizer/RNG再恢复，内容hash精确一致。用同一固定probe和恢复RNG比较预测或loss，FP32容差预设，不增加optimizer步。
4. 4步、finite、更新与保存恢复全部成功且自有进程退出/组清理和预算通过才completed。OOM/NaN/无梯度/checkpoint不符/超时均failed保留产物和traceback，不扩大步数/额外预算自动重试。
5. Adapter权重、GT、缓存tensor和原caption不公开；只发布代码、测试与无身份聚合结论。

## 4. 预算和执行

单卡GPU0，最长7200秒=2 allocated GPUh，前flight、hash、模型加载、等待、训练、CPU保存/恢复、probe前向、收尾验证/回收及报告余量均计coordinator墙钟。顶层及--help/错误预算不加载torch；真实CPUpreflight为缓存数学验收可加载CPUtorch，但不得初始化CUDA。owned worker新session，TERM/grace/KILL+wait含组内子进程，不能pkill其他项目。

先CPU测试和真实缓存preflight，再实际GPU执行。代码/框架复制至本轮新私有目录且pin，未存在输入拒绝、禁止shared fallback/下载，不覆盖已有运行目录。极端OS/D-state无法即时回收需要explicit failed/overrun，不能截断成本。

不为了满32卡扩大本技术试验。四节点可在后续已冻结任务中分工，但本次只需验证一条真实训练链路。GPU数量授权不等于全部科研阶段授权，用户暂停/新节点分配仍优先。

## 5. 当前验证状态

- 同6窗双标签CSV已私有准备并准确hash/parsed文本复验，原CSV不变。此次smoke只用现M1positive校订cache，未做双版本训练对照。
- 主流程独立CPU微验证使用从原文件AST提取的LoRA注入和FlowMatchSFTLoss，微型线性层4次真实AdamW更新，bf16base/FP32adapterautocast兼容、基座权重不变、adapter/optimizer/RNG序列化恢复和同RNGprobe loss一致。CUDA未初始化；**不是完整5B训练通过**。证据LoRA_native_CPU_microcheck_20261020.json。
- 最终runner与15项CPU测试已验收，完整5B真实4更新、冻结基座和checkpoint恢复技术验收通过，详见第7节。公开测试读取随快照提供的同hash原loss/训练模块，不依赖私有M1目录或跳过原API测试。
- Teacher自动FAIL，用户review pending，正式标签与独立开发来源门禁保持。

## 6. 实际启动记录（2026-10-20 19:55 CST）

最终runner/hash及15项专项CPU测试已独立验收，4更新smoke通过Bash托管启动（任务ekWZiz，父PID56299）。运行目录/dockerdata/wm_tmp/teacher_training_smoke_20261020_run01。此时尚在CPU预检，未核验任何优化器更新；启动/单测不证明完整5B反向、峰值显存、checkpoint恢复或质量目标通过。失败与实际allocated墙钟均按原上限保存，禁止启动重复任务。

## 7. 完整真实技术验收（2026-10-20）

- 唯一托管任务ekWZiz exit0，4个固定窗口各1次native FlowMatchSFTLoss/AdamW更新；600个FP32adapter共80,609,280参数。每步loss/clip前梯度范数有限非零且adapter hash改变，基座全量参数hash前后相同。
- 四步loss为0.0656504、0.1441248、0.2330641、0.8501397，梯度范数0.0053285、0.0060683、0.0089475、0.0091320。**不同窗口及随机t/noise的四个loss不是收敛曲线，不能宣称训练收益或原/校订优劣。**
- batch1、1280×704、121帧，缓存VAE/T5不驻GPU、activation checkpoint与offload开启。PyTorch峰allocated **21.5941GiB**，reserved **22.9355GiB**；不是整卡总显存，也不能外推在线encoder、全量SFT、DDP/FSDP或长训练容量。
- 唯一1卡完整coordinator墙钟315.7451秒，含预检/全部hash/setup/训练/保存恢复/probe/回收及1秒报告余量：**0.087707 allocated GPUh**，低于2GPUh硬上限。自有PGID已消失，现场GPU0显存归零。GPU释放后的独立CPU复验和GitHub发布不算继续占卡。
- 私有checkpoint 968,043,701字节，仅adapter、AdamW和Python/CPU/CUDA RNG，无全量基座。真实扰动adapter与exp_avg/exp_avg_sq、独立reseed后严格loadback；各状态hash精确恢复，同RNG/minibatch预测最大差0、loss差0，额外2次前向且**0新增optimizer更新**。此为同worker恢复验收，未验证独立新进程接续下一优化步的trajectory。
- 主流程独立CPU重读真实checkpoint：600adapter全部FP32/finite，80,609,280参数；全部AdamW moment形状/finite/非负二阶矩、step=4，文件和全状态hash、最后adapter与第4步hash、228项冻结运行源码hash、四步JSON哈希链与回收/预算均通过。未新增GPU。
- 前后全部10模型cache内容hash、6commons、18states及source复核通过。原GT/CSV/标签与已有结果未改，权重/视频/缓存/原caption不公开。无身份聚合结果见eval/diag/teacher_training_smoke_public_results_20261020.json；私有复验证明仅本地留存。

## 8. 下一质量候选的剩余门禁

此次只消除了单卡cached-input LoRA梯度/保存恢复的技术不确定性。正式label版本、独立父来源train/dev隔离、独立第二语义复核仍未全部验收（第二助手两次上游空流后未自动重试）；17例主流程试审不是human gold。Teacher仍自动FAIL/用户review pending，不以本checkpoint选择公共测试、WM108或固定val120推理参数，不启动FT3/Student/蒸馏/AR，也不把4步loss称作质量提升。
