# 33 Teacher 优化：内部开发诊断与晋级计划

> 冻结协议日期：2026-10-19。长期目标尚未达到；完成上一轮插值评测和发布不等于Teacher质量验收。
> 固定Wan2.2-TI2V-5B、完整dense caption。未通过自动条件和用户review前，不启动Student、蒸馏、流式/AR或未经验证的FT3大训练。

## 1. 实测差距与优先假设

文档32的三个候选均head wins0/7、全部Teacher自动FAIL。A07的GT条件通过只相对于base与已有三外部基线，不包含R4/FT1。随FT1插值比例提高，验证loss与真实后续匹配改善，Physics/WMB退化；A05公共Total*最高不能被视为全面最好。

优先验证的**假设**：SFT正向验证loss不能观察推理负向预测变化，CFG5可能放大不利偏移。CFG3相对CFG5是否能改善一致性/画质，同时不牺牲动作和真实后续？这是可证伪推理诊断，不是已证明训练退化根因。

七维各候选最好到头部各维最大值仍有缺口。最接近的背景跨帧一致性缺口约0.069pp，平滑度约0.450pp，但各维最好来自不同权重，不能拼接虚拟模型。美学约2.894pp、画质约1.925pp与主体缺口需要后续训练/数据证据，不承诺仅CFG达到所有目标。

## 2. 数据审核与新的devlabel协议

现有16条dev清单源自round2_val且语义/同源审计未通过，旧192jobs不启动。

从FT2训练CSV按每个来源首次候选固定顺序查看8条，不根据任何模型新输出选易/难样本。完整解码121帧后，助手人工查看0/0.5/1.5/3/5秒时间点。该**非随机**小审阅发现原caption描述的不实或未经观察支持事件：Charades蓝色簸箕被描述为灯，MIRA游戏车辆被描述为炮台发球及不符HUD，RoboMIND被加上连续抓取绿/黄物体动作等。不能据此声称全库错误率，也不能把原标签直接当gold truth。

第一轮仅六来源：direct、robomind、hoigen1m、matrix、charades、easycom，各一独立父来源。UKdash/MIRA复杂事件候选不纳入。

**新协议**：助手从真实可见帧保守校订每条135–149词完整英文dense devcaption，所有case的4个cell绑定同一校订文本和hash；原CSV、原caption、源视频和hash永久保留，不原地重标或覆盖。此诊断不是“原训练caption”条件，也不假装人工全视频gold-standard。GPT/云端VLM不接收原视频。

隔离核查包括：
- round2全val1735、WM108、固定base loss120的精确path及canonical父来源；HOI/通用视频跨时间同YouTube ID、Matrix同recording、MIRA同match、robot同episode跨相机、EasyCom同真实Session而非文件名前缀。
- 固定公开benchmark输入仅作可见重复排除，不读取其得分选参数。五个时间点pHash对所有唯一输入图像核验，Hamming≤4视为需人工检查；已观察候选未出现嫌疑，最小16–18。
- 公共benchmark缺完整源映射，近重复未命中不证明全source/预训练无重叠。

这是**训练可见的内部诊断**，不是未见泛化开发集。只用于排除明显失败与产生下一份冻结假设，不以其小样本涨分冒充目标完成。

私有原case/provenance/review图保存在本地审阅目录，不公开原图片视频或识别性路径。

## 3. 第一轮 P1：24视频，仅改CFG

固定两个模型：base 与 A07（已有完整曲线中领域GT/loss较好的候选，Physics风险单独保留）。FT2机制探针不混入此轮，不能将多候选网格扩大到数百组合。

矩阵：6case × 2model × 2arm = **24视频**。

| 参数 | 标准臂 | 对照臂 |
|---|---|---|
| CFG | 5 | 3 |
| Zero* | 关闭 | 关闭 |
| shift / steps | 5 / 50 | 5 / 50 |
| seed / first frame | 0 / 同首帧crop | 完全相同 |
| caption / negative | 校订完整dense / 原NEG | 完全相同 |
| 视频 | 121帧、1280×704、24fps、bf16、tiled | 完全相同 |
| 编码 | ImageIO FFmpeg quality5 | 完全相同 |

checkpoint、source/script/framework/job/content hash显式冻结。输出新run-id目录，不使用或覆盖原bench结果、claims。失败产物保留，retry换run-id且计入预算，不无限重试。

## 4. 预算与运行约束

首轮总上限 **12 allocated GPU-hours**：生成最多4 GPUh，评分/后续小验证最多8 GPUh。生成4卡×3600秒的预留=4 GPUh，加载、同步、编码和失败计算均计入卡占用墙钟，不叫纯kernel GPU秒。

父进程托管4个独立worker，超时仅TERM/KILL本轮准确PID并wait；不pkill、不停止其它项目。启动前核所选卡空闲和任务来源，不为了满32卡扩大样本。每条记录采样/编码墙钟和显存峰值；单条中断未原子保存需新试验重做，已保存对照保留。

评分在生成退出后占独立空闲GPU，最多4卡×2小时=8 GPUh，CPU DreamSim可并行但不当免费GPU预算。达到上限或出现错误即停止该试验，报告未完成而不扩预算。

## 5. 预注册读数与拒绝“静止骗分”

所有24输出需全帧解码，不用文件存在、进程exit0或缩略图充当成功。

- 主诊断读数：主体跨帧一致性和成像质量；对照同一case/首帧/label/seed，原DINO与MUSIQ权重/算法固定。
- 与真实GT的12个共同时间点（排除t0）DreamSim均值/末帧与绝对log运动量误差；额外保留GT与生成运动量原值，不能只报平滑/闪烁proxy。
- 指令/对象和动作的配对审阅：助手观察与用户盲评分开。稀疏帧审阅只能证明显错或未见明显变化，不当完成全用户review。

P1每个模型CFG3−CFG5配对差；另报 (A07差)−(base差)，区分普遍采样变化与SFT特异放大假设。6个父来源、小样本固定seed，不以不稳定CI宣称跨seed显著性。

预设筛除阈值（操作性，不是原Teacher gate）：候选一致性平均提升≥0.003或画质平均提升≥0.005，至少4/6来源同方向；同时GT DreamSim平均劣化≤0.005、末帧≤0.010、运动误差≤0.05，并在GT有明显运动的case生成运动量不得相对CFG5下降超过30%。任何明显动作/主体违背则不晋级。若主读数涨而guard失败，记录tradeoff/否定；不事后改阈值或主指标。

P1通过只能进入新的独立父来源复核或P2，不直接公开测试扫参。未通过则否定本次CFG路线，结合caption审计/负分支同x_t机制诊断决定新小消融。

## 6. 后续有限步骤

P2备选为CFG5 + Zero*最优缩放，zero_steps0（不置零），相对于CFG5 only改一个因素；总视频数/预算另冻结，不能自动附加大grid。只有明确开发收益且无动作退化，才冻结一份配方并对所有比较模型保留标准协议，做独立全benchmark验收。

训练分支必须先根据数据label质检和受控小梯度/保存smoke验证再提出方案；不因为32卡空闲或Stop hook自动开FT3/改变dense-only。

目标完成仍要求多个非dynamic头部维度、领域质量、统计/协议披露和用户Teacher review，不能用开发样本/生成结束/单Total*替代。每个否定结果保留并发布安全代码/聚合结论，原media/凭据不进GitHub。
