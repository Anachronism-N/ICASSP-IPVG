# 官方规则、数据与待确认事项

事实快照为 2026-10-17。官方规则后续可能更新；本文件将“已核实”“未公开确认”“本项目建议”分开。根目录及实验产物不能包含受限人脸、视频、API Token、签名协议或账号凭据。

## 已核实的比赛内容

IPVG 是 [ICASSP 2027 SP Grand Challenges](https://2027.ieeeicassp.org/sp-grand-challenges/) 的 GC-14。Track 1 是人脸身份保持的视频生成；Track 2 是参考主体与多段时间戳动作条件下的视频生成，主体还可包括动物和物体。[比赛官网](https://hidream-ai.github.io/ipvg-challenge-2027.github.io/)

Track 1 最终测试为 200 个未见身份，每身份 5 条 prompt，共 1000 对；每对至少生成一个视频。Track 2 测试数量尚未公开核实。每队每 track 最多提交 3 个完整 run，必须选 1 个 primary；另外两个 run 不保证评测。不能把“3 个 run”解释为每条样本允许无限重采样。

官方评价混合身份、质量/运动/文本、Track 2 时间线对齐等客观维度和人工研究。当前未核到完整可执行 scorer、具体模型及权重、评分权重或实时 leaderboard 链接。

| 节点 | 官方原始日期 | 当前时区状态 |
|---|---|---|
| 两 track 测试集 | 2026-12-09 | 未写时刻/时区，不擅自补 AoE |
| 结果提交 | 2026-12-16 | 未写时刻/时区 |
| 客观评测 | 2026-12-17—21 | 未写时区 |
| 结果公布 | 2026-12-22 | 未写时区 |
| 论文提交 | 2027-01-07 | 资格、篇幅与具体出版类别待核 |
| 录用通知 / 终稿 | 2027-01-21 / 01-28 | 以主办后续通知为准 |

ICASSP 属 CCF B 不意味着所有 challenge 报告、短文或附属活动自动计为 B 类 Full/Regular；不要在简历或项目台账中提前标成已录用的 B 类论文。[CCF 学术评价目录](https://www.ccf.org.cn/Academic_Evaluation/By_category/)

## 数据快照与下载策略

字节数来自 Hugging Face 公开 metadata 中当前文件清单之和，不是解压后的空间。GB 为十进制，GiB 为二进制；`usedStorage` 可能含不同内容/历史，规划下载以所选文件清单为准。

| 仓库 | 授权 | 固定 revision | 当前文件总量 |
|---|---|---|---:|
| [VIP-200K 注释](https://huggingface.co/datasets/HiDream-ai/VIP-200K) | API gated=false | `acbcbbeddd4f192f933928d1d8a13537cda98c42` | 1,900,127,672 B，约 1.90GB |
| [VIP-200K-Video](https://huggingface.co/datasets/HiDream-ai/VIP-200K-Video) | manual | `e5909911c901960e8907919a3f27f789a9620380` | 327,714,718,666 B，约 327.7GB |
| [ReactID-Data](https://huggingface.co/datasets/HiDream-ai/ReactID-Data) | manual | `ea51455678b58038f416d2f44dc2031295706172` | 8,860,159,868,052 B，约 8.86TB |

VIP 注释当前是 20 个 JSON，文件清单没有卡片文字提及的 face frames 包。视频仓库为 100 个 tar。不能宣称注释下载后已有所有参考图；参考图来源和对应关系需授权后确认。官方赛事页的 50 万视频、数据卡的 200,005 条身份记录及 701,470 captions 是不同统计口径。

ReactID 当前含 4096 个视频分片约 8.80TB；metadata 压缩包约 1.63GB，timeline annotations 约 0.455GB，完整 annotations 约 56.63GB。`ReactID-Eval-SEQ` 有 120 条公开 benchmark 的 JSON 与约 114MB 图包，但不能自动当作本届官方 validation 或隐藏测试。

第一阶段只下载白名单分片和必要注释。选 5 千—2 万条质量/主体/动作适当的训练片段，容量由真实样本统计决定；不得通过无上限 snapshot 把整库搬到服务器。每个下载计划记录 repo、revision、路径、预期字节、SHA256（若提供）、授权和保存目录；先计划与核对空间，再执行下载。

raw、解压、裁剪、VAE/text cache、模型权重和 checkpoints 分开计容量。小规模开发可从 2—4TB 可用共享空间与每节点 1—2TB NVMe 缓存规划；若保留 ReactID 全量及缓存，15—25TB 是更合理的容量估计，但并非本项目开局要求。

## 账号、协议和用途

2027 比赛[注册表](https://docs.google.com/forms/d/e/1FAIpQLSfdQO4gRnPkGBCXL94ziiN59E0RBWC2Qro7U-p4SPsZ5bpLOg/viewform?usp=publish-editor)和数据卡[访问申请表](https://forms.gle/j4Nwq38W9TjtPNgq9)是两个入口。匿名访问比赛注册表需 Google 登录，表内附加条款尚未核到。数据门禁文字仍指 MM2026/旧赛事，不能直接当作完整的 2027 规则。

数据卡称使用 HF 注册邮箱填写申请、约 48 小时处理，这是主办说明而非审批保证。本人接受真实条款并签署协议；本仓库不代签、不绕过门禁、不向外部服务发送申请或邮件。

比赛限制研究用途；HF 门禁增加非商业学术研究条件；[VIP 协议](https://github.com/HiDream-ai/ipvg-challenge-2027.github.io/blob/main/Agreement-VIP-200K.pdf)明确不得向第三方提供数据。HF 卡片的 Apache-2.0 标签不能覆盖独立访问条件。模型代码、基础权重、身份识别权重和数据许可分别核对。

## 数据标准化与隔离

标准化 manifest 至少包含：样本、身份/主体组、源视频、track、参考图路径、prompt、目标视频路径、时间线、文件 hash、来源与 split。具体字段以项目 `data validate --help` 和示例 manifest 为准，不自造 upstream 数据格式。

按身份与源视频构造不可拆分的连通组后划分；近重复/裁剪/重编码同组。保存清单和划分 hash，禁止 agent 更换 holdout 后继续把新分数与旧分数直接比较。身份缺失时补齐或排除出泛化评测，不默认每条都是不同人。

拒绝缺失文件、无法解码的视频、非法时间段、超出长度的事件边界、重复 sample ID、跨 split 身份/源视频，以及参考图/目标媒体错配。预处理失败保留错误记录，不用静默删难例提高分数。

## 正式测试前要核准的问题

1. 公开预训练模型、额外训练数据及闭源生成/评价 API 是否允许，是否需申报与截止日期？
2. 测试参考图的 LoRA/适配、prompt 改写、best-of-N、候选 rerank、编辑/拼接/插帧是否允许？
3. baseline、完整 scorer、独立 validation 的入口和发布日？特征骨干、评分权重与人工评价协议？
4. 输出尺寸、fps、长度、编码、命名、压缩包及提交渠道？12 月 16 日的具体时刻和时区？
5. VIP 参考脸帧在哪里？ReactID-Eval-SEQ 能否用于本届开发？
6. 论文资格、页数、是否独立审稿和最终出版 track？是否有 API/GPU 支持？

官网联系地址为 `panyw.ustc@gmail.com`。此清单不是已发送的邮件；实际答复应作为后续规则版本证据保存。
