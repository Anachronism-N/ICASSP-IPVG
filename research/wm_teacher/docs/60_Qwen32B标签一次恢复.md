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

## 5. 唯一恢复的真实失败结案（2026-11-03）

- 执行源码提交：`174b692c68b7b22c51f65fed319be82b97de1c94`；257 个公开资产 + 9 个变更文件的独立远端克隆字节核验通过。
- 完整公开测试：1435 passed / 3 skipped / 895 subtests passed。缺失的原生 FA3 接口检查明确跳过；模拟 FA3 契约测试仍执行。不是 GPU 能力或标签语义验收。
- **本次恢复失败：0/32 标签，0/32 病例实际执行；16 诊断和 16 盲抽分别都是 0 标签 + 16 未执行错误记录。** 没有 worker.json，没有成功 cost.closure.json，不把 32 个错误占位文件叫完成。
- 启动命令的父进程 argv 使用相对 runner 路径；worker 的严格祖先进程检查要求绝对规范路径。保存的原始 argv 直接违反该要求，CPU 重放对应检查可确定地重现拒绝。此启动错误由本次执行方造成，不归因于模型能力、CUDA 或标签质量。
- 原 worker stderr 只留有统一 ValueError；没有完整活祖先链重放。诊断仅重放首段严格父命令门；不补造实际 worker 阶段、模型加载时间或逐例耗时。原 32 个占位记录的 `not_attempted / worker_exception / duration=null` 保留不改，根因写到独立审计。
- 监督窗口实测 **306.165342s × 4 = 0.340183713 GPUh**，包括预检查/等待/失败/终态保存与清理。按原协议保守计 **6 GPUh**，旧失败 2 GPUh 不退款；标签阶段累计上界 **8.165898211 GPUh ≤ 24**。
- 四个获准占卡进程已精确停止；恢复进程树与四卡 CUDA 进程释放通过原连续查询和终态独立查询验证。其他业务任务未停止；旧集群实验未重启。
- 全部本地原始文件已复制到共享的私有实验快照，并逐文件源/副本 SHA 验收；媒体、原 caption、日志与私有授权不发布。

**一次恢复机会已消耗。禁止删除 claim/锁后重跑，禁止换样本/减帧/改精确门救结果。任何后续实际分配必须是另一次明确新增授权，不复用或重置本次额度。**

### 以后经额外授权执行时的启动要求（不是本次重试授权）

- 必须使用 runner 的绝对规范路径启动父进程；先做 CUDA 前的命令契约核对。
- 已公开的原脚本与完整身份门不改；不能把相对路径偷偷补认成已获准命令。
- 原失败凭据保持不可变，新实验目录/输入/host/runtime/source/费用重新审批。
- 训练标签语义、技术恢复与正式训练预算门仍未通过；正式训练、AR、Student、蒸馏保持关闭。

## 6. 新增授权的独立第二次分配（2026-11-03，执行前冻结）

用户在上一轮已明确失败、次数消耗与重新执行必须新增授权的说明之后回复“我给你所有权限,请继续进行推进”。将其限定为新增一次同范围的4卡BF16、5400s、6GPUh分配，不重置第一次恢复，不自动扩展正式训练。

- 新增分配序号2；先前标签保守费用8.165898211GPUh全部保留，新累计上界14.165898211GPUh≤阶段24GPUh。
- 单独授权文件绑定前次失败收据与原claim原始SHA；新共享目录、新输出和state目录。旧原始材料/claim均不可覆盖。
- 同32固定病例、16帧、原prompt/解码参数。CLI新增显式`--supplemental`，父/worker各自核验新增授权；原模式仍受已消耗claim约束。
- 父命令绝对规范路径检查前置到创建claim和任何分配之前；相对启动不得消耗新机会。worker的严格真实身份门保持不变。
- 新增针对性测试152通过。完整公开测试、源/runtime/model/输入冻结与真实远端字节审批后才允许GPU启动。
- 不自动重试、不换样本、不减少帧数、不放宽语义/结构门；结果另记，正式训练/AR/Student/蒸馏保持关闭。

## 7. 新增第二次分配完整实机结案（2026-11-03，整体准入 FAIL）

- 执行提交 `cf7c2fdd52b1e042542318cafdd4c7c36b3081df`；258 个公开文件及5个改动文件的独立远端克隆字节核验通过。完整公开测试1444 passed / 3明确skip / 895 subtests passed。
- 本次已通过绝对父命令、真实祖先链、四卡UUID与无外部GPU工作门，真实加载Qwen3-VL-32B BF16于4卡。固定病例/源视频字节/16首尾均匀帧/prompt/greedy/max tokens与原pilot一致。
- **32/32病例实际执行、0未执行；13/32结构合格标签、19/32校验失败。整体32标签精确准入FAIL，无成功cost.closure。**
  - 诊断组：5标签 / 11错误 / 16病例。
  - 盲抽组：8标签 / 8错误 / 16病例。
  - 13例不符合40..120词要求；4例命中frame元数据拒绝词；1例命中name拒绝词；1例跨实体手/持物关联检查拒绝。
  - 失败阶段：caption_validate17，facts_parse2；没有EOS截断报错、模型加载报错或预算提前停止。
- 原worker的`stop_reason=completed`只表示32病例循环结束；`success=false`、原始cost.failure及19条错误仍保留。不能把该字段当整体成功或从失败记录伪造caption。
- 13个保留标签由未改语义门的第一次pilot验证器再次CPU复核通过；尚无独立语义review批准。字数/固定拒绝词并不证明语义错误；模型文本不符精确接口与真实语义正确率应分开，拒绝词可能存在误拒，失败原文已按原协议不落盘，不能事后断言方向/词数或具体场景语义。
- 加载22.674818s；32例耗时总1324.605135s，单例中位40.910325s，范围21.998675–67.968097s。全部生命周期allocated **1502.934155s×4 = 1.669926839 GPUh**，计入预检/加载/等待/每例失败/保存/清理。按协议保守计 **6 GPUh**；先前累计8.165898211保留，新累计上界 **14.165898211 GPUh≤24**。不退款或挪用其他阶段预算。
- 自有进程树与GPU0..3释放经原连续查询和独立终态查询通过，监控器也正常退出；其他节点/h3任务未动，旧集群未生成或评分。
- 私有共享实验目录保留32标签/错误、所有原始输入副本、parent/worker/supervision/cost与完整SHA审计；公开只发脱敏聚合结果。前次失败claim和新分配claim均保留。

下一步应是**新的受约束标注接口设计与独立语义验证**，不是继续用同设置重跑或将13部分标签当全量完成。任何改变prompt/结构输出/解码的实验必须另立有限预注册及新分配；不得改本轮字数/实体/精确门回填结果。正式训练、AR、Student/蒸馏仍未启动，没有新Teacher质量或SOTA成绩。

## 8. 肯定式输出策略v2（执行前预注册）

持续实验目标下的新协议开发实验，不重跑原失败配置、不事后放宽精确门：新分配序号3，4卡BF16、5400s/6GPUh，一次，不自动重试。原累计14.165898211保留，本轮累计上界20.165898211≤标签阶段24GPUh。

唯一协议策略改动是两阶段的肯定式输出规范：facts仍用原exact实体/events结构，附加简洁场景描述、按entity_id分别写动作、只陈述支持的内容而非列不可读信息；caption明确四句、每句12–22词/共48–88词、直接场景描述，不叙述frame等输入元数据，不靠编造动作补词数。该策略同时改变两阶段prompt，不称可单独归因的单因素试验；原40..120词、拒绝词、跨实体、EOS完整性等验证器全部不变。

固定原32病例与16首尾均匀帧，原模型/runtime/greedy/1024+384token/4卡不变；病例已被开发诊断，blind仅沿用原分组名，不再称新的独立泛化证据。不换样本、不按分数选输入。主要工程验收仍32/32结构合格；任何失败整体不通过。语义验证独立，接口通过不能推断动作/物理语义正确。

新目录、原始模型与来源字节、两阶段prompt SHA、授权/账本、source/runtime和远端公开提交再次冻结；原各轮claim/失败媒体与费用不改。155项针对性测试通过，完整公开测试与远端核验后才加载GPU。剩余标签预算不因持续目标无限扩张，正式训练/AR/Student/蒸馏仍受原门禁。

## 9. 输出策略v2实机完整结案与独立语义诊断预注册

执行提交7207489。32/32病例真实执行：23结构标签、9校验失败（诊断11/16，原blind来源组12/16），0未执行。相同病例原策略13/32，新策略23/32，但六例facts拒绝词、一例caption词数、两例caption元数据仍失败；32/32精确准入整体FAIL。该开发集合重复使用，不能当独立泛化显著收益；两阶段prompt同改，不能单独归因。

实测1578.111905秒×4 = 1.753457673 allocated GPUh，加载22.992886秒。原协议保守计6GPUh，累计20.165898211≤24。原失败/claim不改，无成功closure；四卡与进程树已释放。未训练或评测新Teacher权重。

### 独立8B语义诊断（不是人工gold或训练许可）

预算单独冻结1卡3600s/1GPUh；计入同标签24阶段，上界21.165898211。复用另一个新集群节点已有Qwen3-VL-8B-Instruct BF16，15文件源/副本SHA匹配后本地部署，不调用外部视频/视觉API。模型不同于32B标注器但同Qwen家族，相关偏差如实报告。

固定对两轮原32病例的全部保留标签评审：原策略13、新策略23，共36段；所有64条病例记录完整分母，接口失败并非可伪造的caption，不纳入假语义评分。评审不见方法名、原标注facts或原caption，只读该候选段落和16源帧。七字段严格yes/no/uncertain：实体、动作、结局、无幻觉、无重大遗漏、无运动擦除、无跨主体误归因；由固定规则导出adequate/needs_edit/uncertain。

先做4个程序化静态/移动红圆正负控制，必须全部正确，失败即停止实际标签评分，不自动重试。控制仅证明基本证据敏感性，不证明真人视频语义可靠；新评审源、模型、病例、runtime、费用和远端字节另冻结。64例分母和所有逐例error保留。AI诊断不自行释放正式训练/AR/Student/蒸馏门，也不宣布SOTA差距缩小。

## 10. 独立8B语义诊断完整结果与前瞻约束解码

原评审提交809bda0；4/4程序化运动正负控制通过，36/36保留标签实际评审，无评审错误。原策略13保留标签中10 adequate/3 needs_edit；v2的23保留标签中19 adequate/4 needs_edit。换到各32源病例的完整分母分别10/32与19/32；不可将条件10/13、19/23的比率当全32覆盖率。稀疏帧、同模型家族与重复开发病例限制保留；这是AI诊断，不是人工gold或SOTA收益，没有自动训练批准。

评审生命周期233.205073s、实测0.064779187 allocated GPUh，保守计1GPUh；累计21.165898211，GPU已释放。全部原始判断、控制、源标签/模型/源码字节和费用可私下复核。

### 最后剩余标签预算内的新解码实验（前瞻，不改旧结果）

保持v2两阶段prompt、同32病例/16帧、BF16、greedy和token上限；仅新增原生bad_words_ids与caption的EOS最少40词logits控制。facts禁止词不含frame，以免毁掉required frame_start/frame_end JSON键；caption按原拒绝词规范限制frame等元数据。没有事后剪裁/补词/重写，不放宽原40..120词、实体归属、敏感/grounding拒绝或EOS精确门。超过上限或其它校验失败仍完整保留，不称成功。强迫段落继续可能引入多余或不实内容，必须独立语义审阅，不能只看结构过率。

新一次4卡2400s = 2.666666667GPUh，旧累计21.165898211完全保留；随后单卡540s=.15GPUh同七字段/四控制AI审阅，总累计上界23.982564878≤标签24GPUh，没有阶段借用、无限重试或训练预算。既有最坏单例约68s、上轮实际1578s仅用于运行预算可行性参考，不承诺本轮吞吐或保样救结果。新解码/后续审阅都有独立授权、目录、source/runtime/model/输入/公开字节审批与claim；原3个标注claim与原评审账本不改。

## 11. 约束解码实机失败与资源独占冲突（2026-11-03）

执行提交ed5d64d。第一例产出结构标签，单例472.863967s；最坏单例×剩余例数超过预注册2400s窗，原精确预算门停止，31例未执行。1标签+31未执行错误占位不是32例完成，没有自动重试，也未放宽停止门。

原父进程观察647.219667s，原cost.failure保守计2.666666667GPUh，累计23.832564878≤24。原released=false、actual_allocated_GPUh=null必须保留：自有树已退出，但原连续GPU查询出现4个外部占卡PID，原空卡释放凭据失败。后续按授权精确停止已确认hold_gpu程序、另写独立清理收据并确认GPU释放，不改原false/null。

四个重新出现的外部占卡进程，其/proc启动ticks换成同机monotonic时点为分配开始后约138.38–139.30s；第一例文件到开始后约644.08s才落盘，worker约644.11s。该时间关系证明实验执行期间四卡独占条件被破坏，不只是正常退出后的占卡恢复。不能把472s慢例唯一归因于NoBadWords、EOS控制或某个kernel，也不能按受干扰吞吐声称方法优劣。排查已读到原生NoBadWords逐prefix分支；CPU独立批量掩码原型145个有限score/重复prefix/目标冲突检查与原生掩码torch.equal一致，CPU单call约4.63ms→0.309ms；该原型尚未接入执行源码、未做GPU速度/质量验证，不能据CPU测速宣称实机加速。

后续540s独立评审要求输入实验32/32实际执行，而本次只执行1/32，前置条件失败，因此**未启动、未计费、不删除claim救结果**。剩余标签额度约0.167435122GPUh，不能继续分配原6GPUh实验或自动借用其它阶段预算。下一阶段须明确追加有限预算，并协调新集群其他会话停止自动恢复占卡；仅反复杀PID不构成稳定独占保障。

持续SOTA目标未完成。已经请求新增有限168GPUh（标签/缓存96、技术准入8、受门控候选训练与评测64），这是待确认的下一阶段方案而非已获得授权；未确认不超原24。正式训练必须等数据语义与精确恢复技术门通过，AR/Student/蒸馏继续关闭。当前仍没有新Teacher权重或同协议SOTA胜负证据。

### 无新增GPU分配的最小安全修复

已在原监督父进程补每5s的4UUID原始PID检查；允许当前受监督树内的CUDA进程，发现外部重入即保存原查询、正常清理自有进程并保留失败/全额费用，不向外部PID发信号。这不是独占资源协调的替代品，也不是实机复跑或成功声明。新增CPU单测验证仅记录侵入PID且无外部kill接口；原执行源码21文件已经私有逐SHA保留，不事后把新防护说成原实验已具备。

## 12. 持续目标被拒后追加只读核查（2026-11-03）

目标未完成，原24GPUh标签预算剩余约0.167435122。现已向用户一次明确提出新增168GPUh上限和暂停其他新集群会话自动占卡/提供可安全暂停控制入口的请求；待答复，不将“继续目标”自动变成超原阶段帽或正式训练许可。168的96标签/cache、8技术、64门后训练/评测只是有限下一阶段推进范围，不保证64覆盖新Teacher+base+外部SOTA的官方全集；必须先用新增授权下的有限实机测量冻结后续可行分配，预算不足时不做半批验收或放宽规则。

### 占卡重入实际控制原因

现场launcher GPU0..3空闲。原pair-manager的resources_safe原证据显示训练结束/锁释放，现进程阻塞在wait旧holder，并没有循环重启逻辑；它不是新orphan holder的可证明调度来源。新占卡程序不读od_reserve标记、不取得ms12k flock、不检查已有CUDA任务。既有GPU0..3四个ms12k锁可以只读打开后临时取得/释放，文件均0bytes，未修改；所以仅创建新lease/锁并不能阻止该脚本占卡。未改共享外部项目源码、未向管理器发信号，尚未获得外部会话控制入口。需要会话层停止重复启动或经明确资源协调让所有占卡入口遵守同一lease；不能“删旧锁/杀全机python”假装解决。

### 真实质量协议与预算复核

独立读取官方cal_i2v_final_score与constant，camera位于I2V组、权重0.1/I2V总权2.1，Quality仍含dynamic0.5。base的355原病例适用八维分数集合全部精确、有限（dynamic是246个原bool判定，不当类型损坏）：Total*=89.860075、同公式去dynamic=91.135529。缺camera的原官方9维Total必须保持null；不调用官方缺维补零分支。完整官方core每pair5视频为1775，加camera109×7×5=3815，合5590视频，不等于目前本地core355。

同时核对Physics原66scene CSV行及原final_score_orig字段，base30.130379%、E2 31.526205%；本轮没有重跑IQTable或重评分。WMB350/原2800判定（instruction350、物理1750、常识700）集合/类型通过、独立原公式8.297143；WM108原GTref全字段有限、DreamSim均值0.174638。官方外部精确checkpoint/输入模式/原caption或po/候选数和采样规格尚未对齐；同协议SOTA gap未确定。本次是原始结果复核，不是新模型或新GPU评分。

C276原子集真实生成28.483680GPUh、评分1.100152GPUh；历史两seedALG/Dy每视频约360–389s仅作为旧机证据，不能承诺A800速度，不能据此声称64GPUh必定覆盖新模型完整四基准。已保存私有quality_protocol_readiness.audit.json，所有引用源码/输入SHA可复核，不改变原评分与训练门。

### 已准备的私有人工盲审材料（未伪造审阅完成）

复用已有serve_label_review.py的白名单输入契约，新准备两轮全部36段保留标签（原策略13、v2策略23）；固定seed20261103打乱顺序、隐藏方法名与AI结论。26个去重源视频均CPU完整解码121帧，共3146帧，分别生成0–60、61–120两页contact sheet，同时保留原视频观看入口。标签/video/pages原SHA核验及现有load_cases验证通过，原64条来源记录分母与28个缺标签记录边界另记。

私有目录eval/teacher_label_semantic_review_20261103/human_review_pack，包含index/audit/隐藏mapping及preparation.receipt；权限700/600。没有启动网络listener、没有上传媒体、没有写任何human verdict；人工审阅完成和训练语义批准仍false。此准备解除的是素材/接口缺口，不是语义真值或正式训练门。后续可经现有本地只读check-only入口复核后供真人审阅；不把同族8B的adequate标记自动改写成人类判断。

## 13. 新集群FA2运行时CPU准备（未GPU恢复或训练）

为解决下一轮A800真实技术准入的环境缺口，已从官方v2.8.3 release选择Python3.12/torch2.7/CXX11ABI true/CUDA12 wheel。256,028,810bytes下载SHA256与GitHub release digest严格相同；解压后实际FA2 native库与wheel相应成员字节SHA一致。隔离安装在新本地site目录，不覆盖已有32B标注venv、不修改原FA3协议授权。

硬阻断CUDA初始化和CPU导入阶段网络后，实际Wan模块、WanVideoPipeline及DiffusionTrainingModule导入通过；实际wan.FLASH_ATTN_3_AVAILABLE=false、FLASH_ATTN_2_AVAILABLE=true，链接到核实的flash_attn_2_cuda原生库。forward支持deterministic关键字，torch注册backward schema具有bool deterministic，dropout_p默认0.0。**默认deterministic仍false**，所以原生导入成功不能当精确恢复已完成；新GPU协议还须显式deterministic=True、固定flags/env、4-reference vs 2-prefix+2-freshresume、原adapter/AdamW/RNG精确门，且独立预算/source/model/runtime/host审批。不能把FA2替代偷偷说成旧FA3相同轨迹或旧恢复权限未用。

79项实际加载源码与selected runtime资产SHA已冻结，CUDA_initialized=false、model_loaded=false、attention_kernel_executed=false、GPU授权false。本环境未装deepspeed，checkpoint分支与旧runtime不同，后续须明示冻结且验证，不能冒充旧runtime完整复现。FA2及最小Wan导入依赖只装新隔离目录；训练源、原失败与标注环境不改。

另有CPU复制三片原Teacher DiT到新本地目录的受管任务，复制途中逐hash绑定原CACHE_PINS、输出独立SHA和safetensors头核查；目前尚在执行，不当完整模型部署证明。完成事实必须以独立base_content.proof.json和三个pin相符为准。全程未加载权重到GPU、未执行LoRA update，没有新Teacher质量或SOTA结果。当前仍等待此前已明确提出的新增预算和跨项目占卡入口协调授权。

实际基座头另外CPU核验：三片原safetensors含825个F32 tensor、4,999,787,712参数元素，30个block与全部self/cross q/k/v/o投影存在，配置head_dim128。源文件F32不等于运行时F32：现有build_pipe明确torch_dtype=bfloat16加载且LoRA可训练tensor转float32；后续须按实际加载检查。头读取不等于全内容SHA或GPU模型加载，当前CPU复制未完成状态不改为true。

## 14. 用户扩大预算后的新阶段冻结（2026-11-04）

用户在168GPUh阶段请求后明确回复“可以扩大预算.当前一共8节点可用,资源肯定是充足的”。新增阶段硬帽168GPUh：标签/cache96、技术8、语义与精确技术准入后Teacher训练/评测64，不无限扩张、不借用、不抹掉原23.832564878标签费用或旧失败claim。每子实验另冻结timeout、卡数、单次claim、模型/runtime/输入/源码与终态费用。资源充分不保证同预算/全协议SOTA成功。

首项新实验保持v2 prompt、同32病例/16帧和原精确验证门，把NoBadWords逐prefix实现换成精确等价向量化匹配/布尔scatter掩码，不改EOS最少40词规则。用独立新模式vectorized，4卡内层5300秒，资源守护总5400秒/6GPUh；为启动与compute-mode恢复留100秒，所有加载/等待/失败/清理计入资源总账。CPU有限掩码包含重复prefix/同目标/EOS、原旧模式回归已通过，不等于GPU速度或语义提升。

新集群四卡临时使用EXCLUSIVE_PROCESS，已确认hold_gpu脚本按精确PIDFD/argv/设备归属暂停；守护只对这些已授权holder发SIGTERM，不杀业务任务、不改跨集群共享外部脚本。进入实验后CUDA上下文占有四卡会拒绝新增holder；既有每5秒外部PID检查保留。结束恢复原compute-mode并写独立账本；复原失败不算完成。原集群不动，非分配GPU不改。

先完成此新协议的32/32结构与独立语义审核；FAIL不自动重试/放宽。正式训练仍须数据/cache语义门及独立FA2精确恢复验证后，按已批准的条件64GPUh范围冻结单候选。全基准和外部同协议对手不足时如实报告，不把工程通过当SOTA。

### 新阶段allocation001终态与独立allocation002

首轮db5f9cd在4卡EXCLUSIVE_PROCESS下完整尝试32病例，32例facts_generate均为统一RuntimeError、0标签；没有外部GPU侵入记录。资源守护323.902487s/约0.359891652实际allocatedGPUh、父266.194952s；按总上限保守计6GPUh，新阶段标签用6/96、总阶段6/168，旧23.832564878不退款。原错误没有保留详细traceback，不能凭泛化RuntimeError断言某个CUDA kernel唯一因果。

CPU代码复核发现HF分片模型input_ids位于embedding卡，而输出scores位于lm_head卡；首次向量化掩码沿input_ids.device建prefix却向scores设备scatter，构成明确跨设备代码错误。新实现将prefix token与cache显式对齐scores.device；补单设备to调用及meta异设备回归，原掩码与原生CPU等价验证仍通过。此修复为新独立allocation002的实现版本，原failure/claim/实际源码快照保持不变，不删除claim自动重跑。新002单独从已批96内预留6GPUh，旧失败全额保留，53xx内层工作/5400s总资源窗、同32病例和16帧、原精确门不放宽；先实机证实代码路径，再谈结构/语义。

四卡原compute_mode已正常恢复Default，已知holder仅精确PIDFD TERM，无共享外部源码改写/旧集群信号。资源独占这次实际生效，不把实现错误归因于卡不足。

### 分配002资源门拒绝与分配003的启动前lease方案

分配002在模型加载前遇到holder重入，5秒外部PID门正确停止自有工作，0病例/32未执行，不可说GPU设备修复已证实。原released=false/actualGPUh=null保持，另有资源guard正常复原与随后无GPU进程事实。分配002保守计6GPUh；新阶段用12/168、标签用12/96，旧费用保持，总标签历史35.832564878。原始argv、查询、执行源码及错误全部私有留证。

EXCLUSIVE_PROCESS只有CUDA上下文占有后才排斥新holder，CPU预检窗仍存在竞争。现在在已确认纯占卡入口添加**节点本地、指定UUID、最长5400s**的lease等待，导入torch前查询；没有lease的节点保持原程序行为，旧集群没有写lease。原共享纯holder文件私有原字节备份，最小等待分支不改任何业务训练/生成入口；资源guard只创建自有限时lease、结束身份核验后移除，不覆盖别的锁/lease。已CPU验证等待发生在torch导入前且expiry后返回。新独立分配003用同v4解码与旧精确标签门，另冻结6GPUh/新目录/claim，不删除002claim。

另在已批技术8GPUh内先行固定0.2GPUh、GPU4单卡720s原生FA2合成前后向探针：seq128/1024、24head×128、BF16、dropout0/explicit deterministic=True，各重复2次forward/backward精确torch.equal。真实native文件SHA/官方wheel/源码/publication绑定，单卡原compute模式复原，旧FA3失败费用不动。此探针不加载Teacher权重或更新参数，不是模型精确resume或质量门。正式恢复仍须完整4reference+2prefix+2fresh模型/AdamW/RNG精确验收后才能发技术准入。

原生技术探针001先做CUDA掩码即明确报错：torch2.7的Bool scatter_reduce amax CUDA kernel不支持，未进入FA2前后向；退出7.802832s、保守0.2GPUh，GPU4释放及原compute_mode恢复。此8秒独立诊断避免再次用完整32B找同错误。现在改int32 0/1原子max再bool.masked_fill，重复prefix/目标冲突/EOS掩码与原生CPU精确回归仍通过；独立探针revision2另从已批8内预留0.2GPUh、旧001费用与claim不改。

allocation003尚无GPU claim，原32视频CPU准备已保存旧freeze副本，再用当前int32源码重新SHA冻结同媒体，不生成新标签或覆盖实验结果。revision2真实CUDA掩码与FA2确定性内核通过后才加载32B；不把CPU回归再称GPU通过。技术探针尚不证明完整模型精确恢复，也不开放正式训练门。

技术探针revision2被GPU4另一已确认纯holder占用阻止，未启动worker，退出约2.389s/保守0.2GPUh，原released=false不改；技术累计0.4/8。已核argv为相同共享纯holder另root，故独立revision3识别两类明确已授权holder，并先写GPU4本地UUID限时lease再启动原生probe。不是杀未知任务，不复用revision2claim。revision3另0.2GPUh，必须原生CUDA int32掩码与FA2重复前后向真实通过后才运行未启动003。与大模型执行冻结源码一致的CUDA兼容修复和资源拒绝都留账，不因预算扩大隐藏工程失败。

### 已取得实际原生A800验证，完整模型恢复另行冻结

native3实机GPU4在lease下成功：CUDA掩码3个长度与原生NoBadWords精确等价，CPU输入token与GPU输出scores对齐通过；BF16/24heads/head_dim128/seq128与1024各重复两次，4forward/4backward exact全部通过。实际9.979866s、按0.2GPUh计费，含两次旧失败技术累计0.6/8，GPU4与原mode释放恢复。这不是完整Wan模型精确resume，不能自动改正式训练门。大模型003已以同已核int32实现和同32源病例启动，保持原精确门，结果另审计。

新增full-DiT FA2技术准入范围：5040s单卡=1.4GPUh，技术累计上界2.0/8，留6给后续multinode技术。使用三片原20GB本地base SHA、原train_step/checkpoint/load/FixedOrder/synthetic_sample；FA2 explicit deterministic=True/dropout0、torch确定性flags与CUBLAS工作空间、600adapter/AdamW/Python+CPU+CUDA RNG精确门。物理4reference+2prefix+2freshresume，逻辑horizon4，seed20261104，offload on、lambda.5。每segment真实fresh子进程，旧FA3失败不重跑、lambda/样本/状态门不临时放宽。模型forward不是标签语义验收，full-MoDel probe不消费真实标签，不是候选训练或SOTA质量成绩。源码/模型/runtime/授权/公开字节验收后才启动。


## 15. allocation003完整结案与真实训练API修订（2026-11-04）

固定32病例均实际执行，30条结构合格caption、2条caption_validate词数失败：诊断16/16，沿用原blind来源组14/16。原40..120词、16帧、EOS、实体关联门均未放宽；失败模型原文未保存，不能凭token数猜测词数方向或事后补写。所有32例、两条失败、claim和原执行源码/媒体字节已私有保留并独立核验。32/32精确结构门仍FAIL，30/32不是全量或独立泛化收益。

父生命周期1661.801988秒、实际1.846446654GPUh；完整独占资源窗口5104.808620秒、实际5.672009577GPUh，不能只按短父进程计费。保守计6GPUh，新标签阶段18/96，历史标签累计41.832564878GPUh。原compute模式已恢复，没有原始外部侵入记录；旧失败的false/null与费用不回填或退款。

完整Wan5B FA2精确恢复001在模型加载后、任何训练update前失败：引用不存在的train_e3_lora.FixedOrder，真实训练器类名为DataOrder。不是已证实的数值不确定性；原AssertionError终态与reference.log保留，计1.4GPUh，技术累计2.0/8GPUh。模型加载不等于精确恢复。

独立修订002改用真实DataOrder与Comm，在模型加载前核对真实train_step/checkpoint/load接口。新增真实API的CPU微模型4reference vs 2prefix+重新构造模型resume2回归，adapter/AdamW/Python和CPU RNG及后两次更新记录精确相等；该CPU检查不替代A800/CUDA完整模型门。新002单独冻结1卡5040秒/1.4GPUh，预留后技术累计3.4/8，前次失败SHA、原源码和费用绑定，不复用claim、不自动循环重试，完整精确门不变。

allocation003首次独立8B诊断评审另冻结1卡3600秒/1GPUh：只评审30条原保留caption，完整32病例分母与2条接口失败保留；诊断组和原blind组分开报告。沿用同七字段/四程序化正负控制/16首尾均匀帧，AI判断仍非人工gold且不自行批准语义门。启动前GPU5的不同占卡入口未符合当前精确识别/lease门，未发信号或消耗分配；改为新集群GPU0使用节点本地限时lease与精确已确认holder释放，终态须恢复原模式。标签阶段预留后19/96，旧41.832564878费用不重置；不重新生成或修改任何旧caption。

针对恢复与评审的20项CPU测试通过。完整公开测试和独立远端字节核验后才允许两项新GPU入口。正式Teacher训练、AR、Student/蒸馏依然关闭，没有新Teacher全集质量评测或同协议SOTA结论。


## 16. 模型Python缺PIDFD的真实失败与控制解释器修订（2026-11-04）

提交d34d683ccf86680e59409ec672dfabeb3e54ad7a完整公开1484 passed/3 skipped，271白名单文件及7变更经独立远端克隆逐字节核验。第一次测试命令漏带已部署媒体工具PATH导致5失败，原失败日志保留；补PATH后完整重跑通过。测试产生的pytest缓存没有发布，原测试目录保留，仅复制已测试白名单的相同字节，未放宽发布精确门。

完整恢复002与独立评审003都在资源控制阶段、任何模型/评审worker启动前以AttributeError失败。所选模型Python3.12.13没有os.pidfd_open或signal.pidfd_send_signal；从两项原执行源码对非holder自身PID安全CPU重放，均在pidfd_open处复现。没有模型update、没有新8B判断，不能将模型API的CPU通过当完整恢复成功。恢复002原released=false保留；两项原compute_mode均恢复，旧holder未被误称自有模型进程。恢复002计1.4GPUh、技术累计3.4/8；评审003计1GPUh、新标签累计19/96、历史标签累计42.832564878。各原claim、终态和执行源码私有保留，不退款、不删除锁或自动重跑。

控制修订只将PIDFD操作委托给已核验的隔离系统Python3.11.6（-I），不替换模型/训练Python，不用普通PID kill降级。实际系统解释器对存活自身PID执行pidfd_open+signal0探针通过；原精确argv、GPU placement、UID、PIDFD TERM和退出等待仍在控制进程内执行，未知任务仍拒绝。新增真实控制probe、非holder自身拒绝、参数约束与桥接契约测试；控制器7项CPU测试通过，恢复/评审控制漂移在创建GPU claim前拒绝。

独立恢复003另冻结1卡5040秒/1.4GPUh，新技术预留累计4.8/8；独立评审004另冻结1卡3600秒/1GPUh，新标签预留累计20/96。前项失败SHA和已消耗费用继续绑定。两项冻结系统解释器、signal/select/json/subprocess等关键文件的原始字节与真实PIDFD能力，CPU准入在新claim前复验。仍为同完整4reference+2prefix+2freshresume和同30段首次实际8B评审/完整32病例分母，不重生成caption、不改原精确门、七字段或四控制。完整当前公开测试和远端字节核验后才允许新GPU分配。

以上不是Teacher质量或SOTA收益；结构32/32、人工语义准入和完整模型精确恢复均未通过，正式训练、AR、Student/蒸馏保持关闭。


## 17. 完整精确恢复通过与第三轮标签实际语义诊断（2026-11-04）

恢复003的完整Wan5B在A800/FA2显式deterministic=true下实际8次更新完成：4reference、2prefix、另起进程2resume。原adapter、AdamW、Python/CPU/CUDA RNG与后两次更新loss/LR/梯度范数/样本次序全部精确相等，冻结基座hash未变；独立CPU重读三个checkpoint文件SHA及state hash、完整轨迹逐项核验通过。939.933422秒/实际0.261092617GPUh，保守计1.4GPUh，技术累计4.8/8，GPU释放且模式恢复。仅证明本冻结配置的合成单卡训练恢复技术条件，不是Teacher质量、跨节点NCCL准入或旧FA3轨迹复现。

评审004实际30/30段完成，4/4控制通过，0评审错误：23 adequate、7 needs_edit。完整32来源分母为23/32，不用23/30条件率冒充全量；诊断11/16 adequate、5需改，原blind来源12/16 adequate、2需改、2无caption。七字段有6段动作不匹配、2段运动擦除（部分重叠），AI不报实体转移或幻觉并不证明人类语义真值。223.013092秒/实际0.061948081GPUh，保守1GPUh，新标签累计20/96、历史43.832564878；原输入及review字节独立核验，GPU已释放。结构32/32和人工语义准入仍未通过，不启动正式训练。

### 同病例新前瞻标签协议，不修改旧结果

allocation004_temporal独立冻结4卡5400秒/6GPUh，先前20/96全部保留；仍原32病例和16首尾均匀源帧，实际索引0,8,...,120、24fps、640×352，不重新抽样。输入从16独立图像改为原生video token＋真实时戳，do_sample_frames=false，原生temporal patch2覆盖同16帧；实际CPU处理器验证video_grid_thw=[8,22,40]与8组时戳通过，无模型加载。此输入表示与caption协议同时改变，不声称单因素因果或新盲泛化。

facts仍使用原v2提示、1024 token与exact实体/events验证器。caption改为exact JSON四字符串，提示每句12–22词、共48–88词；仅以单空格连接原字符串，禁止裁剪、补词、重写，之后原40..120词/grounding/元数据拒绝门仍执行。greedy、384 token、EOS完整要求不变；任一失败整体32/32 FAIL，所有病例/失败费用保留。不使用AI判断作为自动修订指令，不事后把旧caption改成功。

新独立入口复用原媒体、模型、运行时和精确验证工具；创建GPU claim前检查原病例与原冻结视频/model/UUID完全相同、控制器真实PIDFD能力和公开远端字节。所有卡的自有限时lease、精确已确认holder TERM、未知任务拒绝、终态原模式恢复和预算包括完整资源生命周期。CPU协议回归和完整公开测试及远端独立核验后才可执行。本协议不保证语义改善，实机结果及随后独立review另记；正式训练/AR/Student/蒸馏保持关闭。


## 18. 官方完整评测协议接线（2026-11-04，仅CPU开发）

原build_eval_jobs.py只生成355非camera、seed0作业；原vbench_eval.py只期待8维的-0.mp4，并把每维结果解释为[mean,rows]。官方camera真实返回[mean,type_means,rows]，不能直接复用该解析器或把已有355视频当官方完整评测。

官方metadata现有1118图文对＝355core＋763camera，camera涵盖109图像×7指令；每对5视频，共5590/冻结模型（core1775、camera3815）。当前355张本地16:9输入均存在，文件名归一化无冲突。完整CPU覆盖清单在eval/teacher_progress_campaign_20261104/vbench_official_complete_coverage.preflight.json准备，逐图像hash仅绑定当前本地字节，不自行证明官方裁剪来源或分辨率对齐；五个seed是独立、非按结果选择的前瞻提案，不是实际执行注册。

vbench_eval.py新增显式--official-five：使用独立videos_official5、staging与scores_official5目录，官方全部九维/5590集合严格检查，不能借用旧355或旧八维结果。camera按三项结构验证逐视频集合、七类型均值、总体均值和分类对应0/1分数一致性；dynamic原bool格式保持，其它结果必须是有限数值。已有无效官方结果保留并拒绝自动重评分；原默认legacy流程与原结果目录不改。15项CPU回归通过，包括camera缺项/重复/不一致、全量五次集合、视频不足时评分前停止、无效原结果不覆盖。没有GPU评分或新质量收益证据。

实际camera后端的现有官方代码使用浮动torch.hub co-tracker/cotracker2并包含TLS失败后的关闭证书验证分支。正式执行前仍必须冻结适配backend源码、原权重和正常TLS/离线加载路径，不能直接运行该浮动网络分支。未加载该camera模型或下载权重；未宣称外部榜单checkpoint或采样对齐完成。完整质量评测仍受候选/语义/监督cache/远端源码与有限预算门约束。

官方五采样模式另补评分输入按模型字节绑定：同名staging视频必须与源SHA一致；每模型score_inputs绑定视频、官方metadata和评分脚本SHA。已有评分没有绑定、metadata或视频字节变化均保留原评分并拒绝复用，不能用旧同名结果污染新候选。CPU测试覆盖跨模型独立绑定及旧默认结果契约，未GPU评分。

### 完整生成计划构建（CPU，不是模型实验）

原build_eval_jobs.py仅有355×seed0的legacy作业构建。新增--official-five-plan显式CPU模式，读取已核完整5590项覆盖清单，核对官方metadata的prompt/图像/五采样行和本地图像SHA，模型由独立JSON映射显式指定checkpoint并绑定文件SHA，禁止从旧MODELS中的mtime最新权重自动选臂。多模型使用同图文和相同五个seed，输出分离到videos_official5；CPU计划及逐作业execution_allowed=false，不运行gen_jobs、不创建视频或GPU claim。旧cases/jobs不覆盖，prompt优化与旧tag选项禁止混用。正式候选checkpoint、sampler预算/backend、输入裁剪来源、语义/监督cache及远端源码门仍须另行通过，不把此准备当完整评测或SOTA证据。

## 19. Git认证恢复与原生视频标签后续评审（2026-11-04）

用户提供新增Git授权后，现有待推送正常提交a825e3f已push；独立新克隆逐字节核验275白名单文件和1542完整公开测试凭据通过。原认证阻断已解除，不再称凭据缺失。使用临时askpass/子进程环境，不把凭据存仓库、Git配置或长期记忆。

原生视频标签allocation004按冻结20d6cc7计划执行，一次4卡5400秒/6GPUh，原32病例和16帧、所有结构与安全门保持不变。case循环仍运行，任何部分成功均不是全量通过；错误不自动重试、旧失败与预算不退款。终态及独立审计完成后才启动新评审。

semantic_review005预注册单卡3600秒/1GPUh，使用同独立8B、七字段、四控制和16源帧。源实验必须有32/32实际尝试、完整终态成本与资源恢复；仅评审全部原始success标签，原失败保留完整32分母，不挑选高分标签、不重写错误caption。原native输入改变与四句JSON效果只能在全体原病例分母上报告，不能当新盲泛化证据。AI评审不自行生成semantic_training_approval，正式训练/AR/Student/蒸馏仍需原门禁通过。

### Encoder/cache实际API技术准入（非监督缓存）

encoder_cache_probe001独立冻结单卡GPU4、720秒/0.2GPUh，计入技术8阶段，先前4.8保留，新预留5.0/8。只使用程序化完整121帧1280×704常色视频与固定合成描述，不用任何未准入真实标签。复用实际cache_e3_inputs的T5/VAE加载、原tiled尺寸和encode函数；要求BF16 CPU输出x0[1,48,31,44,80]、first[1,48,1,44,80]、context[1,512,4096]有限，重复T5 context精确一致。输出仅是私有合成技术张量，不生成监督cache_index或训练样本清单、不执行DiT update，semantic_training_approval与formal_training_allowed仍false。原encoder六pin、环境、源码、资源身份、完整公开回归与远端字节门通过后才运行，不自动重试。

## 20. 原生视频004失败结案与新解码预注册（2026-11-04）

004同32病例全部实跑，14成功/18接口失败（8 JSONDecodeError、9 DenyMatch、1 ValueError；caption_validate12、facts_parse6），整体FAIL。1466.188847秒/实际1.629098719GPUh，保守6GPUh；释放和原模式复原通过，原claim、错误与执行源码已逐SHA私有保留。没有据此启动训练，旧失败不退款、不重跑原设置。独立review005四控制通过、14段全部实际评分：13 adequate/1需改，完整来源覆盖13/32；旧策略23/32，新策略13/32，不能用13/14条件率声称泛化更好或SOTA收益。

真实encoder/cache技术探针001另已成功：合成完整121帧、原VAE/T5 API产生x0/first/context原形状的CPU BF16有限张量，重复T5 context精确相等，文件与张量SHA独立CPU复核。实际0.030331711GPUh、保守0.2，技术累计5.0/8。它不使用真实未准入caption、不写监督cache_index、不执行DiT update，不替代监督缓存准入。

### 独立allocation005_schema，不是原失败自动重试

从已批准96标签阶段另冻结6GPUh，先前27全部保留，新上界33/96。同32病例、原16帧时戳、相同两阶段prompt/greedy/1024+384token/原40..120词与实体关联、安全、EOS门；唯一解码协议改动是实际JSON schema约束＋已验证的固定拒绝词掩码。四字符串JSON仅单空格连接、不剪裁补词重写。facts语法明确五个原键、实体与事件类型、显示索引0–15整数枚举；顺序实体ID与跨实体事件关联仍由原验证器核验，不靠schema假装语义正确。库数值minimum/maximum在真实CPU测试中未生效，已改明确整数enum而非放宽范围；实际库回归22项通过。

LM-format-enforcer0.11.3、interegular0.3.3和纯Python pydantic1.10.24只放新隔离site，官方wheel和成员SHA留证；未改运行中venv或旧环境。真实Qwen tokenizer四字符串合法/非法字符与token路径CPU探针通过，EOS与完整数组边界对应；不会据CPU测试宣称实际模型输出通过。新worker须先在真实CUDA3 BF16 logits与CPU input_ids上执行短语法/拒绝词掩码准入，通过后才加载32B，避免再用完整模型查掩码设备错误。任何失败整体保留，不自动重试、不复用004claim。

旧004独立审计曾误用系统Python3.11导致PEP701源码解析失败；改用已冻结Python3.12仅做CPU审计后通过，没有重跑GPU或篡改原结果。

### schema输出的独立review006前瞻门

另冻结1GPUh/3600秒，在schema005费用上界33之后计入标签阶段34/96。与review005同8B模型、七字段、四合成控制和原16源帧；原schema源必须实际32/32尝试、完整终态费用与资源恢复，评审全部保留success标签、失败不补写、不从高分子集选样。每例原图文/label字节和源计划SHA重新绑定，语义训练批准始终另行检查，AI adequate不自行变成人工gold。


## 29. schema005完整失败与boundary006前置修复（2026-11-04）

- schema005固定32例均实际执行，0有效标签/32错误；每例facts均到1024 token且未EOS。两组分别0/16。不能凭未保留的原文猜测具体循环内容。
- 实测完整生命周期3.617963761 GPUh，保守记6；新增标签阶段累计33/96，旧23.832564878 GPUh不退款。释放与模式恢复、独立审计通过。
- review006源码已完整1566测试通过、279文件独立远端字节核验；该发布不是实际评审。零标签不分配review GPU，不生成任何质量或语义批准。
- 纯CPU真实Qwen tokenizer发现LMFE自由字符串快捷路径遗漏合法跨闭合引号token，例如句号+引号+逗号；新增小型边界trie只补入逐字符通过原JSON语法的token，原拒绝词掩码和输出验证器不变。
- 修复后facts 87-token和caption 9-token完整路径/原EOS门纯CPU BF16验证通过，CUDA未初始化；26项针对性回归通过。尚不代表模型实际生成能够完成或语义正确。
- boundary006为独立新协议，绑定schema005原claim/审计，原32例/native16帧/prompt/greedy/1024+384 token及40..120词门保持不变。模型加载前同时核验facts/caption路径；首次未EOS或AssertionError即停止剩余病例，不重试，剩余按未执行错误保留完整32分母。
- 预留独立6GPUh后新增标签阶段39/96，技术5/8，正式训练评测0/64。完整测试、白名单脱敏发布和独立远端验证前不开新GPU任务。正式Teacher训练仍未启动，SOTA未证明。


## 20. schema005失败与boundary006候选修复（2026-11-04）

- schema005完整32例实际尝试，0标签、32错误；每例facts生成1024 token后未EOS。独立审计确认诊断/盲来源各0/16，实际3.617964 GPUh，保守6 GPUh，四卡释放和模式恢复通过。原失败、claim与费用不改。无有效标签，不启动review006 GPU评审。
- CPU合成完整facts路径发现lm-format-enforcer自由文本快捷路径拒绝合法跨闭合引号token；候选修复仅为小型边界token树，逐字符沿原JSON语法检查，不后处理文本、不放宽原词数/实体/EOS门。该发现不是实机失败唯一根因或语义质量证据。
- boundary006独立协议绑定schema005审计/claim与旧review，保留同32病例、16源帧、prompt、greedy和1024/384 token。模型加载前验收完整合成facts及四句caption的CPU输入/GPU BF16 logits路径。首个未EOS或AssertionError即停止后续病例，32分母与未尝试记录保留，不自动重试。
- 新分配最多6 GPUh，标签阶段已耗33，新预留39/96；技术5/8，正式训练评测0/64。完整测试、正常push与独立远端字节门通过前不开新GPU任务。语义批准、监督cache、正式训练和对齐SOTA证据仍缺失，不将CPU回归称为模型成绩。


## 30. boundary006早停失败与有界facts后续（2026-11-05）

- boundary006已通过1570完整测试、279文件远端独立字节核验并实际加载四卡BF16。首例facts耗尽1024 token、grammar_can_end=false、inside_freetext=true；不能从未保留原文推断具体循环内容或宣布唯一根因。
- 早停执行：1例尝试失败、31未尝试、0/32标签。实际0.352718508GPUh；保守记6，标签累计39/96，技术5/8，正式训练评测0/64。原时点释放和模式恢复通过；后续新占卡不算本实验泄漏。
- 原独立审计因未尝试记录缺failure_reason，None与字符串键混合排序失败；不完整原件保留，另存v2完整CPU审计并绑定原执行源码归档与原释放凭证。未重复GPU。
- 新有界facts仅在独立后续协议使用：scene/description160字符，held_object80，event action120，net difference200，合法实体E1..E32；仍保持32实体/128事件、原16源帧/32病例/prompt/greedy/1024+384token/40..120词及原实体与EOS验证。不是事后裁剪/补写或语义批准。
- CPU adversarial160字符继续token被拒、闭合引号允许但EOS仍拒，完整合法facts路径EOS通过；模型加载前需同掩码真实CUDA复核。初稿6GPUh未执行；核对当前GPU0/1新占卡身份不符合原准入后，改为仅原物理GPU2/3的两卡BF16方案，新的上限3GPUh，预留计42/96；原初稿文件保留、未消耗费用不作为退款。首例未EOS或断言仍立即停止。所有标签/监督cache/正式训练与SOTA门仍未通过。


### 有界标签后续语义复核和安全官方评分准备（2026-11-05）

独立review007前瞻模式复用原8B、七字段/四控制/native16证据，只接受有界源全32病例实际尝试、完整费用审计及释放/模式恢复，全部保留success标签进入复核；失败和未尝试仍计完整32分母，不选择高分子集、不修补caption。指定原授权GPU2、最多1GPUh，计划通过后才登记预留。AI adequate不会自行成为真人gold或semantic_training_approval；全121帧本地盲审材料和原视频入口另行准备，未启动listener或填写真人判定。

正式VBench camera离线加载候选保持原CameraPredict推断/分类/评分函数不变，只替换模型构造，固定官方CoTracker2权重SHA和完整本地源码，禁止浮动torch.hub和TLS证书校验关闭；score_inputs绑定离线后端凭证SHA，变更后拒绝复用旧成绩。缺失依赖置于隔离site，不改冻结训练venv的torch/numpy。真实CPU权重、模型构造、隔离依赖文件SHA和26项评分契约测试通过；未执行GPU视频评分，也未证明完整质量/SOTA。

公开DreamX-World-5B-Cam双向checkpoint清单已按不可变修订做CPU盘点，文档默认121帧/704×1280/24fps/50步/CFG3，需显式camera动作。公开5B-Cam与旧榜单DreamX-World-1.0精确checkpoint映射及官方五采样协议未证明；仅作可复现参考，不能据该映射宣称SOTA对齐。未下载参考模型权重、未运行AR。
