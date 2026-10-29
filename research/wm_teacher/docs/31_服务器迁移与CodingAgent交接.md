# 正式恢复准备实证（2026-10-29，GPU资源已获授权）

- 四节点各9模型当前真实内容核验完成，共36资产，统一digest 501bbc43c93e55695a1ded0e6910621027f6c055967c76caafbbfdfa0b9423ff；外套32占卡已按另行明确授权精准TERM，旧文件/其他项目未改。
- 新真实生成/评分代码完成PID重用、身份读取失败清理、正时长/顺序释放查询、原子账本、硬截止后费用失败证据、native已知路径/真实filedescriptor、130秒尾buffer/候选最坏case投影及host-scoped评分owner验证；最终相关295项CPU回归通过。仅支持已声明受控loader入口，不声称任意native IO/预捕获函数别名都被拦。
- 实际评分venv与模型/依赖全内容冻结完成：52模型/缓存文件、16804依赖文件、106169环境库文件；最终v3 bundle SHA 7ccd524bbbcb55d8f0ba5b99e2c9b0b1e04ebf1f39a05cabeeb36a62a5ea2887。官方local=True及新本地缓存路径固定，数学/8维/GT K12不改。实际发现空trusted_list属于TorchHub缓存元数据，只为该元数据及合法源码限定例外，空权重/工具/媒体仍拒。
- root0实际scoring_owner内容证明已通过CPU全核构造；非评分node不复制整套env，仅核已批准shared证明，自己的9模型/runtime/source/输入继续实际SHA核。四node本地CPU探针各40唯一更新、退出0，对应最终recovery源码。
- 前一版仅CPU预冻结计划因真实owner字段与local/cache合同不兼容已原字节归档，记录teacher_prelaunch_source_amendment_1793223312547205996.json；没有GPU attempt、没有实验输出、没有删除或覆盖原始证据。当前重新冻结同94×seed0/1、同原预算，执行仍只有recovery_a1一次，不借旧失败费用或变参数。
- 正式生成尚未启动；需要先完成公开树全回归/推送/远端字节核验、三计划comparison和6个owner CPUproof/全16配额sidecar，再受控执行原生base/ALG/DyMoS正式分片0，各12视频计入原188、只读机制/吞吐不看中途质量，合格后才剩余15。人审实际认证未完成，数值可报告但JSON结构不自动晋级/不启AR。

---

# 当前资源恢复与正式执行准备（UTC 2026-10-28T19:51:00.719Z）

用户明确“四节点都可以使用、继续实验”，解除下方历史GPU暂停；另明确批准释放ms_gpu_hold_20261028_v1占卡。已精确停止32个该脚本worker及rank0专属shell看护，仅SIGTERM、无强制kill、无文件删除，不终止tmux服务器或其他项目任务。实际无GPU计算进程，初始显存/利用率退场采样需要后续启动前再查，不把瞬时低显存数当长期空闲保证。

本轮仅恢复已授权Teacher固定94×双seed/原预算：base20生成+5评分；ALG/Dy各smoke1/gen24/score16/failure4。旧失败费用上界继续扣生成，实际未知不能清零。新真实生成/parent释放/有界评分接口正在补齐并独立CPU审查，必须新source/实际模型runtime/原日志退役proof及完整公开测试发布通过后才GPU；不是一条资源授权就绕过BLOCKED。三个正式plan目录尚未创建、不重启旧共享账本任务、不盲训AR/Student/蒸馏。

拟冻结base ranks0/3、ALG ranks1/2第一批；DyMoS ranks0/3在base释放后执行，各方法固定首正式分片12视频作机制/费用准入，计入原188、不额外smoke、不看中途质量。准入不通过即停止对应方法，不能重挑参数/case或借阶段预算。实际sourcefreeze和dispatch sidecar最终为准。资源授权私有记录/srv/wm_teacher/eval/diag/teacher_resource_resume_1793217060719.json，不公开节点私有日志。

---

# CPU阶段实施更新（2026-10-28）：以下GPU暂停状态为上一快照

> 发布后本地回执补记：CPU快照3b3f3b2f05b536c07d55128e8322a16de6e847b6已正常推送，21增量文件远端字节＋211manifest SHA通过，发布树干净。回执/dockerdata/wm_tmp/remote_cpu_components_20261028_3b3f3b2.json。本行仅为提交后的本地运维记录；公开内容以该commit及manifest为准，不冒称此行已在该commit中。

用户“请继续推进”按紧接调研建议仅执行CPU工程与验收，不解释为解除资源让位。未启动CUDA、训练、生成、实际评分、占卡或管理监督器；旧视频、模型、缓存、失败账本与旧计划保留。**当前新生成数仍base/ALG/DyMoS各0；Teacher默认仍base s5_c5，AR/Student/蒸馏门禁关闭。**

## 本轮已实现的CPU组件

- scripts/prepare_teacher_recovery.py：独立新recovery_a1目录协议、本地fcntl固定lock inode、唯一temp/文件与目录fsync/rename、每node仅own8可变、global16不可变额度；ALG旧生成上界5926.31142393779秒＋16×5000秒=23.868419840GPUh，Dy9559.88809741661＋16×4800=23.988857805GPUh。旧上界不是精确费用，不挪到failure或清零。
- 旧run_teacher_devset_node退出汇总修复：账损坏仍先清理准确owned session，原损坏bytes不重建；true_elapsed_GPUh在未知时为null，actual_known与charged_upper分开，缺finalized分片保留cap，不因finished-only求和显示0就称免费。
- scripts/gpu_hold.py --monitor-only：只管理PID/UUID/租约健康，不spawnCUDA占卡worker；idle/external/pause/queryerror CPU模拟均验证。**本轮没有实际启动manager，也未声称新实验监督已实机集成。**
- scripts/run_teacher_base.py：原生NONE、94×双seed188、生成20＋评分5=25GPUh独立CPUprepare/verify；内部native helper不装ALG/Dy context，失败/媒体/参数SHA证据保留。helper不是parent释放或真实生成完成证据。
- scripts/prepare_teacher_comparison.py：新base与两新recovery候选共同cases/image/GT/prompt/NEG/models/runtime/common-source/score bundle内容冻结；拒旧candidate schema/root、未知费用清零及旧预约假退役，不以同commit代替内容证明。
- scripts/score_teacher_devset.py：完整显式依赖/评分模型/runtime bundle、流式SHA、不限制真实模型为64MiB；生成媒体/GT/image/cache/source/stage/score绑定，same-size＋还原mtime的媒体漂移仍拒。
- scripts/summarize_teacher_devset.py：原8维/GT/Total公式与每seed四护栏，dynamic>0和Total>=0的预注册资格；固定2000/20261025、同case两seed一起重采样，94主结果及剔除egoexo6后的88父源已解析敏感性并列。6例父源未知明确UNKNOWN，不把188当独立样本。完整视频本地盲审rubric、独立两reviewer及冲突第三裁决仅为协议，schema不是human gold。

## 独立审查发现并已修复的两项真实缺口

1. cache_binding原先只抄计划媒体SHA，独立same-size/还原mtime修改可绕缓存接口；现先逐媒体实际内容check_record，binding旧字节不重写。
2. 费用collector原先可被手造零elapsed/owner/释放布尔receipt减掉预留；现独立parent释放/身份/时钟 verifier未集成前默认闭门，零/正伪receipt均不释放任何cap、actual继续未知；reported超cap保守计入而非截断。CPU原子账本PASS不能当真实费用或GPU释放证明。

## 实机与验收范围

两真实节点仅使用/dockerdata本地隔离CPU探针：每node4worker×10次，各40唯一更新，合计80；全部worker退出0，读者持续严格JSON完整/计数一致。最终源SHA7256abbc0842ea9ba31fc86be494186d83b30626c38ba9f509bef24ddd527d01，测试SHA cc5c783f6c9c28e8f4f83899250060aa9b3dcd75d8257de396a5231b9f4b53b8。私有证据eval/diag/teacher_local_cpu_verified_1793154121297.json（不公开hostname/路径/原日志），旧失败40/59/40报告仍保留。

最终相关CPU联合回归（本轮六新组件＋旧开发监督/E1/E3/导出器）265 passed、exit0；独立11项合成edge攻击测试零失败。公开树完整CPU回归826 passed、3 skipped、exit0；额外核对默认收集包含全部138个新增组件测试。独立最终9项攻击＋6项管理scope mocks共15通过且未mock parent release，源SHA未漂移。公开依赖缺失已通过补入当前devset与只读audit模块修复，不跳过失败。远端发布仍须最终receipt确认。

**这些只是本地互斥/持久写CPU primitive实机PASS，不是新恢复GPU driver准入、native base预算通过或质量PASS。**前一批并发源码变更导致source-hash测试失败保留，稳定后须最终重跑；CPU回归与公开发布数量以本轮最终证据为准，不沿用旧计数。

## 未完成且禁止自动执行的环节

- 所有新run入口、verify_generation_completion、verify_scoring_execution、verify_parent_release目前明确BLOCKED；旧共享账本驱动不能直接用于新recovery计划。尚无全生命周期GPU监督、旧完整终止日志认证与reservation retirement amendment、新node-ready桥接、实际评分stage账本或正式canary调度。
- 新base20生成/5评分尚无真实同工作负载吞吐准入；候选原24生成/16评分/4失败/1烟测阶段不得挪用，Dy总余量约40.112GPU秒不能当每片可超40秒。
- 尚无3×188真实生成/评分completion、实际盲审或候选质量结论；旧base.allowed=False不改，C276不重评不晋级。
- 用户明确恢复GPU并核实节点范围前，不启动GPU或占卡。合格开发候选后的独立公共确认评测/完整1366/SOTA对照仍需另行范围与预算，不复用已暴露C276冒充独立验收。
- 本轮发布仅CPU组件及无身份结论；方案/代码/分析仍走白名单→干净树→公开全回归→正常提交推送→远端字节核验。本文写入不等于已发布，最终receipt才是依据。

---

# 当前暂停状态与最终实验交接（2026-10-26）

## 0. 用户暂停优先：不得自动恢复实验或占卡

用户最新指令：“当前暂时终止所有进程,先把显卡空给另一个任务.在这个期间你可以总结之前的实验情况”。本项目已暂停；早前持续推进目标、四节点资源授权及闲置占卡要求不构成现在恢复的许可。用户明确恢复前，不启动训练、生成、评分、占卡或恢复代理；仅整理已有结果与交接。

- 暂停执行时，四节点各停止1个本项目占卡监督器＋8个占卡子进程；仅SIGTERM，未使用SIGKILL。没有删除视频、模型、缓存、权重或失败账本，没有停止其他项目进程。
- 本次只读复核UTC：2026-10-26T14:35:17.624Z。四节点均无本项目残留进程、无GPU计算进程，32卡利用率均0%。rank0/rank2各卡0MiB；rank1各卡1MiB；rank3 GPU0/1各4MiB、GPU2–7各1MiB。驱动少量显存残留不等于本项目仍占卡；这是一时点快照，不保证其他任务随后不使用GPU。
- 被暂停的节点本地恢复代理未恢复。当前阶段任务为总结，不能把旧活任务句柄或文档中的“运行中”当作恢复入口。

## 1. 已完成的实验与结论

### E3三臂双seed开发集

A原caption；B约束性修正caption；C修正caption＋基座保真anchor λ=0.5。相同1856训练窗、254更新、LoRA rank32。seed0/seed1每臂各94视频及完整评分、GT、配对和独立核验均已完成。

- seed0：A的DreamSim护栏失败；B/C通过，按预注册规则固定选择C。
- seed1：B的DreamSim护栏失败；C通过。不能跨seed平均掩盖失败，不能重选臂。
- C对base Total*差：seed0 −0.135646pp，seed1 −0.106154pp，描述性均值−0.120900pp。只是获候选资格，未证明整体提升或SOTA。详见docs/49。

### C276固定四基准子集：已闭环，不重跑

固定seed0/s5c5/50步/121帧/1280×704@24fps；raw71＋po71＋WM22＋Physics14整场景三视角42＋WMB70=276。全部276输出、33396帧全解码、source/local SHA和零残留通过；五suite完整评分/同子集历史base配对/2000次bootstrap、独立9884检查零失败完成。

| 集合 | C | 匹配base | 差值 | 95% CI |
|---|---:|---:|---:|---|
| VBench raw Total* | 90.105419 | 89.934739 | +0.170679pp | [-0.218970,+0.608091]pp |
| VBench po Total* | 89.679281 | 89.493260 | +0.186020pp | [-0.024820,+0.488302]pp |
| WM22 Total* | 87.464093 | 87.380459 | +0.083635pp | [-0.147141,+0.414795]pp |
| Physics-IQ Original | 26.892795% | 26.111923% | +0.780872pp | [-0.767229,+2.240003]pp |
| WorldModelBench70 | 8.342857/10 | 8.500000/10 | −0.157143分 | [-0.500000,+0.157143]分 |

WM22四项容忍护栏通过，但raw主体一致性/平滑、po成像、WMB形变及部分领域下降；以上总分差的区间均包含0。**不全面晋级C，保留base s5_c5整体默认，不称完整1366或SOTA验收。**详见docs/58。

实际生成28.483680308GPUh（原六次26.542880255＋补片1.940800053）≤批准30；评分1.100151777≤12；总29.583832085≤批准42。旧失败/超时/重复均保留，原40预算plan与用户42授权侧车不互相覆盖。

### ALG / DyMoS：机制及吞吐完成，开发集质量未完成

- ALG：前10/50步执行参考低通三分支引导；真实机制烟测完整50步/121帧及基座参数前后SHA通过。
- DyMoS：前10/50步在正条件self-attention对非参考query→首参考key施加gamma=0.6偏置；融合FlexAttention机制、完整媒体及参数SHA通过，不退回dense S²矩阵。
- 同一pipeline一次加载两seed实测：ALG推理＋保存解码389.460/386.507s，累计失败/机制/性能0.433371GPUh；DyMoS368.060/360.618s，累计0.413873GPUh。实测只证明机制与速度，不证明质量。详见docs/55、57。
- 正式各188双seed开发集首次启动因共享账本截断/丢更新失败，截至暂停各0个有效开发集视频；旧账、计划、日志及receipt保留。旧正式失败费用保守上界ALG1.646197618GPUh、DyMoS2.655524472GPUh，与上述烟测/性能费用分列，不能当精确费用或清零。
- 真实两host CPU共写测试应完成80更新，三方案实际40/59/40，全失败。本机单测通过不能授权该共享锁方案继续GPU。证据：eval/teacher_ledger_actual_twohost_failures_20261026_1410_629ce184.json。

## 2. 未完成事项：仅在用户明确恢复后推进

1. 节点本地可变账本＋固定全局额度＋结束集中汇总；真实跨节点CPU验收及恢复执行尚未完成。保留旧unknown费用上界，不覆盖旧账，不依据损坏JSON的首对象恢复。新的运行源码/计划须先按发布门禁验收。
2. ALG/DyMoS各94例×双seed=188完整生成、全帧/SHA、原8维＋GT、每seed护栏及严格base配对、预算与结论发布。
3. 用户另批严格新base188条，上限25GPUh=生成20＋评分5；尚未生成。旧来源审计strict_baseline_reuse_allowed=False不能被改为True：输入匹配和AST一致不足以证明历史实际执行版本、模型及runtime哈希链。
4. DenseDPO真实偏好数据/新独立语义review仍未过门禁。解析器结构修复不是human gold；不盲训。Teacher达到SOTA及用户review前，不启动AR/Student/蒸馏。
5. 完整1366条基准/180GPUh提案未获授权，不自动启动；276子集不能替代该验收。

## 3. SOTA与发布边界

- 尚无新Teacher全面达SOTA证据。VBench固定8维Total*：历史base89.860075，DreamX公开行重算90.787722，描述差−0.927647pp；逐维dynamic公布最高差−26.912439pp。不是同seed/分辨率/时长的严格排名；camera未评测，疑似错列的美学/成像最高不当可靠SOTA。
- Physics Original/Verified、single/BoN、I2V/V2V不得直接减分；WMB混模态旧榜与70子集不当严格差距；WM108无同集合公开SOTA。详见docs/52。
- C276最终结果/监督代码此前已发布b53a941，10 changed files/198 manifest远端字节核验通过。本次核对本地发布树干净、HEAD=b53a941；未重新核验远端，不把旧回执叫本次核验。
- 本暂停更新目前仅写入本地docs/31，未导出、测试或推送。后续发布仍须白名单→发布树干净→公开树测试通过→提交推送→远端字节核验。
- 本次确认docs/59*及docs/60*不存在；此前被中断写入不能当成已完成文档或预注册。

---

# 历史现场续接（2026-10-26，以下为暂停前快照，旧运行状态不得作为当前指令）

- E3 seed0和seed1三臂各282视频及全部评分/独立配对已完成，固定C不重选；C两seed Total*差均值−0.120900pp，尚无全面收益或SOTA晋级。
- C276全部视频已补齐并全帧验收：276/276、33396帧、source/local SHA和零残留通过，global_completion_a1真实exit0。生成累计28.483680308GPUh（旧6次26.542880255＋一次修复1.940800053）≤批准30；评分预算仍12、总42。原plan和旧失败记录保留，禁止重生成。当前raw/po各8维已完成，其余WM/Physics/WMB评分在运行。
- 全局完成器和评分预算接口修复已eccfdb1远端13文件/182manifest字节验收，64相关CPU回归、697公开回归通过。活评分托管任务KlfuMW（句柄仅当前会话），最终summary及结论发布未完成，不宣称质量通过。
- ALG/DyMoS真实修复后attempt02均运行机制PASS：各50步/前10步active/121帧/全解码，基座逐参数前后SHA相等。含attempt01失败累计ALG0.171741013、DyMoS0.164733928GPUh均<1。结果fc445d8远端5文件/184manifest验收。仍无94双seed质量结论，开发评测需真实吞吐准入，不能捏造旧分段计时。
- recaption完整根扫描/attrs=list[str]/v2 resume门禁修复24CPU通过，旧1854中571根fragment＋36 hands=null结构拒绝607，旧数据不重写。schema不是human gold，DenseDPO真实偏好标签/新独立语义review仍未通过，不盲训。
- 四节点空卡已按用户要求持续计算占卡（interval0、约1GiB、6小时硬限）；让位真实实验、占卡费用单列，不称实验有效利用率。SSH三远端均root@且36000。
- 最近远端验收：88856fc E3/四seed；ca880e8 子集代码；faaee9e恢复/ALG/SOTA协议；89e0e3a seed1与双策略机制；b8c23ea解析修复与42授权记录。另eccfdb1运行接口修复、fc445d8真实烟测结果均远端逐字节验收。各次白名单/公开全回归/回执保留；C276最终评分和开发策略质量验收未完成。

---

# 最新交接说明（2026-10-25，本轮接管更新，以下为上一快照）

**先读**：docs/48（背景/全景/结论）、docs/47（idea与差距）、docs/43-46（各实验预注册与结论）、本文件下方历史段。会话记忆在 PRIVATE_AGENT_MEMORY

## 正在进行（唯一活任务）
E3 seed0三臂282条生成、24维评分及GTref三份各94、全帧解码/hash/配对独立复算均通过：A DreamSim FAIL，B/C PASS，按冻结规则选C；C Total*比base低0.135646pp、dynamic差0，**只是候选资格，非质量晋级**。seed3 188条完整解码/16维评分/GT已完成，四seeddynamic均值差−0.265957pp，样本std2.193141pp；旧三seedstd3.29错误已修正为2.214538。

用户明确选择先进行**C臂276条分层子集诊断**（docs/50，硬上限40GPUh=生成28＋评分12，四节点32卡并行）；完整1366条/180GPUh提案未授权、不启动。继承E3seed1三臂各94已生成并从DiffSynth子树SHA复制到标准eval/e3_devset_eval_seed1_20261025/videos，原目录保留，独立解码/评分待完成。本轮四关发布中，尚未核验的提交不提前称发布。

## 新agent接手后的下一步（按序）
1. **已完成seed0评分，不重跑**。现优先发布E3/四seed结果与C276子集预注册，冻结清单/权重/代码/hash和全CPU测试后才生成。继承seed1完整解码后评分，仅报告稳定性，不重选C。原评分入口保留供复核：
   - 每臂staging到评分目录后用 vbench_eval.py --suite vbench --model <arm>（公开树流程）或 score_e1.py（本地树流程，--root eval/e3_devset_eval_20261024，CONFIGS需加三臂名）
   - GT指标：score_e1.py gtref（DreamSim/运动误差）
   - base参照组：eval/e1_devset_20261021 的 s5_c5 seed0 已有全部评分，直接复用
2. **护栏判定已独立通过复算**（docs/43第3节）：A失败、B/C合格、选C。先276子集诊断，不在公开数据调整臂/参数，不冒称完整/SOTA验收；完整质量review与AR门禁未通过。
3. **结果发布GitHub**：按既有流程（导出白名单→发布树干净→公开树测试全绿→提交推送→远端字节核验），发布脚手架在 /dockerdata/wm_tmp/publish_*.py 与 verify_remote_*.py 可复制改造
4. **seed3完整评分已完成，不重跑**。scripts/summarize_e1_seeds.py严格8组×94验收，公开四seed聚合已生成待四关发布。继承E3seed1全解码后按原入口评分，base seed1仅合法同条件复用。
5. 之后攻击面按docs/48第4节：DyMoS/ALG实现→DenseDPO

## 关键路径与产物
- 三臂LoRA：/srv/wm_teacher/models/train/e3_arms_20261023/{armA_orig,armB_fixed,armC_fixed_anchor}.safetensors（已按路径哈希部署四节点 /dockerdata/models/wm_checkpoints/）
- 三臂训练run（私有）：三节点 /dockerdata/wm_tmp/e3_train_arm*_20261023/（summary/checkpoint/export/progress齐全，254步×3、预算内、冻结基座hash不变）
- 训练缓存：四节点 /dockerdata/wm_tmp/e3_cache_pool_20261023/（1856样本×orig/fixed双caption、cache_index.json sha 5dbb7fa0）
- 重标注记录：training_metadata/e3_recaption_20261022/（pool_records_shard*.jsonl + fix2 + pilot17）
- 评分环境：VBench用 /dockerdata/venv_vbench/bin/python（含dreamsim）；生成用 run_gen.sh（NODE_RANK/NNODES，checkpoint缓存键=绝对路径sha256）

## 节点与运维备忘
- rank1=ssh root@PRIVATE_NODE_ADDRESS -p 36000（免密可用，用户名形式必须root@），rank2=ssh root@PRIVATE_NODE_ADDRESS -p 36000，rank3=ssh root@PRIVATE_NODE_ADDRESS -p 36000（均免密；当前22端口拒绝，不用旧22连接）
- 共享盘高负载时避免多进程读大文件（页锁D状态）：模型/帧解码先拷本地盘（Qwen本地副本 /dockerdata/models_qwen25_local_20261023）
- 用户要求：不用资源时挂占卡任务；无用CPU进程已清（8791网页服务、fileWatcher）

## 已发布记录（GitHub分支wm-teacher-experiments-20261018，全部远端核验）
f14d6a1(docs/43) → a2fc2f3(docs/44) → 4e40023(E1代码) → 276a102(E1/E1b结论) → ab2f992(E2结论) → 9f36de4(三seed方差+E3代码) → c5ae0b3(docs/47+守卫) → c1d103e(docs/48全景)

---

# 最新技术验收更新（2026-10-20）

单卡cached-input LoRA真实4更新及保存恢复技术smoke已独立验收PASS：0.087707allocatedGPUh，峰PyTorch allocated21.5941GiB/reserved22.9355GiB，基座hash不变，adapter/AdamW/RNG扰动后严格恢复且同probe预测/loss差0。自有进程退出、GPU释放；详见文档42与无身份聚合JSON。这不是原/校订训练对照、FT3质量收益或Teacher通过。公开15项CPU测试读取随快照提供的原framework，不依赖私有缓存。

四节点GPU数量由助手自主安排，仍保留预算/失败计费/数据和用户门禁；独立第二语义复核两次上游空流仍阻塞，不能用主流程试审替代独立review或human gold。

# 最新资源与技术smoke更新（2026-10-20 19:35 CST）

用户重新授权四节点全部可用，rank3不再保留给其他实验；下方原让位段仅为历史。GPU数量选择由助手自主，不再逐小实验询问，但仍自限阶段预算且用户暂停优先。

下一任务仅单卡LoRA技术smoke，最多4optimizer更新/2allocatedGPUh，复用M1已核真实121帧校订text/VAEcache。代码及CPU验收完成后才实际GPU；独立语义复核未通过，不用它绕过正式训练数据门禁。详见文档42，不是全量FT3/Student/AR/质量晋级。

# 资源让位更新（2026-10-20）

用户要求空出一个节点：rank3（worker-2）整机留给其他实验。已核8卡均0MiB且无GPU计算进程；本项目后续只用rank0/1/2，不在rank3训练、生成、评分或占卡，直到用户重新授权。无进程需要终止，无文件删除。实际节点地址与约束见私有eval/diag/wm_resource_allocation_20261020.json。

# C1完整验收更新（2026-10-20）

- C1新12原caption视频及12评分完成，旧12校订对照复用并完整hash/解码；原算法公式和pair diagnostics独立复算一致。实际新增1.894074 allocated GPUh/上限6，详见文档39。
- 校订减原平均：base主体+1.765pp/画质+1.321pp/DreamSim−0.026690，A07主体+2.459pp/画质+4.672pp/DreamSim−0.033748。仅输入版本效应，机器人动作失败仍在，不称模型训练成功/泛化。
- 本轮仅主节点4卡，未充分利用四节点；四节点现状需要现场核验，不把历史32卡或占卡当本轮吞吐。原raw/po/CSV/旧机保留，Teacher自动FAIL、用户review pending。下方运行ID和实时状态均为有时间戳历史。

# 当前语义校准与C1推进（2026-10-20）

- M1实际18状态/72前向及独立54,378数值复算完成，0.182380 GPUh/上限2。五项文献/完整M1与L1解码报告已发布f4d175a，95文件和5远端关键文件核验通过。
- L1固定17个首样本的全部121帧agent校准试审已完成：13例至少一可见矛盾、12例动作类矛盾。非340总体错误率，非人类gold/Teacher review；原340语义审核仍pending，原CSV保留。
- C1已实际启动4卡原caption条件推理：六case×base/A07×CFG5新增12视频，对照复用既有P1校订caption12视频。仅caption版本变化，最大新增生成2+评分4=6 allocated GPUh。真实run=/dockerdata/wm_tmp/teacher_C1_20261020_run01，托管wwKpv9，父PID141332；生成完整解码和预算通过才评分。C1专项17、源241、发布236测试已实跑通过，运行/单测不等于12视频评分完成。
- C1不是训练，不证明FT1/A07训练caption归因，不混raw/po或公开benchmark扫参。Teacher自动FAIL、人审pending；不盲FT3/Student/AR。下文历史进度按时间理解。

# 当前已验证状态（2026-10-19）

- P1六来源×base/A07×CFG5/3共24视频和24原算法评分全部完成，完整decode/hash/公式独立复验PASS，实验实际2.912882 allocated GPUh/上限12。
- 降CFG不是A07安全修复：主体+1.224pp但画质−3.453pp；末帧/运动guard失败、机器人motion比0.418441且抓取失败。base机器筛选PASS不等于语义/Teacher通过。见[完整结论](34_Teacher_CFG开发消融结论.md)。
- 完整P1方案代码分析与结论已推送指定Teacher分支b5ba1c6c29c224a4a56749ef5cb8557da02b4af2，远端3关键文件逐字节及81manifest提交文件核验通过。
- 后续先同x_t正/负预测机制诊断与随机分层标签审计，再决定小训练或Zero单因素；新GPU试验预算/实现须冻结后才执行。Teacher自动仍FAIL、人审pending，不启Student/蒸馏/AR或盲FT3。
- 一小时独立空闲占卡授权已到期并释放，不擅自续占；旧机保留。下方旧进度均按时间戳理解。

# 当前 Teacher 优化接续（2026-10-19 10:52 CST）

- 上一轮全部插值评测与最终发布已完成；质量自动 FAIL、人审 pending，见文档32。下文10-18生成/评分任务均为历史，不得重启。
- 文档33 P1：六来源×base/A07×CFG5/3共24视频已实际启动。4张A100采样中（首批30–31/50步），每卡约29GB；此时最终视频0/24，不称完成。
- 清单、原/新caption、CSV实际物理行和视频内容hash已独立核验；源码与DiffSynth运行包已复制到本轮本地目录并逐文件hash冻结。独立183项单测通过不代替真实生成/评分。
- 本轮生成上限4 allocated GPUh，评分上限8、合计12；仅24条完整解码/zero exit/预算通过才启动原算法评分。操作状态见eval/diag/teacher_opt_state_20261019.json，任务句柄只在当前会话有效，接续须实查。
- 用户另授权四节点空闲占卡最多1小时：28张未用于P1的卡已验证低功耗预留（约681MiB/卡），独立最多28 GPUh，真实实验优先；不把占卡称为实验利用率/质量进展。到时自动释放，必要时用私有账本RELEASE_ALL标记提前释放。
- P1新代码/方案/结果发布尚未核验；不能沿用上一轮GitHub提交宣称本轮已发布。

# 完整插值评测验收更新

> 2026-10-18 23:13:15 CST：原3033视频完整解码、三模型全部WM/公共VBench8维、GTref108、Physics198/66、WMB350与三份固定120loss已严格通过；summary和Teacher页重建完成。见docs/32_插值Teacher完整评测结论.md。实验执行/覆盖完成不等于Teacher质量自动或人工验收通过；下文旧快照按时间解释。

# 31 服务器迁移与 Coding Agent 交接

> 用途：让另一台服务器上的 coding agent 不依赖聊天历史，就能审阅并接着推进项目。
> **当前状态权威快照：2026-10-18 21:38 CST（UTC+8）**。三插值模型共3033视频生成并完整解码通过；旧机01:21暂停183条仅为历史边界。
> 本文含本地运维路径。公共快照脱敏发布，凭据、私有日志、数据与权重不公开。

## 当前状态与接续入口（2026-10-18 21:38 CST）

- 新集群四节点各8张A100-40GB。全部3033固定jobs输出及三模型各1011视频完整121帧/24fps/1280×704解码通过；不更改checkpoint、prompt或seed，不重跑Phase E。

| 模型 | 完整视频/GTref覆盖 | DreamSim↓ / 运动误差↓ | Physics-IQ Original raw↑ | 固定120 val loss↓ |
|---|---|---|---:|---:|
| α0.3 | 1011完整解码；GTref108/108 | 0.16940009 / 0.94077642 | 29.634157 | 0.19359857 |
| α0.5 | 1011完整解码；GTref108/108 | 0.16556586 / 0.90095284 | 28.169852 | 0.19132043 |
| α0.7 | 1011完整解码；GTref108/108 | 0.15975603 / 0.80539289 | 26.875848 | 0.18940758 |

- 三模型Physics均核验原198输入及66个唯一三视角场景。三份loss已用原base120视频**同顺序**、5时间位置/固定噪声独立验证，逐clip、per-t、per-dataset和总体均值一致且有限；base loss0.19753256。
- WM-Bench VBench8维三模型已出完整逐case评分；公共355-case VBench-I2V仍在评分。WMB三模型分别在launcher、两个worker并行（0KHxiL/TDIiHu/kc50Z7），日志进度不是已发布350-case结果。最终gate JBurrX仍pending，不认定完整实验/Teacher通过。
- 原Phase E3913输出、14评分组和raw/po结果保留。扩写prompt可能改变语义，不能用po对raw归因训练收益；本地固定seed与榜单5seed区别持续披露。
- 完整解码记录：eval/diag/decode_m_full1_a{03,05,07}_20261018.json；GTref108、Physics66 CSV/metrics、三loss原始JSON可现场核查。计数/抽查不代替完整评分覆盖。
- 配对统计：eval/diag/merge_all_analysis_20261018.json（全部三模型），保留原两模型merge_a03_a05_analysis_20261018.json。A03物理比A05+1.464305分、CI[0.030637,3.105535]，但末帧和运动误差更差；与base/R4物理差CI含0。固定案例/seed、未按来源聚类、多重比较未调整，不据单指标推荐Teacher。
- 已验证持久代码score_merge_cpu.py、monitor_merge.py、wmb_merge_score.py、val_merge_score.py、merge_final_gate.py及配对分析CLI；独立全回归测试通过，具体测试数量/时间以每次报告为准（CPU入口最初22项只是历史子集）。完整门禁要求两full manifests、三模型全解码、48维case集合、WMB350/1750/700、Physics66、GTref108及固定120loss均通过后才重建summary/Teacher页。
- 托管monitor YQ9EwY、VBench eTQBxB、WMB0KHxiL/TDIiHu/kc50Z7、final JBurrX仍活跃；m9OYH5验证loss已完成，8po4Fs CPU已完成。旧r60iAs/rwO4Pz/4C9bK1/00qFx9/5HCYQa为已替换历史任务，不能再启动。句柄仅本会话有效，重新接手核对实际PID/日志和输出。
- 代理已解决GitHub连接，认证身份和目标仓库推送权限已验证。目标Anachronism-N/ICASSP-IPVG，独立分支wm-teacher-experiments-20261018，实验快照放research/wm_teacher；不推原含cookies历史，不覆盖比赛项目文件。**2026-10-18 21:42首批安全快照已推送**，远端分支hash040814269512ddad93cc9734811940bd8de54630与本地一致；已推送的是当前方案、代码和CPU分析，不代表GPU评分/实验完成。最终评分结束后追加完整报告并核验新远端hash。旧服务器未销毁。

> 以下章节是有时间戳的历史快照；当前接续以上节和eval/diag/merge_resume_20261018.json为准。

## 旧机暂停交接（2026-10-18 01:21历史冻结快照）

> 本节仅保留旧机冻结证据及当时恢复要求。新机之后已恢复并实际出视频/CPU评分；183输出、2850待恢复、旧15卡和旧会话任务不再是当前状态。

- Phase E完整完成：3,913视频及14组benchmark/model评分严格覆盖PASS，原始/扩写分数和Teacher页面已保存；Teacher质量验收仍未通过。
- 插值推理暂停于**183/3,033**：α0.5的WM **99/108**、VBench **84/355**；α0.3/0.7尚未生成，Physics-IQ/WMB插值输出均未开始。剩余**2,850**个生成任务，插值评分与3份固定噪声val loss未完成。
- 暂停证据：`eval/diag/migration_pause_20261018.json`，记录旧节点、原manifest SHA256、完成输出清单、15个残留claim及恢复限制。
- 仅恢复缺失推理：`eval/diag/jobs_merge_pending_20261018.jsonl`。最终覆盖验证仍使用**完整**`eval/jobs_merge_all.jsonl`，不能只用pending清单宣布整轮完成。
- 原始jobs、checkpoint、prompt与seed未更改，已完成视频全部保留。SIGTERM后部分CUDA退出需等待，未退出的准确自有进程经SIGKILL终止，最终两节点确认无生成/调度、显存0；不是无损迁移运行中进程，未保存的单条任务需要重做。
- 15个claim与可能的`.tmp.mp4`保留，未批量删除。新Agent先重新确认旧节点无对应活跃生成，再仅对暂停记录中的残留claim处理或等待过期（默认40分钟）；绝不能删除别的仍在运行任务的claim。仅最终`.mp4`路径视为已完成。
- 新集群按32张健康卡重新分片，使用相同模型/输入/seed，先smoke。不要重启旧机历史`pipeline_ft2.sh`或原调度器，不重训FT1/FT2。
- 先保留共享ROOT与本地`/dockerdata/models`、`wm_checkpoints`、各venv、benchmark代码/权重/GT及必要日志，**备份完成之前不能销毁旧机器**。GitHub尚未推送，clone不包含最新未提交材料。
- 旧会话task id已停止或失效，原事件监控/CPU提前评分不会再自动接续。新Agent重新安排WM99/108补齐后的GTref，以及其余bench和验证loss；不得把暂停当插值实验完成。

### 新集群接续准备与评测里程碑（2026-10-18历史过程记录）

> 以下按当时恢复/评分/监控里程碑保留，部分早期段落仍写“尚未恢复/尚未smoke/尚无分数”。这些不是当前待办；现已完成恢复及α0.3/α0.5的CPU全量验证，旧CPU任务r60iAs已由顶部持久化入口替代。

- 用户确认四节点：rank0 `PRIVATE_NODE_ADDRESS`、rank1 `PRIVATE_NODE_ADDRESS`、rank2 `PRIVATE_NODE_ADDRESS`、rank3 `PRIVATE_NODE_ADDRESS`；均为8张A100-40GB。此前新集群Agent报告SSH连通、32卡bf16测试通过，不是生成/评分smoke。旧集群本会话于10-18尝试BatchMode访问新rank0遭认证拒绝，故本会话未验证新节点可读取私有备份，也未执行新机解包；新集群Agent应从本机共享挂载确认访问权限。
- 共享ROOT保持不变；三个worker最初未挂载，已按用户授权的remount/taiji_client命令补挂PRIVATE_SHARED_MOUNT，现四节点暂停JSON均可读。挂载工具同时尝试gy4别名并报只读，目标gy2正常；未为无关别名改动权限。
- 四节点torch-base已有torch2.7.1；已补齐缺失的av19.0.1，未升级其他包。rank0框架导入及cfg_zero_star/cfg_zero_steps接口检查通过。
- 共享盘已有生成所需5B DiT、converted T5/VAE、tokenizer与三份merge权重。rank0复制完成，12个必需文件源/目标SHA256一致，三份merge另与固定迁移摘要一致。已从rank0本地盘分发到三个worker；四节点各约62.79GB缓存均通过12文件大小/SHA256验证，worker共享框架导入也通过。校验清单位于各节点/dockerdata/models/generation_cache_sha256.json。
- 首轮rank0独立生成smoke（MsCTXe）完成50步采样及VAE解码，但save_video保存失败：新机缺imageio-ffmpeg，ImageIO落到PyAV后端并拒绝quality参数；不能算smoke通过。已四节点补齐与旧机导出一致的imageio-ffmpeg0.6.0，四节点真实save_video quality5+121帧/1280×704完整编码解码测试通过，run_gen启动检查加入该依赖。
- rank0 GPU完整smoke已通过：独立输出/dockerdata/wm_tmp/migration_smoke_retry_1792289824830.mp4，121帧/24fps/1280×704完整解码PASS；同输出用正确配套FFmpeg另生成_physiq_5s_verified.mp4，120帧/24fps/1280×720预处理完整解码PASS。VrFZCi原任务exit1仅是末尾绑定旧FFmpeg导致转换失败，已单独修复，不重跑已生成视频。三个worker同case/seed单条GPU生成及121帧/24fps/1280×704完整解码也均PASS（77tkNp/XMUqrN/7o8Wvw）；四节点本案例解码RGB帧SHA256均为074a9b667962052b4ca2b2155734b8d5d3284bc9c3ef063e159293593d563940。只证明本案例重现，不是多seed/全评测结论。随后2850条正式恢复已启动（见下方调度记录）；正式评分尚未开始，原183视频、Phase E分数与原始清单保留。
- 新机定向检查最初未找到Physics-IQ完整GT、VILA/judge及独立评测环境；两节点旧机密码认证失败。旧集群本会话仍能用既有BatchMode SSH访问worker，已按用户请求直接导出到仓库外私有`PRIVATE_SHARED_MOUNT/share_302533218/cedricnie/wm_migration_private_20261018`（根700、文件600）。worker6份归档（94,357,647,360字节）和launcher9份归档（169,117,040,640字节）及环境导出均已写完，另含VBench外链真实权重1份。共16归档264,197,908,480字节（约264.20GB/246.05GiB）。**2026-10-18 12:15:23 CST最终验证已通过**：16包SHA256、15个目录源内容比较PASS，额外CLIP/DINO真实目标对比PASS，207外链目标全覆盖，无partial且700/600权限通过。完成证据`manifests/backup_complete.json`与`manifests/final_verification.log`，恢复说明`RESTORE_PRIVATE.md`。新Agent可从共享备份恢复所列资产，不再需要密码登录旧机下载；回收旧机前仍需确认新机访问权限和项目ROOT以及其它任务资产无遗漏。凭据未写入本文，归档可能有历史敏感日志，只可私下使用，禁止公开上传。
- 用户提供私有归档路径及manifests/SHA256SUMS后，新机按RESTORE_PRIVATE允许的拟恢复项独立校验继续部署，无需把全备份marker缺失变成无限等待。原自有等待任务4qWpRQ已停止、未解包；新任务F6PVxU交叉核对角色/单归档/SHA256SUMS，顺序复制5份评分归档（venv_vbench、venv_piq、conda_vila、wm_eval、vbench_cache），完整源/目标SHA256通过且tar成员安全范围验证后才解到空目标。不会覆盖现有torch-base、models，也不恢复外链归档到共享盘。这里只证明选中评分资产，不冒充全备份/源比较完成，旧机仍保留。逐项恢复记录在新机/dockerdata/wm_migration_private_20261018/scoring_restore_verified.json；归档/导入通过也不等于正式benchmark完成。
- score_extra.sh的FFmpeg从失效的另一共享盘硬编码改为imageio_ffmpeg.get_ffmpeg_exe()解析的配套v7.0.2可执行文件，允许FFMPEG显式覆盖。实测/opt/conda/envs/torch-base/bin/ffmpeg虽有滤镜但没有libx264，不能用于原协议编码。配套FFmpeg的libx264/pad/fillborders检查及真实120帧/24fps/1280×720转换、完整解码、上下8px边缘smear核对均通过。原120帧+pad/smear+crf12协议未变；GPU重试任务VrFZCi此前绑定了缺libx264的旧路径，生成若已通过但后续转换失败，应单独用正确配套FFmpeg验证已有输出，不重跑生成。
- scripts/run_gen.sh已改成**单节点前台托管入口**，必须显式设置NODE_RANK和NNODES；每节点8个子进程由父进程等待并在退出时回收，不再SSH旧worker或nohup/setsid脱管。四节点分别设置NODE_RANK=0/1/2/3、NNODES=4，分片为0..31。原历史pipeline未设置这些参数，会在启动前失败；不要重启它们。
- 入口要求本地模型目录及checkpoint缓存已就绪，并将gen_jobs.py复制到独立本地目录，避免覆盖运行脚本。bash语法、非法rank/nodes/tag拒绝、mock全局分片16..23、子进程非零退出传播以及SIGTERM回收8个准确子PID测试通过；实际GPU生成和评分仍待smoke。生成进程退出不等于完整manifest或评分通过。
- 新机15:33:46实测生成1176/3033：α0.5全部1011已保存并逐帧完整解码PASS，α0.3已保存165，α0.7尚未开始；32卡仍正常生成，零发现错误、零过期活跃claim。α0.5已启动CPU GTref108条（5mL5xv）及Physics-IQ198条（nR3p2p），后者预处理120帧/1280×720完整解码通过后已进入原协议scorer。正式分数尚未齐，不宣称模型验收。
- 五份评分归档恢复F6PVxU已完成：源/本地SHA256与清单一致、tar范围检查及空目标解包完成，三独立环境实际torch/numpy/transformers/flash-attn导入通过；新机生成环境与缓存未被覆盖。正式VBench本地commit fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490。已配置离线TorchHub和CLIP缓存，避免默认模式隐式联网。
- 后续不争卡调度：4C9bK1等launcher生成退出、输入齐且GPU0空闲后做私有WMB单案例smoke并依次评三插值；eTQBxB等同条件且GPU1..7空闲后做WM+公共VBench8维评分。两者分卡，不互相等全部显存。00qFx9等新worker0生成退出且GPU0..2空闲后，各卡跑一插值的固定120条验证loss。旧等待任务pH34dv/GanBeU已停止且未执行评分，不能重复启动。GPU任务尚未开始，仅CPU评测在运行。r60iAs将按α0.3/α0.7各1011输出齐后依次补全逐帧解码、GTref和Physics-IQ，Physics scorer串行避免共用GT缓存冲突。5HCYQa仅在三模型完整解码、3份120条loss、两份完整manifest的逐case/维度/场景/judge覆盖全部PASS后更新summary和Teacher页并写merge_eval_complete_20261018.json；不能以运行退出或文件存在代替验收。
- 15:50:28新机实测1275/3033视频已保存（α0.5 1011，α0.3 264），32卡继续且无发现错误。α0.5 Physics-IQ Original raw已完成198输入/66唯一场景覆盖：28.169852分；同raw base30.130379、R4 27.326698、FT1 24.819238、FT2 28.337988。这是单seed均值，配对场景bootstrap尚未做，不能把超过R4均值说成可靠改善或达到Teacher验收。GTref已53/108，尚未完成。
- 新增α0.5 Physics-IQ同raw配对统计：66场景×3视角整体重采样5000次，点值及前3次重算与官方IQTable1e-8内一致。α0.5−base=-1.961分，95%CI[-3.532,-0.582]；−R4=+0.843，CI[-1.409,3.061]；−FT1=+3.351，CI[1.798,5.138]；−FT2=-0.168，CI[-1.930,1.462]。本固定seed案例支持高于FT1、低于base，尚无可靠高于R4/FT2证据；区间不含seed/judge/训练随机性，多重比较未调整。证据eval/diag/physiq_merge_a05_paired_bootstrap_20261018.json。15:55:22生成1305/3033（α0.3 294），GTref81/108，零发现生成错误。
- α0.5 GTref108/108已完成并覆盖验证，DreamSim0.16556586、末帧0.19522358、运动量误差0.90095284。5000次案例配对bootstrap的DreamSim差（越低越好）：−base=-0.009072，95%CI[-0.014715,-0.003860]；−FT2=-0.006644，CI[-0.012697,-0.000624]；−R4=+0.008648，CI[+0.003025,+0.014616]；−FT1=+0.005933，CI[-0.000098,+0.011949]。固定seed案例支持真实后续匹配好于base/FT2但差于R4；与FT1均值差不确定，运动误差仍差于R4/FT1。案例级非来源聚类重采样，不含seed/judge/训练波动，多重比较未调整。证据eval/diag/gtref_merge_a05_paired_bootstrap_20261018.json。完整VBench/WMB/loss尚未齐，Teacher验收未通过。
- 新机α0.3的WM108及VBench355生成子集已齐，CPU领域GTref不必等全1011输出：nZuhAW已通过WM108完整逐帧解码并提前计算原GTref，不占生成GPU。初始o67rVi的shell全文grep误匹配自身命令，未评分便退出；已改为/proc精确argv匹配防重。后续r60iAs在全模型齐后按已保存case恢复/跳过，不重复改写已完成GTref。
- α0.3提前GTref108/108完整覆盖/有限值通过（nZuhAW）；DreamSim0.16940009、末帧0.20471305、运动误差0.94077642。固定seed案例5000次配对bootstrap：DreamSim−base=-0.005238，95%CI[-0.008755,-0.002250]；−R4=+0.012482，CI[+0.004530,+0.020721]；−FT1=+0.009768，CI[+0.001376,+0.018307]；−FT2=-0.002809，CI[-0.006653,+0.000773]；−α0.5=+0.003834，CI[-0.000280,+0.008116]。本案例集合支持优于base但差于R4/FT1，与FT2/α0.5差异不确定；不含来源聚类/seed/judge/训练方差，多重比较未调整。诊断gtref_merge_a03_paired_bootstrap_20261018.json，不能视为完整Teacher验收。
- gen_jobs.py已修复捕获生成/保存异常仍退出0的问题：实际异常累计后收尾非零退出，继续处理后续任务且finally释放本任务claim；正常成功/已保存skip不失败。独立main mock的成功、编码失败、采样失败、已有输出恢复四种路径均通过，父调度器可看到真实错误；仍不能以进程退出替代全清单覆盖。
- 2026-10-18新机自主推进：固定pending清单2850条在四节点本地SHA256均核对为4b5817ded0af1ec8b52c8ae29fee4d33d8ce5f6b6128f8c5e6cc8aded0db1a7d。四节点前台父进程已正式启动32生成分片，托管任务rank0 GnHeOh、rank1 3rFbn2、rank2 UUwSkV、rank3 mjByh1；启动记录不等于产出已完成。状态保存在eval/diag/merge_resume_20261018.json。
- 新机GTref真实CPU单案例及VBench主体一致性GPU单案例评分smoke均通过，结果只保存在/dockerdata/wm_tmp/score_smoke_52427/gtref_one.json、vbench_smoke_67600/result.json。使用校验过的共享DreamSim/DINO及共享VBench源码；正式旧机VBench/WMB/Physics版本归档仍在恢复，不将单例当全评测或改写Phase E分数。DreamSim缓存缺TorchHub源码/基础权重已补齐，解决无网络时隐式GitHub加载失败。
- 启动后worker初期多进程出现d_alloc_parallel共享目录查找等待；随后四节点全部32卡已实际占GPU显存并进入加载/采样，launcher与两个worker8卡采样时接近100%，rank2部分卡仍在加载。首轮12:40:37监控计数183/3033（新首批尚未落盘），32日志、11采样日志、零发现错误；不把进程/GPU利用率当已产出数或满吞吐承诺。父进程和本地脚本未覆盖，后续按真实输出增量/解码核查。
- 五分钟托管监控rwO4Pz持续读取完整3033输出、claims、四节点GPU及32日志，逐批抽查新视频解码，记录eval/diag/merge_monitor_latest.json与merge_monitor_events.jsonl；会话巡检每5分钟自调度，初始cron cc2deb16，安全上限3天。样本解码/文件非空计数不是全评分覆盖。
- 原始完整manifest哈希与183初始已保存输出保留；15暂停claim已实测全部超过40分钟，无fresh残留。未批量删claim/tmp，正式生成仅按原claim过期机制回收对应缺失任务；模型、prompt、seed与Phase E未改。

## 0. 新 Agent 先读这一页

1. 当前阶段是 **Wan2.2-TI2V-5B 双向 Teacher 优化和评测**，不是流式 Student 训练。
2. 固定使用 5B 模型。用户要求对标 DreamX-World-1.0，在多个非 dynamic 的 VBench-I2V 维度领先头部模型；不以 dynamic 或总分单项为唯一目标。
3. Teacher 自动指标和用户 review 没有通过之前，**不能自行开始流式/因果 Student、蒸馏或长序列 AR 训练**。
4. 后续训练只用完整 dense caption。评测时可以扩写 prompt，但必须保留原始协议结果，并审计扩写语义是否改变。
5. FT1/FT2 已训练完；Phase E原始与扩写评测完整结束。插值已在新4节点32卡接续，α0.3/α0.5的CPU全量验证通过、α0.7仍生成；完整评分/loss尚未齐。**FT3 仅有候选数据清单，未训练**。
6. 旧机器原用15张健康A100-40GB，本项目旧机计算已停止；故障仅指旧worker GPU7。新4节点32卡已通过bf16与生成smoke，不能把旧坏卡排除或15分片沿用到新机。
7. **GitHub 不等于当前完整状态**：本地有未推送提交和后续未提交改动；数据、模型权重、评测输出、多个环境都不在 Git 仓库内。
8. 用户希望常规步骤自主推进、充分使用资源、中文汇报；结果必须区分实测/统计证据/推断。不要反复只报“已排队”，也不要为了满足目标把未完成任务标完成。

阅读顺序：本文 → `docs/30_Teacher模型训练实验记录.md` → `PROGRESS.md` → 第 7 节脚本与第 9 节现场核查。

## 1. 项目目标与验收

### 1.1 项目背景

项目目录名为 `wm_dataset`，长期目标是以人为中心的全双工、多模态交互世界模型。现阶段先准备多源视频与 dense caption，适配一个能从首帧和描述生成合理后续的双向视频 Teacher，验证效果后才进入流式阶段。

当前直接训练条件是 **首帧图像 + 英文 dense caption → 视频**。不能把这称为已实现实时交互、动作控制、相机控制或多模态流式输出；这些是后续目标。

### 1.2 当前验收原则

- 模型仍用 `Wan2.2-TI2V-5B`，不换更大 backbone。
- 在多个非 dynamic 维度上超过 DreamX 和头部榜单模型，而不只是超过单个参照或维持 base 本来的优势。
- 同时保留领域任务的质量与真实后续接近程度。
- 本地 benchmark 目前主要为 1 seed，榜单为 5 seed；差距需披露，不轻率宣称 SOTA。
- 当前审阅脚本要求至少 3 个非 dynamic 维度不低于指定头部对照，以及验证/GT参考指标通过、人工盲评通过。这是操作性 gate，最终用户确认不可省略。
- **目前 FT2 自动 gate 未通过，盲评未完成。**

## 2. 已完成实验及结论

详细数据选择、超参和每轮解释见 `docs/30_Teacher模型训练实验记录.md`。下表均为当前本地测量；“未测”不是分数为 0。

| 模型/轮次 | 训练方式/初始化 | 数据与规模 | 步数 | val loss↓ | WM DreamSim↓ | 运动量误差↓ | VBench-I2V↑ | Physics-IQ raw↑ | WMB raw↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | 原始 Wan2.2-5B | — | — | 0.1975 | 0.1746 | 0.994 | 89.86 | 30.13 | 8.30 |
| Round1 | base，LoRA r32，lr1e-4，49帧，8张H20 | 59,682条（12源），dataset_repeat=2 | 14,921 | 0.1922 | 0.1599 | 0.881 | 未测 | 未测 | 未测 |
| Round2 | R1，LoRA r32，lr1e-4，121帧 | 96,555条5s窗口 | 约6,437 | 0.1879 | 0.1607 | 0.743 | 89.53 | 未测 | 未测 |
| Round3 | R2，LoRA，lr5e-5+cosine | 30,001新窗口 | 2,001 | 0.1876 | 0.1577 | 0.730 | 89.42 | 未测 | 未测 |
| Round4 | R3，LoRA，lr3e-5+cosine | 24,650，美学筛选 | 1,644 | 0.1876 | **0.1569** | 0.730 | 89.47 | 27.33 | 7.10 |
| FT1 | base融合R4 LoRA后，全参lr5e-6 | 33,040，round5数据 | 2,203 | 0.1880 | 0.1596 | **0.672** | 原始版未测 | 24.82 | 7.09 |
| FT2 | 从base，全参lr5e-6 | 32,365，39%补充数据，dense caption | 2,158 | 0.1967 | 0.1722 | 0.922 | **89.82** | 28.34 | **8.15** |

重要历史修正：

- Round5 原计划 LoRA 训练没有真正执行，数据配方转给 FT1；不要虚构一轮 R5 模型结果。
- **FT1 实际用了约30%短 caption（9,846条）**，这来自更早生成的 round5 CSV。后续用户要求 dense-only，FT2 已遵循；不要把 FT1 说成 dense-only。
- “全参训练”只训练 DiT，不是连 T5/VAE 一起训练。保存参数里出现默认 `lora_rank=32` 不代表 FT2 实际用了 LoRA，要看 `lora_base_model`、`trainable_models` 和 checkpoint keys。
- FT2 和 FT1 改了初始化、数据分布、筛选与caption，不是单因素消融。不能下“通用数据必然稀释领域收益”或“全参普遍无效”的因果结论。

### 2.1 当前对目标的距离

DreamX快照原始榜单Total为90.49，去camera按本地相同8维重算Total*为90.79（旧报告曾误把重算值叫原始榜单分）。FT2 raw同口径Total*89.82，在七个非dynamic维度对DreamX领先2/7：i2v_background、motion_smoothness。主体一致性、背景跨帧一致性、美学、画质仍有缺口；没有达到验收目标。

FT2 领域 WM-Bench Total87.11，略高于base87.06，但其真实后续DreamSim仍远不及R4。不能用一个指标替代全部领域收益。额外CPU扫描FT2 WMB原始350视频：跳变proxy6.6%、flicker0.90、帧差3.41（base7.1%/1.02/3.71，R4 12.3%/1.64/5.57），说明时序变化量已收敛；不是物理正确性的独立证明。

### 2.2 已有统计证据

`eval/diag/ft2_paired_bootstrap.json`：固定现有seed，对相同案例5,000次配对bootstrap。

- VBench FT2−base = −0.04分，95%区间[-0.231,+0.150]，不能认定可靠变化。
- VBench FT2−R4 ≈ +0.352分，区间[+0.086,+0.616]，该组案例支持回升。
- WMB FT2−R4 ≈ +1.046，区间[+0.869,+1.220]；FT2−base ≈ −0.151，区间[-0.283,-0.029]。
- DreamSim FT2−R4 = +0.01529，区间[+0.00669,+0.02468]，FT2真实后续匹配仍差于R4。

这些区间不包含seed波动、judge波动或训练随机性，不保证对新分布泛化。

新增Physics-IQ统计（10-17 20:49）：198视频是66物理场景×3视角，以66场景配对bootstrap5,000次。FT2 raw−base raw=-1.792分，95%区间[-2.801,-0.810]；FT2 raw−R4 raw=+1.011，区间[-1.560,+3.657]包含0。FT2 po−base po=-1.055，区间[-1.838,-0.328]。因此不能把raw均值上升1分概括成可靠超过R4，也不能以po分数宣称超过base。完整公式核对及区间见 `eval/diag/physiq_paired_bootstrap.json`。

## 3. 旧机流水线与任务关系（2026-10-17至10-18冻结前历史）

> 本节运行数量、任务ID和分工仅是旧机历史快照，所有旧机本项目生成/调度已停止；当前新机任务与严格完成边界见顶部，勿按本节重复启动。

### 3.1 历史快照

截至 **2026-10-17 23:02 CST**：

- `eval/jobs_ft2_all.jsonl`：**3,913/3,913**视频于23:00生成齐；2026-10-18 00:51:51所有评分完成、主pipeline退出。独立最终覆盖检查PASS，验证3,913输出和14个benchmark/model groups，汇总及Teacher页已更新；Teacher自动gate仍FAIL。
- FT2 raw 的 WM、VBench、Physics-IQ、WMB 完成；FT1_po、FT2_po视频完成。
- Physics-IQ已提前用CPU得到：FT1_po31.29、FT2_po31.43。
- base_po Physics-IQ 已完成：**32.48**，高于同po协议的FT1 **31.29**和FT2 **31.43**；仍需R4_po最终对照。po输入存在语义变化，不能当作严格原始协议。
- `eval/jobs_merge_all.jsonl`：**118/3,033**（2026-10-18 00:52快照）。worker7卡与launcher2–7六卡已真实生成，主phase E现已完成，调度器接续加入launcher0/1。插值评分未出，不能说已改善。
- `eval/summary.md`、`eval/summary.json`、`demo/teacher.html`已更新FT2 raw，但不会因更新页面而变成全pipeline完成。

2026-10-17 23:03:05已对全部3,913个phase E输出逐个解码检查，均121帧/24fps/1280×704、零损坏，`eval/diag/phase_e_integrity_final.json`。这不评价语义/物理正确性，且不是最终benchmark评分完成标志。

### 3.2 旧机历史运行关系

旧机器的主流水线：`/dockerdata/wm_tmp/pipeline_ft2.sh`。

1. Phase A：FT1结束；B：dense caption与扩写prompt；C：FT1 raw评测；D：FT2全参训练。上述已完成。
2. Phase E：3,913视频生成 → Physics-IQ CPU打分、WMB GPU打分、VBench GPU池 → 重建summary/Teacher页。
3. 新修复的 `eval_merge.sh`：确认phase E已经进入打分且3913输出都存在/非空后，先启worker7卡的merge生成；VBench打分结束后先加launcher2–7，WMB保留0/1；主pipeline真正结束且phase E覆盖通过后再加launcher0/1。生成齐3033后打分，worker0/1/2分别补3个模型固定120条val loss；只有生成/分数覆盖检查与loss均齐、审阅页生成成功后才写MERGE EVAL DONE。
4. 已设置进度事件监控；CLI task id和PID属于旧会话，**不具备跨服务器恢复能力**。新Agent用日志/进程/文件重新确认。

冻结前旧会话托管任务参考（已停止/失效）：merge流水线`wLxB1o`（RESUME_GENERATION=1保留已有jobs，不重复启动worker）（替换尚未开始生成的旧等待调度器，收尾补验证loss及严格覆盖）；事件监控`aql7jl`（替换旧`nd6h5c`）；主pipeline退出后覆盖验收`hDX7I9`。旧pidfd等待曾因Python缺os.pidfd_open失效，后改内核syscall434；不要把旧task id当作仍正常。base_po提前评分`5NYE4U`已结束；插值权重CPU抽样校验`m2ax4x`已结束，3份checkpoint均825keys、各20个抽样tensor有限且符合插值公式（最大差0）。这些id可能失效，不作为实验真相。

### 3.3 未完成事项

- phase E的所有po变体和最终覆盖已完成；保留同raw/po协议限制和最终结果，不重新生成。WMB po完整对照：base8.86/R4 8.40/FT1 8.30/FT2 8.82。
- 完成3个merge模型的WM、VBench、Physics-IQ、WMB；生成结束后利用worker空闲GPU补同120条固定噪声val loss（名称loss_m_full1_a03/a05/a07.json），再选Pareto点。未测loss不能省略Teacher gate。
- 审核prompt语义偏移；不能把当前po涨分当作物理能力提高。
- 做必要的小规模推理参数/一致性优化验证，再决定FT3配方；不得直接“猜一个lr”就开新大训练。
- 取得用户Teacher review确认，之后才规划流式阶段。
- 当前git-lfs3.4.1已恢复；待GitHub网络与安全认证/提交身份可用，选择性发布独立脱敏快照到`Anachronism-N/ICASSP-IPVG`。尚未推送成功，不跳过hook。

## 4. benchmark协议与解释边界

| 项目 | 真值/输入与规模 | 输出/证据位置 |
|---|---|---|
| val loss | round2留出CSV，120条，固定5个时间位置+噪声；后续模型同一集合 | `models/train/round2_5s_lora_r32/eval/loss_*.json` |
| WM-Bench | 108条领域留出，GT首帧+caption | `eval/wm_bench/cases.jsonl`、`videos/`、`scores/`、`gtref_*.json` |
| VBench-I2V | 355图文case，不含camera_motion；109条background适用、246条subject适用，画质/美学355条 | `eval/vbench_i2v/cases.jsonl`、`scores/`；榜单快照`eval/vbench_i2v_leaderboard_20261011.csv` |
| Physics-IQ | Original198条，首帧图像+原始描述；不是Verified协议 | `eval/physiq/inputs/`、`videos/`、`videos_5s/`、`results/` |
| WMB | 350条，7领域；首帧+text_instruction，VILA judge | `eval/wmb/inputs/`、`videos/`、`results/*.json` |

- Physics-IQ生成121帧1280×704，评分前取120帧、补上下8px边到1280×720，避免与GT空间不对齐；详见`score_extra.sh`。
- WMB总分 = instruction（0–3）+5×physical pass rate+2×common-sense pass rate。7问回答“No”表示未检测到违规，不是回答“Yes”才通过。
- WMB“常识”两问在此实际是Poor Aesthetics和Temporal Inconsistency，不代表完整语义常识推理。
- GT DreamSim在匹配时间点比真实后续，跳过首帧；分数较好不等价于无物理错误。
- 相关基线没有完全同尺寸/时长：CogVideoX720×480/49帧@8，Wan2.1 1280×720/81帧@16，Hunyuan实际960×544/129帧@24。报告比较时应披露而不是声称所有baseline设置完全一致。

### 4.1 扩写prompt的关键新发现

现有`rewrite_prompts_vbench.py`初版只限制长度/要求忠实，未自动验证事件约束。10条固定抽查发现：

- Physics case0013：仅指定释放篮球，扩写加了压缩并停在箱内结局。
- case0138：仅指定释放纸团，扩写加了下沉杯底。
- case0150：灰网球/白灯罩被改成棕球/白盒子。

因此当前`_po`是扩写条件下探索，不是保证语义等价的官方协议。保留已生成raw和po，不覆盖、不静默换prompt。工作区脚本现已加入更严格的保约束说明与`--out`独立输出参数，但旧po文件与视频仍由初版生成，**没有被重新改写或重测**。下次必须显式指定新路径（例如`prompts_strict_v1.json`），原动作/对象/颜色/材料/方向不可改，未给定的碰撞/沉浮结局不可加。所有比较模型必须使用同一新版prompt，原始结果始终保留。

不能用FT2_po31.43直接对base_raw30.13宣布超过base；也不能仅凭raw→po涨幅宣称学到物理能力。现已取得完整Physics-IQ扩写对照：base_po32.48、R4_po32.47、FT1_po31.29、FT2_po31.43。R4_po与base_po相差仅−0.01，FT2_po与FT1_po仍分别落后1.05/1.19分。扩写仍有语义偏移，不把同po接近等同原始物理任务恢复。

## 5. 数据与存储

### 5.1 原领域来源

hoigen1m、mira、celebvhq、charades、direct、vfhq、matrix、easycom、noxi、seamless、egoexo4d、openvidhd、h2o。

数据统一为5s窗口；总共约2,764,495个已切窗口，但每轮只取3万–10万条。**不必为接着评测搬完全部数据**。

- 原数据及加工窗口：`dataset/`，尤其`dataset/wan22_5s/<source>_5s/`。
- 完整manifest：`dataset/wan22_5s/final_5s_{landscape,portrait}.jsonl`。
- 新增原始/切片来源的caption清单：`dataset/wan22_5s_extra/captions_5s.jsonl`与`manifest_5s_landscape.jsonl`、`SLICE_DONE`。
- 训练CSV保存`video,prompt`，视频路径一般相对于项目ROOT，不是相对于CSV位置。

### 5.2 FT2补充数据

实际训练总数32,365，原领域19,799、补充12,566：OpenVid通用6,600、UKdash1,610、RoboMIND2,244、EgoDex2,112。

切片/caption完整补充池为28,510：OpenVid12,000、UKdash4,510、RoboMIND6,000、EgoDex6,000；注意池规模≠实际训练规模。

筛选：原领域去掉高镜头平移窗口（每来源上20%，绝对阈值8px/4帧@320宽），direct/egoexo4d减半。phase correlation只是全局平移proxy，不是精确相机姿态。

caption词数审计：最少61、中位145、最大403，训练视频无重复；与120条val视频精确重叠0。现有manifest中15,810个已映射训练sequence与110个val sequence交集0；不排除跨数据集复制/预训练重叠。

### 5.3 HF状态

- `nzc-415/wan22-5s`：955 shards、约7.79TB；覆盖2,483,000/2,764,481个窗口。
- 10-12触发公共存储配额403后停止，尚有281,495个未上传窗口。
- `NZC415/wan22-5s-trainval-overflow`补传了R2/R3训练验证中缺失的24,162窗口。
- 旧`wan22-processed-clips`已重建为空，不能当当前5s训练集来源。
- FT2新补充池不保证已上传HF。共享盘是当前完整权威来源。
- RoboMIND/EgoDex等有NC/ND等许可限制，未得到许可审查前不要公开上传视频或处理衍生物。不要因“接手项目”就批量同步到公开仓库。

## 6. 旧机器环境与必须保留的文件

### 6.1 机器/路径

| 内容 | 当前路径/信息 | 新机器处理 |
|---|---|---|
| 项目ROOT | `/srv/wm_teacher` | 同共享盘挂载优先；否则改所有硬编码ROOT和JSON绝对路径 |
| launcher | `PRIVATE_NODE_ADDRESS`，8卡 | 改训练/SSH脚本MASTER、LOCAL_ADDR、WORKER |
| worker | `PRIVATE_NODE_ADDRESS`，GPU0–6健康、GPU7故障 | 新节点重新测试全部卡；更新GPU列表 |
| 网络 | eth1，当前无IB | 新环境按真实网卡设置，不盲目沿用NCCL_IB_DISABLE |
| 原模型缓存 | `/dockerdata/models/Wan-AI/`、`/dockerdata/models/DiffSynth-Studio/` | **两者都保留**，DiffSynth解析后使用converted缓存 |
| 微调/插值缓存 | `/dockerdata/models/wm_checkpoints/` | 按源绝对路径SHA256命名，复制前后需校验内容SHA256；改ROOT后缓存键也需更新。多进程mmap共享checkpoint曾导致worker7卡D状态/空转，顺序预缓存避免此问题 |
| Qwen VL | `/dockerdata/model/Qwen3.5-35B-A3B` | 只在需caption/扩写时搬；当前无须抢占生成GPU重新启动vLLM |
| 框架 | `ROOT/DiffSynth-Studio/` | 基准commit+本地patch，不能只装最新版pip包 |
| benchmark代码 | `/dockerdata/wm_eval/VBench`、`/dockerdata/wm_eval/bench/` | 本地盘、git版本、数据与judge一起保留 |
| VBench权重 | `/dockerdata/vbench_cache`、`/dockerdata/wm_eval/VBench/models` | DreamSim在后者；仅搬代码不够 |
| 本地脚本副本 | `/dockerdata/wm_tmp/` | 有额外临时修复脚本；可作为排障参考，运行时仍以核对后的脚本为准 |
| 本地日志 | `/dockerdata/wm_logs/` | 旧机器回收后丢失，应私下备份必要日志 |
| 共享日志 | `ROOT/dataset/download_logs/` | pipeline/train/gen/scoring权威日志 |
| 演示 | `ROOT/demo/`、`demo/media/` | 非公开审阅；端口8765，配置由新机器决定 |

旧worker挂载需平台授权（用户之前给过taiji_client mount命令），新Agent应向用户/平台获取当前挂载凭据，不从历史聊天/日志提取token，也不要把token写进本文。

### 6.2 Python环境快照（现场读取，不是推荐随意升级）

| 用途/路径 | Python | 关键包 |
|---|---|---|
| 训练/生成 `/opt/conda/envs/torch-base` | 3.12.11 | torch2.7.1，transformers4.57.1，accelerate1.10.1，deepspeed0.18.1，flash-attn2.8.1，diffusers0.31.0，numpy1.26.4，triton3.3.1，av19.0.1 |
| VBench/GT-ref `/dockerdata/venv_vbench` | 3.12.11 | torch2.4.1，transformers4.45.2，numpy1.26.4，triton3.0.0，accelerate1.15.0 |
| Physics `/dockerdata/venv_piq` | 3.12.11 | numpy2.5.4，opencv5.0.0.93；其他依赖需导出实际环境 |
| WMB `/dockerdata/conda_vila` | 3.10.14 | torch2.3.0+cu121，transformers4.46.0，flash-attn2.5.8，numpy1.26.4，triton3.1.0，accelerate0.34.2 |
| caption `/dockerdata/venv_vllm` | 3.12.11 | vllm0.31.0，torch2.13.0，transformers5.17.0，numpy2.3.5，triton3.7.1 |

以上只是包版本快照，不是完整锁文件。迁移前用各环境`python -m pip freeze`和conda显式导出生成**私有环境备份**，先审查其中是否含带凭据的安装URL再共享。二进制flash-attn、CUDA、驱动、CPUAdam编译与硬件关联，不保证直接复制venv即可运行。

新机器将训练、VBench、VILA、vLLM放独立环境，不把所有依赖塞入一个环境。

### 6.3 框架版本与补丁

- DiffSynth-Studio基准commit：`fb337fbb90945ff829de69dbd44ded618f73e889`。
- 本地补丁：`patches/diffsynth_studio_fb337fb.patch`，涉及5文件（最新版本已重新导出并reverse-check）：
  - `diffsynth/core/data/unified_dataset.py`：12次退避读数据；失败后换下一样本，**不等于原样本严格重试**。
  - `diffsynth/diffusion/parsers.py`：cosine/warmup/min_lr_ratio参数。
  - `diffsynth/diffusion/runner.py`：区分DeepSpeed与Accelerate调度计步。
  - `examples/wanvideo/model_training/train.py`：LoRA注入后再load checkpoint，避免LoRA keys被忽略。
  - `diffsynth/pipelines/wan_video.py`：实验性CFG-Zero*推理，默认关闭（不改变当前benchmark采样），job显式`cfg_zero_star=true`启用，`cfg_zero_steps`为真正要置零的初始步数，0表示不置零。CPU公式/分批/默认保持测试已过，GPU生成与效果尚未验证。
- 新机器在确认基准commit后先`git apply --check`；如果已有相同修改则不能重复apply。
- `DiffSynth-Studio`是独立Git仓库，外层提交不包含其完整源码，只带patch。

### 6.4 关键checkpoint（相对ROOT）

| 模型 | 路径 | 体积约 |
|---|---|---:|
| R1 | `models/train/round1_lora_r32/step-14921.safetensors` | LoRA |
| R2最终 | `models/train/round2_5s_lora_r32/run2/step-1937.safetensors` | LoRA；step计数为该run局部 |
| R3 | `models/train/round3_anneal_lora_r32/run1/step-2001.safetensors` | 161MB |
| R4 | `models/train/round4_aes_lora_r32/run1/step-1644.safetensors` | 161MB |
| FT1 | `models/train/full1_5b/run1/step-2203.safetensors` | 10.0GB |
| FT2 | `models/train/ft2_5b/run1/step-2158.safetensors` | 10.0GB |
| 插值 | `models/train/merge/full1_a03.safetensors`、`full1_a05.safetensors`、`full1_a07.safetensors` | 各10.0GB |

FT1/FT2全参checkpoint有825个DiT tensor，不含VAE/T5。插值为`base + α*(FT1-base)`，不是合并R4 LoRA scale，也不是Cosmos多领域Soup的严格复现。

关键迁移校验摘要已生成：`eval/diag/migration_checksums.json`，包含FT1/FT2、3个merge、R4、2份jobs、FT2 CSV和框架patch共10个文件（约50.2GB）的SHA256、大小与mtime。源文件以后若修改必须重算对应摘要；不是视频评测覆盖证明，也不代表已迁移。

迁移时必须带`training_args.json`、CSV、日志、step时间和输出清单；不要只看文件名选“数字最大”的step。不同run会重置step，ckpt注册目前还依赖mtime，拷贝改变mtime可能选错；恢复应优先显式指定checkpoint。

**训练resume不是完整优化器状态恢复**：现有safetensors主要恢复权重，未看到可据此恢复Adam动量/RNG的完整checkpoint。旧pipeline续训重建LR和样本清单，是近似续训，不是bitwise exact resume；移机途中仍在训练则应先说明这个限制。

### 6.5 迁移优先级

**只review**：本文、实验记录30、PROGRESS、scripts、框架patch、training_args、summary与逐案例分数即可；不需要搬TB视频。

**继续当前评测**：再加jobs JSONL、case清单、首帧输入、所有已生成视频与评分文件、benchmark代码/权重/GT、相关本地环境、base+FT1/FT2+merge权重，以及§6.4的R4最终LoRA（保留历史模型复现；round4_po已完整结束，不再有该待生成任务）。WM评测需108条GT原视频。保留Physics-IQ `videos_5s`或在新机器重新按相同方法生成。

**再开FT3**：再搬所选CSV引用的训练视频、caption、候选池与筛选分数。FT2/FT3候选清单引用约3万条，不必先搬完整几百万片段。

**HF上传/数据扩展**：单独迁移上传状态、shard索引、许可审查、完整数据清单，并重新授权HF；不能擅自恢复已被配额阻断的重复上传。

## 7. 脚本入口与关键依赖

| 任务 | 入口/依赖 |
|---|---|
| 多节点训练 | `train_round2_16gpu.sh`（LoRA）、`train_full_16gpu.sh`（全参）、`launch_round2.sh` |
| 数据配比 | `build_round{1,2,3,4,5}_mixture.py`、`build_ft2_mixture.py` |
| 切片/manifest | `reslice_5s.py`、`finalize_5s.py`、`inventory_5s.py`、`slice_new_sources.py` |
| 补充caption | `caption_5s_extra.py`、`deploy_qwen35_vl_a100.sh`、`caption_ctl.sh` |
| 历史总体调度 | `pipeline_ft2.sh`（A–E已结束，**不要重启；Phase E无需重跑**） |
| 插值评测 | `merge_ckpt.py`、`eval_merge.sh`（历史总调度依赖旧日志；当前新机按顶部托管任务接续） |
| 生成 | `gen_jobs.py`（图+prompt）、`run_gen.sh`（当前单节点前台托管，显式`NODE_RANK`/`NNODES`；新4节点全局32分片） |
| CPU完整解码/GTref/Physics | `score_merge_cpu.py`（持久化恢复入口，默认α0.3/α0.7，按严格验证skip；当前任务`8po4Fs`勿重复启动） |
| 分数 | `vbench_eval.py`、`score_pool.py`、`score_extra.sh`、`eval_val_loss.py`、`gt_ref_metrics.py` |
| 诊断 | `scan_cuts.py`（帧差/亮度proxy）、`score_camera.py`（phase correlation） |
| 扩写 | `rewrite_prompts_vbench.py`（当前版本有语义风险，先review实际diff） |
| 汇总/页面 | `summarize_eval.py`、`build_teacher_review.py`、`build_demo.py`、`serve_demo.py` |

历史FT1/FT2训练设置：121帧、1280×704、bf16、microbatch=1/GPU、accumulation=1；旧15卡global batch15。当前新32卡是独立视频生成分片，不表示已改成32卡训练或复现了训练配方。DeepSpeed配置`training_metadata/full/ds_zero2_offload.json`：ZeRO2，CPU optimizer offload，gradient clipping1.0，overlap_comm=false，reduce/allgather bucket5e7。

双节点脚本硬编码`ROOT/WORKER/MASTER`、接口eth1及端口29511/29513。**新服务器只有一节点时，不能原样执行nnodes=2脚本**；GPU数量/拓扑变化必须先调整launch，单卡smoke验证再训练。

## 8. 工程风险与修复状态

1. **共享fuse慢/返回旧脚本**：把环境/权重放本地盘，启动前复制脚本；别覆盖运行中的shell脚本，否则它继续按字节偏移读会产生异常。FT1收尾exit127可能相关，但未独立复现；最终权重已保存。FT2exit0。
2. **读失败**：数据退避补丁会替换下一样本，有助稳定但可能影响有效样本分布；不能声称全部CSV每条精确训练一次。
3. **GPU7硬件错误**：仅指旧worker的卡，曾Xid31非法内存访问，已排除。新机器按实际测试选择健康卡。
4. **torchrun FQDN**：旧网络需显式`--local_addr`，否则rendezvous失败。不同node卡数旧torchrun已跑通；新版本仍需smoke。
5. **apex sm_80**：旧apex会触发no kernel image，基线venv用ImportError stub屏蔽；不能在运行环境随意删共享系统包。
6. **全参保存OOM**：已关overlap_comm并缩bucket，不靠降低lr解释显存改善。首次存checkpoint也要测试。
7. **val loss与生成抢同卡**：gen_jobs启动检查同GPU的eval_val_loss并等待；独立串行GPU安排更可控。只等20秒/扫描proc仍不是通用调度锁。
8. **phase E/merge互等**：主pipeline旧`wait_gone(gen_jobs.py)`范围过宽。merge已改为等评分日志存在+3913文件完整才启worker，且自身wait匹配jobs文件，避免在主pipeline等待时插入新生成。
9. **WMB/VBench显存竞争**：score_pool已检测WMB占用卡、每GPU互斥评分。仍需核查实际进程先后，proc检测不是跨机器资源管理器。
10. **claim任务认领**：`.claim`按共享FS原子O_EXCL；默认40分钟判过期，新generator修复“全为旧claim直接skip”的快捷路径。当前无过期阻塞。**不要批量删claim或.tmp文件，先确认所有相关节点无对应活跃任务。** 新机器和旧机器同时生成同一输出目录也有风险。
11. **评分partial被当完成**：VBench验证所有case输出存在/非空、每维文件名集合与唯一数量符合expected、逐案例分数有限；不同适用维度不是都355。`summarize_eval.py --validate-jobs`还检查GTref必需数值、WMB题数和Physics66场景。别按`[pool done]`单独判断，旧脚本曾不传播子进程错误。
12. **Physics日志覆盖**：summary过去从共享score_physiq.log解析标量，覆盖日志会使旧分数在表中消失；已改追加，迁移必须保留log与metrics JSON。不要依赖summary当唯一原始证据。
13. **Teacher页面**：已修复Full-FT标签匹配和盲评KEYMAP被分维度变量覆盖；自动FAIL保留。当前页面选择最近候选不等于最优模型已确认。
14. **baseline依赖patch**：HunyuanI2V曾有diffusers设备bug，Wan2.1大权重需offload/layerwise fp8。若重跑baseline，保留原venv或其源码diff，别认为只装同package版本就可复现。
15. **时间与状态**：用旧服务器`date`/日志确定CST快照；聊天里的预计结束时间不是承诺或真值。

## 9. 接手后的核查与恢复步骤

> 当前挂载、模型缓存、评分环境与smoke已完成；以下恢复示例供后续迁机/失败恢复参考，**不是要求再次迁移或重跑Phase E**。新Agent先核顶部托管任务和日志，仅补真正缺失的插值任务，不与当前生成/评分同时启动同输出。

### 9.1 先查现场，不启动新任务

在当前共享盘或路径映射完成后执行（以下ROOT仍是旧路径，按实际改）：

```sh
export ROOT=/srv/wm_teacher
date
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
python3 - <<'PY'
import os,json,pathlib,collections,subprocess
r=pathlib.Path(os.environ['ROOT'])
for name in ('jobs_ft2_all.jsonl','jobs_merge_all.jsonl'):
    p=r/'eval'/name
    if not p.exists():
        print(name,'MISSING'); continue
    jobs=[json.loads(line) for line in p.open()]
    done=sum(os.path.isfile(j['out']) and os.path.getsize(j['out'])>0 for j in jobs)
    print(name,done,'/',len(jobs))
    print('pending by model',dict(collections.Counter(j['model'] for j in jobs if not os.path.isfile(j['out']))))
for name in ('pipeline_ft2.log','eval_merge.log'):
    p=r/'dataset/download_logs'/name
    if p.exists():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-3000))
            print(name,f.read().decode(errors='replace'))
for line in subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines():
    if any(x in line for x in ('gen_jobs.py','score_pool.py','evaluation.py','pipeline_ft2.sh','eval_merge.sh')):
        print(line[:300])
PY
```

若旧服务器仍活跃，先确认由哪个Agent负责推进，避免两个Agent修改相同脚本/输出。不得擅自停止旧服务器所有python进程。

### 9.2 私有迁移交付步骤

1. 用户确认切换时点/新服务器路径与挂载权限。
2. 优先保持共享ROOT不变，避免海量数据搬运；`/dockerdata`是本地盘，需要单独备份。
3. 在旧任务写入稳定边界（当前完整视频结束、训练checkpoint保存后）协调停机；只停止自己管理的准确PID/托管任务，不用`pkill python`。
4. 对代码/文档/patch与少量元数据先拷贝一份私有快照，记录版本和工作区diff。**不用`git add .`、不用整仓公开归档**；cookies和带token的origin配置不能公开。
5. 再迁移所需checkpoint/benchmark权重/视频/GT，保留mtime或明确固定checkpoint路径。对关键文件做sha256校验。大视频只拷贝所需输出与训练CSV引用窗口。
6. 新机器重建各独立环境、验证CUDA/ffmpeg/模型加载、跑1条生成和1个scorer smoke。
7. 检查完整性和claim，再恢复缺失jobs；只在旧机器不再写这些输出时恢复。
8. 最后确认所有期望case+scores数量、重建报告、检查Teacher gate。不要把迁移完成当成实验完成。

### 9.3 框架/环境smoke

```sh
# 先确保 DiffSynth 基准commit正确且没有同样补丁，再执行。
git -C "$ROOT/DiffSynth-Studio" rev-parse HEAD
git -C "$ROOT/DiffSynth-Studio" apply --check "$ROOT/patches/diffsynth_studio_fb337fb.patch"
# check通过且确认尚未应用时，才运行下面这条。
git -C "$ROOT/DiffSynth-Studio" apply "$ROOT/patches/diffsynth_studio_fb337fb.patch"

/opt/conda/envs/torch-base/bin/python -c 'import torch; print(torch.__version__,torch.cuda.is_available())'
/opt/conda/envs/torch-base/bin/python "$ROOT/DiffSynth-Studio/examples/wanvideo/model_training/train.py" --help
```

`train.py --help`可能还需设置PYTHONPATH和运行目录，按下面生成配置设置。不要为让smoke通过而跳过真实CUDA/CPUAdam依赖检查。

### 9.4 单节点smoke与四节点托管恢复示例（仅无活跃重复任务时）

先把ROOT和job中的绝对路径映射正确，确认旧任务已停止且所选输出没有活跃claim。该命令每张GPU一个进程，不同时跑val loss/评分。

```sh
export PATH=/opt/conda/envs/torch-base/bin:$PATH
export PYTHONPATH="$ROOT/DiffSynth-Studio"
export DIFFSYNTH_MODEL_BASE_PATH=/dockerdata/models
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# 先只取1条任务，输出到独立smoke目录，避免直接执行整份benchmark。
# 本地临时目录须已创建且确认位置正确。
python3 - <<'PY'
import os,json,pathlib
r=pathlib.Path(os.environ['ROOT'])
j=json.loads((r/'eval/jobs_merge_all.jsonl').open().readline())
j['out']=str(r/'eval/migration_smoke/video.mp4')
p=pathlib.Path('/dockerdata/wm_tmp/job_smoke.jsonl')
p.write_text(json.dumps(j)+'\n')
PY
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/torch-base/bin/python \
  "$ROOT/scripts/gen_jobs.py" /dockerdata/wm_tmp/job_smoke.jsonl 0 1

# 四节点正式恢复示例：只在顶部现有任务确已退出且核清缺失jobs后执行。
# 每节点用自己的rank=0/1/2/3，NNODES=4；父进程前台等待本节点8个子进程。
# TAG为位置参数，须是新且合法的独立标签；不能覆盖正在运行的本地脚本。
NODE_RANK=0 NNODES=4 \
  bash "$ROOT/scripts/run_gen.sh" "$ROOT/eval/diag/jobs_merge_pending_20261018.jsonl" merge_recovery_verified
```

当前`run_gen.sh`必须显式`NODE_RANK`/`NNODES`；4节点各8卡时全局分片为rank×8+localGPU=0..31、nshards=32，work-stealing帮助完成缺失项。每节点通过托管后台工具调用该前台父入口，不使用nohup/setsid或SSH旧worker。若旧claim残留，先证明没有旧机活跃生成，不自动删所有claim。

`pipeline_ft2.sh`包含等待/重新训练/caption等历史阶段，**Phase E已完成，不要重启或为当前插值恢复重跑E**。历史pipeline未设置新版`NODE_RANK`/`NNODES`会在生成入口失败；当前按固定merge jobs和顶部已有托管任务接续。

### 9.5 恢复评分与汇总

当前CPU任务`8po4Fs`和GPU/loss等待任务已托管，不重复执行。仅未来恢复且确认原任务已退出时，可用持久化入口；CPU全套要求每模型1011条输入齐，再验证完整解码、GTref108与Physics66场景，完整有效结果skip：

```sh
# 仅无对应活跃任务时，通过托管后台工具运行；不与8po4Fs重叠。
/opt/conda/envs/torch-base/bin/python -u \
  "$ROOT/scripts/score_merge_cpu.py" --models m_full1_a03,m_full1_a07

# 仅三模型完整分数和固定120条loss齐后使用；未齐会报错，不刷新summary。
python3 "$ROOT/scripts/summarize_eval.py" \
  --validate-jobs "$ROOT/eval/jobs_ft2_all.jsonl" \
  --validate-jobs "$ROOT/eval/jobs_merge_all.jsonl"
/opt/conda/envs/torch-base/bin/python "$ROOT/scripts/build_teacher_review.py" --candidate m_full1_a05
```

WM/VBench、WMB与固定loss分别由顶部已存在任务在对应节点生成退出、输入齐且指定GPU空闲后接续；不在32卡生成时抢卡，不重跑已完成Phase E raw/po。`score_pool.py`与GT-ref的本地代码/benchmark路径需保持恢复版本；Physics-IQ使用imageio-ffmpeg配套且支持libx264/pad/fillborders的FFmpeg，当前解析已修复，不再依赖失效旧共享路径。

完整覆盖仍验证两份full manifests，不能只验pending清单或按运行退出/非空文件宣布完成。`eval_merge.sh`是依赖旧pipeline日志的历史入口，不原样等待旧触发器；未测loss/维度显示pending。可审阅候选α0.5，但候选选择不等于最优或通过Teacher gate，最终用户确认不可省略。

**可版本化的恢复入口（独立回归已验证）**：四脚本已经验证完整/缺失/非有限结果、原始case和固定loss顺序、互斥和失败传播；实际GPU/评分进度见顶部，不把测试通过当完整模型评分或Teacher验收。当前正在使用持久入口，恢复前核对原任务退出，禁止重复启动。

| 入口 | 使用边界 |
|---|---|
| `scripts/monitor_merge.py` | `--root ROOT --state eval/diag/merge_resume_20261018.json`；默认300秒、72小时上限，只读生成/资源检查并保存诊断快照。`--once`单次检查；采样与非空计数不能代替完整覆盖。 |
| `scripts/wmb_merge_score.py` | 在launcher独立GPU调用，外部显式`CUDA_VISIBLE_DEVICES`；先独立单案例judge smoke，再原350题全量。脚本不自行等待GPU释放，调用方必须先核显存/生成进程；现有WMB任务勿重启。 |
| `scripts/val_merge_score.py` | 在目标worker调用，等本机`gen_jobs.py`退出与GPU0..2空闲后并发原3模型固定120条loss。保持原模型/时间步/噪声协议，不与现有loss任务重叠。 |
| `scripts/merge_final_gate.py` | 默认300秒、72小时上限；`--once`立即检查，缺项非零退出。只在完整decode、3份120loss及两份full manifest逐case/维度/场景/judge覆盖均通过后更新完成证据；实验覆盖通过不等于Teacher质量gate或用户review通过。 |
### 9.6 路径迁移检查

下列信息包含绝对路径，单改ROOT不够：

- jobs JSONL中的`image/out/ckpt`。
- WM cases中的`gt/image`、val结果中的`video`、逐视频分数里的`video_path`。
- `scripts/*.py`与`*.sh`硬编码ROOT、MASTER/WORKER、环境和ffmpeg。
- VBench16:9裁剪图路径和官方full_info；DreamSim当前工作目录。
- 模型缓存/符号链接目标；旧服务器Python环境可能含绝对shebang。
- 最新DiffSynth补丁已包含默认关闭的CFG-Zero*采样接口；新版gen_jobs会传其默认参数。新机必须应用5文件补丁或同步对应框架，不能只搬gen_jobs配旧框架。

建议先私有拷贝到新位置，再进行明确前缀映射；保留旧数据快照，不原地替换运行中job文件。

## 10. 优化方向：有文献参考，仍需我们的验证

按用户要求，优先参考相关工作，而不是直接盲试FT3。

1. **权重插值/Soup方向**：Cosmos-Predict2.5 arXiv2511.00062，§4.2.1，多领域分别SFT后合并领域模型与cooldown模型。我们的base↔FT1 α0.3/0.5/0.7不是完全同方法，先用当前评测验证是否存在领域与通用的Pareto点。
2. **推理参数**：先在独立开发子集扫CFG/shift等、固定seed和图像，入围配方再全量测。不在公开测试集反复选参数后直接报无偏泛化。
3. **VCD一致性奖励**：arXiv2510.19193，文中Wan1.3B一致性提升但dynamic下降、imaging未升。Open-Sora的LoRA末步反传不能未经核实直接当Wan2.2配方。
4. **受约束偏好微调**：VideoAlign arXiv2501.13918的常数β Flow-DPO。其VBench Total上涨但Quality略降，不承诺对本项目有效；base-vs-FT偏好可能损伤领域收益，要单独验证。
5. **DreamX阶段化策略**：arXiv2606.16993，camera阶段冻结主干，event阶段fullDiT+非event数据，RL加KL；未公开lr等，不能虚构超参，也不在Teacher未验收前照搬AR阶段。
6. **FT3仅候选**：`training_metadata/ft3/ft3_cand_motion.csv`，matrix×0.3、mira×0.5，保持dense-only；未训练。高帧差与退化相关不代表matrix/mira是因果根源，应控制初始化/LR/其它变量做小消融。

## 11. GitHub与安全发布

- 用户授权的新公开目标：`Anachronism-N/ICASSP-IPVG`。旧`Anachronism-N/wm_dataset`仅历史目标，不再当当前发布目的地。
- 旧机冻结时本地HEAD历史记录：`c0383042c663c15d13436715cdca1639027c554e`，不是新目标已接收提交的证明。
- 旧机冻结时本地origin/main历史记录：`e329f763350c5bdff208c306c4b3ea3d9089337e`，不代表新目标远端状态。
- 旧push曾因认证失败；新集群当前git-lfs3.4.1已恢复。GitHub443/22网络未连通、尚未配置提交身份，安全认证与网络通后才能发布；**目前没有push成功，不把本地提交或脱敏准备说成公开成功。**
- 最新文档、评分检查/调度/持久化CPU入口和其它工作区改动仍需选择性审查；公开使用独立脱敏快照，不原地清洗本地运维交接，clone旧远端无法取得完整当前状态。
- `.git/config`的旧remote曾嵌token，日志/命令不得输出原origin字符串；用户应自行安全更新凭据，必要时轮换已暴露token。不要把私有`.git/config`打入公共交接包。
- `cookies_netscape.txt`已有用户改动；其它如`build_unified_manifest.py`和`rewrite_prompts_vbench.py`可能包含并行/既有改动，接手先看diff、保留，不恢复覆盖。
- git-lfs3.4.1已恢复，LFS hook继续正常执行，不能`--no-verify`；不为发布绕过安全检查。
- git提交只选本次相关文档/脚本/配置，排除视频、权重、cookies、token、私钥、受限数据；不`git add .`。

## 12. 新集群配置与冻结前迁移规划（含历史快照）

当前四节点地址/挂载、本地生成缓存、独立评分恢复及生成smoke已经核验，并已实际运行32卡生成；详情见顶部。下节01:09数据为旧机暂停之前历史，不能当作当前15卡或未迁移状态；CPU/RAM/网络拓扑仍应在未来同步训练前现场核对。

### 12.1 冻结前历史边界（2026-10-18 01:09 CST，不是当前状态）

- phase E于00:51:51完整结束：3,913输出和14个benchmark/model评分组独立覆盖PASS，视频完整解码零损坏，摘要与Teacher页已更新。Teacher自动gate仍未通过；完成实验阶段不等于质量验收。
- VBench Total*（raw / po）：base89.86 / 89.79，R4 89.47 / 89.60，FT1 raw未测 / po89.54，FT2 89.82 / 89.78。Total*均为去camera8维重算，不是原始榜单Total。
- Physics-IQ（raw / po）：base30.13 / 32.48，R4 27.33 / 32.47，FT1 24.82 / 31.29，FT2 28.34 / 31.43。
- WMB：raw base8.30/R4 7.10/FT1 7.09/FT2 8.15；po完整分数base8.86/R4 8.40/FT1 8.30/FT2 8.82。00:51:51已完成phase E且严格覆盖通过；merge仍在运行。
- merge已实际生成162/3,033（01:09快照），其中α0.5 WM89/108、VBench73/355。00:53:25最后旧launcher0–1也加入，当时旧15张健康卡都生成；3个插值模型完整评分尚未完成。
- 三份merge权重均已复制到两节点本地并校验内容SHA256。此前7个worker多进程mmap共享权重处于D状态且GPU空闲；改用本地checkpoint缓存后已恢复实际产出，不能再沿用旧“0输出/共享盘阻塞中”的结论。

### 12.2 能迁移什么，不能直接迁移什么

可迁移文件状态：代码+未提交diff、固定checkpoint、jobs与prompt、已完成视频、逐case分数、benchmark数据/权重、训练CSV和必要GT。新机按同输入/seed/checkpoint继续缺失jobs，不重训FT1/FT2，不重跑已完整评分。

不能直接迁移运行中的GPU进程、显存状态、旧会话task id或shell后台进程。旧judge脚本主要最终保存，若中途停止会损失judge进度；现在Phase E/WMB已保存齐并通过校验，是合适切换窗口。仍应先让新集群环境/挂载/本地权重准备完，完成独立输出的单条生成与评分smoke，再协调停止旧机的插值调度器和准确自有生成PID，等待显存和对应进程释放、确认不再写输出后冻结清单。保留已完成视频和分数，不必等待全部3,033插值任务完成后才迁移。

当前正在生成的单条视频如果无法等待自然完成，准确终止其进程可能损失这一条尚未原子落盘的计算；已完成视频不会因此回滚。现有进程不一定支持新加的“完成当前条即停止”功能，不能声称能自动无损排空。禁止使用`pkill python`、删除整批claim或覆盖jobs。新集群正式恢复前，要确认旧机对应任务全退出并检查残留claim；最后确认32卡产生新输出且日志正常后，再释放旧机器。这是冻结前切换方案；实际旧机已于01:21按用户要求暂停，新集群之后已完成准备并接续。不得重启旧机生成。

优先在新机挂载同共享ROOT，避免迁移几百万视频。即使共享ROOT可见，每节点仍需本地模型缓存（base/T5/VAE及merge）、框架与独立环境；先复制+校验再并发启动32进程。不要只git clone：未推送代码、模型、评测输出和环境不在远端。

### 12.3 吞吐提升的条件与限制

1. **视频生成/独立评测最容易扩展**：每卡一条独立任务，无跨节点all-reduce。相同单卡速度、数据供应和负载均衡下，32卡相对旧15张健康卡的理想吞吐比例为32/15=2.13。此前旧13张生成卡+2张judge阶段的比例32/13=2.46不是当前分工；01:09旧15张均生成也仅历史，当前实际新32卡生成。这只是资源数量的理想比例，不是新机实测、完成时刻承诺或单视频加速倍数。
2. **单视频仍是40GB卡推理**：卡数增加不自动降低单视频延迟或扩大单进程可用显存；仍需保持已验证的121帧/1280×704配置。若需要跨卡单视频并行，要单独验证sequence parallel，不能假定现有脚本支持。
3. **全参训练不保证线性扩展**：4节点all-reduce、CPUAdam/CPU offload和数据读取可能限制吞吐。历史旧环境禁IB走eth1；新集群若有可用IB/RDMA，不应盲目复制NCCL_IB_DISABLE=1。先核对NVLink/节点间带宽、CPU核数/RAM、CPU-GPU传输和本地盘，再比较小规模训练samples/s与保存显存。
4. **多卡改变训练语义**：历史FT1/FT2采用microbatch1×15卡×accum1=global batch15。直接32卡变成batch32，同一32,365样本epoch的优化器步从2,158变约1,012。样本数相同不等于更新次数/优化效果相同；不能把步数、warmup、save_steps或LR机械减半/倍增后当原实验复现。配置决定必须记录，并在后续小消融验证。
5. **建议先分工而非全32卡单一训练**：先把剩余生成按独立任务扩到新集群；后续训练可先2节点16卡，另1节点评测，另1节点caption/奖励或开发消融。只有训练扩展测试通过才决定4节点同步训练。不会因新资源而越过Teacher验收开始流式阶段。

### 12.4 四节点恢复的必要配置

- `run_gen.sh`已改为单节点前台托管，显式设置`NODE_RANK=0/1/2/3`、`NNODES=4`；每节点8子进程，global shard=rank×8+localGPU（0..31）、nshards=32。四节点独立生成smoke和正式缺失jobs恢复已执行，checkpoint、图像、prompt和seed保持不变；不要沿用旧15卡或重复启动。
- `train_full_16gpu.sh`目前`--nnodes 2`、两个LOCAL_ADDR和固定MASTER，不能只改GPU_LIST。四节点需要nnodes4、各节点nproc8、统一rdzv endpoint/id、四个本机地址与实际网卡；单节点/双节点/四节点都先做加载、有限loss/梯度和保存checkpoint测试。
- `score_pool.py`当前只运行launcher subprocess，跨节点评分需要显式节点任务分配及一致的benchmark/model/cache路径，避免所有32卡最终只用了node0。
- `eval_merge.sh`依赖旧pipeline日志触发，迁移后应review为“固定jobs恢复→输出齐→评分→严格覆盖”流程，不从头运行历史A–E以重复训练或caption。
- 每节点本地缓存按源绝对路径SHA256命名；ROOT变化要重建缓存键，内容校验仍使用`eval/diag/migration_checksums.json`。

四节点地址、同共享ROOT挂载和旧机切换时点已落实；不再重复要求这些恢复步骤。未来4节点同步训练仍需现场核对IB/RDMA/以太网、CPU/RAM/本地盘和保存开销；当前只接续独立生成/评测。凭据不写入本文，旧机本项目计算已停止，新机现有任务保持运行。

## 13. 给下一位 Coding Agent 的开场任务

> 请先阅读本文与实验记录30，并重新核查实时日志/生成数量/旧机是否还活跃。继续Wan2.2-TI2V-5B双向Teacher优化，不开始流式训练。保留已完整结束的Phase E3913输出/14组评分，不重跑；继续α0.7缺失生成与三插值模型完整WM/VBench、WMB、固定120条loss和最终严格覆盖，已VALID的CPU结果跳过，保留raw结果，明确prompt语义偏移限制。检查模型、输入、逐维分数完整性后更新summary和Teacher页面，客观比较领域/通用收益；在单seed不确定性内不宣称SOTA。根据文献和实测决定小规模推理/受约束微调验证，FT3候选配方尚未训练。迁移时保留本地未推送代码、数据清单、关键权重和benchmark依赖，不泄漏凭据、不与旧机重复写相同输出。遇到挂载/认证/许可等真正阻塞再请用户协助，常规步骤自主执行并中文报告。
