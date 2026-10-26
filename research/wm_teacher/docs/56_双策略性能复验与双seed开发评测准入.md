# 56 双策略性能复验与双seed开发评测准入（2026-10-26）

真实ALG/DyMoS完整规格单视频机制已通过（docs/55），但此前没保存模型加载/推理/保存的可靠分段，不能猜旧日志或把一次加载反复乘188。下一步只采用独立同pipeline双视频测量，不再另执行单视频03。

每方法新增性能测量最多0.75 allocated GPUh（timeout2600秒、kill30秒、外层2660秒），仍在原docs/51烟测累计1GPUh内：ALG原累计0.171741013，DyMoS0.164733928。失败、加载、编译、保存、哈希/清理全计费，不借C42预算。原attempt01/02记录和结果不覆写。

固定E1首个安全案例，同一base NONE pipeline一次加载、seed0和seed1两条，原50步/shift5/CFG5/121帧/1280×704/24fps、ALG factor0.4或DyMoS gamma0.6/phi1、前10步active，不看质量分选择参数。CPU准备新private request并绑定旧机制文件、当前模型/输入/代码/运行环境哈希；真实run显式单GPU授权/实时监督器和硬超时。

计时为实际monotonic分段与原始起止记录；完整两个视频/参数哈希/50steps/退出0才能成为可复算测量证据。旧02helper源码曾追加纯观察计时，因此旧源码只与已发布eccfdb1不可变manifest source_sha和snapshot_sha分别核对；其他算法/模型/input/current源码不放松，不假称当前字节等于旧版本，也不伪造旧throughput侧车。

依据实际同pipeline一次load＋两视频最大推理/保存用时、编译与未分配开销保守估188例与每shard余量。没有真实测量/PASS或估计超过生成24GPUh时，开发评测保持BLOCKED，不降步数/分辨率、不给每shard24额度。若准入通过，94固定cases×seed0/1共188、源码与全部shard先预留、总生成24/评分16/失败4＋累计烟测1≤45。每seed四护栏独立判断、缺覆盖/错hash拒绝，不用均值掩盖失败。

既有base复用须同图/prompt/seed/配置/权重来源和完整decode/评分证据；缺少历史生成版本证明时base.allowed=false，不能称公平质量比较已完成。必要同协议base重生成需另外明确预算，不从候选24静默借额度。

新measure/devset脚本仅CPU回归129项通过，rank3历史来源和当前9模型/runtime真实CPU检查通过；尚无真实性能或188条质量结果，方案/代码先走四关发布再GPU。公开不含request/prompt/video/逐case/私有目录。
