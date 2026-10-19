# 35 Teacher：机制诊断与随机标签审计

> 2026-10-19预注册，2026-10-20验收更新：P1执行/发布完成，M1实际18状态/72前向及独立54378项复算通过，0.182380 GPUh/上限2；详见文档37。以下方案保留预注册边界，不启动Student、蒸馏、AR或全量FT3。

## 1. 已确认的计算事实

标准CFG为g_s=n+s(p−n)：CFG5=5p−4n，CFG3=3p−2n。P1的n使用原非空NEG且共享首帧图像，不是纯无条件分支。当前SFT代码只对正caption前向施加FlowMatch损失，当前固定验证也是正captionMSE；训练损失乘scheduler training_weight而固定验证不乘，不能说加权标量完全同objective。不能据当前代码断言base预训练或全部历史训练未有负文本/dropout。共享DiT更新如何改变NEG预测仍待实测。

DiffSynth文件noise_pred实际是FlowMatch velocity：x_t=(1−sigma)x0+sigma eps，target=eps−x0，更新x_next=x_t+(sigma_next−sigma)v。首帧覆盖锁定；主统计必须排除future latent之外的首帧。

## 2. M1同状态72前向诊断

固定P1六case的原顺序/校订label/首帧/原NEG/seed0；训练可见诊断，非泛化验收。一次VAE encode真实GT得到x0，复用同首帧z与独立positive/NEG context。不能从cfg1训练输入直接取“nega context”，它可能只是pos副本。

冻结50步、shift5实际scheduler表index[5,25,45]，记录原sigma和传入bf16 timestep以及add_noise实际选值；索引比例不是sigma。每case使用一份CPU fp32 seed0噪声后转bf16，三个t和两个模型共用，构造同GT前向加噪x_t并覆首帧。每个model_fn输入clone，eval/no_grad、cfg_merge=false、TeaCache关闭，不进行去噪递推，不输出新视频。这是GT forward-process诊断，不是P1中间rollout重放；输出MP4无法恢复原rollout x_t。

6case×3t×2model×2branch=72次前向。仅一次构造正负预测，离线fp32计算g3/g5及Zero投影，不因多个CFG额外前向。

记录future latent RMS(p/n/p−n/g)、夹角/零normNA、跨模型delta p/n/d/g、每latent帧分解和真实一步|delta sigma|RMS(g)。对正预测记录raw target MSE；负文本本就不对应正caption GT，不用负MSE定义负文本“质量”。局部latent时间差不是光流/动作成功。

交叉G(P,N;s)=sP+(1−s)N：AB−BB=s delta p，BA−BB=(1−s)delta n，AA−BB为二者和。记录交叉项，不把RMS相加当总RMS；有限性、18同状态输入/72输出、公式fp32残差和source/cache hash均需验证。离线交叉不是混合分支生成质量证据。

候选硬预算最多2 allocated GPUh：优先2卡×3600s（base/A07各build一次）或4卡×1800s（各两个worker分case）。同预处理缓存，GPU加载/编码/等待/前向/传输/失败/cleanup均计墙钟；deadline内预留grace，准确owned组清理，不自动retry/扩额。必须先实现单测、实cache pin、本地code/framework冻结、同输入hash与新目录验证后才启动；本文件不是已执行证明。

## 3. Zero不是默认下一轮

k=<p,n>/(||n||²+eps)，g*=sp+(1−s)kn，非简单normalize且k不保证非负。必须先按生产算法在完整p/n（含首latent）归约k，再仅对未来latent统计；不能先切首帧再算k。early-zero在两支前向后清零，不省DiT计算且改轨迹，不能和projection同开声称单因素。只有跨case/t重复出现负贡献幅值失配、projection降低异常更新且无未来时间成分坍缩，才考虑CFG5固定、projection开启、zero_steps0的12追加视频。生成/评分预算另冻结，禁止自动大grid；仍须画质/动作/GT护栏，不凭张量局部值晋级。

## 4. L1标签审计：先冻结抽样，再审语义

总体为FT2原CSV全部32365窗口/17来源；原CSV SHA256=2fd1615c3c538d9d9462ad166b78129f8c349174bffd5ce1085888d5f001a0a6。

采用明确可执行的**来源分层、窗口等概率不放回**设计：各来源抽min(20,N_s)窗，RNG20261019，来源名和CSV行稳定排序。17源均不少于20，总340窗，pi=n_s/N_s；父来源可能重复，不声称340独立父源。抽样不读模型输出、不替补missing或已知坏例，原P1案例不按结果排除；模型输出不能影响抽样。

父来源canonical审计作为单独字段，跨窗口相关及UNKNOWN单报；源级相关未厘清前，不给基于独立伯努利假设的精确错误率CI。将来如改为父源均匀抽样必须另版本，不能把本清单临时解释为父源采样。需要总体窗口错误率时按源N_s加权，不把宏平均当训练混合错误率。

冻结完整抽样row/index、视频相对路径、原prompt/content hash和入样概率，只存私有；原CSV/label不覆写。审阅四类：对象属性、动作时间序列、相机、虚构终局，supported/contradicted/unverifiable与severity/时间区间。全decode不等于语义审阅；五帧只导航，“未见”不等于全clip不存在。先单agent试审且明确限制；20%独立第二审阅未获实际人审前不能伪造一致性。未知/不可解码保留原样，不能换易例。

CPU冻结清单可以先执行；不向外部VLM发原media，不把词数审计当标签准确率。已有<80词仅文本长度诊断，不据此认定错误。

## 5. 训练分支的条件

若正分支误差/标签问题更突出：用新随机审计支持派生校订版本，再做基座初始化、保守更新的小数据单因素训练；先梯度/保存/恢复smoke，控制source与文本协议，禁止多因素一起改。若负分支投影机制支持：先单因素小生成诊断，再新来源复核。

两种路线都需要独立来源开发复核、真实动作与画质不退化，才冻结候选做一次完整benchmark验收。固定loss只是必要读数，不以其下降保证质量。质量目标仍为原Teacher多non-dynamic头部条件、GT/loss和用户review，不能随诊断结果降低目标。

## 6. 已实测进度（2026-10-20）

- L1固定340窗口完整CPU解码全部通过：17来源各20、全部实测121帧/24fps，原CSV/抽样RNG/逐video/JSON/contact hash与权限复验通过，8 CPU worker每decoder1线程，实际76.160217秒。全340来源identity/canonical仍UNKNOWN、语义NOT_REVIEWED；不能将decoder通过当caption准确或父来源独立。安全聚合见eval/diag/teacher_label_audit_L1_public_summary_20261020.json。
- 首样本导航试审发现可见caption矛盾，例如人物朝向与终局折叠、静止视角描述；只保留私有triage，不据四个非全clip审阅估全库错误率，不原地重标。
- 外部五项已核验原文/实现/许可边界，见[相关工作](36_Teacher优化相关工作与可证伪方案.md)：基座保真、VideoDPO、OpenVid数据原则、Zero*、APG。没有任何一项已证明本5B动作/画质/物理齐升。
- M1已实际执行72前向并独立复算全部指标；25专项CPU测试通过。共享文件系统全CSV resolve预检阻塞已修为完整路径lexical匹配后只resolve唯一命中，原SHA校验保持。Zero k接近1，暂不支持盲开投影生成网格；不把本地前向诊断当rollout因果证明。
