# 36 Teacher优化：相关工作核验与可证伪方案

> 2026-10-20接续。仅记录核验的论文/实现与候选实验，不冒充本Wan2.2-TI2V5B已验证。P1低CFG失败；Teacher自动FAIL、用户review pending。固定5B，不迁入更大模型，不公开原媒体。

## 1. 决策摘要

- **立即推进**：L1随机窗口语义审核、M1同GT前向噪声状态的正负velocity分解。L1解码已340/340通过，实际76.160217秒，17源各20；语义标签仍0、父来源UNKNOWN，不能宣称caption准确率。
- **首选训练假设**：清洁标签子集上的小规模SFT，与同预算、同初始化的基座prediction anchoring比较。此为从图像持续学习外推的假设，不是已验证Wan方法；原Teacher限制中的“不得Student/蒸馏”仍有效，不自行启动该训练。
- **首选推理假设**：只有M1测量支持，才考虑固定CFG的Zero*投影单因素；不重开CFG网格。
- **后置**：视频偏好训练先检验pair排序是否奖励静止；APG先验证native clean-latent转换，不直接套图像默认配置。

## 2. 基座保真：Generative Distillation

**Continual Learning of Diffusion Models with Generative Distillation**，Masip等，2023初稿/v2 2024，正式PMLR2025。

- 原文：https://arxiv.org/html/2311.14028v2 （§3.2式9–10、§4.3、§5）。正式页：https://proceedings.mlr.press/v274/masip25a.html 。
- 官方实现：https://github.com/Atenrev/diffusion_continual_learning ；已观察commit `0d88ac5dea1ed7595b3f1f2f333b3bc1ce555f53`。入口train_cl.py及diffusion_full_gen_distill配置；未核验到LICENSE，不能复制代码后自称Apache许可。
- **原文事实**：Fashion-MNIST/CIFAR10 class-incremental DDPM/DDIM，teacher生成replay再加噪，匹配teacher noise预测。没有text-video、FlowMatch或Wan实验。current-data LwF/Gaussian约束可减轻退化，但仍有diversity与当前任务偏置，不能当完整生成回放复现。
- **本项目候选**：同5B frozen base、同x_t/t/c的velocity保真项 `L_flow + lambda ||v_model - stopgrad(v_base)||²`。与纯SFT比较，单lambda预注册，不用公榜调。当前数据anchor与原法replay不同，须明确标外推。
- **成本与风险**：每anchor样本增加base no-grad前向及权重内存；完整replay另需生成。可能保住基座坏物理或抑制新动作；anchor loss下降不能证明画质/动作保持。标签与新来源动作门禁未完成前不启动训练。

## 3. 视频偏好：VideoDPO与Diffusion-DPO

**VideoDPO: Omni-Preference Alignment for Video Diffusion Generation**，Liu等，2024/CVPR2025。

- 原文：https://arxiv.org/html/2412.14167v1 （§4、§5.1表1）。官方：https://github.com/CIntellifusion/VideoDPO ；观察commit `1febdb42db147824ba4c96eb5e787bbd1295afc7`，lvdm/models/ddpm3d.py及vc2_dpo/t2v_turbo_dpo入口。代码LICENSE未核验到，不从底层项目推定新增代码许可。
- **原文事实**：VideoCrafter2/T2V-Turbo/CogVideoX系text-video实验，非本Wan flow/I2V。OmniScore包含美学/画质、一致性、flicker/smoothness/dynamic、文本对齐；4候选best/worst与频率histogram重加权，不只是分差权重。
- **反证保留**：表1Turbo semantic73.93→73.81、HPS0.262→0.260，CogVid semantic67.10→66.99；overall涨不保证各维度。
- 基础论文：**Diffusion Model Alignment Using Direct Preference Optimization**，Wallace等，2023，https://arxiv.org/html/2311.12908v1 ；官方 https://github.com/SalesforceAIResearch/DiffusionDPO ，LICENSE.txt核验Apache-2.0。图像SD1.5/SDXL验证，路径KL/ELBO理论不能仅epsilon换velocity就声称FlowMatch推导成立。
- **本项目候选**：先在已授权本地pair检查综合评分winner是否动作失败/近静止，不新增生成或训练。只有action-vetted偏好与新来源盲审支持，才比较winner-only SFT与same5B ref偏好训练。pair生成/评分、winner/loser训练、reference前向均算成本。动作/物理作为否决门槛，不是调reward权重就保证。

## 4. 数据质量：OpenVid原则而非直接照抄

**OpenVid-1M: A Large-Scale High-Quality Dataset for Text-to-video Generation**，Nan等，2024/ICLR2025。

- 原文：https://arxiv.org/html/2407.02371v1 （§3、§6.3、AppA5）。官方：https://github.com/NJU-PCALab/OpenVid-1M ，观察commit `215fa7dc8adbc76cb183b4704c463b312181d3d1`。核验train/inference/download入口，未核验完整公开过滤/recaption脚本。数据文中CC-BY-4.0仍受上游条件限制，代码许可未知，数据许可不覆盖代码。
- **原文事实**：美学、CLIP邻帧、UniMatch运动、DOVER清晰度、cut检测和长caption；STDiT/MVDiT视频数据消融，不是Wan本项目微调验证。
- **不能照抄**：过滤high-speed/high-flow、只保moderate motion可能删掉本项目需要的快速操作/接触；长caption不证明动作真实。
- **本项目候选**：随机审核对象、动作方向/次序/接触、相机与虚构终局，保高低运动分层覆盖。之后同source/tokens/更新预算比较原caption与校订caption；quality-only独立另一消融，不能一次改label+quality+混合权重。无需启34B标注器或下载新数据。

## 5. CFG-Zero*：已有Wan2.1证据，但无动作保证

**CFG-Zero*: Improved Classifier-Free Guidance for Flow Matching Models**，Fan等，2025。

- 原文：https://arxiv.org/html/2503.18886v2 （§4.1、§4.2、§5.2、AppA3.2）。官方：https://github.com/WeichenFan/CFG-Zero-star ，Wan实现观察commit `9bc45a75939be9372f7d4e66391d29751ccb05ba`。作者README声明Apache-2.0，未见单独LICENSE，保留声明与核验范围。
- **原文事实**：每样本完整张量展平，`k=<p,n>/(||n||²+eps)`，`g*=kn+w(p-kn)`。有FlowMatch图像及Wan2.1视频，未报告Wan2.2-TI2V5B。小Wan表中总分80.52→80.91、成像65.40→68.13，但其他维度也有退化，绝非动作保持定理。
- **API陷阱**：作者Wan代码`i<=zero_steps`，zero_steps0会清首步；本地`progress_id<cfg_zero_steps`，0无零步。必须用“清几步”明确协议，不能静默改已有P1或声称默认完全一致。k理论正域与闭式解/代码不截负也有边界，照代码数学记录，不擅改clamp。
- **本项目候选**：M1完整p/n含首latent算k，统计排首latent；若跨case/t支持才做CFG5固定、projection only、清零0步的新小生成。early-zero若研究另单因素，不能同时开投影与清零当单因素。global norm可能受静态区域主导；画质涨但动作幅值/完成下降即否定。

## 6. APG：clean-latent投影，不是velocity直接投影

**Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models**，Sadat等，2024。

- 原文：https://arxiv.org/html/2410.02416v1 （§3–4、Fig12、AppB/D），作者代码为AppD snippet，未核验独立作者repo。可用HF实现：https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/guiders/adaptive_projected_guidance.py ，文件头Apache-2.0，不冒称作者官方实现。
- **原文事实**：EDM2/SD1.5/SDXL/Lightning及SD3 flow-image，未见视频/Wan。先clean-latent `d=x0_pos-x0_neg`，momentum、norm、projection，`x0_guided=x0_pos+(w-1)*(d_perp+eta*d_parallel)`。作者纠正noise-space误用见 https://github.com/huggingface/diffusers/pull/9626 。
- **本项目候选**：native flow `x0_hat=x_t-sigma*v`，正确转换回v；首帧锁定/sigma边界单独检查。只方向组件eta0与eta1恢复公式先CPU数学验证，不同时叠momentum/clip。HF formulation defaults不是去parallel时原cond anchor公式，需明确original formulation与w-1。图像默认norm阈值不能直接用于大视频张量。
- 不多模型前向但有变换/norm/momentum内存；改善过曝不等于恢复动作，更不能把纯v投影称原APG复现。

## 7. 当前选择与验收

上述5项都是可证伪候选，无一保证本5B画质、动作、物理齐升。先L1语义与M1机制，后基座保真小训练；视频偏好后置，guidance仅测量支持才小试。

所有新GPU实验硬预算、固定输入/代码/权重hash、准确owned进程清理、完整输出与否定结果保存；raw/po与用户Teacher review分开。源级/跨seed确认不能被训练可见6case、单reward排名或rawval替代。
