# 55 ALG与DyMoS真实机制Smoke结论（2026-10-26）

## 1. 结论边界

两个独立策略各1个固定开发集首例，在真实Wan2.2-TI2V-5B baseline NONE上完成完整50步/121帧/1280×704/24fps/shift5/CFG5，真实退出0、全帧ffmpeg -xerror与视频SHA、DiT逐参数前后SHA一致通过。**这是实际运行机制可用性，不是开发集质量、SOTA或Teacher晋级。**没有据此改gamma/factor、重新挑案例或启动AR。

## 2. 首次失败与实际修复

attempt01保留：ALG累计失败0.026576836GPUh，融合raw参考赋值到BF16 latent后错误用未转换raw做严格bits比较；DyMoS失败0.025954998GPUh，在pipeline内部set_timesteps之前错误要求scheduler已有timesteps。真实基座已加载但0预测。两个旧request/receipt原样保留，不将其改成功。

修复对齐原5B代码：首latent比较effective_z=raw_z.to(sample.dtype,device)，raw参考对象与bits仍每步核对不变；ALG仅低通effective首参考而保持原scheduler/sample与锁回行为。DyMoS在首次预测时严格核50步与时序，不改pipeline初始化。数学公式不变，原95测试全部保留＋11真实融合/lifecycle CPU测试，共106通过；公开联合697通过4个既有环境skip，eccfdb1远端13文件/182manifest字节核验通过后才启动attempt02。

## 3. 完整规格实际结果

| 方法 | attempt02实际GPUh | 两次累计GPUh | PyTorch峰allocated GiB | steps/active | 实际forward | 运行机制 |
|---|---:|---:|---:|---|---:|---|
| ALG | 0.145164177 | 0.171741013 | 26.350533 | 50/10 | 110 | PASS |
| DyMoS | 0.138778930 | 0.164733928 | 26.350218 | 50/10 | 100 | PASS |

ALG仍为U_original＋5×(C_lowpass−U_lowpass)，factor0.4，前10步3branch，其余2branch；原图/reference和future噪声不永久改变。DyMoS仅正支前10步30层self-attention执行融合score_mod，300biased＋2700original调用，R880/S27280；没有构造全尺寸S²矩阵或eager dense fallback。

两视频均121帧、24fps、1280×704；DiT前后完整逐参数SHA相等、预测/latent有限；自有执行进程释放。两次累计都小于独立每方法1GPUh烟测门槛，加载/编译/保存/失败占用已计费。占卡单列，不算真实实验收益。

## 4. 尚未完成

- 仅1例seed0不能证明motion/主体/背景/美学/成像/物理改善；没做完整94×双seed评价，不能称所有维度追平SOTA。
- DyMoS编译内核在实际尺寸可运行，但对原flash后端数值误差分布或其他分辨率未专门验收，不外推。
- 依据docs/51，下一步仍须预算准入、94例两seed固定单配置、相应base配对与每seed四护栏。生成24、评分16、失败4阶段不自动扩大；1视频含加载的墙钟不能直接当重复188次加载的吞吐模型。
- C276仍在42GPUh授权内补片/全帧验收，评分及最终结论未完成；正式DenseDPO因独立真实偏好标签门禁未过不启动，AR门禁保留。

无身份聚合结果：eval/diag/teacher_inference_smoke_public_results_20261026.json；私有request/video/逐参数证据不公开。
