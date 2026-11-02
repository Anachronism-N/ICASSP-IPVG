
# 59 实验全景、最佳 Teacher 与 SOTA 差距

> 更新：2026-11-02。依据原始评分、当前源码和本次重新读取的官方公开表。
> 双向 Wan2.2-TI2V-5B Teacher；AR / Student / 蒸馏仍未启动。

## 1. 核心结论与重要更正

**当前整体默认仍是原生 base + shift5 / CFG5 / 50步，但它不是所有指标的本地第一。没有新 Teacher 已被证明整体超过同协议 SOTA。**

- VBench355 raw Total* 的已测自有局部最高：base–FT1插值α0.5，89.892408；base为89.860075。这点微差不抵消其Physics/WMB退化，也不证明多seed稳定收益。
- WM108 Total*、WMB350 raw、Physics-IQ Original最终分的已测自有局部最高均为E2 shift12：约87.312424、8.397143、31.526205%。但公开VBench主体时序一致性退化、dynamic未迁移，并未通过全部质量/用户review门，不自动晋级。
- 最新完成的候选多基准子集诊断闭环仍是C276。总分均值有小幅改善，但CI含0、WMB及部分一致性退化，C不全面晋级。
- 新标签pilot、CPU恢复测试和工程修复不是新模型质量成绩，不能说SOTA差距因此收窄。

### 重要勘误：以前E2 Physics取错字段

本次直接复核原文件：

- eval/physiq/results/physics-IQ-benchmark/results/base_s12c5-op-run_01_metrics.json:2 的 score_spatiotemporal = 0.2841228754；这是时空子项。
- 同文件:8 的 final_score_orig = 0.3152620481；最终Original综合分是 **31.526205%**。
- base原文件同字段 final_score_orig = 0.3013037869，即30.130379%。

以前docs46/47/48及部分状态说明把E2子项28.41%当最终分，写成“Physics下降1.7pp”，**该理由错误，现明确撤回**。真实E2最终分相对base **+1.395826pp**。对各66scene CSV用原官方IQTable独立CPU重算与saved final差为0；事后5000次scene配对bootstrap（RNG20261018）CI约[−0.397640,+3.082520]pp含0，不称显著改善，也不事后修改原选择标准。不改原始评分文件，不把纠错当新实验收益；E2是否默认晋级仍需按完整质量门重新解释，不能靠错误负向理由或单项正向结果决定。

## 2. 我们确实在对比SOTA吗？

**比较了官方榜单领先模型；尚未完成多数项目的同协议SOTA验收。** “官方领先参照”与“严格可比SOTA对手”必须分开。

|证据层次|可以支持的结论|不能支持的结论|
|---|---|---|
|同病例/输入/seed/评分配对|候选相对base差、CI、预注册护栏|自动等价为超过外部SOTA|
|本地完整集合单seed评测|固定输出下的本地描述性优劣|微差是稳定收益|
|官方榜单公布值|当前页面领先行、公布值描述差|没对齐规格/预算的正式排名|
|去dynamic自定义复算|解释非运动质量构成、按同公式比较各行|超过官方原Total SOTA|
|标签/工程准入|数据结构、恢复、吞吐、费用闭合|模型训练后质量已经提升|

本文统一Δ=我们−参照；百分制差用pp，WMB用分/10。严格协议未对齐时，同协议SOTA gap写“未确定”，不会以小数精度制造比较确定性。

### 本次实际刷新来源（2026-11-02）

1. https://vchitect-vbench-leaderboard.hf.space/config ：I2V component69，36模型。官方原Total最高仍DreamX-World-1.0 90.49%，第二Vivy-R-Vision-3B 90.21%。
2. https://physics-iq.github.io/ ：Original I2V无BoN最高参照Cosmos3-Super43.8%；I2V+WMReward BoN48.9%。跨模态64.5%属于Magi-1+GeoPhys多帧V2V+BoN，不当I2V单采样对手。
3. https://worldmodelbench-team.github.io/ ：最高judge行仍KLING9.10，表日期2024-11-15。页面不足以核实2026现代统一I2V、精确版本和采样预算下的SOTA。

机器复算证据：eval/diag/teacher_experiment_sota_review_20261102.json，保存本次36行原八维、源响应SHA、时间、归一化和本地结果文件SHA。官方SVD-XT1.0/1.1原Total缺失记null；不补零。八个core维度均可复算。

## 3. 固定设置与四种benchmark

- 默认生成：1280×704、121帧、24fps、50步、sigma_shift5、CFG5，原生双向TI2V-5B。
- raw和prompt-optimized（po）分别报告。po是输入扩写，不是新模型；可能引入物体/动作/终局偏移，raw→po上涨不能归为权重训练收益。
- VBench-I2V：core355图文；主体/时序/运动适用246，背景109，美学/成像355。本地通常每pair1视频，官方每pair5视频；另有camera109图×7指令×5视频。
- WM108：本项目108条领域留出，用GT首帧+caption续视频，**不是WorldModelBench350**。外部没有同108集合可直接减分的公开SOTA。
- Physics-IQ Original：66完整scene×3视角=198；原120帧@24fps、pad1280×720/边缘smear/crf12、双take GT与原IQTable非线性聚合。Original≠Verified。
- WorldModelBench：350条、7域、原judge8问；instruction + 5×physics_good_rate + 2×common_good_rate，满分10。judge≠human表。
- GT：DreamSim均值/末帧，K12排除t0；abs(log(generated_motion/GT_motion))为运动误差，越低越好。距离和运动量匹配不充分证明动作/物理正确。
- 独立loss：120条父来源留出、5固定t和噪声；loss下降不是生成质量的充分条件。

## 4. 以前做过的实验：训练、机制和失败都列出

### 4.1 领域LoRA、全参训练与插值

来源：docs30:31/56/80–133；docs32:14/29/44。以下历史结果不是本次新运行。

|实验|idea / 初始化|数据与更新|主要真实结论|
|---|---|---|---|
|R1|base领域LoRA r32，先学2秒|59682条，49帧，lr1e-4，repeat2，14921更新|loss0.1922；DreamSim0.159858；公开355/Physics/WMB未测，不能叫SOTA|
|R2|从R1延到5秒/121帧|96555窗，约6437更新，lr1e-4|loss0.1879；DreamSim0.160706；VBench89.532359，通用能力下降|
|R3|R2续训全新窗口并退火|30001窗，2001更新，lr5e-5|loss0.1876；DreamSim0.157656、末帧0.181804最低；VBench89.416782|
|R4|R3上美学筛选/来源限额|24650窗，1644更新，lr3e-5|DreamSim0.156918最低；VBench89.465387，Physics27.3267%，WMB7.10，通用退化|
|R5方案|美学+运动联合选样，30%短caption|33040窗|**没有独立R5 LoRA训练**；改用于FT1，不虚构R5成绩|
|FT1|base融合R4后全参|33040窗，2203更新，lr5e-6|运动误差0.671924最好；Physics24.8192%，WMB7.091429；完整VBench raw未测|
|FT2|从base重启全参，39%补充数据/dense-only|32365窗，2158更新，lr5e-6|VBench89.819070，WM87.105983，Physics28.338%，WMB8.145714；恢复但不全面超过base|
|base–FT1插值|W=(1−α)base+αFT1，α0.3/0.5/0.7|无需新训练，各1011视频，共3033|α0.5 VBench89.892408略高；Physics28.169852%，WMB7.814286，整体门未过|

插值详细对照：

|raw候选|VBench Total*|WM Total*|DreamSim↓|motion error↓|Physics Original%|WMB/10|
|---|---:|---:|---:|---:|---:|---:|
|base|89.860075|87.064230|0.174638|0.994161|30.130379|8.297143|
|α0.3|89.814224|86.904217|0.169400|0.940776|29.634157|8.031429|
|α0.5|89.892408|87.056111|0.165566|0.900953|28.169852|7.814286|
|α0.7|89.772426|86.909158|0.159756|0.805393|26.875848|7.511429|
|FT2|89.819070|87.105983|0.172209|0.921575|28.338%（历史值）|8.145714|

FT2同时改变初始化、数据混合、镜头筛选、第一人称比例、短caption策略，不是单因素“通用数据稀释领域收益”实验。旧R1–FT2完整allocated成本未在本次只读范围重新追溯，不编GPUh。

### 4.2 P1 / M1 / C1：诊断不能替代泛化

|线|做法与规模|结果|
|---|---|---|
|P1|6来源×base/A07×CFG5/3=24，校订prompt|A07低CFG主体一致性升，但成像−3.4526pp、末帧/运动误差退化、抓取失败；base低CFG机器资格PASS但不全面晋级；实际2.912882GPUh|
|M1|6例×3t=18同状态，base/A07正负共72forward|A07正支MSE18/18下降；CFG放大参数delta，但无负支范数压倒正支证据；不直接证明动作成败，实际0.182380GPUh|
|C1|6例×base/A07，对原/校订caption反事实24视频，其中12新增|两模型DreamSim6/6改善，机器人动作仍失败/畸变；输入修正不是重训练收益，新增1.894074GPUh|
|2026-10-20训练smoke|4固定cached-input窗4次更新，LoRA600tensor/80.6M参数|峰21.5941GiB、base不变、同worker扰动后恢复；不证明新进程下一步resume或模型质量，0.087707GPUh|

这些有限样本有训练配方可见/成员关系边界；不能概称全部未见泛化。来源：docs34:5/19/32；docs37；docs39:5/17/38/79；docs42:15/53。

### 4.3 E1 / CFG-Zero* / seed复验 / E2

|线|固定变量与规模|结果|
|---|---|---|
|E1|94开发例×shift{3,5,8,12}×CFG{4,5,6}=1128|选s12c5；Total87.430717→87.503466，dynamic+2.127660pp，仅开发资格|
|E1b z0|94例，CFG投影不early-zero|Total87.416853，四护栏过但略低于base，中性不采用|
|E1b z2|94例，前两步特殊CFG|成像+1.709692pp，但DreamSim+0.0181015超+0.005护栏，FAIL|
|E1四seed|94×2配置×4seed=752，包含原seed0不是全新752|dynamic差+2.127660/−2.127660/+1.063830/−2.127660pp，均值−0.265957，样本std2.193141；seed2 DreamSim FAIL，不平均藏失败|
|E2|冻结s12c5一次四bench：355raw+108+198+350=1011|VBench89.795634低于base、subject时序差−0.283879pp CI全负、dynamic−0.813008pp未迁移；WM/WMB/**最终Physics**改善，仍未完整晋级|

E2完整结果（纠正取错Physics子项后）：

|项目|base|E2 shift12|Δ|
|---|---:|---:|---:|
|VBench355 Total*|89.860075|89.795634|−0.064441pp|
|VBench dynamic|56.097561|55.284553|−0.813008pp|
|VBench subject_consistency|95.448874|95.164994|−0.283879pp，CI[−0.410,−0.158]|
|WM108 Total*|87.064230|87.312424|+0.248194pp（同公式历史aggregate参照）|
|Physics Original final|30.130379%|31.526205%|**+1.395826pp**，非此前“下降”|
|WMB350|8.297143|8.397143|+0.100000分|

E2没有po355输出，不把计划中raw/po各355写成已运行。单seed正向Physics/WMB不能证明全面或SOTA；纠错也意味着不能再将“全部免训练采样均在所有bench退化”概括为事实。

来源：docs43/46；原E2各维scores标量；Physics原metrics最终字段；eval/wmb/results/base_s12c5.json完整350/1750/700子问重聚。

### 4.4 E3：相同预算下标签修正和anchor三臂

1856窗口、每臂254优化更新，LoRA rank32，base冻结、顺序/t/noise一致。A原caption；B约束性修正caption；C=B+λ0.5基座保真anchor，L_FM+λMSE(vθ,stopgrad(vbase))，同latent/t/context、排除首参考帧。

当时修正标签不是human gold，短caption约18%回退；后续结构/语义审计不能反向修复旧C权重。

评测是同94开发例/模型/seed：seed0与seed1各三臂282新增视频，base各94历史参照；不是1856训练窗口的评测，也不是355公开集。

|模型|seed0 Total*|seed0 dynamic|四护栏|seed1 Total*|seed1 dynamic|四护栏|
|---|---:|---:|---|---:|---:|---|
|base|87.430717|57.446809|对照|87.615127|54.255319|对照|
|A|87.196237|59.574468|FAIL DreamSim|87.560470|58.510638|PASS|
|B|87.280703|58.510638|PASS|87.579524|59.574468|FAIL DreamSim|
|C|87.295070|57.446809|PASS|87.508973|55.319149|PASS|

原资格规则四门：smoothΔ≥−0.003、subjectΔ≥−0.010、DreamSimΔ≤+0.005、motion errorΔ≤+0.05，候选−base。合格中选最高Total*，原规则没有要求Total*>base，不能事后改门。

C被选中只是资格通过；两seed Total*差−0.135646/−0.106154pp，均值−0.120900；dynamic差均值+0.531915pp。不是稳定训练收益或最佳Teacher。

完整生成/评分覆盖不等历史全runtime/权重元组溯源完备。历史base来源复用审计仍有严格执行版本证据欠缺，配对结果不能升级成严格同版本SOTA因果证明。

### 4.5 C276：最后完成的多基准子集诊断闭环

唯一C/seed0/s5c5，盲抽raw71+po71+WM22+Physics14完整三视角scene42+WMB70=276，33396帧全解码，配对/非线性公式/评分覆盖/独立9884checks/费用闭合。不补称完整1366。

|集合|C|同子集base|Δ|95% CI|
|---|---:|---:|---:|---|
|VBench raw Total*|90.105419|89.934739|+0.170679pp|[−0.218970,+0.608091]pp|
|VBench po Total*|89.679281|89.493260|+0.186020pp|[−0.024820,+0.488302]pp|
|WM22 Total*|87.464093|87.380459|+0.083635pp|[−0.147141,+0.414795]pp|
|Physics Original14scene|26.892795%|26.111923%|+0.780872pp|[−0.767229,+2.240003]pp|
|WMB70|8.342857|8.500000|−0.157143分|[−0.500000,+0.157143]分|

总分差CI全含0。raw dynamic+6.122449pp，但raw主体时序−0.291414pp和平滑−0.060143pp的CI全负；po成像−0.176327pp。WMB形变good-rate差−0.114286，CI[−0.2,−0.028571]全负，自驾/工业/游戏退化。

WM22四容忍护栏通过不表示全部质量提升。实际29.583832085GPUh≤批准42。**不全面晋级C**；不能拿71子集90.105419和355图DreamX90.787722减出假的SOTA收窄。

来源：docs58；eval/diag/teacher_C276_public_results_20261026.json，left=C/right=匹配base。

### 4.6 ALG / DyMoS：真实机制通过，正式质量对照失败

- ALG：U_original + CFG×(C_lowpass−U_lowpass)，首参考latent bilinear down/up factor0.4，前10/50步三分支；不永久修改参考/未来噪声，总110forward。
- DyMoS：仅正支前10步，scaled-QK之后softmax之前非参考query→首参考key减γ0.6；R880/S27280/30层融合score_mod，不构造S²矩阵；100forward，300biased+2700originalattention。
- 固定1例完整121帧机制smoke真实PASS，峰约26.35GiB；ALG含失败累计0.171741GPUh，Dy0.164734。后续同pipeline双seed吞吐也通过，但这不是质量分数。
- 本地5B适配不是文献14B官方复现，论文涨分不能搬给本地。

|正式尝试|base完成/12|ALG完成/12|Dy新正式视频|结论|
|---|---:|---:|---:|---|
|a1|10|10|0|canary失败，余15分片不启动|
|唯一恢复a2|9|10|0|再次失败，恢复权限耗尽|

39个完整诊断视频留存，不混为严格188对照。累计生成费用保守上界：base2.380587GPUh、ALG4.294023GPUh、Dy历史2.655524GPUh；这些上界不冒称累计actual全部精确已核，未知actual不清零。没有完整双seedquality分数或候选赢家。

### 4.7 新训练准入：还不是新Teacher成绩

新32GPUh分阶段授权：label/cache24、singleGPU训练恢复技术2、multinode技术6，不借阶段；正式单候选训练预算未批准。

旧1856标签全v1，1249原结构通过/607失败，不能当独立语义gold。四节点Qwen14assets实际核验后第二标签pilot16/16原结构记录，但AI独立8抽帧审阅只有2 adequate/14 needs_edit，9例引用不存在frame8/9；不能放行全量。pilot含首失败累计0.091830960GPUh。

v3修复了索引、无依据mostly-still短句兜底、主体动作归属提示和帧元数据拒绝；112CPU+独立76mock/3CLI通过，不证明新标签视觉已好。用户已批一次同16例恢复各node810秒，所有原费用+新3240秒≤原1GPUhpilot，仍须新freeze/发布/实际生成/独立语义审阅。

单卡恢复CLI的CPU4reference对prefix2+freshprocess2全adapter/AdamW/RNG/顺序精确相同，共8次tiny CPU更新。后续2026-11-02已实际执行全尺寸80.6M LoRA：reference4、prefix2、freshresume2全部更新并保存，三个CUDA进程、基座不变、prefix checkpoint加载状态精确恢复；但reference/prefix首步backward梯度诊断已不同，最终adapter/AdamW轨迹不精确一致，**技术验收FAIL**，没有成功cost.closure。实际已观测611.482715秒，费用仍保守上界7200秒=2GPUh，不自动退款/重试/放宽容差。原完整费用监督两漏洞已修，不以22+6CPU通过冒实机成功。

只读根因定位：原生FA3优先路径没有显式deterministic backward，接口默认False，符合forward loss相同但backward梯度不同的观测；不是独立GPU A/B证明某个atomic kernel唯一致因。三checkpoint RNG精确一致，LoRA B最终相对L2差8.3918%、Adam一/二阶moment相对差52.3%/74.2%，不能以全adapter总体0.12%的差掩盖B或optimizer变化。新技术专用确定性修复和一次额外有界2GPUh正在准备，旧失败和费用保留，不借标签/多节点预算。

一次v3已实际结束：15结构标签+1错误；独立AI8抽帧审阅3 adequate/11 needs_edit/1 uncertain，训练语义仍未批准。用户选择换更强标注器新pilot；实际Qwen3-VL-32B-Instruct BF16 14分片62.133GiB，rank1当前完整24加载资产流式SHA已核，补README后25资产proof绑定。下一固定rank1四卡/16帧/32病例（16诊断+16盲抽）标签pilot预算2GPUh计原24GPUh，图像再次参与caption生成，实体ID分别绑定持物和事件。仅更大模型、CPU处理或资产存在不证明语义质量提升，仍须一次实际生成及独立盲审。

**没有新Teacher训练后bench结果；工程测试、标签结果不代替SOTA证据。**

## 5. 最佳版本到底是什么情况？

### 整体默认与局部最优分开

|范围|最佳已测自有结果/协议|数值|限制|
|---|---|---:|---|
|整体默认|base raw s5_c5|保留基座|不代表每个单项最高|
|完整355 raw Total*|mergeα0.5|89.892408|仅+0.032333pp，Physics/WMB退化|
|完整355 raw去dynamic|base|91.135529|自定义诊断，不是官方Total|
|完整355 po Total*|base+po|89.791814|输入增强不等raw权重收益|
|历史po去dynamic|FT2+po|91.558873|原Total低于base+po|
|WM108 Total*|E2 s12c5|87.312424|VBenchsubject退化，未全面晋级|
|WM108平均DreamSim↓|R4|0.156918|通用/物理/常识退化|
|WM108末帧DreamSim↓|R3|0.181804|不要误称R4末帧也最好|
|WM108motion error↓|FT1|0.671924|不证明动作正确；公共raw355缺测|
|Physics Original raw最终分|E2 s12c5|31.526205%|此前误报取了时空子项；单seed不代表稳定收益|
|Physics Original po|base+po|约32.48%|历史summary两位小数，不编更精确值/Verified分|
|WMB350 raw|E2 s12c5|8.397143/10|未全面晋级|
|WMB350 po|base+po|8.857143/10|raw/po语义未严格等价|

base raw完整参考：VBench89.860075，WM87.064230，DreamSim0.174638，末帧0.208683，motionerror0.994161，Physics30.130379%，WMB8.297143。

GT的WM87.496428不是生成模型。异构Wan2.1/Hunyuan在某些单项更高，但其他综合/GT项更低；它们只是同108集合已测baseline，不能直接叫2026SOTA。

## 6. 各bench距离SOTA还有多远？

### 6.1 跨bench参照差（不等严格可比gap）

|基准/口径|本地结果|官方领先参照|描述Δ|严格同协议gap|
|---|---:|---|---:|---|
|VBench八维Total* base raw|89.860075|DreamX同八维90.787722|−0.927647pp|未确定|
|VBench八维Total*局部最佳α0.5|89.892408|同上|−0.895314pp|未确定|
|VBench自定义去dynamic base|91.135529|DreamX同公式91.344148|−0.208619pp|非官方指标，未对齐|
|VBench自定义去dynamic base|91.135529|36行最高LTX-2无po92.425997|−1.290468pp|非官方指标，未对齐|
|Physics Original base raw|30.130379%|Cosmos3-Super I2V无BoN43.8%|−13.669621pp|未确定|
|Physics Original局部最好E2 raw|31.526205%|同上|−12.273795pp|未确定|
|Physics Original base raw|30.130379%|Cosmos3-Super+WMReward I2V BoN48.9%|−18.769621pp|不同选择预算，不是同预算gap|
|Physics Verified|未测|历史已核noBoN48.62±0.46 / best8 55.10|不计算|Original不能冒Verified；本次未刷新Verified表|
|WMB350 base raw|8.297143|历史KLINGjudge9.10|−0.802857分|版本/生成模式/预算未对齐|
|WMB350 E2 raw|8.397143|同上|−0.702857分|未确定|
|WMB350 base+po|8.857143|同上|−0.242857分|扩写≠raw；也不等human表|
|WM108|base87.064230/E2 87.312424|无同108公开SOTA|未确定|需确定外部候选同输入重评|

VBench本地1视频/pair、规格1280×704/121帧24fps，官方core5视频/pair、另camera集，各模型分辨率/时长不同；官方5B po例1280×720/81帧16fps。DreamX精确榜单checkpoint/raw-po未充分明示，不用另外的AR版本冒同款。

Physics的prompt/checkpoint/多帧context/评分版本/run/BoN需核；最高V2V64.5不能拿来减本地I2V。WMB最高行是旧混模态表，没有现代精确I2V版本协议；9.10 judge不能换成8.82 human。

**没有“在所有bench仅差一个已精确确定的数”的证据。上表为有明确范围限制的描述参照。**

### 6.2 VBench逐维：对统一DreamX和对不同模型逐维最高

|维度|base raw|DreamX|Δ对DreamX pp|当前公布逐维最高|Δ对最高 pp|
|---|---:|---:|---:|---|---:|
|i2v_subject|97.826929|98.27|−0.443071|LTX-2无po99.38|−1.553071|
|i2v_background|99.093271|99.04|+0.053271|ToMoviee99.64|−0.546729|
|subject_consistency|95.448874|95.35|+0.098874|Animate-Anything98.90|−3.451126|
|background_consistency|97.540727|97.24|+0.300727|VideoCrafter98.79|−1.249273|
|motion_smoothness|98.705313|98.52|+0.185313|STIV Apple99.61|−0.904687|
|dynamic_degree|56.097561|72.11|−16.012439|DynamiCrafter-CIL-51283.01|−26.912439|
|aesthetic_quality|61.971457|63.70|−1.728543|疑似异常70.21；排除此行参照67.12|−8.238543/−5.148543|
|imaging_quality|71.060625|71.46|−0.399375|疑似异常97.07；排除此行参照72.86|−26.009375/−1.799375|

raw base对DreamX7个非dynamic有4项描述性略高，不等对所有逐维SOTA领先4项。每维最高来自不同模型，不能拼成一个实际模型。

CogVideoX1.5行出现smooth40.98/dynamic62.29/aes70.21/imaging97.07等异常，疑似错列；保留原表，不自己交换纠正。上述美学/成像公布最高不能作可靠SOTA验收目标，排除异常后的参照也未同协议复现。

## 7. VBench去除dynamic_degree：推荐计算与真实结果

### 7.1 定义：不是七维简单平均

原分数x是0–1；沿用scripts/summarize_eval.py:23/47的归一化n=(x−lo)/(hi−lo)，不新增裁剪。

|维度|lo|hi|
|---|---:|---:|
|i2v_subject / subject_consistency|0.1462|1.0|
|i2v_background / background_consistency|0.2615|1.0|
|motion_smoothness|0.7060|0.9975|
|dynamic / aesthetic / imaging|0|1|

I2V和Quality两组各50%：

I=(n_i2v_subject+n_i2v_background)/2

Q=(n_subject+n_background+n_smooth+n_aesthetic+n_imaging+0.5*n_dynamic)/5.5

Total*=100*(I+Q)/2

删除dynamic后重新归一化Quality组：

Q_no_dynamic=(n_subject+n_background+n_smooth+n_aesthetic+n_imaging)/5

**Total†_no_dynamic=100*(I+Q_no_dynamic)/2**

这同时去camera和dynamic，明确称自定义七维诊断。不能把原Total减一个dynamic数、把分母仍留5.5，或直接七维平均冒同公式。原Total*、dynamic、GT、Physics/WMB仍必须同时报告，不改变预注册选择/晋级。

dynamic在原Total*有效权重1/22≈4.54545%。固定DreamX参照的−16.012439pp带来约−0.727838pp贡献，占原Total*−0.927647pp差约78.46%。这只是线性构成分析，不是“所有SOTA差距79%来自运动”，也不是模型提升。

### 7.2 完整355结果与36官方行同新公式复算

|模型/协议|八维Total*|I2V*|Quality去dynamic|Total†去dynamic|dynamic|
|---|---:|---:|---:|---:|---:|
|base raw|89.860075|98.113513|84.157545|91.135529|56.097561|
|base po|89.791814|98.427250|84.597219|91.512235|46.747967|
|α0.5 raw|89.892408|98.127599|84.050580|91.089090|57.723577|
|FT2 raw|89.819070|98.131449|83.966302|91.048876|56.910569|
|FT2 po|89.781232|98.481201|84.636544|91.558873|45.528455|
|DreamX官方|90.787722|98.336916|84.351380|91.344148|72.11|
|Vivy-R-Vision-3B官方|90.536783|98.496491|84.194782|91.345636|66.40|
|ToMoviee-2.0官方|90.128386|99.199926|84.455530|91.827728|47.07|
|LTX-2无po官方|88.920763|99.196836|85.655158|**92.425997**|**8.54**|
|Animate-Anything官方|88.510752|98.312426|86.311984|92.312205|**2.68**|
|STIV Apple官方|88.790723|97.596781|86.455131|92.025956|15.28|

必须对36行全部计算而不是只挑DreamX。去dynamic最高是LTX-2无po，不是DreamX；其dynamic只有8.54%。删除运动项会改变排序、偏爱近静态输出，所以不能只用这个分数衡量world model。

base po去dynamic比DreamX该行高+0.168087pp，但仍低于新指标最高约0.913762pp，且输入/规格/seed未对齐，dynamic下降。**这不等于超过SOTA。**

### 7.3 C276同子集去dynamic（不与355互排）

|同子集|C|匹配base|Δpp|
|---|---:|---:|---:|
|VBench raw71去dynamic|90.870981|90.983231|**−0.112250**|
|VBench po71去dynamic|91.485271|91.482766|+0.002505|
|WM22同一数学诊断|88.385492|88.303884|+0.081608|

C raw原Total*+0.170679pp，但去dynamic后反而下降：运动量收益不能掩盖非运动综合退化。此新增诊断只有aggregate复算，没有重新做bootstrap，**没有新增CI/显著性**。各维适用病例不同，不能给所有71病例伪造七维per-case Total。

### 7.4 最小可复算代码（CPU，不改原指标）

从项目根目录运行（避免环境中其他scripts包遮蔽本地namespace）：

~~~python
from pathlib import Path
import sys
sys.path.insert(0, str(Path("scripts").resolve()))
from summarize_eval import NORM, I2V, QUAL, total

def without_dynamic(scores):
    dims = I2V + [d for d in QUAL if d != 'dynamic_degree']
    n = {d: (scores[d] - NORM[d][0]) / (NORM[d][1] - NORM[d][0])
         for d in dims}
    i2v = sum(n[d] for d in I2V) / 2
    quality = sum(n[d] for d in QUAL if d != 'dynamic_degree') / 5
    return 100 * (i2v + quality) / 2

# 输入各维0–1，不是百分数；仍同时报告：
# total(scores)[0]*100 和 scores['dynamic_degree']*100
~~~

## 8. 下一步怎样真正证明超过SOTA？

1. 一次已批v3同16例恢复、实际成本/完整记录/独立语义审阅；不合格即停全量，不用CPU/结构通过当gold。
2. 已批准2GPUh单卡真实固定synthetic更新/新进程恢复及完整费用闭合；6GPUh真实multinode NCCL/各rank状态恢复。技术成功不代替质量。
3. 新训练manifest/cache和父来源留出隔离、独立语义approval；单一Teacher候选/参数/全局batch/更新数/来源冻结。正式训练预算另批，不借历史生成余额。
4. 实际训练后，固定94开发双seed+严格base配对/事前护栏/独立review；不加第三seed救结果、不用公共bench回扫选参。
5. 合格候选完整VBench355 raw/po、WM108、Physics198、WMB350，全视频/全帧/评分覆盖/finite/原公式重算/费用/失败独立验收。
6. 外部对手另冻结精确checkpoint、I2V/V2V、raw/po、分辨率/帧数/fps/时长、seed/video数、sampling/BoN预算、judge/评分代码。获取公开权重或原输出后同输入本地重评；不对齐则继续标“参照”，不假定真正SOTA比较已完成。
7. 同时报原Total*、去dynamic、全部维度/GT/物理/常识。如要官方排名，需要按其camera和5视频/pair协议另完整验收，不是只有八维本地分。
8. 实际可比质量证据和用户review门通过后，才称达到/超过对应协议SOTA；AR/Student/蒸馏仍关闭。

## 9. 统计、溯源和发布边界

- 本次旧完整eval/summary.json不包含后续E2/C276；不能单独当最新总表。E2从原scores和Physics final字段重算。
- 双seed同病例成组重采样，不把seed-case当独立；94开发集6个未解析父来源另给88敏感性，不事后删例救分。
- 官方公布值四舍五入/异规格；极小差不代表显著性。自定义去dynamic未新增CI。
- 历史base runtime/模型元组执行溯源不完备的条目降为描述性；当前sourceSHA不能补造旧执行证据。
- 费用未知记null/保守上界，失败和占卡单列；占卡不算有效实验利用率。
- 原媒体、逐case caption/问答、权重、token/cookie、原业务日志不公开。

主要索引：docs30/32（早期训练/插值）；34/37/39（诊断）；43/46（E1/E2，Physics勘误见本文）；49（E3双seed）；52（SOTA协议）；55/56/57（ALG/Dy机制/吞吐）；58（C276）；31顶部（最新准入）。历史“运行中”不当今天活任务/重试许可。

**本地文档写出不是GitHub发布。** 精确白名单→干净树范围检查→完整公开回归→正常commit/push→远端字节核验仍必需。本文没有新的训练收益或SOTA声明。
