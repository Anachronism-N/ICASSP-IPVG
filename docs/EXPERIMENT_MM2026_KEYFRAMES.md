# MM2026相关共同评价与事件约束关键帧选择

## 2026-10-20预注册与复现边界

用户要求以MM2026评价维度推进效果。本轮将获奖论文实际使用的VBench Motion Smoothness和Imaging Quality接入同一离线面板；旧ArcFace/CLIP仍单列且保留精确协议，不与作者未知checkpoint/采样横比。嘴动只诊断，不作此次主晋级依据。当前并没有完整MM2026官方scorer或人工研究，也没有KeyID基线或其算法提升证据。

最新作者仓库查明：KeyID `d4d68cc7`仅README、无许可证/可运行pipeline；AESR `a2b3d010`有编排及评价脚本但无软件许可，不复制代码。AESR evaluation/README公开候选权重为`.15*(GME+Story)+.20*(Cur+Arc)+.15*(Motion+Imaging)`，是作者自动选片分数，**不是含人工的官方FinalScore**。缺Cur/GME/Story不补0、不重新归一化、不编造总分。Story实际使用Gemini视频描述和KaLM embedding，当前不将研究人脸素材发外部API。GME7B/Cur checkpoint本地未核到；face launcher要求ConsisID根`cal_face_sim.py`而该公开仓库只有`eval/get_facesim_fid.py`，不是直接可运行契约。

ConsisID公开face实现16帧、最大脸、pad重检、负cos裁0、无有效帧0，与我们的单脸8帧/负cos保留/missing另报coverage不同。CLIP公开实现使用16帧、截断和logit_scale，还有BGR/RGB差异；KeyID论文未锁这些细节，不把简单乘100叫相同CLIPScore。新面板为`vbench_mm26_quality_v1`，不冒称官方2026测试成绩。

## 可快速跟进的方案

Stand-In有Apache2代码/公开adapter和VACE接入，但适配Wan14B，当前本地只有5B/1.3B，不能挂到现5B。TPIGE有真正Apache2推理/GE实现和历史评价，但其原14B+其他组件也需要单独固定资源。AnyID已经可运行，却不是已核实的MM2026获奖方案。Chronos作者完整公开代码/权重尚未核实。强任务级baseline（Stand-In14B、或Qwen全图条件关键帧编辑+VACE传播）后续独立引入，不将换大模型本身计算法创新。

## 此次独立优化组件

借鉴KeyID的稀疏关键帧重建结构，只优化anchor selector，不实现身份FaceSwap/IPKE、GPT提示生成、Prompt Relay或LTX2.3。使用许可兼容现成AMT-S替代重建器，称**KeyID-inspired事件约束选锚模块消融**，不能称KeyID复现/SOTA或动作补全。

固定33帧/8fps、K=5，uniform为[0,8,16,24,32]，本身已经含两事件分界16。event_error同样固定0/16/32，每个事件再选一个内部帧：原输入降采样32²RGB，最小化该事件两段相对于锚点线性混合的时长加权MAE，确定性最早索引破同分。不用identity、color detector、MUSIQ或AMT评价结果选择，也不按生成输出重试。

两arm都用原AMT在每个非锚帧的真实比例`t=(i-left)/(right-left)`直接插值，理论与实际计数28calls，output仍33帧320×1928fps。不缩放时间，不改变draft、prompt或seed。libx264rgb/crf0编码并完整解码核验所有anchor像素原样留存；此内部无损H.264 RGB实验格式不是比赛推荐正式输出格式。

## 数据与主评价

- 四个已有VACE-flatten非人草稿（红球/蓝块×seed42/43），可能没有完成两个动作；重建不能补不存在的动作。
- 四个自绘相同主体非线性双事件运动视频用于机制正例，不代表生成质量。
- 每域4inputs×2arm=8重建视频，共16。两域分开报告，不把正例混入生成结论。
- 24个已有Track1视频：4旧dev演员×neutral/happy×seed42，VACE-H1、AnyID-single、AnyID-dual三方法同新quality面板重评。非MM2026官方test，不与获奖论文数字直接横比；所有都是历史输出、不新增这些方法的trial。
- quality总48：24人脸视频＋8非人source＋16重建。全帧AMT原MotionSmoothness=1-MAE/255；MUSIQ-SPAQ原ImagingQuality=全帧平均MOS/100、longer-side缩放<=512，统一每video尺度。
- 对每arm重算原稿fullframe和**共同非锚帧**MAE、运动保真、原固定颜色事件coverage/方向、两事件成功率；source只是输入重建目标，不叫真实动作GT。MM2026没有公开的时间动作总分，此颜色proxy只受控域局部事件保留，不称通用动作评价。

## 决策与预算

MM2026相关VideoQuality为共同主面板，但AMT生成和MotionSmoothness用同模型，存在循环偏好；静态也可能高分，不能单凭平滑分晋级。只有generated-draft上共同非锚MAE改善、事件通过数不下降、位移保留不恶化且图像质量不明显下降，才能保留为重建组件；正例有效但草稿无收益，只结案机制验证，停止扩大重建。不声称身份/动作整体方案已优化。

离线导入前设置VBENCH_CACHE_DIR，显式本地模型path/文件SHA，拦截socket与子进程下载，不调用init_submodules自动wget。独立环境VBench/PyIQA锁见`resources/mm2026_eval.lock.json`，不改其他项目。CPU单测/准备3600s内，无限retry禁止。

最坏GPU上限2.5小时：联合pilot（重建2video+quality1video）1800秒/.5；剩14重建1800秒/.5；48quality两节点各24、各1800秒/.5共1；唯一工程修复1800秒/.5。联合pilot采用父阶段总超时，实际成本独立记账；全部已启动任务跟进明确退出，历史失败不覆盖、新run_id。排除异常卡、运行前核验资源、不DDP。

公共交付只含薄实现、测试、版本锁、方案与聚合正负结果；身份media/embedding/逐身份记录/内部地址/凭据不Git。最终结果与真GPU验收证据完成后追加，不用之前22bdb81成果满足新目标。
