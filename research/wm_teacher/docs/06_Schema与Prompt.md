# 06_Schema与Prompt

> 统一交互样本 Schema 与 Caption/Prompt 规则

> 原始日志、规则派生和VLM标注必须分层保存；VLM不得覆盖真实状态。

| 整合原则                             | 要求                                                           |
|----------------------------------|--------------------------------------------------------------|
| 身份与角色                            | 每个人使用track_id；跨clip身份使用identity_group_id；speaker/listener/ot |
| 动作                               | raw_action保存设备输入；semantic_action保存归一化语义。                     |
| 结果与状态                            | expected_outcome来自指令/规划；state_after必须来自日志、传感器或可见证据。          |
| 切分                               | 保留source_id/episode_id；作品、身份、场景、任务、物体实例与采集批次不得跨split泄漏。      |
| 记忆                               | memory_entities[]保存持久实体、最后观测时间、belief state与置信度。             |
| 统一交互样本 Schema 与 Caption 规则       |                                                              |
| 原始日志、规则派生和VLM标注分层保存；VLM不得覆盖真实状态。 |                                                              |
| 字段                               | 类型                                                           |
| sample_id                        | string                                                       |
| video_path                       | string                                                       |
| audio_path                       | string|null                                                  |
| reference_image                  | string|null                                                  |
| history_video                    | string|null                                                  |
| source_dataset                   | string                                                       |
| source_sequence_id               | string                                                       |
| license                          | object                                                       |
| scene_type                       | string                                                       |
| viewpoint                        | enum                                                         |
| actors                           | array<object>                                                |
| entities                         | array<object>                                                |
| relations                        | array<object>                                                |
| interaction_type                 | enum                                                         |
| action                           | object                                                       |
| action_start_time                | float                                                        |
| action_end_time                  | float                                                        |
| action_stream                    | array                                                        |
| state_before                     | object                                                       |
| expected_outcome                 | object                                                       |
| state_after                      | object                                                       |
| state_delta                      | array<object>                                                |
| interrupt_event                  | object|null                                                  |
| dialogue                         | array<object>                                                |
| audio_events                     | array<object>                                                |
| camera_motion                    | object                                                       |
| caption_short                    | string                                                       |
| caption_detailed                 | string                                                       |
| uncertainties                    | array                                                        |
| quality_scores                   | object                                                       |
| split_key                        | object                                                       |
| 模板                               | 输入要求                                                         |
| 通用                               | 视频+可选音频+动作日志+参考帧                                             |
| 数字人/多人对话                         | 8–20s连续对话；最好有diarization                                     |
| 游戏/第一人称                          | 视频+逐帧键鼠/手柄+可选state                                           |
| 人-物物理交互                          | 动作前后各1–2s；手与目标物可见                                            |