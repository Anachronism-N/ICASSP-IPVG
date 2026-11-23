# HF资产备份上传器（先准备，授权后执行）

程序：`scripts/upload_asset_backup.py`。**本次仅准备代码、文档及离线测试，没有启动真实资产上传。** 计划只代表显式选中的资产，不代表整个项目清点完成。

## 功能与限制

- `plan`、`pack`默认完全离线；`upload`不加`--execute`也只报告计划，不连接HF。
- 每个来源用独立group标识。真实执行必须提供覆盖所有group的授权证据，绑定**计划文件SHA256、目标repo_id和可见性**；取得许可后填写，不能把示例当作真实授权。
- 扫描显式目录或逐行相对路径清单，支持大写.MP4。缺失文件/符号链接会报错，不静默跳过；重复组选择同一文件会拒绝。
- 支持tar.gz（gzip level 3）或不压缩tar；默认每片4 GiB（目标大小，含近似tar开销），最大可设置40 GiB。
- 大于分片大小的原始文件按字节范围分段，**不重新裁切视频、不改源文件**。每段索引记录原路径、offset、原文件大小、SHA256；恢复时按offset拼接。
- 一次提交归档和成员索引，SDK传输线程参数默认6，避免每视频一个commit触发限流；分片按序处理（不是同时上传6个分片），不同时打包整个TB级数据集。
- 已上传分片重新核验远端大小/SHA，已有相同对象不重传；同名不同内容绝不覆盖。缓存缺失/损坏、源文件改变、已验证的远端对象消失均停止。
- 不自动创建仓库、改变可见性、购买容量、清理源文件或删除已有远端数据。
- 凭据只从HF_TOKEN、已有Hub登录缓存或600权限的token文件读取，不内置账号/令牌，不将认证或授权文档上传。
- 排除.git、.ssh、.aws、.env、cookie/token/私钥文件；读取内容时检查token、私钥、Bearer和签名URL。检测到风险会**拒绝上传**，不自动改原件。
- 输入为已有tar/zip/gzip时先检查内部内容与目录安全。嵌套压缩包、链接/设备、路径穿越、重复成员或损坏包会拒绝；需先在安全位置解包并单独选择可备份内容。**因此旧环境/迁移包不保证可原样上传。**
- HTTP403/429/5xx或校验错误均停止并返回退出码2，保留断点与证据，**没有盲目重试**。403需解决容量/权限，429按服务端冷却时间后人工恢复。
- 远端LFS SHA与本地完整归档SHA匹配，并实际下载成员索引对照；默认不再次下载整个大归档。不能把指针、进程结束或局部成功当作全部数据传输完成。

## 1. 配置资产范围（现在可以准备）

把配置、计划、授权文件放在备份源目录之外。工作目录也必须在源目录之外，并有足够空间保留分片；程序**不自动删除本地缓存**。MP4已有压缩，tar.gz未必明显缩小，优先按真实字节量准备容量。

配置示例（路径自行替换）：


```json
{
  "root": "/srv/wm_teacher",
  "repo_id": "YOUR_ACCOUNT/YOUR_EXISTING_DATASET",
  "visibility": "private",
  "prefix": "asset-backup/approved-v1",
  "compression": "gzip",
  "shard_bytes": 4294967296,
  "groups": [
    {
      "id": "legacy_5s",
      "paths_files": [
        "/dockerdata/wm_upload_backup_20261116/legacy_upload_state/remaining_after_overflow.txt"
      ]
    },
    {
      "id": "openvidgen_5s",
      "paths": [
        "dataset/wan22_5s/openvidgen_5s"
      ]
    },
    {
      "id": "eval_remaining",
      "paths_files": [
        "eval_remaining.txt"
      ]
    }
  ]
}
```

路径清单每行一个相对root的文件路径。dirs使用paths，已有剩余清单使用paths_files。exclude_paths_files可提供已经确认远端备份的路径，避免重复。排除会写入计划的excluded，不会假称被排除的文件也已上传。

现有旧5秒窗口清单：`/dockerdata/wm_upload_backup_20261116/legacy_upload_state/remaining_after_overflow.txt`（每行相对wm_dataset根）。OpenVidgen窗口在`dataset/wan22_5s/openvidgen_5s/`。混合评测**最新差集**在`/dockerdata/wm_upload_backup_20261116/REMAINING_ASSETS_COVERAGE_BOUNDARY_20261120.json`的held_for_required_authorization.mixed_eval_remaining.members中，可离线导出其path为逐行清单；不要沿用早期清单重新传已完成文件。

旧迁移包位于项目根的同级目录，单独创建另一个root/plan/group；不要将不同根的绝对文件路径塞进同一相对路径清单。对包含凭据、第三方受限媒体或嵌套包的迁移资产，需要先安全审查/解包，程序不会绕过这些条件。

执行**离线计划**：


```bash
cd /srv/wm_teacher
/dockerdata/venv_qwen32b_20261103/bin/python3.12 scripts/upload_asset_backup.py plan --config /dockerdata/approved_backup/config.json --plan /dockerdata/approved_backup/plan.json
sha256sum /dockerdata/approved_backup/plan.json
```

规划时只统计大小与文件签名（inode、大小、mtime_ns），避免对所有TB级视频先做完整哈希。后续打包时逐个计算内容SHA并检查源是否变化。规划不检查许可原文，也不自动声明你已取得授权；执行前由你提供审查过的授权证据。

可选离线检查/打包首片：


```bash
/dockerdata/venv_qwen32b_20261103/bin/python3.12 scripts/upload_asset_backup.py pack --plan /dockerdata/approved_backup/plan.json --work-dir /dockerdata/approved_backup/staging --limit 1
```

如果打包中断，.partial及其他残片保留；不要直接删或覆盖，检查后可用新的work-dir重新打包。不同计划必须用不同work-dir。完成分片的完整SHA在.packed.json中，恢复运行前会重新校验。

## 2. 取得许可后准备授权文件

示例使用**approved:false**，不能执行真实上传。确认真实权利与范围后设置true，将文件路径和SHA256替换为提供的实际授权/适用许可文件；授权文件路径相对于其所在目录。


```json
{
  "approved": false,
  "plan_sha256": "REPLACE_WITH_PLAN_FILE_SHA256",
  "repo_id": "YOUR_ACCOUNT/YOUR_EXISTING_DATASET",
  "visibility": "private",
  "evidence": [
    {
      "group": "legacy_5s",
      "document": "evidence/legacy_5s_permission.pdf",
      "sha256": "REPLACE_WITH_ACTUAL_DOCUMENT_SHA256"
    },
    {
      "group": "openvidgen_5s",
      "document": "evidence/openvidgen_5s_permission.pdf",
      "sha256": "REPLACE_WITH_ACTUAL_DOCUMENT_SHA256"
    },
    {
      "group": "eval_remaining",
      "document": "evidence/eval_remaining_permission.pdf",
      "sha256": "REPLACE_WITH_ACTUAL_DOCUMENT_SHA256"
    }
  ]
}
```

目标HF dataset仓库须提前存在，可见性必须与配置一致；程序不负责判断一份文字声明是否真的取得了第三方授权。公开或私有备份均需符合适用来源条款。授权文件、原始凭据和私密审阅记录不作为上传payload。

## 3. 授权后运行（当前不执行）

依赖：Python 3.11+，huggingface_hub。现有环境：`/dockerdata/venv_qwen32b_20261103/bin/python3.12`，无需为了准备程序安装/降级依赖。

先不带execute查看：


```bash
/dockerdata/venv_qwen32b_20261103/bin/python3.12 scripts/upload_asset_backup.py upload --plan /dockerdata/approved_backup/plan.json --work-dir /dockerdata/approved_backup/staging
```

取得有效授权并配置已有HF认证后：


```bash
/dockerdata/venv_qwen32b_20261103/bin/python3.12 scripts/upload_asset_backup.py upload --plan /dockerdata/approved_backup/plan.json --work-dir /dockerdata/approved_backup/staging --authorization /dockerdata/approved_backup/authorization.json --token-file /root/secure_hf_token --workers 6 --execute
```

--token-file仅传文件位置，不传令牌值。也可以不加此参数，使用HF_TOKEN或现有登录缓存。未来恢复使用**同一个plan、authorization和work-dir**；每次真实运行会重新检查授权文档、缓存和远端文件，跳过已验证内容。授权或计划发生变化时重新生成版本化计划，使用新工作目录和不同远端prefix，不覆盖旧备份。

仅查看进度：


```bash
/dockerdata/venv_qwen32b_20261103/bin/python3.12 scripts/upload_asset_backup.py status --work-dir /dockerdata/approved_backup/staging
```

文件说明：
- upload.lock：进程锁，阻止同work-dir重复启动；进程退出锁自动释放，不需要删除锁文件。
- status.json：当前阶段packing/uploading/shard_verified/blocked/scope_complete。
- state.jsonl：只有归档+索引全部验收通过后才持久化记录固定HF revision与SHA。
- result.json：只有所选全部分片已验收时scope_complete:true；whole_project_complete始终false，**全项目是否完整需另做总清单验收**。
- error.json：最近错误，保留历史；status命令同时显示当前状态和最后历史结果/错误，不根据历史result单独判断当前成功。
- *.index.json：每个成员/字节段的恢复映射，上传时不包含本机root、inode或授权文档内容。

## 4. 恢复

按state.jsonl中的固定revision下载分片及索引，核对完整SHA256；拒绝绝对/越界/链接/重复成员，只提取索引列出的普通文件到新的空目录。每成员核对SHA与大小。分段文件按offset顺序拼接，确认所有段连续、无重叠且总大小等于source_bytes。程序不会覆盖原文件，恢复必须自行选择新的空目录。

## 5. 本次验证

离线回归测试：


```bash
CUDA_VISIBLE_DEVICES= /usr/bin/python3 -m unittest discover -s tests -p test_upload_asset_backup.py -v
```

测试使用本地临时小文件和模拟Hub，覆盖授权缺失、源改变、分段重组、秘密检测、压缩包目录安全、重复启动锁、远端冲突、断点续传及429错误退出，不会提交任何真实HF对象。
