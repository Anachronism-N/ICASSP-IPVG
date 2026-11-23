# Project Aria 下载日志

> 本文由 `scripts/download_project_aria.sh` 自动更新。完整日志保存在 `dataset/download_logs/project_aria_pilot_20260719_072838.log`。

## 本次执行

- 完成时间：`2026-07-19 07:34:32 +0800`
- 模式：`pilot`
- 队列：`dataset/download_queues/project_aria_previews_pilot.jsonl`
- 输出目录：`dataset/project_aria_downloads`
- 退出码：`10`
- 并发/重试/超时：`4 / 3 / 60s`

## 完整性审计

```json
{
  "files": {
    "partial": 1,
    "missing": 24
  },
  "expected_bytes": {
    "partial": 103410118,
    "missing": 6247263160
  }
}
```

## 日志末尾

```text
[2026-07-19 07:28:39 +0800] Project Aria download started
mode=pilot queue=dataset/download_queues/project_aria_previews_pilot.jsonl output_root=dataset/project_aria_downloads
workers=4 retries=3 timeout=60s
proxy_host=http://star-proxy.oa.com:3128
Checking proxy DNS (a failure is logged, then the real URL request is still attempted)...
PRIVATE_NODE_ADDRESS    star-proxy.oa.com
PRIVATE_NODE_ADDRESS    star-proxy.oa.com
PRIVATE_NODE_ADDRESS    star-proxy.oa.com
PRIVATE_NODE_ADDRESS    star-proxy.oa.com
PRIVATE_NODE_ADDRESS   star-proxy.oa.com
PRIVATE_NODE_ADDRESS   star-proxy.oa.com
Gate 1/2: downloading one item and validating size/SHA1...
[1/1] failed ADT/Apartment_release_clean_seq131_M1292/ADT_Apartment_release_clean_seq131_M1292_preview_rgb.mp4
Pilot gate failed; the remaining queue was not started.
```
