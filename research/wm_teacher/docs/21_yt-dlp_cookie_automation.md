# yt-dlp Cookie 自动化方案总结

> 最后更新：2026-07-19
> 目的：评估各种 YouTube cookie 自动刷新方案并记录可用的手册流程

---

## 一、当前工作方式

### 已部署配置

```bash
yt-dlp --cookies cookies_netscape.txt \
  --js-runtimes deno --remote-components ejs:github \
  --download-archive downloaded.txt \
  --format "18/best[height<=720]" \
  --sleep-interval 2 --max-sleep-interval 8 \
  -a youtube_urls_full.txt
```

关键组件：
- **Netscape 格式 cookies**：`cookies_netscape.txt`（根目录）
- **Deno JS 运行时**：解决 yt-dlp JS 签名脚本执行（`--js-runtimes deno --remote-components ejs:github`）
- **HTTP 代理**：`star-proxy.oa.com:3128`（服务器环境需要）
- **续传机制**：`--download-archive downloaded.txt` 跳过已下载视频

### 当前 Cookie 续传流程（手册）

1. 用户在本地浏览器（Chrome/Edge）登录 YouTube 账号
2. 使用扩展（如 "Get cookies.txt LOCALLY"）导出 YouTube cookies
   - 导出格式：Netscape 格式 或 JSON 格式
3. 如果导出 JSON 格式，在服务器上转换：
   ```bash
   python3 convert_cookies.py < www.youtube.com_YYYY-MM-DD.json > cookies_netscape.txt
   ```
4. 将 `cookies_netscape.txt` 上传/复制到项目根目录
5. 验证 cookie 有效性：
   ```bash
   bash scripts/check_cookies.sh
   ```
6. 重启 yt-dlp 进程：
   ```bash
   pkill yt-dlp
   nohup yt-dlp --cookies cookies_netscape.txt ... &
   ```

---

## 二、自动刷新方案评估

### 方案 1：cookies-from-browser（❌ 不可行）

yt-dlp 内置 `--cookies-from-browser` 参数可从本地浏览器读取 cookies。

```bash
yt-dlp --cookies-from-browser chrome ...
```

**问题**：
- 需要桌面浏览器（Chrome/Firefox）在同一台机器上运行
- 服务器环境无 GUI 浏览器
- 无法在无头环境中工作

### 方案 2：OAuth 认证（❌ 不可行）

使用 `--oauth-account` 通过浏览器 OAuth 流程获取 token。

```bash
yt-dlp --oauth-account default ...
```

**问题**：
- 首次设置需要交互式浏览器窗口
- Token 也会过期且刷新需要重新认证
- 与服务器环境不兼容

### 方案 3：Headless 浏览器自动化（⚠️ 不稳定）

使用 Puppeteer/Playwright/Selenium + xvfb 在服务器上模拟浏览器。

**问题**：
- YouTube 检测 headless 浏览器（bot detection）
- 需要维护浏览器实例，资源占用大
- 反爬策略会不断变化，维护成本高
- Google 账号可能被标记为可疑活动

### 方案 4：定期导出 Cookie 脚本（⚠️ 半自动）

在用户本地机器上定时导出 cookies 到共享存储。

**问题**：
- 依赖用户本地机器始终运行
- 网络中断会导致 cookies 不同步
- 跨机器操作复杂度高

### 方案 5：Session Token 检测 + 通知（✅ 推荐）

通过 `check_cookies.sh` 监控 cookie 状态，发现问题时主动通知用户。

```bash
# crontab 定时检查
0 */6 * * * cd /path/to/wm_dataset && bash scripts/check_cookies.sh
```

**优点**：无需额外工具，利用现有脚本
**缺点**：仍需人工介入更新 cookies

---

## 三、结论与建议

### 哪些方法有效

| 方法 | 状态 | 说明 |
|------|------|------|
| Netscape 格式 cookies + yt-dlp | ✅ 可用 | 当前生产方案 |
| Deno JS 运行时 | ✅ 可用 | 解决 JS 签名问题 |
| HTTP 代理 | ✅ 可用 | 服务器出口网络必要 |
| 定时检查脚本 | ✅ 可用 | `scripts/check_cookies.sh` |
| 手动导出 cookies | ✅ 可用 | 需要浏览器 + 扩展 |

### 哪些方法不可行

| 方法 | 状态 | 原因 |
|------|------|------|
| `--cookies-from-browser` | ❌ | 需要本地 GUI 浏览器 |
| OAuth 自动刷新 | ❌ | 首次需要交互，token 也会过期 |
| Headless 浏览器 | ❌ | Bot 检测风险高，维护成本大 |
| 全自动无人工方案 | ❌ | 当前技术条件下无可靠方案 |

### 推荐操作流程

1. **自动化部分**（已实现）：
   - 使用 `check_cookies.sh` + crontab 定期检查 cookie 有效性
   - 检测到失效时发送通知（邮件/企业微信/Slack）
   - yt-dlp 使用 `--download-archive` 自动跳过已下载视频

2. **人工操作部分**（无法自动化）：
   - 收到通知后从浏览器导出新 cookies
   - 上传/复制 `cookies_netscape.txt` 到项目根目录
   - 运行 `check_cookies.sh` 确认有效
   - 重启 yt-dlp 进程

3. **Cookie 失效时间线**：
   - 会话 cookies（SID/HSID）：需在浏览器活跃时保持
   - 认证 cookies（SAPISID/APISID）：通常 1-2 周有效
   - 安全 cookies（__Secure-*）：可能 30 天
   - **建议人工刷新频率**：每周至少一次，或收到告警时立即办理

---

## 四、已有工具清单

| 工具 | 路径 | 用途 |
|------|------|------|
| Cookie 检查 | `scripts/check_cookies.sh` | 验证 cookie 有效性和关键字段 |
| 下载续传 | `scripts/resume_yt_downloads.sh` | 使用 cookies 续传 YouTube 视频 |
| Cookie 转换 | 需要 `convert_cookies.py` | JSON → Netscape 格式转换 |

---

## 五、参考资料

- yt-dlp 官方文档：https://github.com/yt-dlp/yt-dlp#cookies
- Deno 安装：https://deno.land/
- Cookie 导出扩展（Chrome）："Get cookies.txt LOCALLY"
