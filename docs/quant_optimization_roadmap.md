# 苍鸮量化整体优化台账

更新时间：2026-06-19 19:45 CST
项目根目录：`F:\BB`

## 使用规则

1. 每次继续开发前，先读取本台账并对照 Hindsight 记忆，确认“已完成 / 进行中 / 未完成”。
2. 每次上线后，必须更新本台账：状态、验证结果、Git 提交号、线上复核结论。
3. 不把账号、密钥、SSH 密码、API Key 写入本台账；只记录配置契约和验证结论。
4. 任何“兜底”只能作为临时安全网，最终必须回到根因治理：数据质量、策略规则、执行规则、可观测性。
5. 策略优化不承诺一定盈利，但必须消除错误护栏、脏训练数据、误导状态、小仓无解释、亏损复开无证据等系统性问题。

## 总体目标

第一目标：恢复完整闭环：正常发现机会 → 正确评估收益 → 合理开仓 → 可解释持仓/平仓 → 复盘学习。

第二目标：消灭误导性状态，例如：

- “已进入执行队列”但其实没有提交订单。
- `release_pressure` 被历史亏损误触发。
- `expected_net_return_pct` 只显示一个负数但无法解释。
- 模型监控、分析记录、Agent/Skills 守门显示不一致。

第三目标：让训练数据、模型服务、策略调度、OKX 执行、UI 展示全部可追踪、可验证、可维护。

## 十阶段总规划

| 阶段 | 目标 | 当前状态 | 验收标准 |
| --- | --- | --- | --- |
| 1 | 冻结基线与指标 | 第二批完成并上线 | 能对比修复前后开仓率、拒单率、模型返回率、净收益结构 |
| 2 | 策略护栏重构 | 第一批完成 | 未满仓/低质量为 0 时，不再进入释放压力小仓模式 |
| 3 | 预期收益链路重做 | 第一批完成 | 每条不开仓记录能看到具体负收益来源 |
| 4 | 训练数据治理 | 第二批完成并上线 | 训练链路展示样本结构、动作分布、质量状态、数据新鲜度 |
| 5 | 模型服务稳定性 | 第一批完成，持续监控 | 服务器监控、分析记录、Agent/Skills 守门状态一致 |
| 6 | 开仓与仓位质量 | 第二批完成并上线 | 每个新仓解释为什么开、为什么该仓位、为什么该杠杆 |
| 7 | 平仓与持仓复盘 | 第二批完成并上线 | 平仓详情展示触发条件、持仓时长、手续费后净收益、模型意见、执行步骤 |
| 8 | OKX 执行规则 | 第二批完成并上线 | 区分系统主动拦截与 OKX 拒绝，记录 OKX 规则快照 |
| 9 | UI 与可观测性 | 第三批完成并上线 | 页面能直接看出为什么没开仓、哪一步失败、是否需人工处理 |
| 10 | 安全规范部署闭环 | 持续执行 | 测试、密钥扫描、线上同步、冒烟、Git、Hindsight 全部完成 |

## 第一批已完成

Git 提交：`4e4ddaf`  
分支：`codex/profit-attribution-state-machine`

已完成内容：

1. `release_pressure` 误触发治理：只允许当前真实持仓压力触发，历史亏损不再直接压死新仓。
2. `expected_net_return_pct` 可解释：拆成 AI、本地 ML、服务器盈利、时序、手续费、滑点等来源。
3. 18001 本地量化工具链路：探针鉴权、平台隧道契约、SSH keepalive 长连接、通道失败重连。
4. 分钟级行情沉淀：补齐 `1m/5m/15m/1h` K 线拉取入库，ticker 节流持久化。
5. 平台端口契约复核：平台内部 `18000=qwen3-14b-trade`、`18001=local AI tools`、`18002=deepseek-r1-14b-risk`。

验证结果：

- 本地全量测试：`1009 passed`。
- 密钥扫描：通过。
- 格式与静态检查：`black`、`ruff`、`node --check`、`git diff --check` 通过。
- 线上复核：`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均为 `active`。
- 线上模型链路：`qwen3-14b-trade`、本地量化工具、`deepseek-r1-14b-risk` 均可用。

## 第二批已完成

主题：训练数据污染治理 + 策略执行质量。  
状态：代码完成、本地验证通过、已同步线上、线上冒烟通过，Git 提交已完成；最终提交号以本次 Git 历史和 Hindsight 摘要为准。

### 2.1 建立第二批治理基线

状态：完成。

新增 `scripts/export_quant_optimization_baseline.py`，用于导出最近窗口内的决策、订单、训练样本质量和模型状态指标。

线上运行规则：平台服务器上必须按服务用户 `bb` 执行，并继承 `/etc/bb/bb-runtime.env`，否则会因为 PostgreSQL peer 认证误用 `root` 或 root→bb 被拒绝。

推荐线上命令：

```bash
cd /data/bb/app
set -a
. /etc/bb/bb-runtime.env
set +a
runuser -u bb --preserve-environment -- /data/bb/app/.venv/bin/python scripts/export_quant_optimization_baseline.py --output data/reports/second_batch_baseline_latest.json
```

最新线上复核：

- 服务：`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均为 `active`。
- Dashboard：`http://127.0.0.1:8002/` 返回 `302`，符合登录保护预期。
- 线上基线输出：`data/reports/second_batch_baseline_latest.json`。
- 质量统计：`total=8483`、`included=5742`、`downweighted=2735`、`excluded=6`。
- 最近决策采样：`1210`。
- 最近订单：`13`。

### 2.2 治理训练样本污染

状态：完成。

新增 `services/training_data_quality.py`，统一评估训练样本质量，原则是不直接硬删历史数据，而是隔离、降权、重打标签、版本化质量门。

治理内容：

- 降权普通 `hold`、极低信心、弱证据探针、小额噪声样本。
- 降权几分钟亏损快平、费用主导交易、宽点差、行情特征不足样本。
- 排除人工测试、异常收益、缺关键特征、明显错误状态样本。
- 输出 `quality_report`、`quality_score`、`sample_weight`、`label_version`、`data_quality_version`。

### 2.3 重训并接入质量门

状态：完成。

已接入入口：

- `scripts/train_local_ai_tools_models.py` 手动训练。
- `services/trading_service.py` 自动训练。
- `services/ml_signal_service.py` 本地 ML 训练。
- `services/local_ai_tools_client.py` 训练 payload。
- `scripts/deploy_local_ai_tools_service.py` 远端 18001 `/train` 服务端训练元数据。

验收结果：所有训练入口共用同一质量规则，训练 payload 和模型状态可返回质量统计，避免 hold 过多、探针交易和错误拒单继续污染模型。

### 2.4 优化仓位与复开纪律

状态：完成。

已完成内容：

- `services/entry_loss_cooldown.py` 防止同币种同方向亏损后短时间无强证据复开。
- 修复全局 profile 误伤相反方向的问题，避免“某币亏过一次后全方向都被压死”。
- 仓位解释链路记录小仓原因：账户余额、OKX 最小规则、弱证据探针、收益质量不足、风险压缩等。

### 2.5 修复快平仓规则

状态：完成。

已完成内容：

- `services/exit_fast_risk.py` 要求短时间亏损平仓必须具备强风控证据。
- 平仓判断返回 `fresh_exit_strong_evidence_required` 等字段，让详情页能说明是“新鲜强证据不足”还是“明确风控触发”。
- 避免几分钟浮亏被普通弱信号误杀。

### 2.6 补齐 OKX 规则可观测

状态：完成。

已完成内容：

- `executor/okx_executor.py` 在提交前记录 OKX 规则快照：最小张数、合约面值、最小名义价值、步进精度、杠杆、余额、计划数量、最终数量、保证金。
- 执行记录区分 `system_pre_submit_rejection` 与 `okx_rejection`。
- `web_dashboard/static/js/dashboard.js` 展示 OKX 规则快照，避免只看到“被 OKX 拦截”却不知道具体差哪一步。

### 2.7 第二批验证闭环

状态：完成。

本地验证：

- 乱码与 UTF-8 契约：`2 passed`。
- 全量测试：`1017 passed`。
- 聚焦 `black --check`：通过。
- 聚焦 `ruff check`：通过。
- `node --check web_dashboard/static/js/dashboard.js`：通过。
- 密钥泄露扫描：`source safety scan ok: scanned 454 files`。
- `git diff --check`：通过。

线上验证：

- 同步命令：`python scripts\sync_to_online_server.py --split-services`。
- 同步结果：服务重启成功，模型隧道 `model-tunnels-ok`。
- 线上服务：三个核心服务均为 `active`。
- 线上本地量化工具：`/health` 返回 `200`，`/models/status`、`/profit/predict`、`/timeseries/deep/predict`、`/sentiment/deep/analyze`、`/exit/advise` 调用正常。
- 线上基线：按服务用户 `bb` 导出成功。

## 第三批已完成

状态：代码完成、本地验证通过、已同步线上、线上冒烟通过，Git 提交 `635e9e0` 已推送。

已完成范围：

1. 候选策略实验室排版：策略卡片改为容器自适应网格，状态标签、按钮、chip、footer 和统计格允许换行，避免长文本和中文标签撑出卡片。
2. 系统自检页面：自检详情改为“总览 + 异常/需关注/提示/正常分组”，详情字段可滚动展示，安全修复说明单独高亮。
3. 执行记录详情：执行步骤头部改为自适应网格，原因和数据块支持长文本换行，避免 OKX 规则快照或错误详情挤乱弹窗。
4. 数据源自检增强：新增 `market_ticker_freshness`、`market_kline_coverage`、`news_source_freshness`、`social_source_freshness`，可直接看到行情、K 线、新闻、社媒训练数据是否新鲜和覆盖是否偏。
5. 契约测试增强：新增策略实验室防溢出契约、自检分组契约、数据源自检测试，防止后续 UI 和自检链路回退。

本地验证：

- 全量测试：`1019 passed`。
- 聚焦 `ruff check`：通过。
- `node --check web_dashboard/static/js/dashboard.js`：通过。
- 密钥泄露扫描：`source safety scan ok: scanned 454 files`。
- `git diff --check`：通过。

线上验证：

- 同步命令：`python scripts\sync_to_online_server.py --split-services`。
- 上传文件：`web_dashboard/api/system_health.py`、`web_dashboard/static/css/dashboard.css`、`web_dashboard/static/css/strategy_learning.css`、`web_dashboard/static/js/dashboard.js`。
- 服务状态：`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均为 `active`。
- Dashboard：`http://127.0.0.1:8002/` 返回 `302`，符合登录保护预期。
- 线上自检：`status=ok`，`critical=0`、`warning=0`、`ok=15`。
- 数据源复核：ticker `108` 个、最新约 `0.7` 分钟；K 线 `1m/5m/15m/1h` 均有沉淀；新闻源 `9` 类、样本 `8143` 条；社媒平台 `2` 类、样本 `2619` 条。

### 3.1 Scrapling 外部事件数据源增强与采集管理页

状态：本地代码完成，已新增 Dashboard 管理入口；默认仍关闭，待线上按需安装可选依赖并配置源后启用。

已完成内容：

1. 新增 `data_feed/external_event_scraper.py`：Scrapling 可选适配层，支持 HTTPS 白名单源、同域链接提取、超时、限量、去重、源码级安全降级。
2. 新增 `services/external_event_service.py`：后台采集入库服务，复用 `news_articles`，不进入交易热路径。
3. `DataService` 生命周期挂载后台服务，默认关闭，只有 `EXTERNAL_EVENT_SCRAPER_ENABLED=true` 时启动。
4. 新增 `requirements-scraping.txt` 与 `docs/external_event_scraping.md`，说明安装、配置、安全边界和训练作用。
5. 新增 `tests/test_external_event_scraper.py`，覆盖默认关闭、非公网/非 HTTPS 拦截、页面解析、入库去重。
6. 新增 `web_dashboard/api/data_collection.py` 与侧栏“数据采集”页面：可查看 Scrapling 依赖/运行状态、新闻/社媒/K 线/ticker 新鲜度、文本情绪样本质量、本地量化工具训练样本，并可保存 Scrapling 启停、间隔、超时、源数量和白名单源 JSON。
7. 数据采集设置保存仍走 Dashboard 写权限与 `.env` 配置管理；后端继续拒绝非 HTTPS、localhost、内网、带账号密码等高风险 URL，不暴露密钥。

本地验证：

- `pytest tests/test_external_event_scraper.py tests/test_data_collection_api.py tests/test_dashboard_main_ui_contract.py::test_data_collection_page_is_wired_to_api_and_safe_layout`：`8 passed`。
- `black --check web_dashboard/api/data_collection.py web_dashboard/api/router.py data_feed/external_event_scraper.py services/external_event_service.py tests/test_data_collection_api.py`：通过。
- `ruff check web_dashboard/api/data_collection.py web_dashboard/api/router.py data_feed/external_event_scraper.py services/external_event_service.py tests/test_data_collection_api.py`：通过。
- `node --check web_dashboard/static/js/dashboard.js`：通过。
- 密钥泄露扫描：`source safety scan ok: scanned 461 files`。

## 每次继续开发前检查

1. 读取本台账。
2. 回忆 Hindsight 中 `second-batch`、`training-data-quality`、`strategy-execution` 标签。
3. 检查 Git 工作区，避免把上次未提交改动误当新问题。
4. 对照当前阶段验收标准，只修当前阶段必要问题。
5. 如果发现跨阶段根因，先更新台账再改代码。
