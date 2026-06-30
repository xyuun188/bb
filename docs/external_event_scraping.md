# Scrapling 外部事件采集

`Scrapling` 已按项目内可选子模块集成，不需要单独部署新项目。

## 设计边界

- 默认关闭：未启用或未安装 `Scrapling` 时，交易、训练、看板继续按原链路运行。
- 后台采集：由 `DataService` 生命周期启动 `ExternalEventService`，异步抓取并写入 `news_articles`。
- 不进热路径：交易决策的行情读取、模型调用、OKX 下单不会等待外部网页抓取。
- 数据复用：抓取结果落到现有 `NewsArticle`，本地训练脚本会自动把它转成 `text_sentiment_samples`。

## 安全约束

- 只抓取配置里的 HTTPS 公网页面。
- 拒绝 `localhost`、内网 IP、`.local/.internal/.lan` 等非公网目标。
- 只提取同域链接，不跨域爬取。
- 每轮限制源数量、每源条数、请求超时和轮询间隔。
- 抓取失败只记录日志，不影响交易主循环。

## 启用方式

安装可选依赖：

```bash
pip install -r requirements-scraping.txt
```

配置示例：

```env
EXTERNAL_EVENT_SCRAPER_ENABLED=true
EXTERNAL_EVENT_SCRAPER_INTERVAL_SECONDS=900
EXTERNAL_EVENT_SCRAPER_TIMEOUT_SECONDS=6
EXTERNAL_EVENT_SCRAPER_MAX_SOURCES=31
EXTERNAL_EVENT_SCRAPER_MAX_ITEMS_PER_SOURCE=8
EXTERNAL_EVENT_SCRAPER_SOURCES=[
  {"name":"binance_announcements","url":"https://www.binance.com/en/support/announcement/c-48","weight":0.88},
  {"name":"ethereum_blog","url":"https://blog.ethereum.org/","symbols":["ETH"],"weight":0.72}
]
```

如果不配置 `EXTERNAL_EVENT_SCRAPER_SOURCES`，启用后会使用内置的保守默认源。

## 对训练的作用

- 增加官方公告、项目博客、交易所动态等文本事件样本。
- 改善当前新闻/外部事件来源偏少的问题；社媒平台多样性由 `social_posts` 采集器单独治理。
- 不直接改变开仓策略；收益效果需要进入训练样本后，通过本地模型和复盘链路逐步体现。

## 2026-06-29 数据源边界

- Scrapling 是外部事件/新闻采集器，覆盖官方公告、项目博客、交易所上币、稳定币、安全和监管事件，入库为 `news_articles.source = scrapling:*`。
- 社媒平台多样性单独由 Reddit RSS、Hacker News public search 等 `social_posts` 采集器解决，不能把 Scrapling 页面当作社媒平台计数。
- 外部事件和社媒样本不会直接改变实盘开仓规则，必须经过干净训练样本准入和 Profit-First 治理校验后才影响模型/策略权重。
