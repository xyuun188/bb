# Quant Closed-Loop Eradication Master Plan

> **总控定位：** 这是根治方案主控文档，不是让 AI 自由发挥的开发清单。
> AI 执行任何 batch 前，必须先确认本批范围、禁止事项、验收标准和是否影响真实交易。
> 所有涉及真实开仓、仓位、平仓、模型权重、专家路由、风控阈值的改动，都必须先测试、后实现、再验证、再观察。

---

## 一、最终目标

- 建立一个以“手续费后真实赚钱能力”为核心的量化智能系统。
- 不再默认现有大模型、专家、本地模型、策略规则是正确的。
- 所有模型、专家、策略、训练数据、执行链路都必须被评估、对比、淘汰、替换和持续优化。
- 最终形成“数据可靠 -> 模型可信 -> 策略可解释 -> 执行可对账 -> 复盘能学习 -> 巡检能定位”的闭环。
- 解决这些反复问题：
  - 不开仓；
  - 全是小仓；
  - 快平仓；
  - 平仓亏多赚少；
  - 亏损后重复复开；
  - 影子错过机会没有真正反馈到策略；
  - 本地 ML 长期 `learning_only`；
  - 乱码反复出现；
  - OKX 数据不同步影响判断；
  - 系统巡检只发现表面问题；
  - 修改后这里好那里坏。

---

## 二、核心原则

- **赚钱优先**：最终看手续费后净收益、盈亏比、最大回撤、快亏平比例、小仓质量、错过机会减少。
- **证据优先**：模型或专家建议必须能通过历史回放、影子盘、模拟盘、实盘结果验证。
- **动态组合**：不同市场状态调用不同模型和专家，不固定死流程。
- **能进能退**：模型表现好提高权重，表现差降权，持续拖累就禁用。
- **不迷信大模型**：大模型负责理解、解释、事件推理和结构化输出；数值收益预测必须依赖结构化模型和真实交易数据。
- **不盲目微调**：数据没清理、标签不稳定前，不做大模型微调。
- **不简单放宽开仓**：任何开仓放大必须经过净收益、亏损概率、尾部风险、模型一致性、OKX 规则验证。
- **不让 AI 自行扩题**：AI 只能做当前 batch 明确要求的事，不得顺手改无关策略、阈值、UI 或历史数据。
- **不以测试通过代替真实有效**：本地测试通过只是最低门槛，涉及交易行为的改动必须有线上巡检和策略健康观察。

---

## 三、根治边界

必须一起处理：

- 文本乱码与历史污染；
- 训练数据质量；
- 本地 ML readiness；
- 大模型/专家/本地模型效果评估；
- 影子错过机会闭环；
- 开仓、仓位、平仓策略；
- OKX 数据对账；
- 系统巡检；
- Ruff/Black/静态质量债。

不算完成的情况：

- 只改页面文案；
- 只把 `learning_only` 硬改成 `ready`；
- 只放宽阈值让系统乱开仓；
- 只清源码乱码，不清数据库和模型写入口；
- 只本地测试，不部署线上；
- 没有 Git、Hindsight、线上巡检记录；
- 没有说明本批是否影响真实交易；
- 没有回滚点或失败后处理方案。

---

## 三点五、AI 执行防偏协议

每个 batch 开始前，AI 必须先确认：

- 本批只解决什么问题；
- 本批明确不解决什么问题；
- 本批会触碰哪些真实交易链路；
- 本批是否可能影响开仓、仓位、平仓、杠杆、模型权重、专家路由；
- 本批最小验收标准是什么。

AI 执行时禁止：

- 为了减少不开仓而直接降低开仓阈值；
- 为了让 ML 可用而硬改 `ready`；
- 为了利用 missed opportunity 而绕过风控；
- 为了清数据而删除或覆盖原始历史记录；
- 为了模型效果好看而使用未来函数或事后收益做决策前特征；
- 为了清 Ruff 而一次性全仓格式化；
- 为了页面好看而增加没有真实数据来源的假指标；
- 为了推进下一批而忽略测试失败、线上 `critical`、失败订单、弱证据执行或快亏平新增。

每批必须遵守：

- **Preflight**：检查 git 状态、相关测试、线上健康现状。
- **RED**：先写失败测试或 dry-run 审计，证明问题或契约缺失。
- **GREEN**：最小实现，只解决本批目标。
- **VERIFY**：本地测试、质量门、线上巡检、策略健康脚本。
- **REPORT**：汇报改了什么、没改什么、测试结果、线上结果、是否影响真实交易、回滚点、剩余风险。

---

## 四、Batch A：乱码与文本治理根治

目标：

- 建立统一文本治理模块。
- 所有模型输出、专家意见、执行原因、策略记录写入前做乱码检测。
- 区分终端/PowerShell 显示假乱码，以及源码/数据库/缓存真乱码污染。

治理范围：

- 分析记录；
- 执行详情；
- 策略学习事件；
- 专家记忆；
- JSON 缓存；
- 模型返回。

AI 防偏要求：

- 不得把不确定文本强行猜测修复。
- 不得批量覆盖数据库历史记录。
- 必须保留原始文本和修复报告。
- 必须明确写入前拦截点，而不是只做扫描脚本。

系统巡检新增“运行时文本完整性”：

- 最近新增乱码记录数；
- 来源表；
- 字段；
- 样例；
- 是否可自动修复。

验收：

- 新增记录不能再写入典型 mojibake、U+FFFD replacement character 或异常问号占位这类乱码；
- 源码、接口、数据库、缓存都纳入检测；
- 不再只靠前端替换显示。

---

## 五、Batch B：训练数据治理

目标：

- 清理和隔离会污染模型判断的训练数据。
- 训练系统必须知道哪些数据能吃、哪些只能观察、哪些必须隔离。

清理范围：

- 乱码样本；
- 重复样本；
- OKX 对账缺失；
- 价格异常；
- 手续费缺失；
- 错误平仓状态；
- 极小探针污染；
- JSON 解析失败；
- 错误方向标签；
- 未来函数泄露；
- stale market data；
- shadow/sim/live 模式混淆。

样本分层：

- 正常样本；
- 脏样本；
- 隔离样本；
- 只观察样本；
- 可训练样本；
- 可修复样本；
- 归档审计样本。

标签依据：

- 手续费后净收益；
- 最大浮盈；
- 最大浮亏；
- 持仓时长；
- 是否快亏平；
- 是否低质量仓；
- 是否错过机会；
- 是否来自 shadow/sim/live。

AI 防偏要求：

- 不得删除原始数据。
- 不得把可疑样本直接标成 clean。
- 不得让极小探针、弱证据小仓、快亏平异常污染正常盈利模型。
- 不得用平仓结果、未来收益、事后最大浮盈作为决策前特征。
- 训练读取必须通过干净视图或治理过滤接口。

验收：

- 数据采集/训练面板显示样本总数、可训练数、隔离数、动作分布、币种覆盖、时间跨度、数据新鲜度；
- 本地 ML 训练不再吃明显脏数据；
- 每次训练能解释用了多少样本、排除了多少、为什么排除。

---

## 六、Batch C：ML readiness 状态机

目标：

- 本地 ML 不能长期只显示 `learning_only` 但不给原因。
- ML 是否能参与仓位放大，必须由状态机和指标决定。

状态：

- `learning_only`；
- `shadow_ready`；
- `ready`；
- `degraded`；
- `disabled`。

`learning_only` 必须展示阻塞原因：

- 样本不足；
- 脏样本比例高；
- AUC/PR-AUC 不达标；
- 高分组收益不优于低分组；
- 最近训练数据过旧；
- 影子验证没通过；
- 模型返回不稳定。

必须展示指标：

- 样本数；
- 隔离样本数；
- 脏样本比例；
- AUC；
- PR-AUC；
- 高分组收益；
- 低分组收益；
- 最近训练时间；
- 训练数据版本；
- 下一次训练条件；
- 是否允许参与真实仓位放大。

AI 防偏要求：

- 不得硬改 `ready`。
- 不得为了消除告警而隐藏 blocking reasons。
- 不得让未达标 ML 参与真实仓位放大。
- 指标阈值必须来自配置或测试固定，不能执行中临时调到刚好通过。
- 从 `ready` 退回 `degraded/learning_only` 必须有明确原因。

验收：

- 不能硬改 `ready`；
- 页面能说明“差哪些指标”；
- 策略知道 ML 是否允许参与仓位放大；
- 线上表现变差时能自动降级。

---

## 七、Batch C2：模型/专家全面体检

目标：

- 列出现有所有大模型、专家、本地模型、策略组件。
- 不再默认现有组合最优。

统计最近 24/72 小时：

- 参与次数；
- 方向建议；
- 是否采纳；
- 采纳后收益；
- 错误建议率；
- 平均耗时；
- JSON 错误率；
- 未返回率；
- 对快平仓、小仓、错过机会的影响。

处理状态：

- `keep`：保留；
- `reduce`：降权；
- `shadow_only`：只做影子观察；
- `disable`：禁用；
- `replace`：替换；
- `add_candidate`：新增候选。

AI 防偏要求：

- 本批只体检和标记，不直接大幅调整真实权重。
- 不得因为单次亏损就禁用模型/专家。
- 不得因为单次盈利就提高真实权重。
- 没有足够样本时只能标记为观察中。

验收：

- 系统能回答哪个模型/专家有贡献，哪个拖后腿；
- 每个模型/专家都有调用、收益、耗时、稳定性证据；
- 不再默认现有组合最优。

---

## 八、Batch C3：模型/专家竞赛框架

目标：

- 建立模型/专家和基线策略的持续竞赛机制。
- 决策权重必须来自对比结果，而不是人工感觉。

三层竞赛：

- **离线回放竞赛**：用历史数据比较净收益、回撤、错过机会、快亏平。
- **影子盘竞赛**：模型/专家只给建议，不真实执行，统计如果采纳会怎样。
- **模拟盘 A/B 竞赛**：通过小权重模拟执行，对比手续费后结果。

竞赛指标：

- 手续费后净收益；
- Profit factor；
- 平均盈利；
- 平均亏损；
- 盈亏比；
- 最大回撤；
- 快亏平比例；
- 小仓占比；
- 错过机会收益；
- JSON 错误率；
- 未返回率；
- 平均耗时；
- 资源占用。

处理规则：

- 连续贡献为正：提高权重；
- 只降低风险但不提升收益：保留为风控复核；
- 贡献持续为负：降权；
- 长期未返回：暂停；
- 高耗时低收益：后台复盘；
- 某行情状态表现好：只在该行情启用。

AI 防偏要求：

- 新模型不能直接上线。
- 没有 baseline 对比不得调整真实权重。
- 竞赛统计必须区分 shadow、sim、live。
- 不得只用“建议看起来合理”判断贡献。

验收：

- 每个模型/专家都有基线对比；
- 新模型不能直接上线；
- 低贡献模型不会拖慢主链路；
- 系统能解释权重为什么升、降或暂停。

---

## 九、Batch C4：数字货币特征补强

目标：

- 补齐数字货币交易特征，让系统更懂 crypto 交易结构，而不是只会解释。

特征范围：

- `1m/5m/15m/1h` K线；
- ticker；
- 盘口深度；
- 滑点；
- 资金费率；
- 未平仓量；
- 清算/爆仓风险；
- BTC/ETH 对小币种牵引；
- 板块联动；
- 山寨币高波动风险；
- 新闻公告；
- 社媒情绪；
- 事件日历。

外部事件数据必须带：

- 来源；
- 可信度；
- 时间衰减；
- 影响币种；
- 事件类型。

AI 防偏要求：

- 缺失数据源不得静默当作正常。
- 不得把缺失特征填成有利于开仓的默认值。
- 不得让低可信事件直接驱动真实开仓。
- 特征必须记录时间戳，避免 stale data 污染判断。

验收：

- 数据采集管理页能看到每类数据源状态；
- 缺失数据源不能悄悄影响模型；
- 每条决策能展示关键特征贡献。

---

## 十、Batch C5：模型组合动态路由

目标：

- 不再固定所有专家每轮都跑。
- 根据市场状态、候选质量、风险、readiness 和历史贡献动态选择模型/专家。

动态路由依据：

- 当前市场状态；
- 当前候选质量；
- 持仓风险；
- 模型 readiness；
- 专家历史贡献；
- 当前延迟；
- 是否需要风险复核；
- 是否需要事件分析；
- 是否是高质量候选。

示例：

- 普通低质量候选：少量快速模型 + 规则判断；
- 高质量开仓候选：盈利模型 + 风险模型 + 执行专家 + 必要大模型；
- 高风险行情：风险专家和 DeepSeek 风控优先；
- 新闻事件驱动：事件专家和大模型解释优先。

AI 防偏要求：

- 没有 C2/C3 贡献统计前，不得直接替换主链路。
- 初始动态路由必须先 shadow 或 canary。
- 不得为了降延迟而跳过必要风控。
- 不得让低贡献模型继续拖慢高频主链路。

验收：

- 不再所有专家重复调用；
- 高耗时低贡献专家不会拖慢系统；
- 模型/专家调用原因可解释；
- 路由变化不增加弱证据执行和快亏平。

---

## 十一、Batch D：影子错过机会闭环

目标：

- 影子复盘不能只记录“错过了”，要进入策略学习。

进入策略必须满足：

- 同币种同方向多次错过；
- 错过后收益稳定为正；
- 风险证据低；
- 模型方向一致；
- 市场结构相似；
- 不是单次偶然波动。

满足条件时：

- 提升候选排序；
- 提升 expected net 解释权重；
- 允许受控 probe；
- 进入模型训练正样本。

不满足条件时：

- 只记录为观察；
- 不允许绕过风控开仓。

AI 防偏要求：

- 不得用全局 missed count 推动任意币种开仓。
- 不得把 missed opportunity 当成强制开仓理由。
- 不得绕过净收益、亏损概率、尾部风险、模型一致性和 OKX 规则。
- 受控 probe 也必须有明确仓位上限和退出规则。

验收：

- 错过机会能被策略利用；
- 弱证据执行仍必须为 0；
- 页面能显示为什么错过机会被采用或没采用；
- 策略健康脚本能统计采用数、probe 数、阻塞原因和示例。

---

## 十二、Batch E：开仓、仓位、平仓策略重构

目标：

- 让每个订单都能解释为什么开、为什么这个仓位、为什么平。
- 解决不开仓、小仓、快平仓、亏损复开、平仓亏多赚少。

开仓依据：

- 动态证据；
- 预期净收益；
- 盈利质量；
- 亏损概率；
- 尾部风险；
- 模型一致性；
- 专家历史贡献；
- 同币种历史表现；
- 影子错过机会；
- OKX 下单规则。

仓位 sizing：

- 强证据、正期望、风险低：允许合理放大；
- 弱证据：只允许影子或受控小探针；
- 低收益质量：不抬高仓位；
- 余额/OKX 限制：明确展示。

平仓：

- 几分钟亏损平仓必须有强证据；
- 不能因为一点浮亏就乱平；
- 同币种同方向亏损后，短时间复开必须有新强证据。

AI 防偏要求：

- 不得直接放宽开仓阈值来制造交易量。
- 不得绕过硬风控 veto。
- 不得把小仓问题简单修成大仓问题。
- 不得为了减少快平仓而忽略真实风险恶化。
- 不得让亏损复开缺少新证据。

验收：

- 每个订单能解释为什么开、为什么这个仓位、为什么平；
- 小仓必须有原因；
- 快平仓必须有明确风控证据；
- 失败订单、弱证据执行、快亏平新增必须为 0 或有明确历史遗留说明。

---

## 十三、Batch F：系统巡检升级

目标：

- 巡检能定位问题链路，而不是只给表面 warning。

新增巡检节点：

- 文本完整性；
- 数据同步；
- K线覆盖；
- OKX 对账；
- 模型服务；
- ML readiness；
- 模型/专家效能；
- 影子错过机会利用率；
- 小仓原因分布；
- 快平仓原因分布；
- 失败订单；
- Ruff/security 质量门。

巡检页面区分：

- 已修复；
- 未修复；
- 观察中；
- 历史遗留。

AI 防偏要求：

- 不得做没有数据来源的展示型假指标。
- 每个巡检节点必须有 `status/state/evidence/next_actions/owner_path`。
- 没有验证证据不能标记为已修复。
- UI 不能只展示原始 JSON 让用户自己判断。

验收：

- 后续不是靠截图发现问题；
- 系统巡检先告诉问题在哪条链路；
- 每个问题能看到当前状态、最近证据和下一步动作。

---

## 十四、Batch G：Ruff/格式/质量债治理

目标：

- 把质量债纳入门禁，但不制造巨大无关 diff。

先固定 bug/security 门禁：

- `F`；
- `B`；
- `E9`；
- `S608`；
- `E722`。

再分目录处理：

- `scripts/`；
- `services/`；
- `web_dashboard/api/`；
- `tests/`。

每批处理后：

- 跑相关测试；
- 跑 Black；
- 跑 Ruff；
- 部署验证。

AI 防偏要求：

- 不得一次性全项目格式化。
- 不得修改无关业务逻辑。
- 不得为了 Ruff 通过删除必要逻辑。
- touched files 必须局部格式/静态检查，历史债按目录分批清。

验收：

- 质量门逐批收紧；
- 不留下“以后再说”的静态问题；
- diff 可审查、可回滚。

---

## 十五、Batch H：线上部署与持续评估

目标：

- 不再改完就结束，每批都有线上验证和记录。

每批固定流程：

- 本地测试；
- 密钥扫描；
- Ruff/Black；
- 同步线上；
- systemd 服务状态检查；
- Dashboard 访问检查；
- 系统巡检；
- 策略健康脚本；
- Git 提交推送；
- Hindsight 同步；
- 文档更新状态。

观察节奏：

- 2 小时：看系统是否正常跑、有没有失败订单、是否还快平仓；
- 24 小时：看策略是否改善、小仓原因、错过机会是否下降；
- 72 小时或至少 20 笔已平仓订单：看盈利闭环是否改善。

AI 防偏要求：

- 部署前必须记录当前 commit 和服务状态。
- 涉及真实交易行为的 batch 必须有回滚点。
- 线上出现 `critical`、失败订单、弱证据执行、快亏平新增时，停止下一批。
- Hindsight 记录以项目 watcher/git hook 为主；重大里程碑、架构决策、watcher 异常或用户要求时手动同步。

验收：

- 每批都有本地验证、线上验证和文档记录；
- 线上服务 active；
- Dashboard 可访问；
- 策略健康脚本指标可解释；
- Git 和 Hindsight 有记录。

---

## 十六、最终验收标准

- 模型知道自己什么时候不能用。
- 专家知道自己是否赚钱。
- 策略知道为什么开仓/不开仓。
- 仓位知道为什么大/小。
- 平仓知道为什么现在平。
- 训练知道哪些数据不能吃。
- 巡检知道问题在哪条链路。
- Git 和 Hindsight 记录每批完成情况。
- 新模型/专家不能直接上线，必须先证明比基线更好。
- 如果模型/专家拖累净收益或增加快亏平，必须自动降权或暂停。
- 系统最终以手续费后真实赚钱能力为准，而不是以模型解释好听、页面指标好看、交易次数变多为准。

---

## 十七、当前计划执行顺序

- `Batch A`：乱码与文本治理根治。
- `Batch B`：训练数据治理。
- `Batch C`：ML readiness 状态机。
- `Batch C2`：模型/专家全面体检。
- `Batch C3`：模型/专家竞赛框架。
- `Batch C4`：数字货币特征补强。
- `Batch C5`：模型组合动态路由。
- `Batch D`：影子错过机会闭环。
- `Batch E`：开仓、仓位、平仓策略重构。
- `Batch F`：系统巡检升级。
- `Batch G`：Ruff/格式/质量债治理。
- `Batch H`：线上部署与持续评估。
- `二期优化标的`：在一期交易契约、巡检、防偏与线上观察基础上，继续治理 OKX/本地口径、强机会识别、候选集中、旧仓释放、ML/server_profit 拖累和收益闭环。

---

## 十七点五、二期优化标的：真实口径、强机会、容量轮动与收益闭环

本节是一期总控之后的二期优化目标，不等同于当前 batch 已完成项。任何 AI 执行二期前，必须先确认一期交易执行契约、OKX 同步、系统巡检和策略健康观察没有新增 `critical`、失败订单、弱证据执行、快亏无强退出或绕过风控行为。

二期总目标：

- 先保证 OKX 与本地订单、成交、持仓、盈亏口径真实一致，再谈策略放大。
- 把 USAR 这类真实强信号抽象成可审计的强机会识别器，但不得复制其历史口径异常。
- 解决 market-entry 最终候选长期集中、同方向偏空、旧盈利小仓占容量导致不开仓的问题。
- 让旧仓释放从“只处理低质量亏损/低效探针”升级为“容量轮动与方向拥挤治理”。
- 治理 local ML readiness、server_profit 与影子错过机会之间的证据链，最终以手续费后真实已平仓 PnL 证明有效。

阶段 1：OKX/本地交易口径统一。

目标：

- 统一 OKX 原始订单、成交回报、交易所持仓、本地 `Order`、本地 `Position`、执行记录和历史收益样本的字段口径。
- 统一交易对名称映射，例如 `USAR-USDT-SWAP`、`USAR/USDT`、`USAR/USDT:USDT`，禁止 HUSDT/WLFIUSD 这类错配继续污染历史和收益模型。
- 统一合约张数、币数量、合约面值、价格小数位、名义金额、杠杆、手续费、实现盈亏。
- 修复类似 USAR 的口径异常：OKX 原始订单是 10 张、23.21，本地不得再变成 100 数量、2.31 或错误拆分持仓。
- 对历史异常样本进行可审计标记、修正或隔离；未确认真实口径前，不得把异常收益喂给 server_profit、影子复盘或强机会学习。

验收：

- 任意一笔 OKX 订单，本地显示的交易对、方向、数量、均价、名义额、杠杆、手续费、实现盈亏能和 OKX 后台或 OKX API 对齐。
- 执行记录不再重复展示同一真实成交；订单来源能区分系统决策、OKX 成交回报、交易所手动/外部同步。
- 交易对 alias 和合约面值异常有只读巡检节点与修复脚本 dry-run 报告。
- USAR 类样本能被分析为强信号，但其数量/价格/拆分异常不会继续作为收益放大证据。

AI 防偏要求：

- 看到一单收益很高时，必须先核对 OKX 原始成交、合约面值、张数、币数量、价格小数位和本地拆分持仓；不得直接把本地异常收益当成可复制能力。
- 看到 OKX 与后台不一致时，优先查同步和归一化链路，不得先改策略阈值、仓位或杠杆。
- 任何历史修正必须先备份、dry-run、输出影响行数和回滚方式；不得静默覆盖历史。

阶段 2：强机会识别器。

目标：

- 把 USAR 的真实强信号条件抽象为 `strong_opportunity` 级别，而不是简单放宽普通开仓阈值。
- 强机会必须同时满足多源同向、费后综合预期净收益为正、高盈利质量、亏损概率可控、尾部风险可控、行情快照一致、OKX 可交易、交易口径无异常。
- 强机会可以提高开仓优先级和允许中等仓位；普通机会仍保持小仓、探针或观望。
- 强机会必须单独统计胜率、手续费后收益、最大亏损、持仓时长、平仓原因和回撤。

验收：

- 每笔强机会都能解释：哪些来源同向、expected net 如何组成、profit quality 为什么达标、loss probability 和 tail risk 为什么可控。
- 强机会数量不会泛滥；不能把 `shadow_memory` 或 AI 单源乐观预期当成强机会。
- 强机会开仓后有单独闭环报表，能和普通 probe/exploration 单分开评估。

AI 防偏要求：

- 不得把 `expected_net > 0` 单独当作强机会。
- 不得让影子错过机会绕过 selected-side expected net、entry evidence、ML readiness、仓位、杠杆、OKX 和风控契约。
- 强机会只允许在交易口径真实一致之后启用；若 OKX/本地口径仍异常，强机会只能 shadow 观察。

阶段 3：候选集中与不开仓治理。

目标：

- 继续保留全市场扫描、特征拉取和 ranker 的广度，不通过盲目扩大预算或降低 evidence 制造成交。
- 固化 market-entry 集中度监控：`market_unique_symbol_count`、`market_entry_unique_symbol_count`、`market_entry_top3_share`、`market_entry_skip_kind_counts`、`market_entry_tier_counts`。
- 当市场覆盖正常但 entry 只有少数交易对时，必须定位瓶颈发生在 feature/ranker/AI 吞吐/evidence/expected net/server_profit/ML readiness/组合敞口哪一层。
- 对 CRWV、FIL、YGG、LAB、ETH、RESOLV 等影子复盘反复错过的交易对建立错过机会复核队列，但必须再过实时行情一致性、收益质量和风控契约。

验收：

- 系统能解释“为什么不开仓”：没有候选、候选被过滤、证据不足、收益质量不足、server_profit 反向、ML degraded、组合拥挤、OKX 规则或容量不足。
- entry 候选不长期集中在 3-5 个 symbol；若集中，必须输出具体集中原因和未入选样本的后续收益回放。
- 错过机会进入复核闭环，而不是只记录日志。

AI 防偏要求：

- 看到“候选只有 5 个”时，必须先区分全市场扫描、feature fetch、rank selected、AI processed、market-entry 五层；不得把最终 entry 集中误判为全市场扫描失效。
- 不得通过降低 volume/notional、放宽 evidence、提高杠杆或扩大仓位解决候选集中。
- 如果被过滤或预算外 symbol 后续确实出现正费后收益，只能先进入离线复核、shadow 或 canary 设计。

阶段 4：旧仓释放与容量轮动。

目标：

- 在现有低质量亏损仓、低效探针释放之外，增加盈利旧仓容量轮动策略。
- 当同方向持仓过度集中、旧盈利小仓继续占用容量，且有更高质量新机会被 `crowded_side_cap` 拦住时，系统必须能识别应该优先锁盈/全平/部分平哪些旧仓。
- 释放优先级应考虑持仓年龄、名义额、浮盈、手续费倍数、继续持有证据、方向拥挤、是否阻挡强机会。
- 亏损仓不得粗暴全平；必须区分硬风险恶化、小浮亏观察、长时间低效亏损和结构反转。

验收：

- `crowded_side_cap` 触发时，报告能指出造成拥挤的仓位、候选被拦原因、建议释放对象和不释放理由。
- 老、小、低继续收益的盈利 short 仓不会长期无解释占用容量。
- 释放动作有 position review 决策、执行结果、手续费后收益和回滚证据；不能无声平仓。
- 容量释放后，新的高质量机会能获得评估和执行名额，而不是继续被普通 probe 小单占满。

AI 防偏要求：

- 不得为了多开仓而盲目平掉所有盈利仓；赢家仍需看继续持有证据和收益质量。
- 不得把小浮亏当作必须全平理由；硬风险、结构反转和资本效率要分层处理。
- 任何释放策略必须先有只读诊断、shadow/canary 或明确测试，不能用页面按钮或人工截图驱动平仓。

阶段 5：ML readiness、server_profit 与影子错过机会治理。

目标：

- 治理 long 侧 ML readiness 长期 degraded 的问题，检查样本标注、dirty sample、训练窗口、long/short 样本不平衡和 top return。
- 拆分 server_profit 的负向贡献来源：真实不支持、样本不足、历史口径污染、交易对错配、手续费/滑点过高或方向标签错误。
- 让影子 missed opportunity 从“只加分”变成可审计复核链路：错过原因、后续收益、风险质量、是否应进入强机会或 canary。
- 在阶段 1 数据口径修好前，不得盲目信任 server_profit 的负向或正向结论。

验收：

- local ML readiness 的阻塞项有明确改善路径和训练评估报告；不得硬改 ready。
- server_profit 对每个 entry 候选输出正向、反向、缺失或忽略的原因。
- missed opportunity 报表能区分方向错配、证据不足、收益质量不足、组合拥挤、OKX 不可交易、容量不足和行情快照异常。

AI 防偏要求：

- 不得为了让 long 多开而降低 PR-AUC、top return、样本质量或 readiness 门槛。
- 不得把影子复盘高 missed rate 直接等同于“应该当时开仓”；必须看手续费后收益、最大不利波动、可执行价格和风险质量。
- 若历史 OKX/本地口径污染未清理，server_profit 和影子复盘只能作为观察证据，不能作为实盘放大依据。

阶段 6：二期上线验证与闭环指标。

目标：

- 每个二期阶段都必须形成本地测试、线上只读验证、真实交易影响说明、回滚点和剩余风险。
- 二期完成标准以手续费后真实已平仓收益、回撤、快亏、复开纪律和执行契约为准。

验收：

- OKX/本地订单口径 mismatch 为 0。
- `trade_execution_contract.status=ok`，新增 contract violation、weak evidence executed、negative expected executed、fast_loss_without_strong_exit 均为 0。
- `market_unique_symbol_count` 正常，`market_entry_unique_symbol_count` 不长期过窄；若过窄，原因可解释。
- 旧仓释放有明确记录，容量和方向拥挤能被主动治理。
- 强机会有独立胜率、平均收益、最大亏损、持仓时长和手续费后收益统计。
- 平仓收益与 OKX 差异可解释，不能靠本地错误口径证明盈利。

二期推荐执行顺序：

1. 先做 OKX/本地交易口径统一。
2. 再做旧仓释放与容量轮动。
3. 再做强机会识别器。
4. 再做候选集中与错过机会复核。
5. 最后治理 ML readiness、server_profit 和影子错过机会的长期质量。

二期停止规则：

- OKX/本地口径未对齐时，不允许放大强机会仓位。
- 旧仓释放没有审计记录时，不允许以容量不足为由强开新仓。
- 发现新增弱证据执行、负期望执行、快亏无强退出、OKX 规则绕过、交易对错配或历史收益污染，必须停止二期放大逻辑并回到对应阶段。
- 没有手续费后已平仓样本证明前，不得宣称“已解决不赚钱”。

---

## 十七点六、二期未完成闭环台账与工程基线门禁

本节记录截至北京时间 `2026-06-26 07:35` 复核后的二期未完成闭环。后续 AI 继续执行总控时，必须先阅读本节，再决定是否改策略、部署或清理数据。

当前复核口径：

- 线上系统巡检为 `warning`，`critical=0`，问题台账 `fixed=10`、`unresolved=0`、`observing=10`。
- 当前没有 `unresolved`；`strategy_closed_loop` 仍是收益/ML 有效性观察项，不能当作盈利闭环完成。
- `okx_reconciliation=ok`、`okx_trade_fact_integrity=ok`、`position_price_integrity=ok`、`trade_execution_contract=ok`。
- 近窗口新增 contract violation、weak evidence executed、negative expected executed、fast_loss_without_strong_exit 均为 0。
- 当前这些结果只能说明“没有新增硬执行事故”，不能说明“不赚钱、不开仓、小单、历史脏数据、乱码遗留、ML/server_profit 和强机会实盘化已经根治”。

### 1. 不开仓、小单、不赚钱闭环

状态：未完成。

现象：

- 最新窗口没有新增执行事故，但也没有形成稳定开仓。
- 开仓候选存在，但多数被 `entry_evidence_wait`、收益质量不足、证据分不足、ML degraded、server_profit 反向拦住。
- `strategy_closed_loop` 仍是唯一未解决硬项。
- 近 120 分钟只有 1 个平仓样本，AAVE/USDT 亏损约 `-0.861742U`，不能证明策略有效。

下一步：

- 继续治理 `strategy_closed_loop`。
- 对 entry 候选按 expected net、profit quality、loss probability、tail risk、evidence tier、ML、server_profit 分层复盘。
- 建立“候选 -> 下单 -> 持仓 -> 平仓 -> 费后收益 -> 训练反馈”的真实闭环报表。

验收：

- 新增弱证据执行、负期望执行、快亏无强退出均为 0。
- 连续观察窗口内有足够已平仓样本，费后收益、回撤、胜率、快亏、复开纪律可解释。
- 不再只靠“有候选”“有诊断卡”“单笔大盈利”宣称完成。

禁止：

- 不得直接降低 evidence、profit quality、ML readiness 门槛来制造开仓。
- 不得硬改 ML ready。
- 不得用单笔 USAR 类大盈利样本推导整体放大仓位。

### 2. OKX、本地口径与历史脏数据闭环

状态：当前窗口 ok，历史清理未完全闭环。

已确认：

- 当前 `okx_trade_fact_integrity=ok`。
- 当前 `position_price_integrity=ok`。
- 当前 `trade_execution_contract=ok`。
- 当前 `okx_reconciliation=ok`，14 天候选平仓单 `247`、实际扫描 `247`、缺失闭仓 `0`、完整巡检耗时约 `4.45s`，不再复现 dry-run 超时。
- 近窗口没有交易对错配、弱证据执行、负期望执行、快亏无强退出。

未闭环原因：

- 历史执行记录、历史持仓、历史收益样本是否全部修正、隔离或标记，还不能说彻底结束。
- `okx_reconciliation` 的 dry-run `TimeoutError` 已收口；后续若再次超时，应先看 `candidate_close_order_count`、`scanned_close_order_count`、`duration_seconds` 和巡检调度分组。
- 历史脏样本如果继续进入 server_profit、影子复盘或训练，会继续污染策略。

下一步：

- 对历史订单、成交、持仓、收益样本做只读审计。
- 按订单 ID、OKX `ordId`、OKX `fillId` 优先关联，交易对 alias 只作为辅助。
- 对无法确认的历史样本标记为隔离或不参与训练/收益模型。
- 保持 OKX 历史对账 dry-run 为 ok；若再次超时，先查候选平仓单过滤、数据库慢查询和完整巡检并发调度，不把超时当正常完成。

验收：

- 历史异常样本有清单、备份、dry-run、影响行数和回滚方式。
- OKX 后台、OKX API、本地 `Order`、本地 `Position`、执行记录、历史收益口径一致。
- 不再出现 HUSDT/WLFIUSD、张数/币数量/价格小数位错配污染收益。

### 3. 乱码代码与乱码数据闭环

状态：当前巡检 ok，源码遗留未确认彻底清完。

已确认：

- `visible_text_encoding=ok`。
- `runtime_text_integrity=ok`。
- 当前运行时没有新增疑似乱码写入。

未闭环原因：

- 之前多次发现或提到源码里仍有历史乱码片段。
- 巡检 ok 只能说明当前扫描规则未发现裸乱码，不等于全仓历史乱码都已经人工确认清除。
- 乱码如果存在于业务文案、reason、错误分类，会继续影响诊断和 AI 判断。

下一步：

- 做源码级乱码专项扫描。
- 区分真实乱码、编码显示误判、历史测试样本三类。
- 只修真实业务文案和运行时写入路径，不盲目改测试样本或外部原始数据。
- 新增测试防止 reason、execution_source、系统巡检文案再次写坏。

验收：

- 源码、前端静态资源、运行时写入文本三层都有扫描结果。
- 真实业务乱码为 0。
- 不能再因为乱码上下文导致补丁失败或误判。

### 4. ML readiness、server_profit 与影子错过机会闭环

状态：诊断已上线，治理未完成。

当前证据：

- local ML 仍是 `degraded`。
- `allow_live_position_influence=false`。
- 阻塞项为 long/short top return 低于阈值。
- server_profit 在最新 entry 里仍多为反向或负贡献。
- shadow missed opportunity 仍只能作为观察或复核证据，不能直接开仓。

下一步：

- 查训练样本、标签、dirty sample、long/short 样本结构、top return。
- 拆解 server_profit 反向原因：口径污染、方向标签、手续费/滑点、样本不足、真实无效。
- 把 missed opportunity 做成复核队列，而不是直接变成开仓理由。

验收：

- ML readiness 阻塞项有训练报告和改善证据。
- server_profit 每个候选能解释正向、反向、缺失、忽略原因。
- missed opportunity 能区分错过原因和后续费后收益质量。

### 5. 候选集中与市场扫描闭环

状态：部分完成，未闭环。

当前证据：

- 不是全市场扫描坏了。30 分钟 market unique 约 25，120 分钟约 39。
- 但最终 entry 仍很窄，30 分钟只有 2 个 market entry，且都 blocked。
- ranker 会选出约 8 个，但进入可交易 entry 后被 evidence/收益质量挡住。

下一步：

- 固化五层诊断：全市场扫描、feature fetch、rank selected、AI processed、market entry。
- 对被过滤和预算外 symbol 做后续收益回放。
- 如果确有费后正收益机会，再进入 shadow/canary，不直接放宽过滤。

验收：

- 能解释“候选集中”到底卡在哪一层。
- entry 候选不长期只剩 3-5 个，若集中必须有原因和后续回放。
- 不用扩大预算、降低质量底线来制造成交。

### 6. 旧仓释放与容量轮动闭环

状态：只读审计完成，真实释放闭环未完成。

当前证据：

- 当前 open positions 约 4，不是满仓卡死。
- 旧仓释放不是当前不开仓主因。
- release decision -> close order -> filled order -> position closed 的完整执行链还没有实盘闭环验证。

下一步：

- 找出未闭环 release decision。
- 明确哪些仓位应该继续持有、部分锁盈、全平或观察。
- 释放动作必须有决策、执行、收益和回滚证据。

验收：

- crowded side 或容量压力出现时，能指出具体阻塞仓位和释放对象。
- 释放后新高质量机会能获得评估和执行名额。
- 不允许无声平仓或截图驱动全平。

### 7. 强机会识别器闭环

状态：只读 shadow 已完成，实盘化未完成。

未闭环原因：

- `strong_opportunity` 现在只是识别和解释，不驱动 live sizing。
- 没有独立胜率、费后收益、最大亏损、持仓时长、平仓原因统计。
- USAR 不能直接作为放大模板，因为它曾混有历史口径异常。

下一步：

- 先观察 strong candidate 和 near miss。
- 建立强机会独立报表。
- 满足样本、收益、回撤要求后，才能进入 canary。

验收：

- 强机会不是 expected net 单因子。
- 必须多源同向、收益质量、亏损概率、尾部风险、OKX 可交易、口径一致。
- canary 前不得提高真实仓位。

### 8. 本地工作区脏状态、Git 与部署一致性闭环

状态：未完成，必须前置处理。

当前问题：

- 本地工作区有大量改动和未跟踪文件。
- 这会导致后续 AI 不知道哪些已上线、哪些只是本地、哪些是临时文件。
- 也会影响回滚、部署、测试和总控执行判断。

下一步：

- 分组所有改动：已上线代码、未上线代码、测试、文档、临时文件。
- 对已上线文件做线上校验。
- 删除明确无用临时文件，不删除不明业务文件。
- 跑测试、格式和安全检查。
- 建立 Git checkpoint。
- 每阶段结束必须工作区干净，或者列明未提交原因和风险。

验收：

- 本地 Git 状态可解释。
- 线上代码、总控文档、Git commit 三者能对齐。
- 后续 AI 不会读错版本或重复改同一问题。

### 推荐继续执行顺序

1. 工程基线先行：清理工作区、确认线上/本地/Git 对齐。
2. 历史脏数据和 OKX 对账超时。
3. 乱码源码与运行时文案专项。
4. 策略闭环：不开仓、小单、不赚钱。
5. ML readiness、server_profit、shadow missed opportunity。
6. 候选集中和 filtered-out 回放。
7. 旧仓释放真实执行链。
8. 强机会从 shadow 到 canary。
9. 二期阶段 6：用费后真实已平仓样本验收。

### 二期继续执行硬约束

后续 AI 不得把以下内容当作完成：

- 诊断卡上线。
- 页面 warning 减少。
- 单笔订单赚钱。
- 当前窗口 OKX 口径 ok。
- 没有失败订单。
- 工作区里有代码但没提交。
- 本地测试通过但线上没有观察样本。

真正完成必须满足：真实执行链可解释、OKX 口径一致、历史污染隔离、无新增执行事故、费后平仓收益可验证、工作区和部署状态可回滚。

---

## 十八、停止规则

- 某批测试失败，不进入下一批。
- 线上出现 `critical`，先修硬故障。
- 新增失败订单、弱证据执行、快亏平，停止放大逻辑，回到对应批次定位。
- 连续 3 次局部修复仍不能解决同一问题，停止补丁，重审架构。
- 没有线上验证、Git、Hindsight、文档记录，不算完成。
- AI 无法说明本批是否影响真实交易时，不允许继续执行。
- AI 无法说明本批回滚点时，不允许部署。
- 当前 batch 未形成可验证结果时，不允许开启下一批。

---

## 十九、每批执行记录表

| Batch | 状态 | 范围确认 | 本地测试 | 线上验证 | Git Commit | 回滚点 | 剩余风险 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A 乱码与文本治理 | 本地完成，真实库 dry-run 通过 | 只改文本治理写入边界、dry-run 审计脚本和系统巡检节点；不影响开仓/仓位/平仓/杠杆/模型权重/专家路由 | `pytest` 35 passed；脚本 `--help` 通过；Ruff/Black 局部通过 | 临时启动 DB 隧道后 dry-run 扫描 822 条：疑似记录 0、疑似字段 0、未修改数据；隧道已关闭 | 本轮统一提交（见 Git 历史） | 回滚本批改动文件即可；无 DB 迁移、无历史覆盖 | 还未部署线上服务；后续部署后需在系统巡检页面复查 `runtime_text_integrity` 节点 |
| B 训练数据治理 | 本地完成，真实库 dry-run 通过 | 只改训练样本质量评估、shadow 隔离 dry-run、ML/Local AI 训练读入口与数据采集治理快照；不删除原始数据，不影响开仓/仓位/平仓/杠杆/模型权重 | `pytest` 42 passed；脚本 `--help` 通过；Ruff/Black 局部通过 | 临时 DB 隧道 dry-run 扫描 200 条 completed shadow：quarantined 0、top_reasons 空，未修改数据；隧道已关闭 | 本轮统一提交（见 Git 历史） | 回滚本批 touched files 即可；无 DB 迁移、无历史覆盖；如规则误伤可回退 `DATA_QUALITY_VERSION` v3 相关改动并重训 | 尚未部署线上服务；Batch C 需基于治理报告进入 ML readiness 状态机；真实交易行为改善仍需后续批次 |
| C ML readiness | 本地完成，只读状态检查通过 | 只改 ML readiness 状态机、训练元数据指标、预测/状态接口 gating 与 dashboard 展示；不硬改 `ready`，不改开仓/仓位/平仓/杠杆/专家路由/真实模型权重 | `pytest` 16 passed；新增预测层 gating 回归测试；Ruff/Black 局部通过 | 只读检查本地模型：`status=learning_only`、`allow_live_position_influence=false`，阻塞项为样本/测试数不足、PR-AUC 缺失、训练数据版本旧、模型过旧；未修改真实交易状态 | 本轮统一提交（见 Git 历史） | 回滚本批 touched files 即可；无 DB 迁移、无历史覆盖；如 readiness 误判可回退 `services/ml_readiness.py` 与 ML status/predict 集成 | 尚未部署线上服务；旧模型因 v1 数据质量与 PR-AUC 缺失被保守锁为 learning_only，需后续重训/线上观察后才可能进入 ready |
| C2 模型/专家体检 | 本地完成，真实库只读验证通过 | 只做模型/专家参与、采纳、收益、耗时、JSON 错误、未返回率体检与状态建议；不直接调整真实权重、不开启/禁用线上模型、不替换主链路 | `pytest` 29 passed；Ruff/Black 局部通过；新增只读完整报告接口 `/model-expert-health/status` 与系统巡检卡 | 临时 DB 隧道只读调用 C2 巡检卡成功：7 个组件、全部 `shadow_only` 复核建议、`live_weight_mutation=false`；未修改数据；隧道已关闭 | 本轮统一提交（见 Git 历史） | 回滚 `services/model_expert_health.py`、系统巡检接入与测试即可；无 DB 迁移、无权重/路由写入 | 尚未部署线上服务；当前真实库体检建议只是观察/复核，后续 C3 需建立基线竞赛后才能把贡献用于权重调整 |
| C3 模型/专家竞赛 | 本地完成，真实库只读验证通过 | 建立只读竞赛报告、baseline 对比、shadow/sim/live 分层状态与权重建议来源；不直接提高/降低真实权重，不让新模型上线，不替换主链路 | `pytest` 18 passed；Ruff/Black 局部通过；新增只读完整报告接口 `/model-expert-competition/status` 与系统巡检卡 | 临时 DB 隧道只读调用 C3 巡检卡成功：`baseline_missing`、shadow 样本 1192、`can_apply_live_weight=false`、未修改数据；隧道已关闭 | 本轮统一提交（见 Git 历史） | 回滚 `services/model_expert_competition.py`、系统巡检接入与测试即可；无 DB 迁移、无权重/路由写入 | 当前真实库缺少可用 baseline outcome，竞赛报告只能阻塞真实权重判断；后续需补足 executed outcome 或模拟 A/B 账本后才可作为 C5 路由依据 |
| C4 数字货币特征补强 | 本地完成，真实库只读验证通过 | 新增数字货币特征覆盖/新鲜度治理报告、数据采集页特征覆盖面板、系统巡检卡与只读接口；只做特征可用性审计，不改开仓/仓位/平仓/杠杆/模型权重/专家路由，不让缺失特征驱动实盘 | `pytest` 75 passed；Ruff/Black 局部通过；新增 `/crypto-feature-coverage/status` 与数据采集页 `feature_coverage` 只读返回 | 临时 DB 隧道只读调用 C4 报告和巡检卡成功：17 类特征、缺失 5、过期 0、已中性阻断 5、观测币种 56、`audit_only=true`、`live_signal_mutation=false`、`can_missing_features_drive_live_entry=false`、缺失策略 `neutral_blocked`；未修改数据；隧道已关闭 | 本轮统一提交（见 Git 历史） | 回滚 `services/crypto_feature_coverage.py`、系统巡检/数据采集接入、前端展示与测试即可；无 DB 迁移、无历史覆盖、无实盘参数改动 | 当前真实库仍缺少部分 C4 特征源，报告会阻断其真实开仓影响；后续如要让事件日历/清算/板块联动等进入可用状态，需先补真实采集来源和时间戳证据 |
| C5 动态路由 | 本地完成，真实库只读验证通过（尚未部署产生线上路由样本） | 新增模型/专家动态路由影子策略、决策 raw 追踪、只读路由报告、系统巡检卡与 `/model-dynamic-routing/status`；初始只生成 shadow/canary 计划，不改变真实专家调用、不跳过风控、不替换主链路 | `pytest` 43 passed（C5 路由、系统巡检、batch expert/错误安全/多样性回归）；Ruff/Black 局部通过 | 临时 DB 隧道只读调用 C5 报告和巡检卡成功：当前历史 route_plan 0（新代码未部署，尚无 `dynamic_model_routing` 样本）、`audit_only=true`、`live_route_mutation=false`、`can_apply_live_route=false`、巡检卡 warning；未修改数据；隧道已关闭 | 本轮统一提交（见 Git 历史） | 回滚 `services/model_dynamic_routing.py`、`ai_brain/ensemble_coordinator.py` 动态路由 raw 写入、系统巡检接入与测试即可；无 DB 迁移、无权重/路由写入 | 真实线上尚未产生动态路由样本，因此“减少重复专家调用”只在影子计划层可验证；必须部署后观察 route_plan、弱证据执行和快亏平指标，再决定是否进入 canary，不能直接 live 替换 |
| D 影子错过机会闭环 | 本地完成，真实库只读验证通过 | 新增 shadow missed opportunity 闭环服务、策略学习接入、系统巡检卡、只读接口与页面展示；只把错过机会作为同币种同方向重复证据后的学习/probe 候选，不强制开仓、不绕过风控、不改杠杆/仓位/模型权重/专家路由 | `pytest` 109 passed（D 服务、策略学习、系统巡检、Dashboard 契约）；`pytest` 66 passed（entry evidence/probe/opportunity/memory/data collection 回归）；Ruff/Black 局部通过 | 临时 DB 隧道只读调用 D 默认报告和巡检卡成功：默认窗口 `24h/200`、主键窗口查询、服务约 8.019s、巡检卡约 6.729s；completed 192、missed 80、adopted 1、probe 2、blocked 27、weak_evidence_executed 0；`audit_only=true`、`live_entry_mutation=false`、`can_bypass_risk_controls=false`、`weak_evidence_execution_allowed=false`、`global_missed_count_can_drive_entries=false`；隧道已关闭 | 本轮统一提交（见 Git 历史） | 回滚 `services/shadow_missed_opportunity_closed_loop.py`、`services/strategy_learning.py`、系统巡检/前端接入与测试即可；无 DB 迁移、无历史覆盖、无真实交易参数改动 | 当前 D 只完成“错过机会可被保守利用”的闭环，真实收益改善仍需部署后观察；巡检卡 warning 是因为存在被阻塞原因，不是弱证据执行；在线报告为性能安全使用 24h/200 近端窗口，不代表全历史统计 |
| E 开仓/仓位/平仓重构 | 本地完成，真实库只读发现历史遗留 critical，需部署后观察 | 新增交易执行契约审计、只读报告和系统巡检/API 接入；补成交确认写入最终 execution reason；审计只回查订单 decision_id 与快亏附近平仓决策，不放宽开仓、不绕过风控、不改杠杆/仓位/模型权重/专家路由、不写真实库 | `pytest tests/test_trade_execution_contract.py tests/test_system_audit_api.py -q` 29 passed；`pytest tests/test_trading_service_boundaries.py tests/test_entry_evidence_policy.py tests/test_entry_evidence_probe.py tests/test_entry_opportunity_scoring.py tests/test_position_quality.py tests/test_new_pair_loss_pause.py tests/test_trade_execution_contract.py -q` 185 passed；Ruff/Black 局部通过 | 临时 DB 隧道只读调用 E 报告和巡检卡成功：服务约 5.895s、巡检卡约 4.538s；`audit_only=true`、`live_entry_mutation=false`、`live_exit_mutation=false`、`can_bypass_risk_controls=false`；补充订单决策 28、补充 exit 决策 7；executed_entry 13、weak_evidence_executed 4、missing_entry_explanation 1、fast_loss 8、fast_loss_without_strong_exit 0、contract_violation 5；隧道已关闭 | 本轮统一提交（见 Git 历史） | 回滚 `services/trade_execution_contract.py`、`services/execution_service.py`、系统巡检接入与测试即可；无 DB 迁移、无历史覆盖、无真实交易参数放宽 | 历史真库仍有 4 条弱证据执行和 1 条缺开仓解释，不能标记线上全绿；当前本地 `EntryPolicy` 已阻断 `weak_conflict_probe/degraded_missing_probe`，成交确认后会写 reason，但需部署后观察新增样本必须为 0，未清零前不得做任何开仓放大 |
| F 系统巡检升级 | 本地完成，真实库只读验证通过但整体仍 warning | 新增巡检 `owner_path/state/state_label` 契约、issue ledger 状态分层、section timeout、失败 fallback 保留原始 section key、模型训练未配置观察态与可见文本乱码源头修复；不做展示型假指标，不改开仓/仓位/平仓/杠杆/模型权重/专家路由 | `pytest tests/test_system_audit_api.py -q` 23 passed；`pytest tests/test_system_audit_api.py tests/test_text_integrity_runtime.py tests/test_runtime_text_integrity_audit.py -q` 30 passed；Ruff/Black 局部通过 | 临时 DB 隧道只读调用完整巡检成功：总耗时约 33.424s；status=`warning`；cards 16、critical 0、warning 12、ok 4、findings 10、nodes 18；issue ledger fixed 4、unresolved 11、observing 1；`visible_text_encoding` ok、offender_count 0；无缺失 `owner_path/state/state_label`，无泛化 `audit_section_*` root cause；隧道已关闭 | 本轮统一提交（见 Git 历史） | 回滚 `web_dashboard/api/system_audit.py`、`services/text_integrity.py` 与对应测试即可；无 DB 迁移、无巡检历史写入、无真实交易参数改动 | 当前线上整体仍为 warning，说明还有 11 个未解决项和 1 个观察项；F 只提升定位和防偏能力，不代表策略收益已改善，也不能覆盖 E 的历史弱证据执行风险；后续批次必须继续读 issue ledger 而不是只看页面摘要 |
| G 质量债治理 | 本地完成，待 H 统一部署/线上观察 | 按总控先固化 bug/security 门，再分目录清理 Ruff/Black；只处理 Ruff/Black 报告中的文件，未全仓一键格式化，未修改开仓/仓位/平仓/杠杆/模型权重/专家路由 | `ruff check . --select F,B,E9,S608,E722` 0 issues；`ruff check .` 0 issues；`black --check scripts services web_dashboard/api tests ai_brain config db executor` 通过；`pytest tests/test_dashboard_auth_accounts.py tests/test_dashboard_security.py tests/test_secret_utils.py tests/test_model_runtime_policy.py tests/test_server_monitor_probe.py tests/test_batch_expert_json_stability.py -q` 76 passed；`pytest tests/test_trading_service_boundaries.py tests/test_market_auto_entry_processor.py tests/test_memory_feedback.py tests/test_order_position_reconciliation.py -q` 134 passed；`pytest tests/test_system_audit_api.py tests/test_text_integrity_runtime.py tests/test_runtime_text_integrity_audit.py -q` 30 passed；关键脚本/模块 `py_compile` 通过 | 本批为静态质量与格式治理，未连接真实库、未写线上数据；线上验证并入 H 部署后 systemd/Dashboard/系统巡检/策略健康观察 | 本轮统一提交（见 Git 历史） | 回滚本批 Ruff/Black touched files 即可；无 DB 迁移、无历史覆盖、无真实交易参数改动 | G 已清零当前 Ruff/Black 门，但它只降低质量债和后续误改风险，不代表策略收益改善；后续新增文件必须继续进质量门，H 部署前仍需密钥扫描和线上巡检 |
| H 线上部署与持续评估 | 已部署线上，进入 2h/24h/72h 持续观察 | 按 split-services 流程同步源码并重启 `bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service`；部署前记录本地 HEAD `375f410dd311f98d20871491dbb05c5430e8cc9e` 与分支 `codex/profit-attribution-state-machine`；本批不放宽开仓、不改杠杆、不绕过风控，所有交易行为改动仍受前序契约审计约束 | 部署前最终闸门：`pytest -q` 1296 passed；`ruff check .` 0 issues；`black --check scripts services web_dashboard/api tests ai_brain config db executor core` 通过；`security_secret_scan.py --fail-on high .` 扫描 514 files OK；`git diff --check` 通过；`node --check web_dashboard/static/js/dashboard.js` 通过 | `sync_to_online_server.py --split-services` 上传 60 个变更文件并重启成功：model tunnels ok，三项 systemd 均 active，Dashboard `302` 健康响应；线上模型服务 `qwen3-14b-trade`、`deepseek-r1-14b-risk`、`local-ai-tools` active，废弃服务 inactive；120 分钟策略健康：240 decisions 全 hold、entry/orders/failed_orders/fast_loss_close 均 0、open_positions 2；真实库只读文本审计扫描 200 条 0 疑似；真实库系统巡检 30.512s 返回 warning、critical 0、cards 16、warning 12、ok 4、owner/state 契约完整；交易执行契约仍为历史遗留 5 violations（4 weak evidence、1 missing explanation），无 fast_loss_without_strong_exit | 本轮统一提交（见 Git 历史） | 首要回滚点为部署前 commit `375f410dd311f98d20871491dbb05c5430e8cc9e`，可回滚线上 `/data/bb/app` 到该代码点并重启三项服务；本批无 DB 迁移、无历史覆盖、无真实交易参数放宽 | 线上仍为 warning，不是全绿：历史 E 契约 violation 未消失，策略近 120 分钟仍 0 开仓，错过机会样本仍多；不得据此做开仓放大，必须继续观察新增弱证据执行、失败订单、快亏平、route_plan 样本和收益闭环，2h/24h/72h 观察未达标时回到对应批次定位 |

---

## 二十、Batch H 补充记录：巡检口径修复与线上复查（2026-06-23）

触发原因：Batch H 上线观察期间，以真实服务用户视角运行系统巡检时，`trade_execution_contract` 曾因 24 小时历史旧样本返回 `critical`。进一步只读核查确认：历史窗口内仍有 5 个契约遗留 violation（4 条弱证据执行、1 条缺开仓解释），但上线后精确窗口没有新增弱证据执行、失败订单或快亏平。因此本次只修复只读巡检口径，不放宽开仓、不改杠杆、不绕过风控、不修改真实库历史数据。

本次代码修复：
- `services/trade_execution_contract.py`：`TradeExecutionContractService.report()` 增加 `since` 当前窗口过滤，`query_policy` 记录 `db_time_filter/since_utc`，用于区分历史 24 小时审计和服务重启后的当前运行窗口。
- `web_dashboard/api/system_audit.py`：`trade_execution_contract` 巡检同时保留历史报告和当前 runtime window 报告；只有当前窗口出现硬违规时才标记 `critical`，历史遗留但当前未复现时标记为 `warning/observing`。
- 新增回归测试覆盖：历史报告有弱证据执行、当前窗口干净时，系统巡检不能误报 current critical；`since` 过滤必须排除旧决策和旧订单。

本地验证：
- `pytest tests/test_trade_execution_contract.py tests/test_system_audit_api.py -q`：33 passed。
- `pytest -q`（显式临时 SQLite，避免触碰线上库）：1298 passed。
- `ruff check .`：0 issues。
- `black --check scripts services web_dashboard/api tests ai_brain config db executor core`：通过。
- `python scripts/security_secret_scan.py --fail-on high .`：扫描 514 files OK。
- `git diff --check`：通过。

线上部署与复查：
- `python scripts/sync_to_online_server.py --split-services` 上传 3 个变更文件并重启成功：`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard `302` 健康响应。
- 模型服务复查：`qwen3-14b-trade.service`、`deepseek-r1-14b-risk.service`、`local-ai-tools.service` active；废弃模型服务 inactive。
- 线上系统巡检复查：整体 `warning`，`critical_cards=[]`；`trade_execution_contract` 状态降为 warning，摘要为“24h historical trade execution contract violations remain; current runtime window has not reproduced them.”
- `trade_execution_contract` 当前窗口：runtime started_at `2026-06-22T22:56:20.865347+00:00`，`current_summary.contract_violation_count=0`、`weak_evidence_executed_count=0`、`executed_entry_count=0`、`fast_loss_without_strong_exit_count=0`；历史窗口仍保留 `contract_violation_count=5`，不得标记全绿。
- 重启后精确窗口约 2.21 分钟：3 decisions，全部 hold；entry/orders/filled/failed_or_rejected/positions_created/positions_closed/fast_loss/weak_evidence_executed 全部 0；strategy_learning_events 3。
- 120 分钟策略健康窗口：242 decisions，全部 hold；entry/orders/failed_orders/fast_loss_close_under_15m 均 0；open_positions 2。该窗口包含重启前样本，只用于确认没有新增执行错误，不代表收益改善。
- `model_training` 仍为 warning，但细节显示 local AI tools 可用、runtime_probe ok、shadow_sample_count 19982、trade_sample_count 1600、text_sentiment_sample_count 8000；硬 warning 来源是 Scrapling 外部事件采集已启用但没有有效 HTTPS 公网采集源。该问题属于外部事件数据源配置风险，不得被当作放宽开仓理由。

追加修复：系统总巡检模型卡片轻量化（2026-06-23）：
- 触发原因：交易执行契约口径修复上线后，单独运行 `model_expert_health`、`model_expert_competition`、`model_dynamic_routing` 都能返回只读报告，但完整系统巡检在 20 秒 section timeout 和并行负载下偶发把模型卡片显示为“巡检模块执行失败”。该噪声会误导后续 AI 把注意力从真实交易闭环问题转向不存在的模型服务故障。
- 本次只改系统总巡检聚合页的查询窗口：新增 `MODEL_EXPERT_AUDIT_HOURS=24`、`MODEL_EXPERT_AUDIT_LIMIT=200`，三张模型卡在总巡检内使用 24h/200 轻量窗口；独立详情接口 `/model-expert-health/status`、`/model-expert-competition/status`、`/model-dynamic-routing/status` 继续尊重调用方传入的 `hours/limit`，用于深度复盘。
- 安全边界不变：三张模型卡仍强制 `audit_only=true`，不得改真实模型/专家权重，不得启用 live route，不得替换主链路，不得把缺失 baseline 或 shadow route 结果当作开仓放宽理由。
- 本地 TDD 验证：先新增参数断言测试并确认红测为缺少 `MODEL_EXPERT_AUDIT_*` 常量；实现后 `pytest tests/test_system_audit_api.py::test_model_expert_health_audit_reports_read_only_state tests/test_system_audit_api.py::test_model_expert_competition_audit_never_allows_live_weight_change tests/test_system_audit_api.py::test_model_dynamic_routing_audit_and_endpoint_force_read_only -q` 为 3 passed；`pytest tests/test_system_audit_api.py -q` 为 24 passed；`ruff check web_dashboard/api/system_audit.py tests/test_system_audit_api.py` 0 issues；`black --check web_dashboard/api/system_audit.py tests/test_system_audit_api.py` 通过；`git diff --check` 通过。
- 线上部署：`python scripts/sync_to_online_server.py --split-services` 上传 1 个变更文件并重启成功：model tunnels ok，`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard `302` 健康响应。
- 线上只读总巡检复查（`record_history=False`）：总耗时 12.585s，整体 `warning`，`critical_cards=[]`，cards 16、warning 11、ok 5；三张模型卡均正常返回，不再是“巡检模块执行失败”。其中模型体检为 7 个组件全部只影子观察，竞赛仍缺 baseline 且 `can_apply_live_weight=false`，动态路由 121 个 route plan 全部影子观察、`can_apply_live_route=false`、弱证据执行 0。
- 仍未清零的 warning：`trade_execution_contract` 仍为历史 24h violation 遗留、当前 runtime window 未复现；`okx_reconciliation` dry-run 本次超时但不证明缺失仓位；这些 warning 不能被 AI 当作“系统已全绿”，也不能被用来放宽开仓。
- 最新 120 分钟策略窗口：241 decisions 全 hold；entry_decisions/executed_entries/orders/failed_orders/positions_created/positions_closed/fast_loss_close_under_15m 均 0；open_positions 2；strategy_learning_events 241；missed_opportunity_sample 369。结论是没有新增执行类事故，但仍未形成收益闭环，下一步应诊断 missed opportunity、候选证据链、expected net、成本/滑点、模型 readiness 和同币种同方向重复证据，不得直接降低风控门槛。

继续观察规则：
- Batch H 仍未完成，必须继续 2h/24h/72h 或至少 20 笔已平仓订单观察。
- 后续如果当前窗口新增 `critical`、失败订单、弱证据执行、快亏平、绕过风控、无强证据亏损复开，必须停止扩大逻辑并回到对应批次定位。
- 当前仍是 0 开仓，不允许为了制造交易量直接放宽阈值；若持续 0 开仓且 missed opportunity 高，应诊断候选证据链、expected net、成本/滑点、模型 readiness、特征缺失和同币种同方向重复证据，而不是绕过风险门。
- 回滚点：本次修复只涉及 `services/trade_execution_contract.py`、`web_dashboard/api/system_audit.py` 及对应测试；线上若需回滚，可回到部署前 commit `7eaa8e4` 或回滚本次提交并重启三项服务。

---

## 二十一、Batch H 补充记录：暂停态与观察脚本防偏（2026-06-23）

触发原因：Batch H 继续观察时发现“0 开仓 + missed opportunity 高”的初步判断会被两个因素带偏：其一，线上曾经 `paused=true`，当时只有持仓复核在跑，没有新币种 market analysis；其二，`scripts/inspect_online_strategy_health.py` 使用固定远端临时文件 `/tmp/codex_strategy_sample.py`，并行跑 10m/120m 观察窗口时会互相覆盖，导致观察输出口径混淆。

本次修复范围：
- `web_dashboard/api/system_audit.py`：`_load_trading_runtime_audit_window()` 透出 `paused/scan_mode/current_stage/market_current_stage/market_round_active/last_market_round_*`；`trade_loop` 巡检在 runtime 心跳新鲜且 `paused=true` 时标记为 warning/observing，并明确 summary 为暂停态，避免把暂停误判成策略卡死或证据不足。
- `tests/test_system_audit_api.py`：新增回归测试，先确认暂停态会被旧逻辑误判为 `critical`，再验证新逻辑把它归为观察项，并在 details 中暴露 `paused/scan_mode`。
- `scripts/inspect_online_strategy_health.py`：远端观察脚本改用 `/data/bb/app/tmp/codex-strategy-health/sample_<minutes>_<token>.py` 和对应 launcher，避免并行窗口覆盖；观察完成后清理临时文件。
- `tests/test_inspect_online_strategy_health.py`：新增测试锁定唯一临时文件路径、窗口替换和非 `/tmp` 固定路径契约。

安全边界：
- 不放宽开仓阈值；不改杠杆、仓位、平仓、模型权重或专家路由；不绕过风控；不写真实库历史数据。
- 暂停态只用于解释 trade_loop 巡检状态，不能被用作“系统已经健康赚钱”的证据。
- 修复观察脚本只为避免观察窗口互相污染，不能把更多 missed opportunity 当作强制开仓理由。

本地验证：
- `pytest tests/test_system_audit_api.py tests/test_inspect_online_strategy_health.py -q`：27 passed。
- `ruff check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py web_dashboard/api/system_audit.py tests/test_system_audit_api.py`：0 issues。
- `black --check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py web_dashboard/api/system_audit.py tests/test_system_audit_api.py`：通过。
- `python scripts/security_secret_scan.py --fail-on high .`：扫描 514 files OK。
- `git diff --check`：通过。

线上部署与复查：
- `python scripts/sync_to_online_server.py --split-services` 两次同步均成功，最终上传 `scripts/inspect_online_strategy_health.py`；`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard `302` 健康响应。
- systemd 环境只读总巡检（`record_history=False`）：总耗时约 14.747s，整体 `warning`，`critical_cards=[]`，cards 16、warning 11、ok 5；`issue_ledger` fixed 5、unresolved 5、observing 6。
- `trade_loop` 当前因刚重启处于冷启动观察，但 runtime details 已正确展示：`paused=false`、`mode=paper`、`scan_mode=auto`、market round active/last_market_round 时间可见；这证明当前 0 订单不是暂停造成。
- `trade_execution_contract` 当前 runtime window 仍为 0 执行违规：`executed_entry_count=0`、`weak_evidence_executed_count=0`、`fast_loss_without_strong_exit_count=0`、`contract_violation_count=0`；历史 24h violation 仍保留为 warning，不得标记全绿。
- 修复后的策略健康脚本串行 120m 复查：268 decisions、267 hold、1 entry candidate、0 orders、0 failed_orders、0 positions_created/closed、0 fast_loss_close_under_15m、open_positions 2。唯一 SOL/USDT short 候选为正 expected net，但 evidence tier `blocked`、score 低于 min、profit quality 约 0.469、loss probability 约 0.510、tail risk 约 0.441，最终 `risk_check:skipped`，没有绕过风控。
- 追加观察口径修复：`scripts/inspect_online_strategy_health.py` 增加 `analysis_type_counts`、`analysis_type_action_counts`、`entry_candidate_evidence_by_type`、`market_decisions`、`position_review_decisions`、`market_entry_decisions`，避免把持仓复核数量误当成新币种 market 扫描数量。
- 修复后 120m 线上只读复查：272 decisions 中 `market_decisions=49`、`position_review_decisions=223`；`analysis_type_action_counts` 为 `position_review:hold=223`、`market:hold=45`、`market:short=4`；`entry_candidate_evidence_by_type.market=49`；orders/failed_orders/fast_loss_close_under_15m 仍为 0，4 个 market short 候选均为 `risk_check:skipped`。

当前结论：
- 本轮修复解决的是观察和巡检防偏，不是收益闭环完成。
- 当前线上没有新增弱证据执行、失败订单或快亏平，但仍未产生真实收益改善样本。
- 后续继续 Batch H 的 2h/24h/72h 或至少 20 笔已平仓订单观察；如果持续 0 开仓，应诊断候选证据链、expected net 组成、成本/滑点、profit quality、loss probability、tail risk、模型 readiness、特征缺失、同币种同方向重复证据，而不是直接降阈值。

回滚点：
- 代码层可回滚 `web_dashboard/api/system_audit.py`、`scripts/inspect_online_strategy_health.py` 及对应测试；线上回滚后重启三项服务即可。
- 本批无 DB 迁移、无历史覆盖、无真实交易参数放宽。

## 二十二、Batch H 补充记录：market 候选证据链统计（2026-06-23）

触发原因：Batch H 继续观察时，线上仍然是 `orders=0`，但新增 `analysis_type` 统计已经证明当前有新币种 market analysis 和 market entry candidate，不能继续把 0 单简单归因于暂停态、脚本窗口互相覆盖或持仓复核数量混淆。因此本次只增强只读观察脚本的证据链统计，让后续定位能区分“没有候选”“候选被风控/质量挡住”“候选已执行但订单异常”。

本次修复范围：
- `scripts/inspect_online_strategy_health.py`：增加 `expected_net_breakdown.components` 解析，并输出 market 候选的 `score_gap`、`profit_quality_ratio`、`loss_probability`、`tail_risk_score` 和 expected net 组件贡献统计。
- `tests/test_inspect_online_strategy_health.py`：新增模板契约测试，锁定上述证据链字段，防止后续观察脚本退化成只看 orders/hold 的粗口径。

安全边界：
- 本批只增加只读诊断字段，不放宽开仓阈值，不改杠杆、仓位、平仓、模型权重或专家路由，不绕过风控，不修改真实库历史数据。
- `expected_net_return_pct > 0` 不能单独视为可开仓；必须同时看 score gap、profit quality、loss probability、tail risk、成本/滑点、模型 readiness 和交易执行契约。
- 观察到 missed opportunity 或 shadow memory 正贡献，只能作为受限证据输入；不能把全局错过机会、单次暴涨样本或影子记忆直接升级成强制开仓理由。

本地验证：
- `pytest tests/test_inspect_online_strategy_health.py -q`：5 passed。
- `ruff check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：0 issues。
- `black --check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：通过。
- `python scripts/security_secret_scan.py --fail-on high .`：扫描 514 files OK。
- `git diff --check`：通过。

线上只读复查：
- 新版策略健康脚本读取 120 分钟窗口：279 decisions，其中 `market_decisions=60`、`position_review_decisions=219`；`analysis_type_action_counts` 为 `position_review:hold=219`、`market:hold=54`、`market:short=5`、`market:long=1`。
- `python scripts/sync_to_online_server.py --split-services` 已同步本次相关的 2 个变更文件：`docs/superpowers/plans/2026-06-22-quant-closed-loop-eradication.md`、`scripts/inspect_online_strategy_health.py`；`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard `302` 健康响应。
- 追加修正：`entry_candidate_evidence_by_type` 之前按所有 decisions 统计 `entry_candidate_evidence` payload，容易把 market 扫描数误读成候选数；已改为只从 `entry_decisions` 统计，并用红绿测试锁定。修正后 30 分钟窗口 `market_entry_decisions=3` 且 `entry_candidate_evidence_by_type.market=3`；120 分钟窗口 `market_entry_decisions=8` 且 `entry_candidate_evidence_by_type.market=8`。
- 修正后 120 分钟窗口有 `market_entry_decisions=8`，全部停在 `risk_check:skipped`；`executed_entries=0`、`orders=0`、`failed_orders=0`、`positions_created=0`、`positions_closed=0`、`fast_loss_close_under_15m=0`，open_positions 仍为 2。
- 8 个 market 候选的 `expected_net_return_pct` 全部为正：min 0.281631、median 0.316030、max 0.337874；但 `position_size_pct` 全部为 0，不能据此认为已经满足开仓质量。
- 8 个 market 候选的 `score_gap = score - min_score_required` 全部为负：min -2.558195、median -1.989693、max -0.753132，说明当前不是下单链路丢单，而是证据评分未达开仓门槛。
- `profit_quality_ratio`：min 0.361259、median 0.423606、max 0.469269；`loss_probability`：min 0.428200、median 0.541700、max 0.560100；`tail_risk_score`：min 0.342714、median 0.408209、max 0.441075。当前候选质量和风险结构仍偏弱。
- expected net 组件拆解：`ai` 固定正贡献 0.15；`shadow_memory` 为正贡献且接近 cap；`local_ml` 全部 0；`server_profit` 全部负贡献；`fee` 和 `slippage` 全部负贡献；`timeseries` 仅小幅贡献。当前正 EV 主要由 AI 与影子错过机会支撑，尚不足以越过评分、质量和风险门槛。
- `local_ml=0` 是 ML readiness 保护而非缺功能：线上模型最新训练于 2026-06-22T23:56:22Z，`sample_count=19982`、`test_count=4996`，但状态为 `degraded`、`allow_live_position_influence=false`。阻塞项包括 long PR-AUC 0.3587 < 0.52、short PR-AUC 0.3912 < 0.52、short top-score bucket return -0.011 < 0.05，以及 dirty/downweighted 样本比例 0.8976 > 0.08。质量报告显示 20,000 条 shadow 样本中 hold 17,947 条，主要原因是 `very_low_decision_confidence`、`hold_observation_downweighted`、`hold_missed_opportunity_downweighted`；不得硬改 ready 或放宽 readiness。
- `server_profit` 负贡献来自真实模型输出，不是字段映射错误：最新 market 候选中 `local-profit-trained-v2` 可用，但候选方向的 expected return 全为负，部分 best_side 只是“两边都负时较不差的一边”，因此 `local_profit_aligned=false`，expected net 组件中 server_profit 维持小幅负贡献约 -0.017 至 -0.028。不得把 best_side 文本当成正收益支持，也不得忽略负 expected return。

手工系统巡检防偏口径：
- 手工调用 `collect_system_audit_status(record_history=False)` 时，不能使用线上裸 `python3`，否则会因缺少依赖得到 `No module named 'fastapi'` 假故障；也不能直接 shell source `.env`，因为复杂列表/映射值会被 shell 误解析。
- 正确口径是：由 root 进程用 Python/dotenv 读取 `/data/bb/app/.env` 与 `/etc/bb/bb-runtime.env`，并让 runtime env 覆盖 app env，然后以 OS 用户 `bb` 和 `/data/bb/app/.venv/bin/python` 执行巡检。否则容易出现 `Peer authentication failed for user "root"`、`Peer authentication failed for user "bb"` 或模型训练假 `critical`。
- 按上述稳定口径复查系统巡检：整体 `warning`，`critical_cards=[]`，cards 16、warning 11、ok 5；`issue_ledger` fixed 5、unresolved 7、observing 4。`trade_loop` 为 warning：最近 2 小时 284 decisions、0 orders、open_positions 2、`market_analysis_paused=false`、runtime heartbeat fresh。`trade_execution_contract` 为 warning：24h 历史遗留仍在，但当前 runtime window 未复现；`can_bypass_risk_controls=false`、`live_entry_mutation=false`、`live_exit_mutation=false`。`model_expert_health/model_expert_competition/model_dynamic_routing/shadow_missed_opportunity/crypto_feature_coverage` 均仍为只读观察或阻塞态，没有 live 权重、live 路由、弱证据执行或风控绕过。

当前结论：
- 当前 0 订单不是暂停态造成，也不是观察脚本窗口互相覆盖造成；线上确实产生了 market 候选，但全部因证据链不足被跳过。
- 这说明前序“弱证据不得执行、不得绕过风控”的约束正在生效；同时也说明收益闭环还没有完成，不能把 0 异常订单等同于策略有效赚钱。
- 下一步应继续 Batch H 观察，并优先定位为什么 market 候选长期 `score_gap < 0`、`profit_quality_ratio` 偏低、`loss_probability/tail_risk` 偏高、`server_profit` 负贡献和 `local_ml=0`；不得通过直接降阈值、改杠杆、放大仓位或硬改 ML readiness 来制造成交。

回滚点：
- 代码层可回滚 `scripts/inspect_online_strategy_health.py` 与 `tests/test_inspect_online_strategy_health.py`；本批无服务行为改动、无 DB 迁移、无历史覆盖、无真实交易参数放宽。

---

## 二十三、Batch H 补充记录：local ML readiness 直连观察（2026-06-23）

触发原因：前一轮已经证明 `local_ml=0` 是 ML readiness 保护，而不是 expected net 字段映射错误；但策略健康脚本本身没有直接输出 ML readiness、阻塞原因和训练质量摘要，后续 AI 仍可能把 `local_ml=0` 误读成“少接了一个功能”，进而走偏到硬改 `ready` 或放宽 readiness。因此本次只把 ML readiness 只读摘要接入策略健康脚本输出。

本次修复范围：
- `scripts/inspect_online_strategy_health.py`：新增 `local_ml_readiness_summary()`，调用 `MLSignalService().status()`，输出 `status/readiness_state/allow_live_position_influence/advisory_enabled/blocking_reason_codes/metrics/quality_totals/quality_top_reasons`。
- `tests/test_inspect_online_strategy_health.py`：新增模板契约测试，锁定 `local_ml_readiness`、`allow_live_position_influence`、`blocking_reason_codes` 和质量原因字段，防止后续观察脚本退化。

安全边界：
- 本批只增加只读诊断字段，不改变开仓阈值、杠杆、仓位、平仓、模型权重、专家路由或风控门。
- `local_ml_readiness.status=degraded` 与 `allow_live_position_influence=false` 必须被视为保护性阻断，不得硬改 `ready`，不得隐藏 blocking reasons，不得让未达标 ML 参与真实仓位放大。
- 影子错过机会、AI 正贡献或 expected net 为正，仍不能绕过 score gap、profit quality、loss probability、tail risk、server_profit 和交易执行契约。

本地验证：
- `pytest tests/test_inspect_online_strategy_health.py -q`：6 passed。
- `ruff check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：0 issues。
- `black --check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：通过。
- `python scripts/security_secret_scan.py --fail-on high .`：扫描 514 files OK。
- `git diff --check`：通过。

线上只读复查：
- 10 分钟窗口（`2026-06-23T00:57:39Z`）：23 decisions，其中 `market_decisions=9`、`position_review_decisions=14`、`market_entry_decisions=2`；2 个 market 候选全部为 `risk_check:skipped`，`executed_entries=0`、`orders=0`、`failed_orders=0`、`positions_created=0`、`positions_closed=0`、`fast_loss_close_under_15m=0`。
- 120 分钟窗口（`2026-06-23T00:57:39Z`）：290 decisions，其中 `market_decisions=98`、`position_review_decisions=192`、`market_entry_decisions=11`；11 个 market 候选全部为 `risk_check:skipped`，`executed_entries=0`、`orders=0`、`failed_orders=0`、`positions_created=0`、`positions_closed=0`、`fast_loss_close_under_15m=0`，open_positions 仍为 2。
- 120 分钟候选证据链：`expected_net_return_pct` 全部为正（min 0.280506、median 0.314058、max 0.337874），但 `position_size_pct` 全部为 0；`score_gap` 全部为负（min -2.578678、median -1.989693、max -0.753132），`profit_quality_ratio` median 0.419975，`loss_probability` median 0.5436，`tail_risk_score` median 0.399856。当前仍是候选质量与风险门未过，不是下单链路丢单。
- expected net 组件拆解：`ai` 固定正贡献 0.15，`shadow_memory` 约 0.35 且为正，`local_ml` 全部 0，`server_profit` 全部负贡献（约 -0.017 到 -0.028），`fee/slippage` 为负，`timeseries` 仅小幅贡献。
- `local_ml_readiness` 直连结果：`available=true`、`status=degraded`、`readiness_state=degraded`、`allow_live_position_influence=false`、`advisory_enabled=false`；阻塞项为 `long_pr_auc_below_threshold`、`short_pr_auc_below_threshold`、`short_top_return_below_threshold`、`dirty_sample_ratio_high`。
- ML 指标：`sample_count=19982`、`test_count=4996`、`dirty_sample_ratio=0.8979`、`long_pr_auc=0.372210098695988`、`short_pr_auc=0.38521014983148383`、`top_long_avg_return_pct=0.10222611240077366`、`top_short_avg_return_pct=-0.10458550472034482`、`training_data_version=2026-06-23.v3`。质量汇总为 total 20000、included 2042、downweighted 17940、excluded 18；主要原因仍是 `shadow:very_low_decision_confidence`、`shadow:hold_observation_downweighted`、`shadow:hold_missed_opportunity_downweighted`。

同步后复查：
- `python scripts/sync_to_online_server.py --split-services` 已同步 2 个变更文件：`docs/superpowers/plans/2026-06-22-quant-closed-loop-eradication.md`、`scripts/inspect_online_strategy_health.py`；`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard `302` 健康响应。
- 部署后 10 分钟窗口（`2026-06-23T01:01:22Z`）：21 decisions，其中 `market_decisions=9`、`position_review_decisions=12`、`market_entry_decisions=1`；该候选为 `risk_check:skipped`，`executed_entries=0`、`orders=0`、`failed_orders=0`、`positions_created=0`、`positions_closed=0`、`fast_loss_close_under_15m=0`。
- 部署后 120 分钟窗口（`2026-06-23T01:01:22Z`）：291 decisions，其中 `market_decisions=101`、`position_review_decisions=190`、`market_entry_decisions=11`；11 个 market 候选全部 `risk_check:skipped`，`orders=0`、`failed_orders=0`、`fast_loss_close_under_15m=0`，open_positions 仍为 2。`local_ml_readiness` 仍为 degraded 且 `allow_live_position_influence=false`。
- 稳定口径系统巡检（`record_history=False`，Python/dotenv 加载 `/data/bb/app/.env` 与 `/etc/bb/bb-runtime.env` 后以 OS 用户 `bb` 执行）：整体 `warning`，`critical_cards=[]`，cards 16、warning 11、ok 5；`issue_ledger` fixed 5、unresolved 7、observing 4。`trade_loop` 最近 2 小时有分析但 0 orders、open_positions 2、`market_analysis_paused=false`、runtime heartbeat fresh；`trade_execution_contract.current_summary.contract_violation_count=0`、`weak_evidence_executed_count=0`、`fast_loss_without_strong_exit_count=0`，历史 24h violation 仍保留为 warning。`runtime_text_integrity` 为 ok，扫描 815 条、疑似记录 0。

当前结论：
- 策略健康脚本现在能在同一份观察报告里解释 `local_ml=0` 的原因，降低后续 AI 把保护状态误读成缺功能的风险。
- 当前没有触发新增失败订单、弱证据执行、快亏平或风控绕过等停止规则；但也没有形成盈利闭环，Batch H 仍未完成。
- 下一步仍应继续观察，并定位高 missed opportunity 与低候选质量之间的断点：训练样本质量、候选评分、server_profit 负贡献、profit quality、loss probability、tail risk、成本/滑点和同币种同方向重复证据，而不是降低阈值或硬改 ML readiness。

回滚点：
- 代码层可回滚 `scripts/inspect_online_strategy_health.py` 与 `tests/test_inspect_online_strategy_health.py`；本批无 DB 迁移、无历史覆盖、无服务行为变更、无真实交易参数放宽。

---

## 二十四、Batch H 补充记录：entry evidence 阻塞口径修正（2026-06-23）

触发原因：前一轮观察已经证明线上有 market 候选且全部未下单，但只看 `score_gap = opportunity_score.score - min_score_required` 容易让后续 AI 走偏，把它误当成唯一硬阻塞，再去直接降阈值制造成交。实际执行口径还必须看 `evidence_score.tier`、`effective_score`、`decision_state_machine.skip_kind`、`shadow_only`、`tradeable_probe`、`hard_block` 和最终 `position_size_pct`。

本次修复范围：
- `scripts/inspect_online_strategy_health.py`：新增 `evidence_components(decision)` 与 `entry_skip_kind(decision)`，输出 `market_entry_opportunity_score_gap_stats`、`market_entry_evidence_effective_score_stats`、`market_entry_evidence_tier_counts`、`market_entry_final_skip_kind_counts`、`market_entry_evidence_component_status_counts`、`market_entry_evidence_shadow_only_count`、`market_entry_evidence_tradeable_probe_count`、`market_entry_evidence_hard_block_count`，并在 entry examples 中补充 `hard_block`、`hard_block_reasons`、`advisory_wait_reasons`、`aligned_support_sources`、`major_opposites`、`weak_opposites`、`strong_opposites`。
- `tests/test_inspect_online_strategy_health.py`：新增模板契约测试，先确认缺少上述字段时测试失败，再锁定脚本必须输出最终入场阻塞口径，避免后续观察退化成只看 orders、expected net 或 score gap。

安全边界：
- 本批只增强只读诊断字段，不放宽开仓阈值，不改杠杆、仓位、平仓、模型权重、专家路由或风控门，不修改真实库历史数据。
- `expected_net_return_pct > 0`、`shadow_memory` 正贡献、`ai` 正贡献或 `score_gap` 接近阈值，都不能单独升级为开仓理由。
- 如果继续 0 订单，下一步必须解释证据有效分、组件状态、ML degraded、server_profit opposite、sentiment opposite、profit quality、loss probability、tail risk 和同币种同方向历史，而不是直接降低门槛或硬改 readiness。

本地验证：
- 已按 TDD 增加 `test_strategy_health_report_exposes_entry_execution_blocking_contract`，先看到缺字段失败，再实现脚本输出后通过。
- `pytest tests/test_inspect_online_strategy_health.py -q`：7 passed。
- `ruff check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：0 issues。
- `black --check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：通过。
- `python scripts/security_secret_scan.py --fail-on high .`：扫描 514 files OK。

线上只读复查（同步前，当前本地脚本远端执行，`2026-06-23T01:19:53Z`）：
- 120 分钟窗口共 310 decisions，其中 `market_decisions=124`、`position_review_decisions=185`、`entry_decisions=12`、`market_entry_decisions=11`；`orders=0`、`failed_orders=0`、`positions_created=0`、`positions_closed=0`、`fast_loss_close_under_15m=0`，open_positions 仍为 2。
- 11 个 market entry 候选全部未执行，`entry_state_counts` 为 `risk_check:skipped=12`；market 候选 `position_size_pct` 全部为 0。
- market 候选 `expected_net_return_pct` 全部为正，但这不是开仓结论；`market_entry_opportunity_score_gap_stats` 全部为负（min -2.578678、median -1.989693、max -0.753132）。
- 新增最终执行口径显示：`market_entry_evidence_tier_counts.blocked=11`，`market_entry_final_skip_kind_counts.entry_evidence_wait=11`，`market_entry_evidence_effective_score_stats` 为 min 29.241435、median 31.25256、max 34.41474。
- 当前不是 hard risk block、shadow-only 或 tradeable probe：`market_entry_evidence_hard_block_count=0`、`market_entry_evidence_shadow_only_count=0`、`market_entry_evidence_tradeable_probe_count=0`。
- 组件状态显示：`ai:aligned=11`、`shadow_memory:aligned=11`、`ml:ignored=11`、`timeseries:aligned=10`，但 `sentiment:opposite=10`、`server_profit:opposite=9`、`server_profit:ignored_negative_expected=2`、`symbol_side_history:opposite=2`。也就是说，当前正 EV 主要由 AI 与影子记忆支撑，ML 因 degraded 被忽略，server_profit 与情绪多数反向，最终证据有效分仍低于可交易底线。

同步后复查：
- `python scripts/sync_to_online_server.py --split-services` 已同步 2 个变更文件：`docs/superpowers/plans/2026-06-22-quant-closed-loop-eradication.md`、`scripts/inspect_online_strategy_health.py`；`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard `302` 健康响应。
- 部署后 10 分钟窗口（`2026-06-23T01:24:53Z`）：32 decisions，其中 `market_decisions=12`、`position_review_decisions=19`、`entry_decisions=2`、`market_entry_decisions=1`；`orders=0`、`failed_orders=0`、`positions_created=0`、`positions_closed=0`、`fast_loss_close_under_15m=0`，open_positions 仍为 2。该 market 候选为 `blocked`，最终 `entry_pre_execution_skip`，实际下单方向费后预期净收益为 -0.071697%，系统未提交订单。
- 部署后 120 分钟窗口（`2026-06-23T01:24:53Z`）：314 decisions，其中 `market_decisions=129`、`position_review_decisions=184`、`entry_decisions=13`、`market_entry_decisions=12`；`orders=0`、`failed_orders=0`、`positions_created=0`、`positions_closed=0`、`fast_loss_close_under_15m=0`，open_positions 仍为 2。
- 120 分钟窗口的 12 个 market 候选全部为 `market_entry_evidence_tier_counts.blocked=12`；最终 `market_entry_final_skip_kind_counts` 为 `entry_evidence_wait=11`、`entry_pre_execution_skip=1`；`market_entry_evidence_effective_score_stats` 为 min 28.76099、median 31.25256、max 34.41474；`hard_block=0`、`shadow_only=0`、`tradeable_probe=0`。
- 120 分钟窗口组件状态：`ai:aligned=12`、`shadow_memory:aligned=12`、`ml:ignored=12`、`timeseries:aligned=11`，但 `sentiment:opposite=11`、`server_profit:opposite=9`、`server_profit:ignored_negative_expected=3`、`symbol_side_history:opposite=2`。`local_ml_readiness` 仍为 `degraded` 且 `allow_live_position_influence=false`。
- 稳定口径系统巡检（`record_history=False`，Python/dotenv 加载 app/runtime env 后以 OS 用户 `bb` 执行）：整体 `warning`，`critical_cards=[]`，cards 16、warning 11、ok 5；`issue_ledger` fixed 5、unresolved 7、observing 4。`trade_loop` 最近 2 小时 310 decisions、0 orders、open_positions 2、`market_analysis_paused=false`、runtime heartbeat fresh；`trade_execution_contract.current_summary.contract_violation_count=0`、`weak_evidence_executed_count=0`、`negative_expected_executed_count=0`、`fast_loss_without_strong_exit_count=0`，历史 24h violation 仍保留为 warning；`runtime_text_integrity` 为 ok，扫描 815 条。

当前结论：
- 当前 0 订单不是暂停态、观察脚本覆盖、下单链路丢单、硬风控误杀或 shadow-only 造成；更精确地说，是入场证据状态机给出 `entry_evidence_wait`，证据 tier 仍为 `blocked`，仓位计算保持 0。
- 这说明“弱证据不执行、风控不绕过”的约束仍在生效；同时也说明收益闭环仍未完成，不能把 0 失败订单误当成策略已经会赚钱。
- 后续优先诊断为什么有效证据分长期停在 29-34 区间：ML readiness、server_profit 方向收益为负、sentiment opposite、profit quality 偏低、loss probability/tail risk 偏高、成本/滑点压力、同币种同方向历史不足或质量不够。不得通过降阈值、放大仓位、强开 probe、硬改 `ready` 来制造成交。

回滚点：
- 代码层可回滚 `scripts/inspect_online_strategy_health.py` 与 `tests/test_inspect_online_strategy_health.py`；本批无 DB 迁移、无历史覆盖、无服务行为变更、无真实交易参数放宽。

---

## 二十五、Batch H 补充记录：拒单与快亏止损证据持久化（2026-06-23）

触发原因：Batch H 继续观察期间，线上 120 分钟窗口开始出现真实订单样本，其中包含 1 条拒单和 1 条 15 分钟内亏损平仓。按停止规则，新增失败/拒绝订单和快亏平必须先核查，不能直接进入下一批功能扩展。本次只做只读诊断和未来事件的可追溯性补强，不放宽任何交易门槛。

本次修复范围：
- `services/execution_service.py`：在交易所确认与未确认两条路径中，把 `ExecutionResult` 的压缩快照写入 `decision.raw_response["execution_result"]`，包含 `order_id/exchange_order_id/status/quantity/price/fee/pnl/exchange_confirmed/exit_progress/raw_response`。这样未来 OKX 拒单、系统预提交拒绝、规则快照、请求参数和 `raw_error` 不会只停留在临时日志里。
- `scripts/inspect_online_strategy_health.py`：新增 `order_execution_result(decision)`、`order_status_counts`、`non_filled_orders`、`rejected_orders`、`pending_or_open_orders` 和 `rejected_order_examples`，并在 entry examples 中输出执行结果快照，避免把拒单误读成已成交或把未成交原因丢失。
- `scripts/inspect_online_strategy_health.py` 同时补充 entry evidence 原始分、有效分、分数 offset、阈值、组件 points、relief 状态和 advisory wait reasons，用来解释为什么某些候选从 `blocked` 进入受控 `exploration/weak_conflict_probe`，而不是让后续 AI 只看 `expected_net > 0` 后去降阈值。
- `tests/test_inspect_online_strategy_health.py` 与 `tests/test_trading_service_boundaries.py`：用模板契约测试和执行服务边界测试锁定上述字段；拒单场景先确认缺少 `execution_result` 时测试失败，再实现持久化后通过。

安全边界：
- 本批不改变开仓阈值、证据 tier、probe 条件、杠杆、仓位、平仓、模型权重、专家路由或风控 veto。
- 不修改真实库历史数据；历史拒单只做只读核查，缺失的历史 `raw_error` 不得补猜。
- 拒单不是成交；拒单订单不得计入已开仓收益样本，也不得被用来证明策略有效或无效，只能作为执行链路诊断样本。
- 快亏平只有在能找到强结构化退出证据时，才可归为风控退出；否则必须触发停止规则并回到 Batch E 定位。
- `exploration` 或 `weak_conflict_probe` 只能按已有证据状态机的小额受控路径执行，不得被 AI 扩展成普遍放宽开仓。

本地验证：
- `pytest tests/test_inspect_online_strategy_health.py -q`：10 passed。
- `pytest tests/test_trading_service_boundaries.py -q`：119 passed。
- `ruff check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py services/execution_service.py tests/test_trading_service_boundaries.py`：0 issues。
- `black --check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py services/execution_service.py tests/test_trading_service_boundaries.py`：通过。
- `git diff --check`：通过。
- `python scripts/security_secret_scan.py --fail-on high .`：扫描 514 files OK。

线上只读核查事实（同步前，使用当前本地诊断脚本读取线上数据）：
- 120 分钟窗口约 `2026-06-23T02:01Z`：`orders=3`、`filled_orders=2`、`rejected_orders=1`、`non_filled_orders=1`、`positions_created=1`、`positions_closed=1`、`fast_loss_close_under_15m=1`、`executed_entries=1`。
- 拒单样本为 order `2552`、decision `117291`、`SAHARA/USDT`、`short`、`status=rejected`、`quantity=0`、`exchange_order_id=null`。该决策没有成交，也没有创建持仓；它是受控 `exploration` probe，不是弱证据强行成交。
- 该历史拒单在现有持久化数据里没有 OKX `raw_error`，journal 中也没有可恢复的完整拒单行。因此精确 OKX 拒绝码不可从当前数据还原，不能臆造。本批代码修复的是未来拒单的持久化缺口。
- 120 分钟窗口的 market entry evidence 阈值为 weak_probe 35、exploration 45、small 60、medium 70、normal 80；tier 统计为 `blocked=12`、`weak_conflict_probe=2`、`exploration=2`；`tradeable_probe_count=2`、`shadow_only_count=2`、`hard_block_count=0`。ML 仍为 degraded，且不允许影响真实仓位放大。
- 快亏样本为 ZETA position `1609`：entry order `2553`、decision `117311`、`ZETA/USDT` short，成交价约 `0.03825`、数量 `500`；exit order `2554`、decision `117320`、`ZETA/USDT:USDT` close_short，成交价约 `0.03829`、数量 `500`；持仓约 `2.43` 分钟，realized PnL 约 `-0.039135 USDT`。
- ZETA exit decision `117320` 存在强结构化退出证据：`forced_exit=true`、`exit_intent=hard_risk`、`close_evidence.forced_exit=true`、`close_evidence.exit_intent=hard_risk`、`position_release_policy.release_reason="severe_loss_pressure; signal_reversal_watch"`、`position_quality.bucket="release_now"`、score `22.0`，并且 `exit_quality.invalidation` 中 `severe/key_break/trend_reversal` 为 true。因此该快亏从当前证据看符合强退出证据要求，但部署后仍必须用系统巡检的 `trade_execution_contract.current_summary.fast_loss_without_strong_exit_count=0` 再确认。

同步后复查：
- `python scripts/sync_to_online_server.py --split-services` 已同步 3 个变更文件：`docs/superpowers/plans/2026-06-22-quant-closed-loop-eradication.md`、`scripts/inspect_online_strategy_health.py`、`services/execution_service.py`；`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard `302` 健康响应。
- 部署后 15 分钟窗口（`2026-06-23T02:33:44Z`）：43 decisions，其中 `market_decisions=17`、`position_review_decisions=26`；`entry_decisions=0`、`orders=0`、`rejected_orders=0`、`positions_created=0`、`positions_closed=0`、`fast_loss_close_under_15m=0`，open_positions 仍为 2。
- 部署后 120 分钟窗口（`2026-06-23T02:33:58Z`）：330 decisions，其中 `market_decisions=133`、`position_review_decisions=196`、`entry_decisions=14`、`market_entry_decisions=13`；`orders=3`、`filled_orders=2`、`rejected_orders=1`、`pending_or_open_orders=0`、`positions_created=1`、`positions_closed=1`、`fast_loss_close_under_15m=1`。该窗口仍包含同步前 SAHARA 拒单与 ZETA 快亏样本。
- 120 分钟窗口中 SAHARA 拒单 example 的 `execution_result` 仍为空，证明历史拒单的 OKX raw error 未能倒推恢复；本批修复只保证未来 `exchange_not_confirmed` 路径会写入 `execution_result.raw_response`。
- 部署后 120 分钟 market entry evidence：阈值仍为 weak_probe 35、exploration 45、small 60、medium 70、normal 80；tier 统计为 `blocked=9`、`weak_conflict_probe=2`、`exploration=2`；raw score median `41.264163`，effective score median `34.40461`，score offset median `10.0`。
- `local_ml_readiness` 仍为 `degraded`，`allow_live_position_influence=false`；阻塞项仍包括 `long_pr_auc_below_threshold`、`short_pr_auc_below_threshold`、`short_top_return_below_threshold`、`dirty_sample_ratio_high`，不得硬改 ready 或让未达标 ML 放大仓位。
- 稳定口径系统巡检（`record_history=False`，继承 dashboard 环境、以 OS 用户 `bb` 和 `/data/bb/app/.venv/bin/python` 执行）：整体 `warning`，`critical_cards=[]`，cards 16、warning 9、ok 7；`trade_loop` 为 ok，最近 2 小时 3 orders、open_positions 2、`market_analysis_paused=false`、runtime heartbeat fresh。
- `trade_execution_contract` 巡检为 ok；全窗口 summary 中 `weak_evidence_executed_count=0`、`negative_expected_executed_count=0`、`fast_loss_without_strong_exit_count=0`、`reentry_without_strong_unlock_count=0`、`contract_violation_count=0`。当前 runtime window 自 `2026-06-23T02:28:34Z` 起：25 decisions、0 executed entries、0 weak evidence executed、0 fast loss without strong exit、0 contract violations。
- `runtime_text_integrity` 为 ok，扫描 810 条记录，疑似乱码记录 0；本轮没有引入新的运行时文本污染。

当前结论：
- 这轮发现的是一个未来拒单可追溯性缺口，不是交易策略需要放宽的证据。历史 SAHARA 拒单原因缺失只能如实记录为不可恢复。
- ZETA 快亏金额很小，且当前 raw evidence 指向硬风控退出；它不应被用来禁止必要止损，也不应被用来放宽入场。真正要守住的是：未来 `fast_loss_without_strong_exit_count` 必须为 0。
- Batch H 仍处于观察阶段，不是全绿完成。后续必须继续看 15m/120m/24h 窗口里的新增拒单、弱证据执行、快亏无强退出、loss re-entry 和收益闭环，不得只因为出现少量成交就推进仓位放大。

回滚点：
- 代码层可回滚 `services/execution_service.py`、`scripts/inspect_online_strategy_health.py` 与对应测试；本批无 DB 迁移、无历史覆盖、无真实交易参数放宽。

---

## 二十六、Batch H 补充记录：系统巡检交易契约卡片优先调度（2026-06-23）

触发原因：Batch H 继续观察时发现，完整系统巡检偶发把 `trade_execution_contract` 包装成 warning，错误详情为 `TimeoutError`。直接调用 `TradeExecutionContractService().report()` 并不慢，24h 报告约 1.766s、runtime 报告约 0.334s，且契约违规计数均为 0。因此根因不是交易执行契约本身异常，而是完整系统巡检同时跑 16 个 section 时，慢诊断段可能与契约卡片争用异步调度/资源，使关键交易契约卡片在统一 section timeout 下被误判为超时。

本次修复范围：
- `web_dashboard/api/system_audit.py`：新增 `PRIORITY_AUDIT_KEYS=("trade_execution_contract",)`，让交易执行契约卡片先于其它慢诊断段完成，再并发执行剩余巡检 section。
- `web_dashboard/api/system_audit.py`：把优先 section 与剩余 section 的结果按 `section_key` 合并回 `result_by_key`，后续仍走原有卡片构建、异常包装和状态排序逻辑，避免因优先调度改变巡检卡片语义。
- `tests/test_system_audit_api.py`：新增 `test_system_audit_runs_trade_contract_before_slow_diagnostics`，模拟慢 `model_training` 持有锁、`trade_execution_contract` 等同一把锁的场景；修复前交易契约卡片会被统一 timeout 包成 warning，修复后交易契约为 ok，慢诊断段仍按原规则被包装为 warning。
- `tests/test_system_audit_api.py`：固定 `test_strategy_closed_loop_audit_separates_active_runtime_window` 的当前时间，避免测试样本随真实日期漂移后离开 24h 窗口。

安全边界：
- 本批只改系统巡检调度可靠性，不改变开仓阈值、证据 tier、probe 条件、杠杆、仓位、平仓、模型权重、专家路由或风控 veto。
- `trade_execution_contract` 被优先调度不代表忽略其它巡检问题；慢 section 仍会按原 timeout 规则返回 warning，并继续进入整体 warning/critical 聚合。
- 线上 `strategy_closed_loop` 仍因历史亏损样本、样本不足和 ML 不可用保持 warning；本批不能被解释为策略盈利闭环已证明。
- 若未来 `trade_execution_contract` 自身真实超时或返回违规计数，仍必须按停止规则处理，不得因为它是 priority key 就压低严重性。

本地验证：
- TDD 红灯：`pytest tests/test_system_audit_api.py::test_system_audit_runs_trade_contract_before_slow_diagnostics -q` 在修复前失败，表现为 `trade_execution_contract` status 为 warning 而不是 ok。
- TDD 绿灯：同一测试在修复后通过。
- `pytest tests/test_system_audit_api.py -q`：26 passed。
- `ruff check web_dashboard/api/system_audit.py tests/test_system_audit_api.py`：0 issues。
- `black --check web_dashboard/api/system_audit.py tests/test_system_audit_api.py`：通过。
- `git diff --check`：通过。
- `python scripts/security_secret_scan.py --fail-on high .`：扫描 514 files OK。

线上复查：
- `python scripts/sync_to_online_server.py --split-services` 已同步 `web_dashboard/api/system_audit.py`；`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard 返回 `302`。
- 部署后完整系统巡检（`2026-06-23T03:10:54Z`，`record_history=False`）：整体 `warning`，`critical_cards=[]`，cards 16，其中 warning 9、ok 7。
- `trade_loop` 为 ok，`trade_execution_contract` 为 ok，`runtime_text_integrity` 为 ok；此前的 `trade_execution_contract TimeoutError` 未再出现。
- `trade_execution_contract` 全窗口 summary：decision_count `517`、executed_entry_count `9`、weak_evidence_executed_count `0`、negative_expected_executed_count `0`、fast_loss_count `5`、fast_loss_without_strong_exit_count `0`、reentry_without_strong_unlock_count `0`、contract_violation_count `0`。
- 当前 runtime window 刚随服务重启从 `2026-06-23T03:10:16.958104Z` 起算，current_summary 当时为冷启动零样本；该零样本只能说明重启后尚无新增执行样本，不能证明策略盈利。
- `strategy_closed_loop` 仍为 warning：历史已平仓 9 笔、0 wins、9 losses、fast_loss_under_15m_count 5，ML usable rate 0.0；结论仍是样本不足且不能证明 ML/策略有效。

当前结论：
- 本轮解决的是系统巡检观测可靠性问题：关键交易契约卡片不再被慢诊断段并发拖成误报 timeout。
- 停止规则当前从巡检关键口径看未被新的 critical 或交易契约违规触发，但 Batch H 仍是观察阶段，不是盈利闭环完成。
- 后续继续推进时，必须优先看 `trade_execution_contract.current_summary`、新增拒单、弱证据执行、快亏无强退出、loss re-entry 和 `strategy_closed_loop` 的真实收益样本，而不是只看系统巡检 overall 是否从 warning 变少。

回滚点：
- 代码层可回滚 `web_dashboard/api/system_audit.py` 与 `tests/test_system_audit_api.py`；本批无 DB 迁移、无历史覆盖、无真实交易参数放宽。

---

## 二十七、Batch H 补充记录：线上观察脚本执行结果分类修正（2026-06-23）

触发原因：继续观察 15m/120m 线上窗口时，`scripts/inspect_online_strategy_health.py` 的 `market_entry_final_skip_kind_counts` 出现 `unknown=2`。只读定位确认这两条不是未知原因：一条是 ZETA 已成交并完成本地同步，另一条是此前已核查的 SAHARA 交易所未确认/拒单。旧脚本只读取 skip_kind，缺少对非跳过执行结果的最终分类，容易让后续 AI 把“成交/拒单路径”误读为“原因不透明”。

本次修复范围：
- `scripts/inspect_online_strategy_health.py`：`entry_skip_kind(decision)` 保留原有 `entry_evidence_shadow_only.skip_kind` 与 state machine stage `skip_kind` 优先级；若无 skip_kind 且 `was_executed=true` 或 `decision_state_machine.summary` 为 `local_sync/completed`，返回 `executed`；若最终为 `local_sync/skipped` 或 `local_sync/failed`，返回 `exchange_not_confirmed`。
- `tests/test_inspect_online_strategy_health.py`：新增模板契约测试，先确认已执行样本仍会被旧口径归成 `unknown`，再锁定观察脚本必须输出 `executed` 与 `exchange_not_confirmed` 分类。

安全边界：
- 本批只修正只读观察脚本的分类字段，不改变开仓、仓位、平仓、杠杆、模型权重、专家路由、证据 tier 或风控 veto。
- `executed` 分类只是说明该候选已经进入执行并成交，不代表交易策略有效；仍必须结合 PnL、快亏强退出、弱证据执行和后续平仓样本判断。
- `exchange_not_confirmed` 分类只是说明本地未确认成交或交易所拒绝，不得计入已开仓收益样本，也不得被拿来证明策略盈利或亏损。

本地验证：
- TDD 红灯：`pytest tests/test_inspect_online_strategy_health.py::test_strategy_health_classifies_market_entry_execution_outcomes -q` 在修复前失败，表现为已执行样本返回 `unknown` 而不是 `executed`。
- TDD 绿灯：同一测试修复后通过。
- `pytest tests/test_inspect_online_strategy_health.py -q`：11 passed。
- `ruff check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：0 issues。
- `black --check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：通过。
- `git diff --check`：通过。
- `python scripts/security_secret_scan.py --fail-on high .`：扫描 514 files OK。

线上复查：
- `python scripts/sync_to_online_server.py --split-services --skip-restart` 已同步 `scripts/inspect_online_strategy_health.py`，未重启交易服务。
- 同步后 120m 观察窗口（`2026-06-23T03:41:22Z`）：346 decisions、12 entry_decisions、11 market_entry_decisions、orders 2、filled_orders 2、rejected_orders 0、positions_created 1、positions_closed 1、fast_loss_close_under_15m 1、open_positions 2。SAHARA 拒单已滚出 120m，ZETA 快亏仍在窗口内。
- 同步后 `market_entry_final_skip_kind_counts` 为 `entry_evidence_shadow_only=5`、`entry_evidence_wait=5`、`executed=1`，不再出现 `unknown`。
- 同步后 `market_entry_evidence_tier_counts` 为 `blocked=5`、`weak_conflict_probe=5`、`exploration=1`；`tradeable_probe_count=1`、`shadow_only_count=5`、`hard_block_count=0`。ML 仍为 `degraded`，`allow_live_position_influence=false`。

当前结论：
- 这轮修复降低了观察误读风险：已执行/交易所未确认路径不会再被归为 `unknown`。
- 当前窗口没有新增拒单；仍有 ZETA 快亏旧样本，因此继续遵守 Batch H 观察规则，重点看未来 `fast_loss_without_strong_exit_count`、弱证据执行、loss re-entry 和真实手续费后收益。
- 不能因为 `unknown` 消失就认为策略闭环完成；这只是诊断口径更清楚。

回滚点：
- 代码层可回滚 `scripts/inspect_online_strategy_health.py` 与 `tests/test_inspect_online_strategy_health.py`；本批无 DB 迁移、无历史覆盖、无服务重启、无真实交易参数放宽。

---

## 四十四、Batch H 补充记录：持仓未满时 market 预算下限修正（2026-06-23）

触发原因：第 38-41 节已经证明候选链不是没有全市场扫描，而是全市场 scan 120、feature fetch 12 之后，动态 `market_symbol_budget` 经常被 `strategy_learning` 压到 2，导致 AI 深度 market 分析面长期偏窄。用户同时指出不开仓、小单和重复候选问题没有被根治，因此本批不能继续停留在诊断层，必须修正“低风险且组合持仓未满时仍只分析 2 个 market symbol”的预算瓶颈。

本次修复范围：
- `services/analysis_budget.py`：低风险、持仓组合未满且 market roster 需要补齐时，`market_symbol_limit` 至少保持 `roster_fill_market_symbol_min`，运行时策略画像不得把该下限压低到配置地板以下。
- `services/strategy_learning.py`：运行时 profile 生成与归一化时保留 `PORTFOLIO_ROSTER_FILL_MARKET_SYMBOL_MIN` 下限，避免学习画像把候选预算长期压回 2。
- `tests/test_analysis_budget.py`、`tests/test_strategy_learning.py`：新增/更新回归测试，锁定低风险未满仓 roster fill 场景下的 market budget 下限。
- `scripts/inspect_online_strategy_health.py`：新增 `--entry-only` 只读窄输出，用于把 entry evidence、仓位、小单、执行契约与 ML readiness 放在一个短窗口里复核，避免被大 JSON 或页面状态带偏。

安全边界：
- 本批只扩大“进入 AI market 分析”的候选预算下限，不改变开仓阈值、entry evidence tier、仓位、杠杆、平仓、ML readiness、模型权重、专家路由、风控 veto 或真实下单接口。
- `market_symbol_budget` 扩大不等于允许强行开仓；候选仍必须经过 expected net、profit quality、loss probability、tail risk、entry evidence、position sizing、风控和交易执行契约。
- 后续 AI 不得把“预算从 2 修到 6/8”写成收益闭环完成；它只证明候选覆盖瓶颈被缓解，真实收益仍要由新 entry 样本和已平仓 PnL 证明。

本地验证：
- `pytest tests/test_analysis_budget.py tests/test_strategy_learning.py tests/test_decision_repo_sanitization.py tests/test_decision_persistence_service.py tests/test_inspect_online_strategy_health.py tests/test_market_auto_entry_processor.py tests/test_trading_service_boundaries.py tests/test_no_mojibake_source.py -q`：216 passed。
- `ruff check` 覆盖本批 touched files：no issues。
- `black --check` 覆盖本批 touched files：通过。
- `python -m py_compile` 覆盖本批 touched Python files：通过。

线上同步与复查：
- `python scripts/sync_to_online_server.py --split-services` 已同步候选预算、策略学习与策略健康脚本相关改动，并重启 split services；三项服务均 active，Dashboard 返回 `302`。
- 部署后 `python scripts/inspect_online_strategy_health.py --minutes 30 --market-symbol-only`（`2026-06-23T14:20:24Z`）显示：77 decisions、26 market decisions、0 market entry decisions、0 orders、0 failed/rejected、open_positions 6、交易执行契约 `ok`。
- 同一窗口 candidate funnel：`scan_symbol_count` median 120、`feature_fetch_requested_count` median 12、`feature_valid_count` median 12、`market_symbol_budget` min 6、median 8、max 8，`rank_selected_count` median 5、max 8。
- 最新 funnel 显示 `market_limit_policy=position_first_low_risk_underfilled`，`configured_market_symbol_limit=8`、`selected_market_symbol_limit=8`、`target_position_groups=12`、`roster_underfilled=true`。这说明此前预算被压到 2 的问题已经被修正到配置下限。

当前结论：
- 候选覆盖瓶颈已经从“每轮通常只送 2 个 market symbol 给 AI”改善为“低风险未满持仓时预算 6-8，窗口 median 8”。这能直接回应用户关于分析交易对重复、全市场筛选不充分的担心。
- 但部署后 15/30 分钟窗口仍没有新的 market entry 样本，因此还不能证明“更多 market 分析”已经转化为高质量开仓；下一步必须继续观察新的 entry evidence、skip_kind、小单原因和 ML readiness。
- 当前 0 订单仍不是下单链路丢单：交易执行契约 `ok`、failed/rejected 为 0、fast loss 为 0；主要剩余瓶颈继续指向 ML degraded、证据强度不足和候选费后收益质量。

后续 AI 防偏要求：
- 解释候选重复时必须先检查 `market_symbol_budget` 是否仍低于 roster fill 下限；如果窗口 median 已到 8，不得继续把问题归因于“每轮只有 2 个 market 分析”。
- 若 market budget 已扩大但 entry 仍少，必须转向 entry evidence、expected net 组件、ML readiness、profit quality、loss probability、tail risk 和仓位 sizing，不得继续只改候选预算。
- 不得为了证明改动有效而降低 evidence 阈值、放大仓位、提高杠杆或硬改 ML ready。

回滚点：
- 代码层可回滚 `services/analysis_budget.py`、`services/strategy_learning.py`、`scripts/inspect_online_strategy_health.py`、`tests/test_analysis_budget.py`、`tests/test_strategy_learning.py`、`tests/test_inspect_online_strategy_health.py`；本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽。

---

## 四十五、Batch H 补充记录：未执行决策终态持久化与小单证据口径（2026-06-23）

触发原因：线上观察出现两类容易让后续 AI 走偏的问题：第一，PLTR/USDT 弱证据候选停在 `ai_analysis:completed`，健康脚本只能归为 `unknown`，容易被误判成下单链路丢单；第二，AI16Z/USDT 已执行订单名义金额约 13.5U，用户看到“小单、不赚钱、平仓快”时需要能追到 sizing 的真实原因，而不是只给表面页面修复。

本次修复范围：
- `db/repositories/decision_repo.py`：新增 `finalize_unresolved_decisions()`，只对未执行且没有终态的决策补写 `risk_check:skipped`；不会覆盖已执行订单、已有终态或真实交易结果。
- `services/decision_persistence_service.py`：新增同名服务方法，作为策略服务的持久化入口。
- `services/trading_service.py`：每轮跟踪 `round_decisions`，在正常结束与异常路径中调用 `_finalize_unresolved_decision_states()`，并保持在 `_fill_missing_decision_reasons()` 之后执行，避免原因补全与终态补全互相覆盖。
- `scripts/inspect_online_strategy_health.py`：`--entry-only` 输出小单相关证据，包括 evidence tier、effective score、quality tier、notional floor、low payoff、probe cap、expected net、profit quality、loss probability、tail risk 和执行契约。
- `tests/test_decision_repo_sanitization.py`、`tests/test_decision_persistence_service.py`、`tests/test_inspect_online_strategy_health.py`、`tests/test_trading_service_boundaries.py`：新增/更新回归测试，锁定未执行决策补终态、已执行不覆盖、健康脚本 entry-only 契约和交易服务边界。

安全边界：
- 本批只补“未执行决策终态”和“只读小单解释”，不改变任何开仓阈值、证据 tier、仓位、杠杆、平仓、ML readiness、模型权重、专家路由、风控 veto 或真实交易执行逻辑。
- `risk_check:skipped` 只表示该轮没有进入执行，不得被用来掩盖真实 rejected/failed orders；已执行与已有终态不允许被覆盖。
- 小单如果由 `low_payoff_quality=true`、`quality_tier=exploration` 或 `strategy_probe_cap_applied=true` 触发，后续 AI 不得直接放大仓位；必须先证明候选质量、费后收益、风险和 ML readiness 达标。

本地验证：
- `pytest tests/test_analysis_budget.py tests/test_strategy_learning.py tests/test_decision_repo_sanitization.py tests/test_decision_persistence_service.py tests/test_inspect_online_strategy_health.py tests/test_market_auto_entry_processor.py tests/test_trading_service_boundaries.py tests/test_no_mojibake_source.py -q`：216 passed。
- `ruff check` 覆盖本批 touched files：no issues。
- `black --check` 覆盖本批 touched files：通过。
- `python -m py_compile` 覆盖本批 touched Python files：通过。

线上同步与复查：
- `python scripts/sync_to_online_server.py --split-services` 已同步 `db/repositories/decision_repo.py`、`services/decision_persistence_service.py`、`services/trading_service.py`、`scripts/inspect_online_strategy_health.py` 并重启 split services；三项服务 active，Dashboard 返回 `302`。
- 部署后 `python scripts/inspect_online_strategy_health.py --minutes 15 --entry-only`（`2026-06-23T14:20:24Z`）显示：50 decisions、15 market decisions、0 market entry decisions、0 orders、0 failed/rejected、open_positions 6、fast_loss_close_under_15m 0，交易执行契约 `ok`。
- 同一窗口没有新的 market entry 样本，因此尚不能用新样本证明 `unknown` 已完全消失；60m 窗口里的 PLTR/USDT `unknown` 是部署前旧样本，不能当作本次修复失败证据。
- 已执行小单复盘：AI16Z/USDT short 约 13.5U 名义金额是受 `low_payoff_quality=true`、`quality_tier=exploration`、`strategy_probe_cap_applied=true` 等质量/探索仓位限制影响，不是杠杆展示字段或下单接口随意缩小。

当前结论：
- 诊断口径已补齐：未执行弱证据候选不应再长期停在 `ai_analysis:completed` 并显示 `unknown`；新样本仍需继续等待线上 market entry 验证。
- 小单问题不能靠直接放大下单金额解决。当前小单是质量与探索仓位保护触发，真实根因仍是候选证据弱、profit quality 边缘、loss probability/tail risk 偏高和 ML degraded。
- 当前核心剩余目标不变：提升候选质量与 ML readiness，证明新 entry 的证据强度和费后收益质量改善，再逐步允许更正常的 sizing；不能用页面修复、诊断字段或阈值放宽替代盈利闭环。

后续 AI 防偏要求：
- 每次看到 `unknown` 必须先看样本时间是否早于本节部署时间；部署前旧样本不能证明新终态修复失败。
- 每次看到小单必须输出 sizing 原因链：`quality_tier`、`position_size_pct`、`notional_floor_blocked`、`low_payoff_quality`、`strategy_probe_cap_applied`、expected net、profit quality、loss probability、tail risk、ML readiness。
- 若 `--entry-only` 在部署后出现新的 `unknown`，才允许回到终态持久化链路排查；否则下一步应继续 ML 样本质量、候选评分和 evidence 组件根因。

回滚点：
- 代码层可回滚 `db/repositories/decision_repo.py`、`services/decision_persistence_service.py`、`services/trading_service.py`、`scripts/inspect_online_strategy_health.py`、`tests/test_decision_repo_sanitization.py`、`tests/test_decision_persistence_service.py`、`tests/test_inspect_online_strategy_health.py`、`tests/test_trading_service_boundaries.py`；本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽。

---

## 四十二、Batch F/H 补充记录：系统巡检问题台账分层修复（2026-06-23）

触发原因：用户指出系统巡检页显示“系统巡检接口异常”、问题台账仍有很多问题未处理。线上同环境复核后确认：`/api/system-audit/status` 后端接口可返回，当前不是 API 整体崩溃；但 issue ledger 把多类只读 shadow/观察状态误归为 `unresolved`，导致页面给人“很多硬问题没处理”的错觉。

本次修复范围：
- `web_dashboard/api/system_audit.py`：`trade_loop` 新增 `orderless_observation` 细分。交易服务心跳新鲜、有大量分析、无 failed/rejected/order 硬错误、但 2 小时 0 订单时，仍保留 warning 提醒，但台账归为观察项，而不是未修复硬故障。
- `web_dashboard/api/system_audit.py`：issue ledger 分层新增只读观察规则：`model_expert_health`、`model_expert_competition`、`model_dynamic_routing`、`crypto_feature_coverage`、`shadow_missed_opportunity` 在 audit_only、无 live mutation、无 bypass、无 unsafe mutation 且不能 live apply 时，归类为 observing。
- `tests/test_system_audit_api.py`：新增健康心跳但 0 订单的 trade_loop 观察态测试；新增 shadow-only governance warning 归 observing 的台账测试。

安全边界：
- 本批不隐藏 warning、不把状态改成 ok、不降低任何交易风控，只修正“问题台账状态标签”的语义。
- `observing` 仍会显示在系统巡检中，表示需要继续观察或补数据源；它不等于功能已收益改善，也不允许跳过后续批次。
- `unresolved=0` 只能说明当前巡检没有未处理硬故障；不能说明策略已经能稳定盈利或应该强行开仓。

本地验证：
- `pytest tests/test_system_audit_api.py -q`：29 passed。
- `pytest tests/test_dashboard_main_ui_contract.py -q`：42 passed。
- `ruff check web_dashboard/api/system_audit.py tests/test_system_audit_api.py`：no issues。
- `black --check web_dashboard/api/system_audit.py tests/test_system_audit_api.py`：通过。

线上同步与复查：
- `python scripts/sync_to_online_server.py --split-services` 已同步 `web_dashboard/api/system_audit.py` 并重启 split services；`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard 返回 `302`。
- 同环境直接调用 `collect_system_audit_status(record_history=False, source='codex_probe')`（`2026-06-23T13:06:41Z`）显示：status=`warning`、cards 16、critical 0、warning 10、ok 6、findings 10、nodes 18。
- 问题台账复查：`fixed=6`、`unresolved=0`、`observing=10`、`total=16`；`critical_card_keys=[]`。
- 当前 observing 包括：`trade_loop`（服务冷启动/有分析无订单观察）、`okx_reconciliation`（对账 dry-run 超时观察）、`strategy_quality` 与 `strategy_closed_loop`（历史遗留当前未复现）、`model_training`（可选增强/学习模式）、`model_expert_health`（只读影子体检）、`model_expert_competition`（baseline 样本不足）、`model_dynamic_routing`（影子阶段）、`crypto_feature_coverage`（缺失特征已中性阻断）、`shadow_missed_opportunity`（保守学习）。

当前结论：
- 用户页面看到的问题台账“很多未处理”不是所有功能都坏了，而是台账分层过严，把只读 shadow/观察项标成了 unresolved。现在已修正为 0 个未修复硬问题、10 个观察项。
- “系统巡检接口异常”如果页面仍出现，应优先判断为前端请求失败/登录状态/缓存/fetch fallback，而不是后端巡检整体失败；线上后端同环境调用已经证明接口能返回完整 payload。
- 当前 warning 仍有价值：它提醒模型仍 degraded、候选 evidence 不足、特征源仍缺、baseline 仍不足、OKX 对账可能需要缩窗或异步化；这些是后续批次要继续处理的观察项，不是可以删除的噪声。

后续 AI 防偏要求：
- 回答系统巡检问题时必须区分三层：API 是否返回、card status 是否 warning/critical、issue ledger state 是 unresolved 还是 observing。
- 不得为了让页面变绿把 warning 改成 ok；只有真实硬故障消失或数据源补齐后才能转 ok。
- 对 `unresolved=0` 的解释必须带上 caveat：当前无硬故障，不代表收益闭环完成。

回滚点：
- 代码层可回滚 `web_dashboard/api/system_audit.py` 与 `tests/test_system_audit_api.py`；本批无 DB 迁移、无历史覆盖、无交易参数改动。线上回滚后需重启 `bb-dashboard.service`。

---

## 四十三、Batch F/H 补充记录：Dashboard 接口异常前端归因修复（2026-06-23）

触发原因：用户指出数据采集页、本地 ML 页和系统巡检页都有“接口异常”。复核发现后端同环境函数可返回，部分“接口异常”来自 Dashboard 登录态/401 或非 2xx 响应被前端 `fetchJSON()` 吞成 `null`，调用方 `.catch()` 不会触发，导致页面把登录过期、接口错误和空数据混在一起显示。

本次修复范围：
- `web_dashboard/static/js/dashboard.js`：`fetchJSON()` 现在先解析 API JSON；遇到 401 时提取后端错误文案、跳转登录并 `throw Error`；遇到非 2xx 时 `throw Error(apiErrorText(...))`；网络异常也继续抛出。
- `tests/test_dashboard_main_ui_contract.py`：新增契约测试，锁定 `fetchJSON()` 不再 `return null` 吞错，确保本地 ML、数据采集、系统巡检等页面已有 `.catch()` fallback 能真正生效。

安全边界：
- 本批只改变前端错误传播，不改变任何后端业务接口、训练逻辑、交易策略、风控、仓位、杠杆或 ML readiness。
- 401 应解释为登录态问题，不得直接写成数据采集、本地 ML 或系统巡检业务接口坏。
- 本地 ML `degraded` 是模型 readiness 状态，不是接口异常；只有请求失败/401/非 2xx 才是接口层异常。

本地验证：
- `pytest tests/test_dashboard_main_ui_contract.py tests/test_data_collection_api.py tests/test_model_server_config.py tests/test_system_audit_api.py tests/test_inspect_online_strategy_health.py tests/test_no_mojibake_source.py -q`：112 passed。
- `pytest tests/test_ml_signal_training_quality.py tests/test_model_artifact_safety.py -q`：16 passed。
- `ruff check web_dashboard/api/system_audit.py tests/test_system_audit_api.py scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py tests/test_dashboard_main_ui_contract.py`：no issues。
- `black --check web_dashboard/api/system_audit.py tests/test_system_audit_api.py scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py tests/test_dashboard_main_ui_contract.py`：通过。

线上同步与复查：
- `python scripts/sync_to_online_server.py --split-services` 已同步 `web_dashboard/static/js/dashboard.js` 并重启 split services；`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard 返回 `302`。
- 后端同环境直接调用 `get_data_collection_status()` 和 `get_ml_signal_status()` 均可返回；当前数据采集特征覆盖为 warning（缺失 5、过期 0），本地 ML 为 `degraded` 且 `allow_live_position_influence=false`，不是接口整体不可用。

当前结论：
- 如果用户页面同时出现系统巡检、数据采集、本地 ML 接口异常，优先检查登录态/401 和前端 fetch fallback；不要直接判定三个业务接口全部坏。
- 页面修复后，401 会引导重新登录，非 2xx 会显示具体 API 错误，业务层 `warning/degraded/observing` 会继续作为状态展示而不是泛化接口异常。

后续 AI 防偏要求：
- 处理 Dashboard 页面异常时必须先分三类：登录态/401、HTTP 非 2xx、业务 payload 的 `warning/degraded/observing`。
- 不得把 ML degraded、feature missing 或 issue ledger observing 写成“接口异常”；它们是业务状态。

回滚点：
- 代码层可回滚 `web_dashboard/static/js/dashboard.js` 与 `tests/test_dashboard_main_ui_contract.py`；本批无 DB 迁移、无历史覆盖、无交易参数改动。线上回滚后需重启 `bb-dashboard.service`。

---

## 二十八、Batch H 补充记录：ML degraded 样本组成诊断补强（2026-06-23）

触发原因：Batch H 持续观察显示 `local_ml_readiness` 仍为 `degraded`，阻塞项包括 `long_pr_auc_below_threshold`、`short_pr_auc_below_threshold`、`short_top_return_below_threshold` 和 `dirty_sample_ratio_high`。只看这些阻塞码容易让后续 AI 误以为可以通过硬改 readiness 或放宽 dirty ratio 让 ML 介入。只读核查线上模型元数据后确认：20,000 条训练窗口中 hold 样本 17,970 条，long 678 条，short 1,352 条；低置信度/hold 观察降权占大头，同时 PR-AUC 也确实低，short 高分组收益为负。因此当前不应启用 ML 实盘影响，而应先让观察报告解释模型为什么不能用。

本次修复范围：
- `scripts/inspect_online_strategy_health.py`：`local_ml_readiness_summary()` 新增 `quality_by_kind`、`quality_top_actions`、`quality_top_timeframes`，让 15m/120m 健康摘要直接暴露训练样本由哪些 action/timeframe 主导。
- `tests/test_inspect_online_strategy_health.py`：新增模板契约断言，先确认健康脚本缺少这些字段时红测失败，再锁定观察脚本必须输出 ML 样本组成。

安全边界：
- 本批只补只读诊断字段，不改变 ML 训练阈值、readiness 状态机、模型权重、专家路由、开仓阈值、仓位、杠杆、平仓或风控 veto。
- `dirty_sample_ratio_high` 不能被简单理解为阈值太严；当前 PR-AUC 未达标且 short top return 为负，必须继续保持 `allow_live_position_influence=false`。
- hold 样本主导说明训练数据结构仍不利于实盘收益判断；后续如果要修训练链路，必须先做 TDD 与离线验证，不能直接让 ML 参与真实仓位。

本地验证：
- TDD 红灯：`pytest tests/test_inspect_online_strategy_health.py::test_strategy_health_report_exposes_local_ml_readiness_summary -q` 在修复前失败，缺少 `quality_by_kind`。
- TDD 绿灯：同一测试修复后通过。
- `pytest tests/test_inspect_online_strategy_health.py -q`：11 passed。
- `ruff check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：0 issues。
- `black --check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：通过。
- `git diff --check`：通过。
- `python scripts/security_secret_scan.py --fail-on high .`：扫描 514 files OK。

线上复查：
- `python scripts/sync_to_online_server.py --split-services --skip-restart` 已同步 `scripts/inspect_online_strategy_health.py`，未重启交易服务。
- 同步后 15m 观察窗口（`2026-06-23T03:48:18Z`）：44 decisions、0 entry_decisions、0 orders、0 rejected_orders、0 fast_loss_close_under_15m、open_positions 2。
- `local_ml_readiness` 仍为 `degraded`，`allow_live_position_influence=false`；metrics 为 sample_count 19,982、test_count 4,996、dirty_sample_ratio 0.8988、long_pr_auc 0.3972、short_pr_auc 0.3765、top_short_avg_return_pct -0.0312。
- 新增诊断字段显示 `quality_top_actions` 为 `shadow:hold=17970`、`shadow:short=1352`、`shadow:long=678`；`quality_top_reasons` 为 `shadow:very_low_decision_confidence=17970`、`shadow:hold_observation_downweighted=10603`、`shadow:hold_missed_opportunity_downweighted=7367`。

当前结论：
- 当前 ML degraded 是真实模型/训练样本结构问题，不是 dashboard 或 readiness 口径误报。
- 这轮补强让后续 AI 能看到 ML 为什么不能用：hold/低置信度样本主导且收益排序指标未达标。
- 后续若要根治 ML，需要回到训练样本构成、候选样本平衡、收益标签和离线评估，而不是硬改 ready、降低 PR-AUC 门槛或让 degraded ML 放大仓位。

回滚点：
- 代码层可回滚 `scripts/inspect_online_strategy_health.py` 与 `tests/test_inspect_online_strategy_health.py`；本批无 DB 迁移、无历史覆盖、无服务重启、无真实交易参数放宽。

---

## 二十九、Batch H 补充记录：ML 训练样本选择与加载性能修正（2026-06-23）

触发原因：Batch H 继续观察后确认，`local_ml_readiness` degraded 不是页面误报，而是训练窗口真实被 hold/低置信观察样本主导。只读分布显示最新 20,000 条 completed shadow 样本中 `decision_action` 为 hold 17,981、short 1,353、long 666；但历史 completed 样本里 `best_action` 为 long/short 的交易结果并不少。若继续只取最近 20,000 条完整 ORM 样本，训练会长期偏向“观察 hold”，同时线上装载 20,000 个完整 `ShadowBacktest` ORM 对象耗时约 150.907 秒，存在自动训练卡住风险。

本次修复范围：
- `services/ml_signal_service.py`：新增 `select_shadow_training_rows()`，在保持最近样本基础上，优先保留 `decision_action in {long, short}` 与 `best_action in {long, short}` 的训练样本；当前 20,000 窗口目标至少保留 25% 原始非 hold 决策样本，并保留 best-action 交易样本。
- `services/ml_signal_service.py`：`load_shadow_training_rows()` 改为只读会话，并只查询训练需要的列，返回轻量 `ShadowTrainingRow`，避免装载完整 ORM 实体、无关 raw 字段和大对象。
- `tests/test_ml_signal_training_quality.py`：新增选择器契约与临时 SQLite DB 入口测试，锁定训练入口必须同时包含 recent、decision trade 与 best-action trade 样本，并且返回轻量训练 row 而不是 `ShadowBacktest` ORM 实体。

安全边界：
- 本批不改变 ML readiness 阈值、不硬改 `ready`、不改变模型权重、不改变专家路由、不改变开仓/仓位/杠杆/平仓/风控 veto。
- 本批没有重训模型、没有替换现有模型文件、没有让 degraded ML 参与真实仓位放大；未来训练后仍必须由 PR-AUC、收益分层、dirty ratio、样本数和模型年龄共同决定 readiness。
- 本批不删除、不覆盖、不修历史 shadow 样本，只改变未来训练读取窗口和读取方式。
- 如果后续重训后指标仍不达标，ML 必须继续保持 `degraded/learning_only`，不得把样本平衡本身解释为模型已可用于实盘。

本地验证：
- 红灯契约：新增 DB 入口与轻量 row 断言后，旧 loader 路径无法通过，暴露训练入口仍停在旧会话/ORM 装载思路；修复后同一测试通过。
- `pytest tests/test_ml_signal_training_quality.py -q`：6 passed。
- `pytest tests/test_trading_service_boundaries.py::test_ml_signal_auto_train_quarantines_before_training tests/test_trading_service_boundaries.py::test_ml_signal_auto_train_uses_completed_cursor_for_new_samples -q`：2 passed。
- `ruff check services/ml_signal_service.py tests/test_ml_signal_training_quality.py tests/test_trading_service_boundaries.py`：0 issues。
- `black --check services/ml_signal_service.py tests/test_ml_signal_training_quality.py tests/test_trading_service_boundaries.py`：通过。

线上只读验证：
- 同步前性能探针显示：`COUNT` completed+returns 138,182 条约 0.275s；最新 20,000 条 `decision_action` 聚合约 0.312s；轻量 recent 20,000 列装载约 0.282s；只查询训练必需列 recent 20,000 约 2.679s；完整 ORM 装载 recent 20,000 约 150.907s。根因是完整 ORM/大对象装载，而不是数据库计数或聚合慢。
- `python scripts/sync_to_online_server.py --split-services --skip-restart` 已先同步 `services/ml_signal_service.py`，未重启交易服务，用于正式 loader dry-run。
- 线上正式 `load_shadow_training_rows(limit=20000)` 只读 dry-run 耗时约 5.117s，返回 row type 为 `ShadowTrainingRow`；样本组成变为 `decision_action`: hold 15,000、long 1,950、short 3,050，非 hold 决策样本 5,000；`best_action`: hold 7,986、long 5,875、short 6,139，best-action 交易样本 12,014。
- dry-run 未重训、未写 DB、未改模型 artifact、未改变线上交易参数。
- 正式同步重启：`python scripts/sync_to_online_server.py --split-services` 后 `bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard `302`；由于代码文件已在 dry-run 前同步，本次正式同步只上传总控文档并重启服务，让新 loader 进入长期服务进程。
- 重启后 15m 健康窗口：32 decisions，全部 hold；entry/orders/rejected/fast_loss 均为 0，open_positions 3；`local_ml_readiness` 仍为 degraded，`allow_live_position_influence=false`。
- 重启后 120m 健康窗口：316 decisions，9 entry decisions，1 filled order，0 failed/rejected，0 fast_loss_close_under_15m，open_positions 3；该 1 笔成交发生在本批正式同步前，不能作为本批新执行样本。
- 重启后系统巡检（`record_history=False`）：overall `warning`，`critical_cards=[]`，cards 16；`trade_execution_contract` 为 ok，current_summary 中 `contract_violation_count=0`、`weak_evidence_executed_count=0`、`fast_loss_without_strong_exit_count=0`、`reentry_without_strong_unlock_count=0`；`runtime_text_integrity` 为 ok。`model_training` 仍为 warning，但详情是学习观察、可选外部事件源未配置和运行探针超时，不是本批引入的交易执行风险。

当前结论：
- 这轮解决的是 ML 训练入口的样本窗口偏斜和加载性能问题：未来训练不会再被最近 hold 样本完全淹没，也不会因完整 ORM 装载 20,000 条而接近或超过超时。
- 这不等于 ML 已经 ready。当前模型是否能参与真实仓位，仍必须看下一次训练后的 readiness 报告和线上观察指标。
- 后续继续推进时，优先观察下一次训练的 `quality_top_actions`、PR-AUC、top/bottom return、dirty ratio、`allow_live_position_influence`，不得把样本平衡当成放宽开仓或放大仓位的理由。

回滚点：
- 代码层可回滚 `services/ml_signal_service.py` 与 `tests/test_ml_signal_training_quality.py`；本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽。

---

## 三十、Batch H 补充记录：ML 安全 dry-run 训练评估闸门（2026-06-23）

触发原因：上一轮已修复 ML 训练样本选择与加载性能，但如果后续 AI 直接运行正式训练脚本，仍可能把不达标模型写入 `data/ml_signal/winrate_model.joblib`，或在未明确验证前隔离训练样本。为避免“样本平衡 = 模型已 ready”的误读，本轮新增不会写模型 artifact、不会隔离样本的训练评估入口，用来先看真实指标，再决定是否进入正式模型替换流程。

本次修复范围：
- `services/ml_signal_service.py`：`train_from_frame()` 新增 `persist_artifact` 参数；默认仍保持原正式训练行为，只有显式 `persist_artifact=False` 时才只返回训练元数据，不写 `MODEL_PATH` 与 `METADATA_PATH`。
- `services/ml_signal_service.py`：训练元数据新增 `training_run_mode` 与 `artifact_persisted`，让 dry-run 输出可被机器和人直接审计。
- `scripts/train_ml_signal_model.py`：新增 `run_training()` 与 CLI 参数 `--dry-run`；dry-run 模式强制跳过 quarantine，返回 `dry_run_no_quarantine_writes`，并调用 `train_from_frame(..., persist_artifact=False)`。
- `tests/test_ml_signal_training_quality.py`：新增 dry-run 契约测试，锁定 dry-run 不得写模型文件、不得写 metadata、不得调用 quarantine、必须把 `persist_artifact=False` 传入训练函数。

安全边界：
- `--dry-run` 只能用于离线评估，不代表模型被部署，也不代表 readiness 变成 ready。
- dry-run 指标通过时，也不得自动启用 ML 实盘影响；必须另走正式训练、artifact 替换、readiness 复查、线上观察和停止规则检查。
- dry-run 指标不通过时，必须继续保持 `degraded/learning_only` 与 `allow_live_position_influence=false`，不得硬改 ready、降低 PR-AUC 门槛、放宽 dirty ratio、强开 probe、放大仓位或绕过风控 veto。
- 本批不改变开仓阈值、仓位、杠杆、平仓、专家路由、模型权重、风险 veto 或真实交易执行逻辑。

本地验证：
- TDD 红灯：新增 dry-run 契约测试后，旧代码因 `train_from_frame()` 不支持 `persist_artifact`、训练脚本缺少 `run_training()` 而失败。
- TDD 绿灯：实现 `persist_artifact`、`run_training()` 与 `--dry-run` 后，同一测试通过。
- `python -m pytest tests/test_ml_signal_training_quality.py -q`：8 passed。
- `python -m pytest tests/test_trading_service_boundaries.py::test_ml_signal_auto_train_quarantines_before_training tests/test_trading_service_boundaries.py::test_ml_signal_auto_train_uses_completed_cursor_for_new_samples -q`：2 passed。
- `ruff check services/ml_signal_service.py scripts/train_ml_signal_model.py tests/test_ml_signal_training_quality.py tests/test_trading_service_boundaries.py`：0 issues。
- `black --check services/ml_signal_service.py scripts/train_ml_signal_model.py tests/test_ml_signal_training_quality.py tests/test_trading_service_boundaries.py`：通过。

线上只读 dry-run 评估：
- 执行方式：以 `bb-dashboard.service` 的运行环境启动，并降权为 OS 用户 `bb` 执行 `/data/bb/app/.venv/bin/python scripts/train_ml_signal_model.py --dry-run --skip-quarantine --limit 20000`。
- 模型 artifact 前后校验一致：`data/ml_signal/winrate_model.joblib` size `10652409`、mtime_ns `1782192227930000000`；`data/ml_signal/winrate_model_metadata.json` size `6838`、mtime_ns `1782192227958000000`。dry-run 前后完全一致，证明本轮未替换线上模型 artifact。
- dry-run 输出明确为 `training_run_mode=dry_run`、`artifact_persisted=false`、`training_quarantine.reason=dry_run_no_quarantine_writes`；`loaded_row_count=20000`、`frame_sample_count=19971`、`completed_shadow_sample_count=138382`。
- 样本质量：total 20,000、included 4,859、downweighted 15,112、excluded 29、effective_weight_ratio 0.6048；`decision_action` 为 hold 15,000、short 3,050、long 1,950。主要降权原因仍是 `shadow:very_low_decision_confidence=15000`、`shadow:hold_missed_opportunity_downweighted=10000`、`shadow:hold_observation_downweighted=5000`。
- dry-run 指标：train_count 14,978、test_count 4,993；`long_pr_auc=0.3570802373417013`、`short_pr_auc=0.3726301250780891`；`top_long_avg_return_pct=0.3144083738290642`、`top_short_avg_return_pct=-0.016619942578254214`；`bottom_long_avg_return_pct=-0.22171420536348094`、`bottom_short_avg_return_pct=-0.340247292118988`。

线上同步与复查：
- `python scripts/sync_to_online_server.py --split-services` 已同步总控文档；本轮代码文件此前已同步到线上，最终同步实际上传 1 个 changed file。同步后 `bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均为 active，Dashboard 返回 `302`。
- 同步后 15m 健康摘要（`2026-06-23T06:09:33Z`）：39 decisions、15 market decisions、0 entry decisions、0 orders、0 rejected orders、0 fast_loss_close_under_15m、open_positions 3；没有触发新增执行风险。
- 同步后 120m 健康摘要（`2026-06-23T06:09:34Z`）：325 decisions、11 entry decisions、9 market_entry_decisions、1 filled order、0 failed/rejected orders、0 fast_loss_close_under_15m、open_positions 3；该 1 笔 filled 属于窗口内历史样本，不证明本批 dry-run 闸门启用了 ML 或制造了新交易。
- 同步后 `local_ml_readiness` 仍为 `degraded` 且 `allow_live_position_influence=false`；阻塞项仍为 `long_pr_auc_below_threshold`、`short_pr_auc_below_threshold`、`short_top_return_below_threshold`、`dirty_sample_ratio_high`。当前 artifact 状态仍在保护系统不让未达标 ML 影响实盘。
- 同步后系统巡检（`record_history=False`）：overall `warning`，但 `critical_cards=[]`；`trade_execution_contract` 为 ok，current_summary 中 `contract_violation_count=0`、`weak_evidence_executed_count=0`、`negative_expected_executed_count=0`、`fast_loss_without_strong_exit_count=0`、`reentry_without_strong_unlock_count=0`；`runtime_text_integrity` 为 ok，suspected_records 0、suspected_fields 0。`model_training` 仍为 warning，原因是可选增强数据源、运行探针或学习观察，不是本批新增交易执行风险。

当前结论：
- 本轮新增的是“先评估、后决定是否替换模型”的安全闸门，解决后续 AI 无法安全查看新训练窗口指标的问题。
- 平衡窗口改善了训练组成与加载速度，但 dry-run 指标仍不能证明 ML 已可实盘影响：long/short PR-AUC 仍偏低，且 short top return 仍为负。
- 当前不能启用 ML live influence，不能替换线上 artifact，不能把样本平衡当作开仓放宽依据。后续若要继续根治 ML，应先定位 short 高分组收益为负的原因，包括收益标签、特征有效性、样本时间分布、成本/滑点、side imbalance、低置信 hold 样本降权策略和候选生成质量。

回滚点：
- 代码层可回滚 `services/ml_signal_service.py`、`scripts/train_ml_signal_model.py` 与 `tests/test_ml_signal_training_quality.py`；本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽。若回滚线上运行代码，需要同步后重启服务以刷新 Python 进程导入。

---

## 三十一、Batch H 补充记录：ML score bucket 诊断补强（2026-06-23）

触发原因：上一轮 dry-run 已证明新训练窗口仍未达到 ML 实盘影响条件，尤其 `short top return` 仍为负，PR-AUC 仍偏低。但只看总指标会让后续 AI 容易走偏：可能误以为继续平衡样本、硬调 readiness、降低阈值或直接替换 artifact 就能解决。为避免这种误判，本轮只在训练元数据中增加高/低分桶诊断，用真实 test bucket 解释哪些样本段拖累收益排序。

本次修复范围：
- `services/ml_signal_service.py`：`build_training_frame()` 保留只读诊断上下文字段 `decision_action`、`best_action`、`missed_opportunity`，不把它们加入 `FEATURE_KEYS`，不作为模型训练特征。
- `services/ml_signal_service.py`：训练 metadata 新增 `score_bucket_diagnostics`，分别输出 long/short 的 top/bottom 分桶摘要，包括 count、avg_model_score、avg_return_pct、win_rate、avg_sample_weight、decision_action counts、best_action counts、horizon counts、data_quality_status counts 和 top_quality_reasons。
- `tests/test_ml_signal_training_quality.py`：新增分桶诊断契约测试，以及 `build_training_frame()` 必须保留诊断上下文的红绿测试，防止后续 AI 只保留空结构或让线上诊断退化成 `unknown`。

安全边界：
- 本批只增加只读/offline 诊断；不改变 `FEATURE_KEYS`，不改变模型选择器、训练行选择、收益标签、样本权重、quarantine、readiness 阈值或 live trading 行为。
- 本批不硬改 `ready`，不降低 PR-AUC 门槛，不放宽 dirty ratio，不放宽开仓阈值，不改变仓位/杠杆/平仓/风控 veto/专家路由/模型权重。
- 分桶诊断只能用于解释为什么指标不达标，不能被当作启用 ML live influence、替换 artifact 或放大交易的理由。
- 如果分桶诊断显示高分组仍被 hold/低置信/降权样本主导，后续应定位样本结构、候选生成、标签、特征和成本/滑点，而不是把观察样本强行解释为可交易信号。

本地验证：
- TDD 红灯：新增 `test_build_training_frame_preserves_diagnostic_sample_context` 后，旧 frame 缺少 `decision_action`，测试因 `KeyError: 'decision_action'` 失败。
- TDD 绿灯：补充 frame 诊断上下文字段后，同一测试通过；`test_train_from_frame_reports_score_bucket_diagnostic_segments` 也通过。
- `python -m pytest tests/test_ml_signal_training_quality.py -q`：10 passed。
- `python -m pytest tests/test_trading_service_boundaries.py -q`：119 passed。
- `ruff check services/ml_signal_service.py scripts/train_ml_signal_model.py tests/test_ml_signal_training_quality.py tests/test_trading_service_boundaries.py`：0 issues。
- `black --check services/ml_signal_service.py scripts/train_ml_signal_model.py tests/test_ml_signal_training_quality.py tests/test_trading_service_boundaries.py`：通过。
- `git diff --check`：通过。
- `python scripts/security_secret_scan.py --fail-on high .`：scanned 514 files OK。

线上只读 dry-run 评估：
- `python scripts/sync_to_online_server.py --split-services --skip-restart` 已同步 `services/ml_signal_service.py`，未重启交易服务；用于只读 dry-run 探针。
- 远端探针以 `bb-dashboard.service` 环境启动，叠加 `/data/bb/app/.env` 与 `/etc/bb/bb-runtime.env`，降权为 OS 用户 `bb` 执行 `/data/bb/app/.venv/bin/python scripts/train_ml_signal_model.py --dry-run --skip-quarantine --limit 20000`。
- dry-run 前后 artifact 完全一致：`data/ml_signal/winrate_model.joblib` size `10652409`、mtime_ns `1782192227930000000`；`data/ml_signal/winrate_model_metadata.json` size `6838`、mtime_ns `1782192227958000000`；`artifact_stats.unchanged=true`。
- dry-run 输出仍明确为 `training_run_mode=dry_run`、`artifact_persisted=false`、`training_quarantine.reason=dry_run_no_quarantine_writes`；`loaded_row_count=20000`、`frame_sample_count=19971`、`completed_shadow_sample_count=138498`。
- 样本质量仍显示 total 20,000、included 4,859、downweighted 15,112、excluded 29、effective_weight_ratio 0.6048；`decision_action` 为 hold 15,000、short 3,050、long 1,950。主要降权原因仍是 `shadow:very_low_decision_confidence=15000`、`shadow:hold_missed_opportunity_downweighted=10000`、`shadow:hold_observation_downweighted=5000`。
- dry-run 总指标仍不达标：`long_pr_auc=0.3486757989761258`、`short_pr_auc=0.379175314541324`；`top_long_avg_return_pct=0.1379226009542219`、`bottom_long_avg_return_pct=-0.18529265356726526`；`top_short_avg_return_pct=-0.0562176316920748`、`bottom_short_avg_return_pct=-0.34500060718564035`。
- 新增分桶诊断显示：long top 桶 998 条，`avg_return_pct=0.1379226009542219`、`win_rate=0.3517034068136273`、`avg_sample_weight=0.4359819639278556`，其中 `action_counts` 为 hold 888、short 81、long 29，`data_quality_status_counts.downweighted=888`，主要原因是 `very_low_decision_confidence=888`、`hold_observation_downweighted=559`、`hold_missed_opportunity_downweighted=329`。
- short top 桶 998 条，`avg_return_pct=-0.0562176316920748`、`win_rate=0.36472945891783565`、`avg_sample_weight=0.43544088176352697`，其中 `action_counts` 为 hold 944、long 39、short 15，`data_quality_status_counts.downweighted=944`，主要原因是 `very_low_decision_confidence=944`、`hold_observation_downweighted=502`、`hold_missed_opportunity_downweighted=442`。这说明 short 高分组收益为负并不是展示误报，而是 test bucket 仍被低置信 hold/降权样本主导且收益排序没有达到可交易条件。

线上同步与复查：
- `python scripts/sync_to_online_server.py --split-services` 已同步总控文档并重启 `bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service`；三项服务均 active，Dashboard 返回 `302`。
- 重启后 15m 健康摘要（`2026-06-23T07:00:53Z`）：36 decisions、10 market decisions、2 entry decisions、1 market_entry_decision、1 order、0 failed/rejected orders、1 fast_loss_close_under_15m、open_positions 6；该快亏样本为 `WLFI/USDT` short，持仓约 13.84 分钟，realized_pnl `-0.064212`，notional 约 `18.5896`。
- 重启后 120m 健康摘要（`2026-06-23T07:01:08Z`）：346 decisions、116 market decisions、15 entry decisions、12 market_entry_decisions、5 orders，全部 filled，0 failed/rejected orders、1 fast_loss_close_under_15m、open_positions 6。
- 重启后 `local_ml_readiness` 仍为 `degraded` 且 `allow_live_position_influence=false`；当前线上 artifact 的 readiness metrics 仍显示 `dirty_sample_ratio=0.7571`、`long_pr_auc=0.34560508478282215`、`short_pr_auc=0.3761940221638387`、`top_short_avg_return_pct=-0.037152709390829576`。
- 重启后系统巡检（`record_history=False`，`2026-06-23T06:59:42Z`）：overall `warning`，但 `critical_cards=[]`，cards 16、warning 9、ok 7；`trade_execution_contract` 为 `ok`，24h summary 中 `contract_violation_count=0`、`weak_evidence_executed_count=0`、`negative_expected_executed_count=0`、`fast_loss_without_strong_exit_count=0`、`reentry_without_strong_unlock_count=0`。
- 当前运行窗口契约摘要：`decision_count=36`、`executed_entry_count=0`、`contract_violation_count=0`、`weak_evidence_executed_count=0`、`negative_expected_executed_count=0`、`fast_loss_count=1`、`fast_loss_without_strong_exit_count=0`、`reentry_without_strong_unlock_count=0`。因此当前存在快亏观察风险，但没有触发“快亏且无强退出证据”的硬停规则；后续如果快亏形成簇、出现无强退出快亏、失败/拒绝订单、弱证据执行或风控绕过，必须先停止推进并定位。
- `runtime_text_integrity` 为 `ok`，扫描 809 条，suspected_records 0、suspected_fields 0；`model_training` 仍为 warning，主要来自可选增强数据源未配置与 runtime probe timeout，不是本批新增交易执行风险。

当前结论：
- 本轮解决的是“dry-run 为什么差”的可解释性问题，不是 ML ready 问题。
- 线上真实数据表明，模型分桶已经能区分部分 long 收益排序，但 short top bucket 仍为负，且 long/short top 桶都被低置信 hold/降权样本大量主导；因此当前仍不能启用 ML live influence，不能替换 artifact，不能把样本平衡当作放宽交易的证据。
- 后续更优先的根治方向是：定位为什么可交易候选样本不足、为什么高分 short bucket 被 hold/低置信样本主导、收益标签是否与成本/滑点/方向一致、现有特征是否能区分 short 盈利场景，以及是否需要在离线评估中加入更清晰的 side-aware 候选质量诊断。

回滚点：
- 代码层可回滚 `services/ml_signal_service.py` 与 `tests/test_ml_signal_training_quality.py`；本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽。若回滚线上运行代码，需要同步后重启服务以刷新 Python 进程导入。

---

这版核心就是：**不再围绕现有死框架修补，而是建立一个模型/专家/策略持续竞赛、淘汰、替换、增强的系统，最终以最懂赚钱、最懂数字货币投资的组合为准。**

新增防偏内容只服务一个目的：让后续 AI 按这个总控执行时，不会偷换目标、不乱放宽交易、不硬改状态、不造假指标、不跳过验证。

---

## 三十二、Batch H 补充记录：策略健康硬停口径防偏与精简摘要（2026-06-23）

触发原因：继续执行 Batch H 观察时，策略健康脚本原先只直接输出 `fast_loss_close_under_15m`、拒单数量和样本列表，没有把同一窗口的 `trade_execution_contract` 硬停摘要并列展示。后续 AI 容易走两种歪路：看到快亏就过度停止必要止损，或看到契约巡检 ok 就忽略拒单/快亏观察风险。因此本轮只做只读诊断口径补强，不改任何交易行为。

本次修复范围：
- `scripts/inspect_online_strategy_health.py`：接入 `TradeExecutionContractService().report(since=since, limit=600)`，在同一份健康报告里输出 `trade_execution_contract`，包含 `contract_violation_count`、`weak_evidence_executed_count`、`negative_expected_executed_count`、`fast_loss_without_strong_exit_count`、`reentry_without_strong_unlock_count`、`can_bypass_risk_controls`、`violations` 和 `fast_loss_samples`。
- `scripts/inspect_online_strategy_health.py`：新增 `json_safe()`，把契约服务返回的嵌套 `datetime` 转为 ISO 字符串，避免 120m 窗口有快亏样本时 `json.dumps()` 因 `datetime` 失败。
- `scripts/inspect_online_strategy_health.py`：新增 `--summary` 模式，并把摘要裁剪下沉到远端模板内执行，只输出停止信号、交易契约、ML readiness、拒单样本和快亏样本，避免完整报告过长被 SSH 输出上限截断后本地 JSON 解析失败。
- `tests/test_inspect_online_strategy_health.py`：新增模板契约和摘要契约测试，先确认缺字段、非 JSON 安全样本、缺少 summary-only 远端模式时测试失败，再实现后通过。

安全边界：
- 本批只改只读观察脚本和测试，不修改开仓阈值、证据 tier、仓位、杠杆、平仓、模型权重、专家路由、ML readiness、风控 veto 或真实交易执行逻辑。
- `fast_loss_close_under_15m` 是观察风险；真正的硬停口径必须同时看 `fast_loss_without_strong_exit_count`。如果后者大于 0，必须停止推进并回到 Batch E/平仓证据链定位。
- `rejected_orders` 或 `failed_orders` 仍是停止观察信号；即使 `trade_execution_contract.status=ok`，也不能把拒单窗口误读为全绿，更不能据此放大开仓。
- `trade_execution_contract.status=ok` 只说明本窗口未发现弱证据执行、负预期执行、无强退出快亏、亏损后无强解锁复开或风控绕过；不等于策略盈利闭环完成。

本地验证：
- TDD 红灯：新增 `test_strategy_health_report_exposes_trade_execution_contract_summary` 后，旧模板缺少 `TradeExecutionContractService` 导入和 `trade_execution_contract` 输出，测试失败。
- TDD 红灯：线上 120m 验证发现 `TypeError: Object of type datetime is not JSON serializable` 后，新增 `test_strategy_health_contract_samples_are_json_safe`，旧模板缺少 `json_safe()`，测试失败。
- TDD 红灯：`--summary` 首版在本地解析完整远端 JSON，遇到 SSH 输出截断后失败；新增 `test_strategy_health_remote_command_can_emit_summary_only`，旧 `_build_remote_command()` 缺少 `summary` 参数，测试失败。
- TDD 绿灯：上述测试在实现后通过。
- `pytest tests/test_inspect_online_strategy_health.py -q`：15 passed。
- `ruff check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：no issues。
- `black --check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：通过。
- `git diff --check`：通过。

线上只读复查：
- `python scripts/sync_to_online_server.py --split-services --skip-restart` 已同步 `scripts/inspect_online_strategy_health.py`，未重启交易服务。
- `python scripts/inspect_online_strategy_health.py --minutes 15 --summary`（`2026-06-23T08:44:31Z`）：50 decisions、0 orders、0 failed/rejected、0 fast_loss_close_under_15m、open_positions 6；`trade_execution_contract.status=ok`，`contract_violation_count=0`、`weak_evidence_executed_count=0`、`negative_expected_executed_count=0`、`fast_loss_without_strong_exit_count=0`、`reentry_without_strong_unlock_count=0`；`local_ml_readiness.status=degraded` 且 `allow_live_position_influence=false`。
- `python scripts/inspect_online_strategy_health.py --minutes 120 --summary`（`2026-06-23T08:44:31Z`）：408 decisions、2 orders、1 filled、1 rejected、positions_created 0、positions_closed 1、open_positions 6、`fast_loss_close_under_15m=1`。
- 120m 拒单样本为 order `2561`、decision `118360`、`TSLA/USDT`、buy、`status=rejected`、`quantity=0`、`exchange_order_id=null`；OKX raw error 为 `sCode=51155`，原因是本地合规限制导致该 pair 不可交易。该样本没有成交，不得计入收益样本。
- 120m 快亏样本为 position `1614`、`WLFI/USDT` short，持仓约 `13.844` 分钟，realized_pnl `-0.0642124`，notional `18.5896`；同窗口 `trade_execution_contract.summary.fast_loss_without_strong_exit_count=0`，因此它是快亏观察风险，不是“无强退出证据快亏”的硬停违规。
- 120m `trade_execution_contract.status=ok`，`can_bypass_risk_controls=false`，summary 中 `contract_violation_count=0`、`weak_evidence_executed_count=0`、`negative_expected_executed_count=0`、`fast_loss_count=1`、`fast_loss_without_strong_exit_count=0`、`reentry_without_strong_unlock_count=0`，`violations=[]`。

当前结论：
- 本轮解决的是策略健康观察口径防偏和输出可靠性问题，不是收益闭环完成。
- 当前 15m 窗口干净；120m 窗口仍包含 1 条 TSLA/USDT 拒单和 1 条 WLFI/USDT 快亏观察样本。拒单/快亏观察风险仍要求继续监控，不能据此推进仓位放大、阈值放宽或 ML live influence。
- 后续 AI 执行总控时必须优先看 `--summary` 输出中的 `rejected_orders`、`failed_orders`、`trade_execution_contract.summary`、`violations`、`fast_loss_samples`、`local_ml_readiness.allow_live_position_influence`，再决定是否继续下一步；不能只看完整报告里的某一个计数。

回滚点：
- 代码层可回滚 `scripts/inspect_online_strategy_health.py` 与 `tests/test_inspect_online_strategy_health.py`；本批无 DB 迁移、无历史覆盖、无服务重启、无模型 artifact 替换、无真实交易参数放宽。

---

## 三十三、Batch H 补充记录：已知不可交易 symbol 执行前拦截（2026-06-23）

触发原因：继续观察 Batch H 时，120m 摘要窗口出现 `TSLA/USDT` 拒单：order `2561`、decision `118360`、buy、`status=rejected`、`quantity=0`、`exchange_order_id=null`。OKX raw error 明确包含 `sCode=51155`，原因为本地合规限制导致该 pair 不可交易。现有 `EntrySymbolBlocklistPolicy` 已能识别 `51155`、`local compliance restrictions`、`can't trade this pair`，`ExecutionService` 也会在 entry 拒单后调用 `remember_untradable_symbol()`，`TradingService._load_untradable_symbol_blocks()` 会从近期 `AIDecision.execution_reason` 恢复不可交易 symbol；但已经生成或排队的 entry candidate 仍可能绕过 market scan 层过滤，走到交易所提交后再次拒单。因此本轮只把“已知不可交易 symbol”的记忆接入 entry execution gate，在提交交易所前拦住复发拒单。

本次修复范围：
- `services/entry_opportunity_gate.py`：`EntryOpportunityGatePolicy` 新增只读依赖 `blocked_symbol_reason`，在 suspicious symbol 检查之后、legacy evaluator 和原有 `_evaluate()` 之前返回已知不可交易原因。
- `services/trading_service.py`：实例化 `EntryOpportunityGatePolicy` 时注入 `self.blocked_symbol_reason`，复用现有 blocklist 记忆，不新增另一套 symbol 状态源。
- `tests/test_trading_service_boundaries.py`：新增 `test_entry_opportunity_gate_blocks_known_untradable_symbol_before_execution`，先锁定没有注入字段时的失败，再确认 gate 会把 `OKX 51155 local compliance restrictions` 作为执行前拦截原因返回。

安全边界：
- 本批只阻断“已经被交易所或历史执行结果证明不可交易”的 symbol 再次进入执行，不修改开仓阈值、证据 tier、仓位、杠杆、平仓、模型权重、专家路由、ML readiness 或风控 veto。
- 这不是收益策略优化，也不是让候选更容易成交；它只减少同一类合规拒单反复提交到交易所的噪声和风险。
- `blocked_symbol_reason` 必须复用 `EntrySymbolBlocklistPolicy` 的已有 TTL/归一化/错误码识别能力，不允许后续 AI 把它扩展成随意拉黑亏损 symbol 或用主观收益判断屏蔽候选。
- 历史 `TSLA/USDT` 拒单仍会在 120m 窗口滚动期内出现；不能因为窗口里还有旧拒单就误判本轮修复无效，也不能把旧拒单计入已成交收益样本。

本地验证：
- TDD 红灯：新增边界测试后，旧 `EntryOpportunityGatePolicy` 因缺少 `blocked_symbol_reason` 参数失败。
- TDD 绿灯：实现注入后，`pytest tests/test_trading_service_boundaries.py::test_entry_opportunity_gate_blocks_known_untradable_symbol_before_execution -q` 通过。
- `pytest tests/test_entry_symbol_blocklist.py tests/test_execution_result_classifier.py -q`：15 passed。
- `pytest tests/test_trading_service_boundaries.py -q`：120 passed。
- `pytest tests/test_trading_service_boundaries.py tests/test_entry_symbol_blocklist.py tests/test_execution_result_classifier.py -q`：135 passed。
- `ruff check services/entry_opportunity_gate.py services/trading_service.py tests/test_trading_service_boundaries.py tests/test_entry_symbol_blocklist.py tests/test_execution_result_classifier.py`：no issues。
- `black --check services/entry_opportunity_gate.py services/trading_service.py tests/test_trading_service_boundaries.py tests/test_entry_symbol_blocklist.py tests/test_execution_result_classifier.py`：通过。
- `git diff --check`：通过。

线上同步与只读复查：
- `python scripts/sync_to_online_server.py --split-services` 已同步 `services/entry_opportunity_gate.py` 与 `services/trading_service.py` 并重启服务；`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard 返回 `302`。
- 部署后 15m 摘要（`2026-06-23T09:12:18Z`）：33 decisions、0 orders、0 failed/rejected、0 fast_loss_close_under_15m、open_positions 6；`trade_execution_contract.status=ok`，全部 violation counters 为 0；`local_ml_readiness.status=degraded` 且 `allow_live_position_influence=false`。
- 部署后 120m 摘要（`2026-06-23T09:12:18Z`）：425 decisions、1 order、0 filled、1 failed/rejected、open_positions 6、0 fast_loss_close_under_15m；拒单仍是部署前旧 `TSLA/USDT` order `2561`，订单时间 `2026-06-23T07:45:10Z`，会随 120m 窗口滚出。
- 当前没有新增弱证据执行、负预期执行、无强退出快亏、loss re-entry 或风控绕过；但 Batch H 仍未完成，不能据此推进仓位放大、阈值放宽或 ML live influence。

当前结论：
- 本轮修复的是“已知不可交易 symbol 的重复拒单复发”问题，不是盈利闭环完成。
- 后续观察时，如果 15m/120m 出现新的拒单，必须先看是否为新 symbol、新错误码、blocklist 未识别、队列竞态或交易所瞬时问题，再决定是否扩展 blocklist 识别；不得用降阈值、放大仓位或硬改 readiness 来掩盖拒单。
- 若 120m 旧 `TSLA/USDT` 拒单滚出后仍出现同 symbol 新拒单，说明执行前 gate 仍有遗漏路径，必须停止推进并回到 entry candidate 到 execution submit 的调用链定位。

回滚点：
- 代码层可回滚 `services/entry_opportunity_gate.py`、`services/trading_service.py` 与 `tests/test_trading_service_boundaries.py`；本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽。若回滚线上运行代码，需要同步后重启服务以刷新 Python 进程导入。

---

## 三十四、Batch H 补充记录：Dashboard 健康面异常收口与防偏（2026-06-23）

触发原因：用户指出系统巡检页、数据采集页、本地 ML 页、服务器监控/系统自检页仍有异常或需关注项。复查确认，之前 Batch H 重心偏向交易闭环和拒单拦截，漏掉了用户直接看到的 Dashboard 健康面收口。本轮按“真实故障、真实降级、误报/旧告警”拆分处理：不能把真实缺失特征和 ML degraded 洗成 ok，也不能让已处理旧拒单、重复重探和旧 secure key 覆盖继续把页面打红。

本次修复范围：
- `services/crypto_feature_coverage.py`：特征覆盖报告从加载完整 `AIDecision` ORM 对象改为轻量投影 `id/symbol/created_at/feature_snapshot`，避免数据采集页和系统巡检因大字段加载超时。
- `web_dashboard/api/data_collection.py`：`get_data_collection_status(include_feature_coverage=True)` 支持调用方跳过重复特征覆盖重探；跳过时返回只读 `skipped_by_caller` 契约，不改变任何交易信号。
- `web_dashboard/api/system_audit.py`：`model_training` 巡检不再重复跑完整 `feature_coverage`，并把模型运行探针超时窗口从 4s 提高到 8s；真实模型不可用仍会进入 hard failure，学习观察态仍为 warning。
- `services/secure_runtime_config.py`：当 systemd runtime env 已注入 `LOCAL_AI_TOOLS_API_KEY` 时，启动加载 secure settings 不再用旧的 `local_ai_tools.api_key` 密文覆盖它；修复 18001 本地量化工具明明 runtime key 正确、Dashboard 启动后却被旧 key 覆盖成 401 的问题。
- `web_dashboard/api/system_health.py`：系统自检的 `recent_failed_orders` 补查订单关联 `decision_id`，并将明确归因为 OKX 51155、合规限制、不可交易 symbol 等已处理终态拒单降为 `info`；未知拒单、未决单、未归因终态失败仍保持 warning。
- `web_dashboard/api/system_health.py`：运行时模型自检只把当前固定专家槽和高风险复核模型视为必需模型；额外/旧配置模型如 `deepseek-v4-pro` 探测失败只作为环境观察项，不再拖红当前系统。
- `web_dashboard/static/js/dashboard.js`：Local ML `degraded/learning_only` 且 `allow_live_position_influence=false` 时展示为“学习观察” warning，不再渲染成红色异常；模型接口不可用时仍然是 bad。

安全边界：
- 本批不修改开仓阈值、证据 tier、仓位、杠杆、平仓、模型权重、专家路由、ML readiness 门槛、风控 veto 或真实交易执行逻辑。
- `crypto_feature_coverage.status=warning` 仍是真实数据质量 warning：当前缺失 `liquidation_risk`、`btc_eth_anchor`、`sector_correlation`、`abnormal_wick`、`event_calendar`，缺失特征继续按 `neutral_blocked` 处理，不能改成 ok。
- 本地 ML `readiness_state=degraded` 且 `allow_live_position_influence=false` 是真实受控降级，不允许后续 AI 为了页面全绿而强行 ready、降低 PR-AUC/样本质量门槛或允许实盘仓位影响。
- `recent_failed_orders` 只对“终态 + 明确已处理归因”的拒单降级；未知失败单、pending/open/partial 订单、没有执行原因的拒单仍必须保留 warning。
- runtime env 保护只针对 `LOCAL_AI_TOOLS_API_KEY`，不改变 OKX、AI provider、高风险复核和数据源密钥的 secure settings 加载规则。

本地验证：
- `pytest tests/test_crypto_feature_coverage.py tests/test_data_collection_api.py::test_data_collection_status_exposes_sources_and_training tests/test_system_audit_api.py tests/test_system_self_check.py tests/test_dashboard_main_ui_contract.py tests/test_dashboard_error_safety.py tests/test_secure_runtime_config.py -q`：118 passed。
- `ruff check web_dashboard/api/data_collection.py web_dashboard/api/system_audit.py web_dashboard/api/system_health.py services/crypto_feature_coverage.py services/secure_runtime_config.py tests/test_crypto_feature_coverage.py tests/test_system_audit_api.py tests/test_system_self_check.py tests/test_dashboard_main_ui_contract.py tests/test_secure_runtime_config.py`：no issues。
- `black --check web_dashboard/api/data_collection.py web_dashboard/api/system_audit.py web_dashboard/api/system_health.py services/crypto_feature_coverage.py services/secure_runtime_config.py tests/test_crypto_feature_coverage.py tests/test_system_audit_api.py tests/test_system_self_check.py tests/test_dashboard_main_ui_contract.py tests/test_secure_runtime_config.py`：通过。
- `python scripts/security_secret_scan.py --fail-on high .`：source safety scan ok，扫描 514 files。
- `git diff --check`：通过。

线上同步与复查：
- 首次全量同步在 SFTP 遍历阶段连接中断，未当作完成；随后按同一 SSH/重启逻辑聚焦同步本批相关文件并重启 split services。
- 最终同步后 `bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard 返回 `302`。
- 复查以 Dashboard 主进程真实环境执行，避免 `.env` 旧值污染：`high_risk_model=deepseek-r1-14b-risk`，`secure_skipped=["local_ai_tools.api_key"]`，`secure_error=""`。
- 数据采集页后端：`duration_sec=1.383`、`error=null`、`feature_status=warning`、`feature_error=null`、`local_ai_status=learning_only`、`local_ai_available=true`。
- 系统巡检页：overall `warning`，但 `critical=0`；cards 16、warning 10、ok 6；`model_training` 为 warning，`hard_failure=false`、`observing=true`、`runtime_probe.status=ok`、`ai_model_count=2`；`crypto_feature_coverage` 为 warning 且 `error=null`。
- 系统自检页：overall `ok`；critical 0、warning 0、ok 15、info 2；`recent_failed_orders` 已降为 info，order `2561` 为已处理终态拒单，`handled_terminal_failure_count=1`、`unhandled_terminal_failure_count=0`、`has_unresolved_order=false`。
- 服务器监控页：`status=ok`、`available=true`、`remote_monitor_available=true`；`qwen3-14b-trade` 与 `deepseek-r1-14b-risk` 均 `available=true`；`local_ai_tools_available=true`、`local_ai_tools_status_ok=true`。
- 本地 ML 页后端：`ml_signal_status.available=true`、`status=degraded`、`readiness_state=degraded`、`allow_live_position_influence=false`、`influence_enabled=false`；`local_ai_tools_status.available=true`、`service_available=true`。

当前结论：
- 本轮已处理用户可见 Dashboard 健康面的“异常/误报/密钥覆盖/旧拒单”问题：数据采集不再抛特征覆盖 TimeoutError，Local AI tools 不再因旧 secure key 被 401，服务器监控恢复 ok，系统自检不再把已处理 TSLA 拒单当成新故障。
- 系统巡检整体仍为 warning 是正确的：还有真实只读观察项和缺失特征，不代表本轮失败，也不能为了页面全绿隐藏。
- 本地 ML 仍是 degraded 且禁止实盘影响，这说明 ML 训练质量问题仍需后续从样本、标签、特征和离线评估根治，不能在 Dashboard 层硬改状态。

后续 AI 防偏要求：
- 后续 AI 看到 Dashboard warning 时必须先分清 `error`、`hard_failure`、`observing`、`info`、真实数据质量 warning，不能把 warning 一律当作代码异常，也不能把 warning 一律改成 ok。
- 复核 Dashboard 必须使用 Dashboard 主进程同等环境，或者明确加载 `/data/bb/app/.env` 与 `/etc/bb/bb-runtime.env` 并调用 `load_secure_settings_into_runtime()`；不能用裸 `.venv/bin/python` 直接导入后拿 `.env` 旧配置下结论。
- 如果后续再次出现 Local AI tools 401，优先检查 `/etc/bb/bb-runtime.env`、模型服务器 `/data/trade_ai/local_ai_tools.env`、secure settings 覆盖顺序和 Dashboard 进程环境，不要先改前端。
- 如果后续再次出现 `deepseek-v4-pro` 类旧模型 critical，先确认它是否属于当前固定专家槽或高风险复核模型；非必需旧配置不能拖红系统自检。
- 如果后续再次出现拒单，必须区分“已处理旧拒单滚动窗口残留”和“新增未知拒单”；新增拒单才回到 entry candidate 到 execution submit 调用链定位。

回滚点：
- 代码层可回滚 `services/crypto_feature_coverage.py`、`services/secure_runtime_config.py`、`web_dashboard/api/data_collection.py`、`web_dashboard/api/system_audit.py`、`web_dashboard/api/system_health.py`、`web_dashboard/static/js/dashboard.js` 与对应测试；本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽。若回滚线上运行代码，需要同步后重启三项服务。

---

## 三十五、Batch H 补充记录：系统自检交易心跳旧错误清理（2026-06-23）

触发原因：继续复核 Dashboard 健康面时，系统巡检没有 critical，15m/120m 策略健康也没有拒单、弱证据执行、负预期执行、无强退出快亏或 loss re-entry；但系统自检页仍有 `trading_service` warning。只读探针显示交易进程心跳新鲜、当前轮未卡死、交易契约为 ok，warning 来源是上一轮 `exchange position reconciliation timed out during market round start; continuing with local position state` 仍残留在 runtime heartbeat 的 `last_round_error`。

本次修复范围：
- `services/trading_service.py`：`record_round_error()` 改为把错误记录到当前 `analysis_scope`，避免 market/position 的临时错误落到全局 full scope 后难以清理。
- `services/trading_service.py`：`_finish_runtime_round(scope, ok=True)` 成功完成时清理该 scope 的 `last_error`；当 market/position/full 都没有剩余错误时，同步清理全局 `_last_round_error`。
- `tests/test_trading_service_boundaries.py`：新增回归测试，确认可恢复的 market 对账超时会先写入 heartbeat，后续同 scope 成功完成后 `market_last_error` 与 `last_round_error` 都清空。

安全边界：
- 本批只修 runtime heartbeat 状态机的错误生命周期，不改变 OKX 对账逻辑、不改变本地持仓降级策略、不改变开仓/平仓/仓位/杠杆/风控/模型权重/ML readiness。
- 对账超时本身仍会记录为错误；只有后续同 scope 成功完成，且其它 scope 没有未清错误时，才清理系统自检 warning。
- 如果 market/position round 真实卡死、心跳不新鲜、运行进程停止、持续对账失败或交易契约违规，系统自检仍必须保持 warning/critical，不允许为了页面全绿清掉真实故障。

本地验证：
- `pytest tests/test_trading_service_boundaries.py::test_successful_runtime_round_clears_recovered_scope_error tests/test_trading_service_boundaries.py::test_parallel_market_position_runtime_state_is_isolated -q`：2 passed。
- `pytest tests/test_system_self_check.py::test_self_check_uses_split_process_runtime_heartbeat tests/test_system_self_check.py::test_self_check_warns_when_split_process_round_is_stuck tests/test_system_self_check.py::test_self_check_warns_when_market_round_is_stuck tests/test_system_self_check.py::test_self_check_uses_position_watchdog_for_position_round -q`：4 passed。
- `pytest tests/test_trading_service_boundaries.py tests/test_system_self_check.py tests/test_system_audit_api.py tests/test_dashboard_main_ui_contract.py -q`：215 passed。
- `ruff check services/trading_service.py tests/test_trading_service_boundaries.py`：no issues。
- `black --check services/trading_service.py tests/test_trading_service_boundaries.py`：通过。

线上复查前置证据：
- 同步前 15m 策略健康：50 decisions、0 orders、0 rejected、0 failed、0 fast_loss；交易契约 ok，`contract_violation_count=0`、`weak_evidence_executed_count=0`、`negative_expected_executed_count=0`、`fast_loss_without_strong_exit_count=0`。
- 同步前 120m 策略健康：377 decisions、0 orders、0 rejected、0 failed、0 fast_loss；交易契约 ok，上述硬停计数均为 0。
- Dashboard 同环境系统巡检：overall warning 但 `critical_cards=[]`；`trade_execution_contract` ok，current summary 中硬停计数均为 0；`visible_text_encoding` ok，`runtime_text_integrity` ok。
- Dashboard 同环境系统自检：`critical_items=[]`，唯一 warning 为 `trading_service`；其 details 显示心跳新鲜、round 未卡死，warning 原因是旧 `last_round_error` 残留。

同步后线上复查：
- `python scripts/sync_to_online_server.py --split-services` 上传 `docs/superpowers/plans/2026-06-22-quant-closed-loop-eradication.md` 与 `services/trading_service.py`，三项服务 `bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard 返回 `302`。
- 15m 策略健康：52 decisions、0 orders、0 failed、0 rejected、0 fast_loss；交易契约 ok，`contract_violation_count=0`、`weak_evidence_executed_count=0`、`negative_expected_executed_count=0`、`fast_loss_without_strong_exit_count=0`、`reentry_without_strong_unlock_count=0`。
- 120m 策略健康：371 decisions、0 orders、0 failed、0 rejected、0 fast_loss；交易契约 ok，上述硬停计数均为 0。
- Dashboard 同环境系统巡检：overall `warning` 但 `critical_cards=[]`；`trade_execution_contract` 为 ok；`visible_text_encoding` 为 ok，扫描 282 files、offender 0；`runtime_text_integrity` 为 ok，扫描 809 records、疑似记录 0。
- Dashboard 同环境系统自检复跑：overall `ok`；total 17、critical 0、warning 0、ok 15、info 2。`trading_service` 已恢复 ok，旧 `last_round_error` 不再把页面打成 warning。
- 中途一次系统自检看到 `server_monitor` warning，经复跑确认是 `server_monitor_refreshing` 并发刷新窗口，不是稳定故障；最终稳定结果为 warning 0。

后续 AI 防偏要求：
- 看到系统自检 `trading_service` warning 时，必须同时看 `heartbeat_age_seconds`、`round_stuck`、`market_round_stuck`、`position_round_stuck`、`runtime_error`、`last_round_error` 和交易契约硬停计数，不能只凭 warning 改前端。
- `exchange position reconciliation timed out ... continuing with local position state` 属于对账降级风险，不等于开仓/平仓契约违规；若后续同 scope 成功完成且契约计数为 0，可清理观察噪声。
- 如果该错误连续出现、导致心跳卡死、出现未解决订单/仓位不一致、或交易契约出现 violation，必须回到 OKX 对账/持仓同步链路定位，不能把它降级成 info。

回滚点：
- 代码层可回滚 `services/trading_service.py` 与 `tests/test_trading_service_boundaries.py`；本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽。线上回滚后需重启 `bb-paper-trading.service`，让 runtime heartbeat 状态机重新加载。

---

## 三十六、Batch H 补充记录：执行记录杠杆展示、系统巡检 JS 作用域与候选池集中诊断（2026-06-23）

触发原因：用户继续复核 Dashboard 时指出两类问题：其一，执行记录里 WLFI/USDT 平仓很快且亏损，平仓行显示 `1.0x`，而开仓行显示 `3.0x`；其二，系统巡检页仍出现“系统巡检接口异常”，问题台账仍有很多未处理项。同时用户补充疑问：分析协作记录里的市场分析交易对经常相似，当前候选分析是否真的在全市场筛选更优交易对，重复候选是否是长期不开仓原因。

本次修复范围：
- `web_dashboard/api/trades.py`：执行记录列表在展示平仓单杠杆时，优先匹配同 symbol、同模型、同执行模式、同平仓方向、成交时间附近的已关闭 `Position`，并使用该仓位真实 `leverage` 作为平仓行展示杠杆；保留 `ai_suggested_leverage` 字段用于追溯平仓决策本身的建议杠杆。
- `web_dashboard/api/trades.py`：交易详情的 `matched_positions` 增加 `leverage`，方便核对“平仓行展示杠杆来自被平掉的真实仓位”。
- `tests/test_trade_history_api.py`：新增回归测试，覆盖 `close_short` 决策建议杠杆为 `1.0x`、被平掉仓位真实杠杆为 `3.0x` 时，列表展示和详情匹配仓位均返回 `3.0x`。
- `web_dashboard/static/js/dashboard.js`：把 `systemAuditShadowMissedOpportunityDetails` 从 `systemAuditGenericDetailsHtml` 的函数体里拆出来，恢复为全局函数，避免系统巡检点击详情时出现 `systemAuditShadowMissedOpportunityDetails is not defined`。
- `tests/test_dashboard_main_ui_contract.py`：新增前端契约测试，锁定 `systemAuditGenericDetailsHtml`、`systemAuditShadowMissedOpportunityDetails`、`systemAuditCardDetailsHtml` 的声明顺序，并确认 shadow missed opportunity 详情函数没有被误嵌套。

候选池真实逻辑复核：
- 当前配置仍为 `scan_mode="auto"`，`auto_scan_symbol_limit=20`。auto 模式不是只扫配置里的 BTC/ETH/SOL，也不是只固定分析少数币。
- `data_feed/okx_rest_client.py::get_available_symbols()` 会读取 OKX active USDT linear swaps，排除可疑合约、贵金属等不适合项，并按 `activity_score` 排序；`activity_score` 由成交量、24h 成交额、涨跌幅、振幅、点差、主流币加分和极端振幅/超小价格惩罚组成。
- `services/trading_service.py` 在 auto 模式下先取更大的候选池：`pool_limit = max(auto_scan_symbol_limit, auto_scan_symbol_limit * AUTO_SCAN_ROTATION_POOL_MULTIPLIER, AUTO_SCAN_ROTATION_POOL_MIN, 30)`，当前中心参数为 `AUTO_SCAN_ROTATION_POOL_MULTIPLIER=20`、`AUTO_SCAN_ROTATION_POOL_MIN=240`，所以不是只取 20 个。
- 拉取 K 线特征前会通过 `_budget_auto_scan_feature_symbols()` 对大池子做预算和轮转，并保留持仓复核 symbol；当前 `AUTO_SCAN_FEATURE_FETCH_POOL_MIN=12`，实际每轮会受时间预算、特征拉取成功率、持仓复核优先级和动态分析预算影响。
- 特征可用后由 `EntryFeatureRankerPolicy` 二次筛选排序：先选流动性、量比、ADX、动量、波动、24h 变化、布林极值、趋势距离更合适的 hard candidates；不足时用 soft candidates 补齐；同时扣减 recent hold、recent analysis 和 no-opportunity rotation penalty。
- 排序后的 market candidates 仍需经过 AI 决策、expected net、entry evidence tier、position size、机会评分、成本/滑点、ML readiness、风控 veto 和执行契约，不是“被分析了就必须开仓”。

线上只读观察结论：
- 2026-06-23 11:42 UTC 只读窗口：最近 120 分钟共 411 条 decisions，其中 market decisions 122 条、market unique symbols 34 个；market top symbols 为 ETH 10、CL 9、BZ 8、SUI 7、XRP 7、LPT 7、CRV 6、HOOD 6 等。说明“只扫几个固定币”不成立。
- 同一窗口最终 market entry candidates 只有 8 条，集中在 3 个 symbol：XRP 4、HOOD 3、SOL 1；动作分布为 short 5、long 3。说明“经过排序/门禁后候选集中在少数币”是事实。
- 同一窗口 entry skip 为 `entry_evidence_wait=6`、`entry_pre_execution_skip=1`、`entry_evidence_shadow_only=1`；evidence tier 为 `blocked=7`、`weak_conflict_probe=1`。同期策略健康仍为 orders 0、failed/rejected 0、positions_created/closed 0、fast_loss_close_under_15m 0，交易执行契约为 ok。
- 本地 ML 仍为 `degraded`，`allow_live_position_influence=false`；阻塞项仍包括 long AUC/PR-AUC/accuracy 低、short PR-AUC/top return 低、dirty sample ratio 高等。当前 0 开仓更像是候选质量、证据强度、预期收益质量、ML 降级和风控门共同作用，不是订单提交链路丢单。

安全边界：
- 本批只修 UI/API 展示一致性、前端 JS 作用域和只读诊断记录，不修改开仓阈值、证据 tier、仓位、杠杆、平仓、模型权重、专家路由、ML readiness、风控 veto 或真实交易执行逻辑。
- 平仓行展示 `Position.leverage` 只解决“用户看到的平仓杠杆应与被平仓仓位一致”；不代表平仓决策本身建议杠杆应被改写，`ai_suggested_leverage=1.0` 必须保留用于追踪原始决策。
- 系统巡检整体 `warning` 与问题台账未清零不等于本轮修复失败；`critical=0` 且 JS 异常消失后，剩余 warning/ledger 项必须按真实阻塞、观察态和历史遗留继续处理，不能为了页面全绿隐藏真实问题。
- 候选重复感不能用“直接扩大下单”“降低 evidence 阈值”“硬改 ML ready”“放大仓位/杠杆”来解决。后续若要优化候选质量，应先做只读诊断：候选覆盖率、特征拉取失败率、rotation cursor、recent-analysis 去重、no-opportunity penalty、entry ranker 入选/淘汰原因、entry evidence 组件和最终 skip_kind。

本地验证：
- `pytest tests/test_trade_history_api.py tests/test_dashboard_main_ui_contract.py -q`：48 passed。
- `ruff check web_dashboard/api/trades.py tests/test_trade_history_api.py tests/test_dashboard_main_ui_contract.py`：no issues。
- `black --check web_dashboard/api/trades.py tests/test_trade_history_api.py tests/test_dashboard_main_ui_contract.py`：通过。
- `node --check web_dashboard/static/js/dashboard.js`：通过。

线上同步与复查：
- `python scripts/sync_to_online_server.py --split-services` 已同步当前代码并重启 split services；`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard 返回 `302`。
- `/api/trades?mode=paper&limit=20&page=1` 复查 WLFI/USDT close 样本：`action="close_short"`、`leverage=3.0`、`actual_leverage=3.0`、`ai_suggested_leverage=1.0`、`hold_minutes≈13.84`，说明平仓展示杠杆已来自真实已关闭仓位。
- 系统巡检前端函数复查：`systemAuditShadowMissedOpportunityDetails` 已在全局声明，且没有嵌套在 `systemAuditGenericDetailsHtml` 内；`/api/system-audit/status` 当前 `audit_status=warning`、`audit_critical=0`、`audit_warning=10`，未再复现该 JS undefined 异常。

当前结论：
- 执行记录“平仓杠杆与开仓杠杆不一致”的展示问题已修复；WLFI/USDT 的短持仓亏损本身仍是交易质量问题，需要继续结合快亏强退出、候选证据、entry/exit reasoning 和真实 PnL 观察，不能被 UI 修复掩盖。
- 系统巡检“接口异常”中这次确认的 JS 函数作用域问题已修复；但问题台账仍有真实未解决/观察项，Batch H 不能标记完成。
- 候选分析确实有全市场自动筛选与轮转，但最终 entry candidates 目前集中在少数 symbol。这个集中现象可能贡献“老是不开仓”的结果，因为如果排名靠前的少数候选长期证据不足，就会反复被挡住；但当前证据不支持把它归因为“没有全市场筛选”或“下单链路坏了”。

后续 AI 防偏要求：
- 后续 AI 看到协作记录里 symbol 重复，必须先区分四层：OKX 大池子覆盖、特征拉取预算池、ranker 入选池、最终 entry candidate；不能把最终候选集中误判成 auto scan 失效。
- 后续 AI 必须同时输出 unique market symbols、market entry symbols、entry skip_kind、evidence tier、position_size_pct、expected net 组件、feature fetch timeout/invalid 数量，再判断是否需要改候选生成。
- 若要继续优化，应优先补“候选生成质量诊断仪表”：每轮 pool size、fetch selected、feature valid、ranked selected、recent-analysis skipped、no-opportunity penalty top、entry rejected reason top。只有这些只读证据证明候选池被预算/排序长期饿死，才允许讨论调整 rotation/ranker；仍不得直接放宽交易门。

回滚点：
- 代码层可回滚 `web_dashboard/api/trades.py`、`web_dashboard/static/js/dashboard.js`、`tests/test_trade_history_api.py`、`tests/test_dashboard_main_ui_contract.py`；本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽。线上回滚后需重启 `bb-dashboard.service`，若回滚 API 代码也需重启对应 split service。

---

## 三十七、Batch H 补充记录：market symbol 漏斗诊断固化（2026-06-23）

触发原因：前一轮已经确认 auto scan 确实覆盖多个交易对，但最终 market entry candidates 集中在少数 symbol。仅靠临时 SSH 查库或人工看分析协作记录，容易让后续 AI 走偏：要么误判为“没有全市场扫描”，要么直接去放宽 evidence/仓位/杠杆制造成交。因此本轮把候选池收敛诊断固化到只读策略健康脚本里，让每次 15m/120m 复查都能直接看到“market 覆盖”和“entry 候选集中度”。

本次修复范围：
- `scripts/inspect_online_strategy_health.py`：新增 `market_symbol_diagnostics`，输出 `market_decision_count`、`market_unique_symbol_count`、`market_top_symbols`、`market_entry_count`、`market_entry_unique_symbol_count`、`market_entry_top_symbols`、`market_entry_action_counts`、`market_entry_skip_kind_counts`、`market_entry_tier_counts`、`market_top3_share`、`market_entry_top3_share` 与 `entry_unique_to_market_unique_ratio`。
- `scripts/inspect_online_strategy_health.py`：新增 `counter_rows()`、`symbol_counter_rows()`、`top_share()`，把 Counter 结果输出成稳定 JSON 结构，避免后续解析依赖 tuple/list 细节。
- `scripts/inspect_online_strategy_health.py`：`summary_report()` 与本地兜底 `_summarize_report()` 均保留 `market_symbol_diagnostics`，避免 `--summary` 模式丢失候选漏斗。
- `tests/test_inspect_online_strategy_health.py`：新增模板契约测试，锁定 market symbol 漏斗字段和只读边界文本；同时更新 summary 测试，确认摘要里仍包含 `market_entry_unique_symbol_count`。

安全边界：
- 本批只改只读线上观察脚本和测试，不修改交易服务、不修改开仓阈值、证据 tier、仓位、杠杆、平仓、模型权重、专家路由、ML readiness、风控 veto 或真实交易执行逻辑。
- `market_entry_top3_share=1.0` 只能说明最终 entry 候选集中在 3 个 symbol，不等于这些 symbol 应该强行开仓，也不等于其它 symbol 被错误过滤；后续必须结合 feature fetch、ranker、recent-analysis dedupe、evidence tier 和 skip_kind 继续定位。
- 该字段用于阻止后续 AI 只凭“重复 symbol”猜测原因；不得把它用作降低风控门槛或扩大仓位的依据。

本地验证：
- `pytest tests/test_inspect_online_strategy_health.py -q`：16 passed。
- `ruff check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：no issues。
- `black --check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：通过。
- `python -m py_compile scripts/inspect_online_strategy_health.py`：通过。
- 关联回归：`pytest tests/test_trade_history_api.py tests/test_dashboard_main_ui_contract.py -q`：48 passed。

线上只读复查：
- `python scripts/inspect_online_strategy_health.py --minutes 120 --summary`（`2026-06-23T11:48:06Z`）：413 decisions、0 orders、0 failed/rejected、0 positions_created/closed、open_positions 6、fast_loss_close_under_15m 0。
- 交易执行契约仍为 ok：`contract_violation_count=0`、`weak_evidence_executed_count=0`、`negative_expected_executed_count=0`、`fast_loss_without_strong_exit_count=0`、`reentry_without_strong_unlock_count=0`。
- ML 仍为 `degraded`，`allow_live_position_influence=false`，阻塞项仍包括 long AUC/PR-AUC/accuracy、short PR-AUC/top return 和 dirty sample ratio。
- 新增 `market_symbol_diagnostics` 当前窗口显示：market decisions 121、market unique symbols 34、market top3 share 0.22314；market entry candidates 8、market entry unique symbols 3、market entry top symbols 为 XRP 4、HOOD 3、SOL 1，market entry top3 share 1.0，`entry_unique_to_market_unique_ratio=0.088235`。
- market entry skip/tier 当前仍为 `entry_evidence_wait=6`、`entry_pre_execution_skip=1`、`entry_evidence_shadow_only=1`；`blocked=7`、`weak_conflict_probe=1`。结论仍是候选末端证据不足，非下单链路丢单。

后续 AI 防偏要求：
- 每次回答“为什么候选重复/为什么不开仓”时，必须同时引用 `market_unique_symbol_count`、`market_entry_unique_symbol_count`、`market_entry_top3_share`、`market_entry_skip_kind_counts`、`market_entry_tier_counts` 与交易执行契约，不能只看页面协作记录。
- 若 `market_unique_symbol_count` 充足但 `market_entry_unique_symbol_count` 长期很低，应继续补 feature/ranker/dedupe 诊断，而不是直接改交易门。
- 若 `market_unique_symbol_count` 本身偏低，再回到 auto scan、OKX symbols、feature fetch budget、blocked/open-position/unclaimed filters 定位；不能把两类问题混为一谈。

回滚点：
- 代码层可回滚 `scripts/inspect_online_strategy_health.py` 与 `tests/test_inspect_online_strategy_health.py`；本批无 DB 迁移、无历史覆盖、无服务行为变更、无模型 artifact 替换、无真实交易参数放宽。线上回滚观察脚本可用 `sync_to_online_server.py --split-services --skip-restart` 同步，无需重启交易服务。

---

## 三十八、Batch H 补充记录：market candidate 前置漏斗落库诊断（2026-06-23）

触发原因：第 37 节已经把“market 覆盖足够、最终 entry 候选集中”固化进健康脚本，但它只能从已落库的 `AIDecision` 汇总末端结果，仍不能解释收窄发生在 feature 拉取、ranker 排序、recent-analysis 去重、动态分析预算还是 evidence 门。为了避免后续 AI 继续靠猜测修参数，本轮把单轮 market candidate 前置漏斗作为只读元数据挂到 market 决策 `raw_llm_response.market_candidate_funnel`，并由健康脚本汇总最新样本。

本次修复范围：
- `services/trading_service.py`：`_rank_auto_feature_vectors()` 保存最近一次 `EntryFeatureRankerPolicy.rank()` diagnostics，用于本轮只读漏斗快照。
- `services/trading_service.py`：新增 `_market_candidate_funnel_snapshot()`，输出 scan/filter/fetch/valid/rank/dedupe/analysis budget 各层数量、rank top symbols、recent-analysis dedupe、`read_only=true` 与 `is_entry_gate=false`。
- `services/trading_service.py`：新增 `_attach_market_candidate_funnel()`，在 fast prefilter hold 和正常 market AI decision 落库前把同一轮 `market_candidate_funnel` 附加到 raw response。
- `scripts/inspect_online_strategy_health.py`：`market_symbol_diagnostics` 增加 `candidate_funnel_sample_count` 与 `latest_candidate_funnel`，从最近 market decisions 聚合新字段；旧决策没有该字段时保持兼容。
- `tests/test_trading_service_boundaries.py`：新增只读漏斗快照测试，锁定 `read_only/is_entry_gate`、feature valid/invalid、rank selected、recent dedupe 与边界文本。
- `tests/test_inspect_online_strategy_health.py`：更新模板与 summary 契约测试，锁定 `latest_candidate_funnel` 不在 summary 模式丢失。

安全边界：
- 本批只增加诊断元数据，不改变 scan pool、feature fetch budget、ranker 算法、recent-analysis dedupe 行为、analysis budget、AI 决策、entry evidence、仓位、杠杆、ML readiness、风控 veto 或真实交易执行逻辑。
- `market_candidate_funnel` 是写入每条 market decision 的只读解释字段，不参与策略评分；后续 AI 不得用它直接放宽 `market_symbol_limit`、扩大仓位或绕过 evidence 门。
- 如果该漏斗显示 feature 请求/valid 足够但 rank/dedupe 后长期很窄，下一步只能先补更细的 ranker 淘汰原因、动态 budget 来源和策略学习压力诊断；不能直接把 `market_symbol_limit` 从 2 提高到大批量制造候选。

本地验证：
- `pytest tests/test_inspect_online_strategy_health.py tests/test_trading_service_boundaries.py::test_auto_scan_feature_budget_rotates_market_pool_and_keeps_positions tests/test_trading_service_boundaries.py::test_market_candidate_funnel_snapshot_is_read_only_and_exposes_rank_dedupe_counts tests/test_trade_history_api.py tests/test_dashboard_main_ui_contract.py -q`：66 passed。
- `ruff check services/trading_service.py scripts/inspect_online_strategy_health.py tests/test_trading_service_boundaries.py tests/test_inspect_online_strategy_health.py web_dashboard/api/trades.py tests/test_trade_history_api.py tests/test_dashboard_main_ui_contract.py`：no issues。
- `black --check services/trading_service.py scripts/inspect_online_strategy_health.py tests/test_trading_service_boundaries.py tests/test_inspect_online_strategy_health.py web_dashboard/api/trades.py tests/test_trade_history_api.py tests/test_dashboard_main_ui_contract.py`：通过。
- `python -m py_compile services/trading_service.py scripts/inspect_online_strategy_health.py`：通过。

线上同步与复查：
- `python scripts/sync_to_online_server.py --split-services` 已同步 `services/trading_service.py` 与 `scripts/inspect_online_strategy_health.py` 并重启三项 split services；`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard 返回 `302`。
- 服务重启时间为 `2026-06-23T12:03:17Z`；重启前窗口中的 market 决策不带 `market_candidate_funnel` 属于旧进程样本，不能当作失败证据。
- 等待新 market 决策后，`python scripts/inspect_online_strategy_health.py --minutes 15 --summary`（`2026-06-23T12:07:14Z`）显示 53 decisions、0 orders、0 failed/rejected、0 fast_loss、open_positions 6，交易执行契约仍 ok，硬停计数均为 0。
- 同一 15m 窗口 `market_symbol_diagnostics`：market decisions 17、market unique symbols 15、market entry candidates 0；`candidate_funnel_sample_count=3`，说明新字段已落库并被健康脚本读取。
- 最新 `latest_candidate_funnel` 显示：`scan_symbol_count=120`、`open_position_filtered_count=6`、`feature_fetch_requested_count=12`、`feature_valid_count=12`、`feature_invalid_count=0`、`market_feature_before_rank_count=12`、`market_symbol_budget=2`、`rank_selected_count=2`、`rank_tradable_candidates=2`、`rank_secondary_candidates=2`、`rank_total_candidates=12`、`recent_analysis_dedupe_count=0`、`market_feature_after_dedupe_count=2`。
- 当前这一轮前置漏斗结论：不是 feature 拉取失败导致候选少；主要收窄发生在 feature fetch 预算只取 12 个，以及动态 `market_symbol_limit=2` 后 ranker 只让 2 个 symbol 进入 AI market 分析。该 budget 来源为 `strategy_learning`，理由是当前有持仓，position loop 并行负责持仓复盘，market 分析只用剩余小批量候选，避免抢占大模型资源。

当前结论：
- 第 37 节回答的是“最终 entry 候选是否集中”；第 38 节进一步回答“末端集中之前，market 分析候选为什么每轮较窄”。当前证据显示 feature 拉取本身健康，动态 budget 和 ranker 是前置收窄主因。
- 这能解释为什么用户在分析协作记录中看到的 market 分析交易对会重复或相似：每轮真正送入 AI 的 market symbol 不是 120 个，而是预算/排序后的 2 个左右；同时最终 entry 还要再过 evidence 门。
- 这仍不代表应该直接提高预算。提高 market_symbol_limit 会增加 LLM/模型资源占用，并可能抢占持仓复盘；必须先评估持仓复盘压力、LLM 延迟、watchdog、命中率和 missed opportunity 质量。

后续 AI 防偏要求：
- 看到 `market_entry_unique_symbol_count` 低时，必须先看 `latest_candidate_funnel.feature_fetch_requested_count`、`feature_valid_count`、`market_symbol_budget`、`rank_selected_count`、`recent_analysis_dedupe_count`，再判断瓶颈。
- 如果 `feature_valid_count` 接近 `feature_fetch_requested_count` 且 `rank_selected_count == market_symbol_budget`，不得继续归因于行情特征异常；应转向动态 analysis budget、ranker 选择和 evidence 门。
- 若要继续优化，应先补策略学习 budget 来源诊断和 ranker 未入选 top/淘汰原因，而不是直接改开仓阈值或把 market_symbol_limit 放大。

回滚点：
- 代码层可回滚 `services/trading_service.py`、`scripts/inspect_online_strategy_health.py`、`tests/test_trading_service_boundaries.py`、`tests/test_inspect_online_strategy_health.py`；本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽。线上回滚后需重启 `bb-paper-trading.service` 让运行时不再写入该诊断字段；已有 raw response 里的只读字段可保留，不影响交易。

---

## 三十九、Batch H 补充记录：候选预算来源与 ranker 淘汰原因诊断增强（2026-06-23）

触发原因：用户追问分析协作记录中 market 分析交易对经常相似，担心候选逻辑没有真正从全市场筛出更优交易对，并指出过程中看到“乱码”描述。经复核，候选链确实存在“全市场大池子存在，但每轮送入 AI 的 market symbol 被 feature fetch 预算、动态 analysis budget 和 ranker 收窄”的现象；同时这次排查里也暴露出一个 AI 防偏点：PowerShell 终端输出会把正常 UTF-8 中文渲染成 mojibake，不能直接当作源码污染。

本次修复范围：
- `services/analysis_budget.py`：`AnalysisBudgetPolicy._result()` 新增只读 `market_limit_diagnostics`，记录 budget source、risk level、market limit policy、configured/selected market limit、position group count、target position groups、roster_underfilled、position review caps 与 market caps。该诊断只解释 AI 分析负载分配，不参与开仓、仓位、杠杆、ML readiness 或风控判定。
- `services/entry_feature_ranker.py`：`EntryFeatureRankerPolicy.rank()` 新增 `ranked_symbol_sample`、`market_symbol_limit` 与 `filtered_out_candidates`，对已选和未选候选输出 score、net_score、recent hold penalty、recent analysis penalty、rotation penalty、selection tier、selected 与 `non_selected_reason`。未进 AI 的候选现在能区分 `outside_market_symbol_budget` 和 `feature_filter_rejected`。
- `services/trading_service.py`：每轮 build strategy context 前清空上一轮 rank diagnostics，避免旧缓存污染当前漏斗；`market_candidate_funnel.analysis_budget` 挂载新的 budget 诊断，`ranked_symbol_sample` 随 market decision 落库。
- `tests/test_analysis_budget.py`、`tests/test_entry_feature_ranker.py`、`tests/test_trading_service_boundaries.py`：新增/更新回归测试，锁定 budget 诊断只读边界、ranker 未入选原因、funnel 中 budget/ranker 字段。
- `docs/superpowers/plans/2026-06-22-quant-closed-loop-eradication.md`：清掉主文档中作为示例写入的裸 U+FFFD 字符，改为文字描述，避免源码乱码扫描把示例本体当成真实污染。

编码与乱码防偏结论：
- 本轮在 PowerShell `Get-Content`/diff 输出里看到的多段 `鎸佽...` 属于终端编码渲染问题；同一文件用 UTF-8 读取时显示为正常中文。
- 不能再凭终端输出直接断言“代码里还有乱码”。必须先用 UTF-8 文件读取、`tests/test_no_mojibake_source.py` 和运行时 `runtime_text_integrity` 只读审计区分真污染与终端假乱码。
- 本轮真实文件级问题只有主文档中 1 个裸 replacement character 示例，已替换为 `U+FFFD replacement character` 文本描述。

本地验证：
- `pytest tests/test_analysis_budget.py tests/test_entry_feature_ranker.py tests/test_trading_service_boundaries.py::test_market_candidate_funnel_snapshot_is_read_only_and_exposes_rank_dedupe_counts tests/test_inspect_online_strategy_health.py -q`：30 passed。
- `pytest tests/test_no_mojibake_source.py tests/test_text_integrity_runtime.py tests/test_runtime_text_integrity_audit.py tests/test_analysis_budget.py tests/test_entry_feature_ranker.py tests/test_trading_service_boundaries.py::test_market_candidate_funnel_snapshot_is_read_only_and_exposes_rank_dedupe_counts -q`：22 passed。
- `ruff check services/analysis_budget.py services/entry_feature_ranker.py services/trading_service.py tests/test_analysis_budget.py tests/test_entry_feature_ranker.py tests/test_trading_service_boundaries.py scripts/inspect_online_strategy_health.py`：no issues。
- `black --check services/analysis_budget.py services/entry_feature_ranker.py services/trading_service.py tests/test_analysis_budget.py tests/test_entry_feature_ranker.py tests/test_trading_service_boundaries.py scripts/inspect_online_strategy_health.py`：通过。
- `python -m py_compile services/analysis_budget.py services/entry_feature_ranker.py services/trading_service.py scripts/inspect_online_strategy_health.py`：通过。

线上同步与复查：
- `python scripts/sync_to_online_server.py --split-services` 已同步 `services/analysis_budget.py`、`services/entry_feature_ranker.py`、`services/trading_service.py` 并重启 split services；`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard 返回 `302`。
- `python scripts/inspect_online_strategy_health.py --minutes 30 --summary`（`2026-06-23T12:24:24Z`）显示：109 decisions、0 orders、0 failed/rejected、0 positions_created/closed、open_positions 6、fast_loss_close_under_15m 0；交易执行契约仍 ok，所有硬停计数为 0。
- 同一 30m 窗口：market decisions 33、market unique symbols 21、market entry candidates 1、market entry unique symbols 1；`candidate_funnel_sample_count=22`。
- 最新 `latest_candidate_funnel` 显示：`scan_symbol_count=120`、`feature_fetch_requested_count=12`、`feature_valid_count=12`、`feature_invalid_count=0`、`market_symbol_budget=2`、`rank_selected_count=2`、`rank_tradable_candidates=2`、`rank_secondary_candidates=1`、`rank_total_candidates=12`。这说明当前瓶颈不是特征拉取失败，而是预算/ranker 收窄。
- 同一漏斗中 `ranked_symbol_sample` 已能解释未选候选：`SUI/USDT` 和 `OP/USDT` 为 `selected_for_market_analysis`，`CHZ/USDT` 为 `outside_market_symbol_budget`。`analysis_budget.market_limit_diagnostics` 显示 `configured_market_symbol_limit=8`、`selected_market_symbol_limit=2`、`position_group_count=6`、`target_position_groups=10`、`roster_underfilled=true`、policy 为 `position_first_low_risk_underfilled`。
- 本地直接跑 `scripts/audit_runtime_text_integrity.py` 因本机 DB 连接被拒绝返回 warning/error，不能当成线上运行时乱码证据；源码层 `test_no_mojibake_source` 已通过，线上系统巡检仍需以 Dashboard 同环境的 `runtime_text_integrity` 为准。

当前结论：
- 现在可以实事求是地回答用户：系统确实有全市场候选池和轮转，最近窗口 market unique symbols 并不低；但每轮真正送入 AI 深度 market 分析的不是 120 个，而是 feature fetch 预算取 12 个，再由动态 market budget/ranker 通常收窄到 2 个。
- 这会造成协作记录里 market 分析 symbol 看起来相似，也可能是长期不开仓的上游原因之一：AI 反复看到少量高排名候选，如果这些候选 evidence tier 长期 blocked/weak，就会持续不开仓。
- 但当前仍不能把问题简单归因于“没有全市场筛选”，也不能直接提高 market_symbol_limit 制造成交；提高预算会增加 LLM 资源占用并可能抢占持仓复盘，必须先评估延迟、watchdog、missed opportunity 质量和 ranker 淘汰分布。
- 当前 0 订单仍不是下单链路丢单：交易执行契约 ok，failed/rejected 为 0，硬停计数为 0；ML 仍 degraded 且 `allow_live_position_influence=false`。

后续 AI 防偏要求：
- 看到候选重复时，必须按五层顺序解释：OKX/auto scan 大池、feature fetch 预算、feature valid、ranker/budget 选中、entry evidence/skip_kind；缺一层都不能下结论。
- 如果 `feature_valid_count == feature_fetch_requested_count` 且 `rank_selected_count == market_symbol_budget`，不得继续归因为行情特征异常；下一步应看 budget source、position pressure、ranked_symbol_sample、recent-analysis dedupe 与 evidence 组件。
- 不得把 `ranked_symbol_sample` 或 `market_limit_diagnostics` 用作交易准入依据；它们只能解释候选为什么没被 AI 分析，不能覆盖 evidence 门、ML readiness、风控 veto、仓位和杠杆控制。
- 后续若要改候选质量，优先做只读 A/B 观察或 shadow 级别模拟；在没有证明 missed opportunity 质量改善前，不允许直接放宽开仓门槛、提高仓位、提高杠杆或硬改 ML ready。

回滚点：
- 代码层可回滚 `services/analysis_budget.py`、`services/entry_feature_ranker.py`、`services/trading_service.py`、`tests/test_analysis_budget.py`、`tests/test_entry_feature_ranker.py`、`tests/test_trading_service_boundaries.py`；本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽。线上回滚后需重启 `bb-paper-trading.service`。

---

## 四十、Batch H 补充记录：ranker feature filter 淘汰原因落库诊断（2026-06-23）

触发原因：第 39 节上线后，最新 120m 线上漏斗出现更具体的瓶颈：`feature_valid_count=12`、`market_symbol_budget=2`，但某些轮次 `rank_selected_count` 只有 1，说明“有效特征存在”之后仍缺少“其余候选为什么没进入 hard/soft 候选”的解释。若不补这一层，后续 AI 仍可能把问题误判成特征异常、直接提高 `market_symbol_limit`，或盲目放宽开仓 evidence。

本次修复范围：
- `services/entry_feature_ranker.py`：新增 `_feature_filter_diagnostic()`，复用当前 hard/soft filter 参数，输出每个候选的 hard filter 原因、soft analysis filter 原因和关键指标，包括 notional、volume_ratio、ADX、volatility、24h change 以及各类 floor/cap。
- `services/entry_feature_ranker.py`：`rank()` diagnostics 新增 `rank_underfilled`、`rank_underfill_reason`、`filtered_out_reason_counts`、`filtered_symbol_sample`；`ranked_symbol_sample` 和 selected symbols 也带上 `filter_reasons/filter_metrics`，用来解释 secondary fill 和预算外候选。
- `services/trading_service.py`：`market_candidate_funnel` 透传 `rank_underfilled`、`rank_underfill_reason`、`rank_filtered_out_candidates`、`rank_filtered_out_reason_counts`、`filtered_symbol_sample`。
- `tests/test_entry_feature_ranker.py`：新增 rank underfill 场景，锁定低量比、低 notional、过高波动/24h change 等过滤原因计数和 filtered sample。
- `tests/test_trading_service_boundaries.py`：更新 funnel 快照测试，锁定过滤原因字段已经从 ranker 透传到 market candidate funnel。

安全边界：
- 本批只解释现有 ranker 过滤结果，不改变 hard/soft filter 条件、排序公式、feature fetch 数量、market_symbol_limit、recent-analysis dedupe、entry evidence、仓位、杠杆、ML readiness 或风控 veto。
- `analysis_volume_ratio_below_floor`、`analysis_notional_below_floor` 只能说明候选不满足当前分析质量底线；不得直接把它们当成“应该放宽量比/成交额门槛”的证据。
- 如果未来要讨论调整 ranker 参数，必须先用只读窗口证明这些过滤造成高质量 missed opportunity，而不是只因为候选少或不开仓就放宽。

本地验证：
- `pytest tests/test_entry_feature_ranker.py tests/test_trading_service_boundaries.py::test_market_candidate_funnel_snapshot_is_read_only_and_exposes_rank_dedupe_counts tests/test_analysis_budget.py tests/test_inspect_online_strategy_health.py -q`：31 passed。
- `ruff check services/entry_feature_ranker.py services/trading_service.py tests/test_entry_feature_ranker.py tests/test_trading_service_boundaries.py`：no issues。
- `black --check services/entry_feature_ranker.py services/trading_service.py tests/test_entry_feature_ranker.py tests/test_trading_service_boundaries.py`：通过。
- `python -m py_compile services/entry_feature_ranker.py services/trading_service.py`：通过。

线上同步与复查：
- `python scripts/sync_to_online_server.py --split-services` 已同步 `services/entry_feature_ranker.py` 与 `services/trading_service.py` 并重启 split services；`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard 返回 `302`。
- 等待新 market 决策后，`python scripts/inspect_online_strategy_health.py --minutes 10 --summary`（`2026-06-23T12:37:18Z`）显示：31 decisions、0 orders、0 failed/rejected、0 positions_created/closed、open_positions 6、fast_loss_close_under_15m 0；交易执行契约仍 ok，硬停计数均为 0。
- 最新 `latest_candidate_funnel` 显示：`scan_symbol_count=120`、`feature_fetch_requested_count=12`、`feature_valid_count=12`、`feature_invalid_count=0`、`market_symbol_budget=2`、`rank_selected_count=2`、`rank_tradable_candidates=3`、`rank_secondary_candidates=4`、`rank_filtered_out_candidates=5`、`recent_analysis_dedupe_count=0`。
- `rank_filtered_out_reason_counts` 当前为 `analysis_volume_ratio_below_floor=4`、`analysis_notional_below_floor=3`。`filtered_symbol_sample` 示例显示 HMSTR 因 analysis notional 不足，LAB/WLFI 因 analysis volume ratio 不足，CAT/IRYS 同时因量比和 notional 不足被过滤。
- `ranked_symbol_sample` 同时显示预算外但可分析的候选：SUI、BTC、ACT、CHZ、ARB 等未入选原因为 `outside_market_symbol_budget`；其中 BTC/ACT/CHZ/ARB 是 secondary fill 或 hard filter 之后的预算外候选，不是特征拉取失败。

当前结论：
- 候选收窄现在能拆成三类：第一，feature fetch 只抓 12 个；第二，ranker hard/soft filter 会因量比、notional 等质量底线淘汰一部分；第三，剩余候选再被 `market_symbol_budget=2` 截断。
- 最新线上样本说明本轮不是“只有 1-2 个币有特征”，而是 12 个特征有效，其中 7 个进入 hard/soft 候选，2 个进入 AI market 分析，5 个被 soft analysis filter 淘汰。
- 这进一步支持“不开仓/重复候选是候选预算、ranker 质量过滤、evidence 门和 ML degraded 共同作用”的判断；不支持直接改下单链路、降 evidence 阈值、提高杠杆或硬改 ML ready。

后续 AI 防偏要求：
- 看到 `rank_selected_count < market_symbol_budget` 时，必须查看 `rank_underfilled`、`rank_filtered_out_reason_counts` 和 `filtered_symbol_sample`；不能只看 `feature_valid_count`。
- 看到 `rank_selected_count == market_symbol_budget` 时，必须查看 `ranked_symbol_sample` 中的 `outside_market_symbol_budget`，判断是否存在预算截断候选；不能把所有未分析候选都归因为过滤失败。
- 如果过滤主因长期是 volume/notional 不足，下一步应结合 missed opportunity 回放看这些被过滤样本是否真的有手续费后收益机会；没有收益证据前不得降低质量底线。

回滚点：
- 代码层可回滚 `services/entry_feature_ranker.py`、`services/trading_service.py`、`tests/test_entry_feature_ranker.py`、`tests/test_trading_service_boundaries.py`；本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽。线上回滚后需重启 `bb-paper-trading.service`。

---

## 四十一、Batch H 补充记录：候选漏斗窗口聚合与窄输出巡检（2026-06-23）

触发原因：第 38-40 节已经能看 latest candidate funnel，但 `inspect_online_strategy_health --summary` 输出包含大量明细，容易被 SSH 20,000 字符输出上限截断。后续 AI 若只看到截断片段，仍可能误判为“没有全市场筛选”“特征异常”或“下单链路失败”。本节新增只读窄输出和窗口聚合证据，专门约束候选重复/不开仓问题的诊断顺序。

本次修复范围：
- `scripts/inspect_online_strategy_health.py`：新增远端 `MARKET_SYMBOL_ONLY` 输出模式和本地 `--market-symbol-only` 参数，远端直接输出精简 market symbol 与 candidate funnel JSON，避免本地解析被截断的远端大 JSON。
- `scripts/inspect_online_strategy_health.py`：`candidate_funnel_window` 已聚合最近窗口的 scan、feature fetch、feature valid、market budget、rank selected、rank filtered out、recent analysis dedupe、budget source、market limit policy、underfill reason、filtered reason、selected/outside-budget/filtered symbol 分布。
- `scripts/inspect_online_strategy_health.py`：窄输出压缩 `latest_candidate_funnel`，只保留 symbol、score、net_score、selected、non_selected_reason、selection_tier、filter_reasons、volume_ratio、ADX、24h change、notional 和 market limit 关键字段，不再输出每个样本的大块阈值配置。
- `tests/test_inspect_online_strategy_health.py`：新增/更新测试，锁定 `--market-symbol-only` 远端分支、关键风控/ML guard 保留、明细样本压缩以及 budget 大字段不外泄。

安全边界：
- 本批只改变巡检脚本输出形态和只读聚合，不改变候选扫描、feature fetch 数量、ranker 参数、market_symbol_limit、recent-analysis dedupe、entry evidence、仓位、杠杆、ML readiness、模型权重或风控 veto。
- `candidate_funnel_window` 和 `--market-symbol-only` 只能作为诊断证据，不得作为交易准入、提杠杆、放宽 evidence 或强制 ML ready 的依据。
- 看到候选重复时，必须先看窗口聚合，再看 latest funnel；不得只凭一条 latest 或被截断日志下结论。

本地验证：
- `pytest tests/test_inspect_online_strategy_health.py -q`：18 passed。
- `ruff check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：no issues。
- `black --check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：通过。
- `python -m py_compile scripts/inspect_online_strategy_health.py`：通过。

线上同步与复查：
- `python scripts/sync_to_online_server.py --split-services --skip-restart` 已同步 `scripts/inspect_online_strategy_health.py`，未重启交易服务；该脚本只用于线上只读诊断输出。
- `python scripts/inspect_online_strategy_health.py --minutes 60 --market-symbol-only`（`2026-06-23T12:56:35Z`）完整输出未再被截断：215 decisions、65 market decisions、4 market entry decisions、0 orders、0 failed/rejected、0 positions_created/closed、open_positions 6、fast_loss_close_under_15m 0。
- 同一 60m 窗口：`market_unique_symbol_count=31`、`market_entry_unique_symbol_count=3`、`candidate_funnel_sample_count=57`；`scan_symbol_count` median 120、`feature_fetch_requested_count` median 12、`feature_valid_count` median 12、`feature_invalid_count` zero 57，说明全市场扫描和特征拉取不是当前主要瓶颈。
- 同一 60m 窗口：`market_symbol_budget` median 2、`rank_selected_count` median 2、`market_feature_after_dedupe_count` median 2、`rank_filtered_out_candidates` p75 8、`rank_underfilled_count=1`。这说明主要收窄发生在 budget/ranker 之后，而不是没有全市场候选。
- 同一 60m 窗口 filtered reason 聚合：`analysis_notional_below_floor=132`、`analysis_volume_ratio_below_floor=127`、`recent_abnormal_wick=12`、`analysis_adx_below_floor=8`、`analysis_volatility_above_cap=4`、`analysis_day_change_above_cap=2`。
- 120m 精简复核：437 decisions、130 market decisions、36 market unique symbols、5 market entry candidates、3 market entry unique symbols、58 candidate funnels；scan median 120、feature fetch median 12、feature valid median 12、market budget median 2、rank selected median 2、feature invalid zero 58、rank filtered out p75 8、recent dedupe positive 2。
- 120m budget 复核：latest `market_limit_policy=position_first_low_risk_underfilled`，`configured_market_symbol_limit=8` 但 `selected_market_symbol_limit=2`，`roster_underfilled=true`。交易执行契约仍 `ok`，`executed_entry_count=0`；ML 仍 `degraded` 且 `allow_live_position_influence=false`。

当前结论：
- 现在可以明确回答用户：当前系统有全市场扫描池，最近窗口不是只盯固定几个交易对；但 AI 深度 market 分析链路会从 120 个扫描候选收窄到 12 个 feature fetch，再由 budget/ranker 通常收窄到 2 个送入 AI。
- 协作记录里 market 分析交易对看起来相似，主要来自 `market_symbol_budget=2`、ranker 质量过滤、position-first 预算策略以及 entry evidence/skip_kind 的共同作用；这可能是长期不开仓的上游因素之一，但不是下单链路丢单，也不是没有全市场筛选。
- 当前仍不支持直接放宽开仓门槛、提高仓位/杠杆或硬改 ML ready；若要优化候选质量，下一步应做只读 missed opportunity/filtered-out 关联回放，证明被过滤或预算外候选确实有手续费后收益机会。

后续 AI 防偏要求：
- 每次解释“候选为什么重复/为什么不开仓”，必须按窗口聚合顺序看：`scan_symbol_count`、`feature_fetch_requested_count`、`feature_valid_count`、`market_symbol_budget`、`rank_selected_count`、`rank_filtered_out_reason_counts`、`outside_budget_symbol_counts`、`market_entry_skip_kind_counts`、交易执行契约、ML readiness。
- 如果 `scan_symbol_count` 和 `feature_valid_count` 正常，不得再笼统归因于“行情特征异常”或“没有全市场筛选”。
- 如果 `orders=0` 且交易执行契约 `ok`、failed/rejected 为 0，不得再把问题写成下单接口失败；应继续追 entry evidence、skip_kind、ML degraded 和候选预算/ranker 质量。
- 涉及乱码判断时，必须先用 UTF-8 文件读取、`tests/test_no_mojibake_source.py` 和线上 `runtime_text_integrity` 复核；不得把 PowerShell 终端渲染、补丁上下文匹配失败或截断日志称作“代码里有乱码”。

回滚点：
- 代码层可回滚 `scripts/inspect_online_strategy_health.py` 与 `tests/test_inspect_online_strategy_health.py`；本批无 DB 迁移、无历史覆盖、无服务重启、无真实交易参数放宽。

---

## 四十六、Batch H 补充记录：过滤/预算外候选结果回放与窄输出二次压缩（2026-06-24）

触发原因：第 41 节已经证明当前不是“没有全市场筛选”，但仍缺少一个关键证据：被 ranker 过滤或被 market budget 截断的 symbol，后续是否真的变成了正费后收益 entry 候选。没有这层回放，后续 AI 仍可能凭“候选少/不开仓”直接降低 volume/notional 底线、扩大预算或放宽 entry evidence。首次加入回放字段后，`--market-symbol-only` 线上输出仍接近 20,000 字符截断上限，因此同步做二次压缩。

本次修复范围：
- `scripts/inspect_online_strategy_health.py`：新增只读 `candidate_filter_outcomes`，把近窗口 `market_candidate_funnel.ranked_symbol_sample/filtered_symbol_sample` 中未入选 symbol，与同窗口后续 market entry 决策做关联，统计 `market_entry_after_filter_count`、`positive_expected_net_after_filter_count`、`executed_after_filter_count`、skip kind、evidence tier、expected net stats 和少量示例。
- `scripts/inspect_online_strategy_health.py`：压缩 `--market-symbol-only` 输出，`candidate_funnel_window` 只保留关键统计中位数/p75/max、Top 原因和少量 symbol；`latest_candidate_funnel` 样本从 5 条降到 2 条，filter/reason Top 从 12 降到 6，避免远端输出截断。
- `scripts/inspect_online_strategy_health.py`：`--summary` 也改用同一套窄版 market diagnostics，避免摘要模式再次输出完整 funnel 样本而被截断。
- `tests/test_inspect_online_strategy_health.py`：新增/更新测试，锁定回放字段存在、只读边界、输出压缩和大字段不外泄。

安全边界：
- 本批只做只读诊断和输出压缩，不改变候选扫描、feature fetch 数量、ranker 参数、market_symbol_limit、recent-analysis dedupe、entry evidence、仓位、杠杆、ML readiness、模型权重或风控 veto。
- `positive_expected_net_after_filter_count` 只能说明“被过滤/预算外候选值得进一步离线复核”，不能作为直接放宽 quality floor、entry evidence、仓位、杠杆或 ML readiness 的依据。
- 如果该计数长期为 0，优先说明过滤/预算外样本没有在同窗口形成正费后 entry 候选；不得继续把不开仓简单归咎为“错过了大量好币”。

本地验证：
- `python -m py_compile scripts/inspect_online_strategy_health.py`：通过。
- `pytest tests/test_inspect_online_strategy_health.py -q`：21 passed。
- `pytest tests/test_inspect_online_strategy_health.py tests/test_entry_feature_ranker.py tests/test_analysis_budget.py tests/test_trading_service_boundaries.py -q`：159 passed。
- `ruff check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：no issues。

线上只读复查：
- `python scripts/inspect_online_strategy_health.py --minutes 20 --market-symbol-only` 已完整返回，未再被 20,000 字符截断；`--summary` 也已改为窄版 market diagnostics 并完整返回。
- 20m 窗口：73 decisions、17 market decisions、0 market entry decisions、0 orders、0 failed/rejected、0 positions_created/closed、open_positions 5、fast_loss_close_under_15m 0。
- 交易执行契约：`status=ok`，`executed_entry_count=0`、`contract_violation_count=0`、`weak_evidence_executed_count=0`、`negative_expected_executed_count=0`、`fast_loss_without_strong_exit_count=0`。
- ML readiness：仍为 `degraded`，`allow_live_position_influence=false`；阻塞项为 `long_pr_auc_below_threshold`、`long_top_return_below_threshold`、`short_pr_auc_below_threshold`；`long_pr_auc=0.335069`、`short_pr_auc=0.407735`、`top_long_avg_return_pct=-0.052489`、`top_short_avg_return_pct=0.172264`。
- 候选窗口：`scan_symbol_count` median 120，`feature_fetch_requested_count` median 48，`feature_valid_count` median 35，`market_symbol_budget` median 8，`rank_selected_count` median 4，`rank_filtered_out_candidates` median 30，`recent_analysis_dedupe_count` median 0。
- ranker 过滤主因：`analysis_notional_below_floor=410`、`analysis_volume_ratio_below_floor=250`、`recent_abnormal_wick=21`、`analysis_adx_below_floor=12`、`analysis_volatility_above_cap=5`、`analysis_day_change_above_cap=2`。
- 过滤/预算外回放：`sampled_symbol_count=71`、`sampled_occurrence_count=140`、`market_entry_after_filter_count=0`、`positive_expected_net_after_filter_count=0`、`executed_after_filter_count=0`。当前窗口没有证据表明被过滤/预算外 symbol 后续形成了正费后 entry 候选。

当前结论：
- 当前线上 20m 窗口不是“只扫固定几个币”：scan 120，feature fetch 48，median valid 35。
- 也不是“下单链路坏了”：orders=0 的同时交易执行契约 ok，failed/rejected/contract violation 都是 0。
- 当前不开仓/候选少的直接证据更偏向：ranker 质量底线（notional/volume/异常影线等）大量过滤、hard/secondary 候选不足导致 rank underfilled、entry 候选为 0，以及 ML degraded 仍禁止影响实盘仓位。
- 因为 `positive_expected_net_after_filter_count=0`，当前窗口不支持通过放宽质量底线、扩大仓位/杠杆或硬改 ML ready 来“补成交”。下一步应继续扩大只读观察窗口，或做离线回放评估过滤样本真实手续费后收益，再决定是否需要调整 ranker/budget。

后续 AI 防偏要求：
- 解释不开仓时必须同时看：`market_entry_after_filter_count`、`positive_expected_net_after_filter_count`、`rank_filtered_out_reason_counts`、`rank_underfilled_count`、`ML blocking_reason_codes` 和交易执行契约；不能只看某个 symbol 重复或 orders=0。
- 如果 `positive_expected_net_after_filter_count=0`，不得把“过滤太严”当成已证明根因；只能说当前窗口未证明过滤漏掉正费后 entry。
- 如果后续该计数转正，也只能进入离线复核或 shadow/canary 设计，仍不能绕过 selected-side expected net、entry evidence、ML readiness、仓位、杠杆和 OKX 风控。

回滚点：
- 代码层可回滚 `scripts/inspect_online_strategy_health.py` 与 `tests/test_inspect_online_strategy_health.py`；本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽。

---

## 四十七、Batch H 补充记录：Scrapling 源配置遗留形态修复与系统巡检台账清零硬故障（2026-06-24）

触发原因：继续复核 Dashboard 健康面时，线上真实 Dashboard 环境下 `/api/system/self-check` 已为 `ok`，但系统巡检 issue ledger 仍显示 `model_training` 为唯一 `unresolved`。同环境只读探针确认根因不是模型服务不可用，也不是系统巡检接口坏，而是 `EXTERNAL_EVENT_SCRAPER_SOURCES` 的历史写入形态异常：整个 JSON 源列表被包进单个 `{ "url": "...整段 JSON..." }` 中，导致 Scrapling 把整段 JSON 当作 URL，报 `URL is too long`，进而把外部事件采集误判为已启用但无有效 HTTPS 公网采集源。

本次修复范围：
- `config/settings.py`：新增 `parse_external_event_scraper_sources_value()` 复用解析器，兼容标准 JSON 源列表、单个 dict、逗号 legacy 列表，以及“整段 JSON 源列表被错误包进 url 字段”的历史形态。
- `services/external_event_service.py`：运行时热加载 `.env` 时复用同一解析器，避免长进程热加载再次把线上源列表读成单个超长 URL。
- `tests/test_external_event_scraper.py`：新增回归测试，锁定 Settings 初始化和 ExternalEventService 热加载都能从该遗留形态恢复为真实源列表。

安全边界：
- 本批只修复外部事件采集源配置解析和诊断准确性，不改变开仓阈值、证据 tier、候选排序、仓位、杠杆、平仓、ML readiness、模型权重、专家路由或风控 veto。
- `model_training` 从 unresolved 变为 observing，不代表 ML degraded 已修好；本地 ML 仍必须按 PR-AUC、高分组收益、样本质量和 readiness 状态机决定是否能影响实盘仓位。
- Scrapling 外部事件源恢复 active 只能增强文本事件样本来源，不能直接驱动真实开仓；事件特征仍需时间戳、可信度、训练样本和影子验证。

本地验证：
- `pytest tests/test_external_event_scraper.py::test_external_event_sources_recover_legacy_json_wrapped_in_url tests/test_external_event_scraper.py::test_external_event_runtime_reload_recovers_legacy_json_wrapped_in_url -q`：2 passed。
- `pytest tests/test_external_event_scraper.py tests/test_data_collection_api.py tests/test_system_audit_api.py tests/test_system_self_check.py -q`：77 passed。
- `ruff check config/settings.py services/external_event_service.py tests/test_external_event_scraper.py`：no issues。
- `black --check config/settings.py services/external_event_service.py tests/test_external_event_scraper.py`：通过。

线上同步与复查：
- `python scripts/sync_to_online_server.py --split-services` 上传 `config/settings.py`、`services/external_event_service.py` 并重启成功；`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard 返回 `302`。
- Dashboard 同环境只读复查：server monitor `status=ok`、`available=true`、`remote_monitor_available=true`。
- 数据采集页 Scrapling 状态：`source_status=active`、`enabled=true`、`valid_source_count=14`、`invalid_source_count=0`、`runtime_active=true`；样例源包括 `binance_announcements`、`coinbase_blog`、`ethereum_blog`、`solana_news`、`okx_latest_announcements`、`okx_new_listings`。
- 系统自检：overall `ok`，summary 为 critical 0、warning 0、ok 15、info 1，problem_keys 为空。
- 系统巡检：overall 仍为 `warning`，但 `critical_keys=[]`；`model_training.hard_failure=false`、`hard_source_warning_count=0`、`source_warnings=[]`，只剩 CryptoPanic、CoinMarketCal、NewsAPI 未配置等可选增强源观察项。
- 问题台账：`fixed=6`、`unresolved=0`、`observing=10`、`total=16`。

当前结论：
- 这次解决的是 Dashboard 健康面中真实存在的外部事件源配置解析硬故障，系统巡检台账已无 unresolved 硬故障。
- 系统巡检 overall 仍为 warning 是正确状态：还有交易无订单观察、OKX dry-run 超时观察、策略历史遗留观察、ML 学习观察、模型/专家影子观察、缺失特征中性阻断和 missed opportunity 保守学习。
- 当前仍不能据此扩大仓位、提高杠杆、放宽 entry evidence、降低 ranker 质量底线或硬改 ML ready。

后续 AI 防偏要求：
- 看到系统巡检 warning 时，必须先区分 API 是否失败、card status、issue ledger state；`unresolved=0` 只说明当前无硬故障，不代表收益闭环完成。
- 看到 `model_training` warning 时，必须查看 `hard_failure`、`hard_source_warning_count`、`source_warnings`、`optional_source_warnings` 和 ML readiness，不得把可选增强源未配置误写成模型服务故障。
- 看到 Scrapling 配置异常时，必须检查 `EXTERNAL_EVENT_SCRAPER_SOURCES` 是否为标准 JSON 源列表，或是否被错误包进单个 `url` 字段；不得通过关闭巡检或隐藏 warning 来处理。

回滚点：
- 代码层可回滚 `config/settings.py`、`services/external_event_service.py`、`tests/test_external_event_scraper.py`；本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽。线上回滚后需重启三项 split services。

---

## 四十八、Batch H 补充记录：Local AI Tools ready 口径与巡检探针环境纠偏（2026-06-24）

触发原因：继续复验 Dashboard 健康面时，裸 `runuser -u bb -- env DATABASE_URL=... python -c ...` 探针一度显示系统巡检 `critical`、问题台账 `model_training` unresolved，并报 `deepseek-v4-pro` 外部 API 401、local-ai-tools 401、模型端点使用公网 `103.85.84.147:21840/21842`。进一步读取 `bb-dashboard.service` 主进程环境后确认：Dashboard 真实运行环境已由 `/etc/bb/bb-runtime.env` 覆盖为 loopback 端点 `127.0.0.1:18000/18001/18002`，local-ai-tools key 正确；前述 critical 是手动探针未加载服务环境造成的假阳性。随后又发现 local-ai-tools `/models/status` 已有训练包和子接口，但顶层 status 缺省/unknown 时，数据采集与巡检把它显示成 `learning_only`，容易误导后续 AI 认为本地工具仍未训练完成。

本次修复范围：
- `services/local_ai_tools_client.py`：当 `/models/status` 返回 `available=true` 且训练包可用，但没有顶层 `status` 时，客户端补充 `status="ready"`；没有训练包但子接口可用时仍保留 `heuristic_fallback_available`，不把未训练状态误判为 ready。
- `web_dashboard/api/data_collection.py`：`_local_ai_training_status()` 将 `model_bundle_available=true` 的 local-ai-tools 展示为 `ready`，并透出 `model_bundle_available`、`service_available`、`trained_at` 和 `raw_status`，避免“样本数很多 + raw unknown”被错误归为 `learning_only`。
- `web_dashboard/api/system_audit.py`：`model_training` warning 摘要改为按真实观察原因拼接；当前只剩可选增强源未配置时，摘要显示“模型服务可用；可选增强数据源未配置。”，不再泛化写“或模型仍在学习观察”。
- `tests/test_local_ai_tools_client.py`、`tests/test_data_collection_api.py`、`tests/test_system_audit_api.py`：新增回归，分别锁定训练包 ready 补状态、数据采集 ready 展示、以及系统巡检摘要不再误报学习观察。

安全边界：
- 本批只修复状态口径、巡检摘要和探针防偏，不改变开仓阈值、候选排序、entry evidence、仓位、杠杆、平仓、ML readiness 阈值、模型权重、专家路由或风控 veto。
- `local-ai-tools ready` 只表示服务器量化工具训练包和子接口可用；不代表策略已稳定盈利，也不代表可以放大仓位或降低证据门。
- 系统巡检 `warning` 仍保留，因为 CryptoPanic、CoinMarketCal、NewsAPI 等可选增强源未配置，以及模型/专家、特征覆盖、missed opportunity 等仍有观察项；不得为了页面全绿隐藏这些 warning。

本地验证：
- `pytest tests/test_local_ai_tools_client.py tests/test_data_collection_api.py tests/test_system_audit_api.py tests/test_system_self_check.py -q`：93 passed。
- `ruff check web_dashboard/api/system_audit.py tests/test_system_audit_api.py services/local_ai_tools_client.py web_dashboard/api/data_collection.py tests/test_local_ai_tools_client.py tests/test_data_collection_api.py`：no issues。
- `black --check web_dashboard/api/system_audit.py tests/test_system_audit_api.py services/local_ai_tools_client.py web_dashboard/api/data_collection.py tests/test_local_ai_tools_client.py tests/test_data_collection_api.py`：通过。

线上同步与复查：
- `python scripts/sync_to_online_server.py --split-services` 先同步 `services/local_ai_tools_client.py`、`web_dashboard/api/data_collection.py`，再同步 `web_dashboard/api/system_audit.py`，均重启成功；`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard 返回 `302`。
- 以 OS 用户 `bb`、加载 `/etc/bb/bb-runtime.env`、使用 `/data/bb/app/.venv/bin/python` 的同环境只读复查显示：`local_ai_training_status.status=ready`、`raw_status=ready`、`model_bundle_available=true`、`service_available=true`、`trained_at=2026-06-23T20:47:53.086240+00:00`、shadow 样本 19991、trade 样本 1604、text 样本 8000。
- 系统自检：overall `ok`，summary 为 critical 0、warning 0、ok 15、info 1，problem_keys 为空。
- 系统巡检：overall `warning`，但 `critical=0`、`unresolved=0`；issue ledger 为 `fixed=7`、`unresolved=0`、`observing=9`、`total=16`。`model_training.summary="模型服务可用；可选增强数据源未配置。"`，`hard_failure=false`、`runtime_probe.status=ok`、`ai_model_count=2`、`model_critical_items=[]`。
- 策略健康 120m 摘要：405 decisions、3 orders、3 filled、failed/rejected/pending 均 0、positions_created 3、positions_closed 0、open_positions 8、fast_loss_close_under_15m 0；交易执行契约 `ok`，`contract_violation_count=0`、`weak_evidence_executed_count=0`、`negative_expected_executed_count=0`、`fast_loss_without_strong_exit_count=0`。
- local ML readiness：`status=ready`、`readiness_state=ready`、`allow_live_position_influence=true`、`blocking_reason_codes=[]`；训练窗口 5541 样本，long PR-AUC 0.5501、short PR-AUC 0.5599、top long return 0.4828%、top short return 0.3177%。
- 候选诊断仍显示真实观察项：120m 窗口 scan median 120、feature fetch median 48、feature valid median 46、market budget median 8、rank selected median 8；过滤主因仍为 notional/volume/异常影线等，`positive_expected_net_after_filter_count=0`。这说明当前不支持通过放宽质量底线、仓位、杠杆或 evidence 来“补成交”。

当前结论：
- 系统巡检/系统自检当前没有硬故障：API 没坏、模型端点没坏、local-ai-tools key 没坏、问题台账 unresolved 为 0。
- 之前裸探针看到的 `model_training critical` 是探针环境错误，不是线上 Dashboard 实际故障；以后必须以服务主进程环境或等价 env 复验为准。
- 本轮修复后 local-ai-tools 不再被误显示为 `learning_only`；但这只解决状态解释，不等于收益闭环完成。

后续 AI 防偏要求：
- 复验 Dashboard、系统巡检、数据采集或本地 ML 页面时，必须使用 Dashboard 主进程同等环境：至少加载 `/etc/bb/bb-runtime.env`，以 OS 用户 `bb` 执行 `/data/bb/app/.venv/bin/python`，并避免只用裸 `DATABASE_URL=... python -c ...` 导入配置。裸探针只可用于极窄 DB 读操作，不能作为 Dashboard 健康结论。
- 如果看到模型端点是公网 `103.85.84.147:21840/21842`、`deepseek-v4-pro` 或 local-ai-tools 401，必须先检查是否没有加载 runtime env；不得立即下结论为线上服务故障。
- 看到 `model_training` warning 时必须看 `hard_failure`、`model_critical_items`、`hard_source_warning_count`、`optional_source_warning_count`、`local_ai_tools.status` 和 local ML readiness；不能只凭 warning 文字说模型坏了。
- 即使 local ML 与 local-ai-tools 都 ready，开仓、仓位和杠杆仍必须继续受 selected-side expected net、entry evidence、profit quality、loss probability、tail risk、OKX 规则和风控 veto 约束。

回滚点：
- 代码层可回滚 `services/local_ai_tools_client.py`、`web_dashboard/api/data_collection.py`、`web_dashboard/api/system_audit.py`、`tests/test_local_ai_tools_client.py`、`tests/test_data_collection_api.py`、`tests/test_system_audit_api.py`；本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽。线上回滚后需重启三项 split services。

---

## 四十九、Batch H 补充记录：已成交 entry 小单/杠杆诊断固化（2026-06-24）

触发原因：继续观察 Batch H 时，用户再次指出“不赚钱、不开仓、开仓都是小单、执行记录里杠杆/亏损/平仓时间看不清”。此前 `--entry-only` 已能看 market entry 样例，但真实已成交 entry 多数来自 `entry_candidate`，没有专门汇总订单名义金额、决策仓位、决策杠杆、证据 tier、收益质量和小单限制原因，后续 AI 容易把小单误判成下单接口失败、杠杆展示错误，或走偏到直接放大仓位。本批只补只读诊断，不改变任何交易行为。

本次修复范围：
- `scripts/inspect_online_strategy_health.py`：新增 `executed_entry_sizing_diagnostics`，在 `--entry-only` 中汇总真实已执行 entry 的 `order_status_counts`、订单名义金额、决策仓位比例、决策杠杆、expected net、profit quality、loss probability、tail risk、evidence tier、sizing quality tier 和 `sizing_reason_tag_counts`。
- `scripts/inspect_online_strategy_health.py`：已成交样本增加紧凑明细：decision `position_size_pct/suggested_leverage/executed_at/execution_price`、order `status/side/quantity/price/notional/filled_at`、evidence、sizing、`sizing_reason_tags`、notional gap/fill ratio、紧凑 execution result。
- `tests/test_inspect_online_strategy_health.py`：新增/更新回归测试，锁定已成交 entry 小单诊断字段、样本最多 12 条、大字段不外泄、远端模板包含该诊断链。

安全边界：
- 本批只做线上健康脚本的只读诊断增强，不改变开仓阈值、候选排序、entry evidence、仓位、杠杆、平仓、ML readiness、模型权重、专家路由、OKX 下单规则或风控 veto。
- 看到小单时不得直接放大 `position_size_pct`、杠杆或 notional floor；必须先证明 selected-side expected net、profit quality、loss probability、tail risk、ML readiness 和交易执行契约都支持更大仓位。
- 如果 `orders>0` 且 `failed_orders=0/rejected_orders=0`、交易执行契约 `ok`，不得把“小单”写成下单接口故障；应优先看 `sizing_reason_tag_counts` 和样本里的 sizing/evidence 原因。

本地验证：
- `pytest tests/test_inspect_online_strategy_health.py -q`：22 passed。
- `ruff check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：no issues。
- `black --check scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：通过。

线上只读复查：
- `python scripts/inspect_online_strategy_health.py --minutes 120 --entry-only`（UTC `2026-06-23T21:05:28Z`，北京 `2026-06-24 05:05:28`）完整返回新字段。
- 120m 窗口：406 decisions、95 market decisions、23 entry decisions、1 market entry decision、3 executed entries、3 orders、3 filled、failed/rejected 均 0、positions_created 3、positions_closed 0、open_positions 8、fast_loss_close_under_15m 0。
- 交易执行契约：`status=ok`、`executed_entry_count=3`、`contract_violation_count=0`、`weak_evidence_executed_count=0`、`negative_expected_executed_count=0`、`fast_loss_without_strong_exit_count=0`。
- local ML readiness：`status=ready`、`allow_live_position_influence=true`、`blocking_reason_codes=[]`、训练窗口 5541 样本，long PR-AUC 0.5501、short PR-AUC 0.5599。
- 已成交 entry 诊断：3 笔均为 `evidence_tier=exploration`、`sizing_quality=probe`、`decision_leverage=3x`；订单名义金额 median 30.408U、min 4.4996U、max 35.1434U；决策仓位比例 median 0.0021；expected net 全为正，median 0.714895；profit quality median 0.99291；loss probability median 0.4863；tail risk median 0.252338。
- 小单原因计数：`low_payoff_quality=3`，其中 `notional_floor_blocked=2`；两笔明确写明“收益质量不足或小盈大亏风险偏高，不抬高仓位”。订单均 `filled`，不是交易所拒单或本地提交失败。

当前结论：
- 最近 120m 的“小单”不是下单接口失败，也不是杠杆随意变化；实际成交订单均 filled，决策杠杆均为 3x，订单金额偏小主要来自探索档/探针档和收益质量保护。
- 这说明“能开仓”链路当前是通的，但系统仍未证明稳定盈利；Batch H 仍然需要继续观察已平仓收益、快亏无强退出、loss re-entry、弱证据执行和手续费后收益。
- 当前不支持通过直接放大仓位/杠杆来解决“不赚钱”；更优先的根因仍是提高候选质量、证据强度、收益质量与风险结构，让 entry 从 `exploration/probe` 升级到更高质量 tier 后再自然放大。

后续 AI 防偏要求：
- 每次解释“开仓小单/杠杆看起来不一致/亏损快”时，必须先跑 `--entry-only` 并读取 `executed_entry_sizing_diagnostics`：`order_status_counts`、`order_notional_stats`、`decision_leverage_stats`、`evidence_tier_counts`、`sizing_quality_tier_counts`、`sizing_reason_tag_counts`、expected net、profit quality、loss probability、tail risk。
- 若样本显示 `low_payoff_quality`、`notional_floor_blocked`、`exploration/probe`，不得把它改成“提高仓位/提高杠杆”的任务；应回到候选 evidence、ranker、ML、收益质量和风险结构。
- 若后续出现 `filled_order_count < executed_entry_count`、`missing_order_count>0`、`failed/rejected>0` 或交易执行契约 violation，才回到下单/订单关联/执行契约链路排查。

回滚点：
- 代码层可回滚 `scripts/inspect_online_strategy_health.py` 与 `tests/test_inspect_online_strategy_health.py`；文档层可回滚本节。无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽、无服务重启依赖。

追加校正（2026-06-24 05:09 北京时间）：
- `scripts/inspect_online_strategy_health.py` 将 `execution_result.raw_response` 中缺失的 `planned_order_contracts/planned_base_quantity` 从 `0.0` 改为 `null`，避免后续 AI 把“历史 execution_result 未持久化计划数量”误判成“系统计划下 0 张”。有真实字段时仍正常四舍五入输出。
- 复跑 `python scripts/inspect_online_strategy_health.py --minutes 120 --entry-only`：412 decisions、25 entry decisions、4 executed entries、4 orders、4 filled、failed/rejected 均 0、trade execution contract `ok`。已成交诊断仍显示 4 笔全部为 `exploration/probe`、决策杠杆 3x、`low_payoff_quality=4`、`notional_floor_blocked=3`，订单名义金额 median 30.408U；planned 字段缺失时已显示 `null`。

追加根因化（2026-06-24 05:24 北京时间）：
- `services/entry_payoff_quality.py`：`EntryLowPayoffQualityPolicy` 新增 `reasons()`，输出稳定原因码：`score_below_required`、`expected_net_below_min`、`profit_quality_below_min`、`raw_expected_return_negative`、`small_win_big_loss_penalty_high`、`hard_contribution_caution`、`evidence_low_payoff_quality`。原 `is_low_payoff()` 改为复用原因码，布尔行为不变。
- `services/entry_profit_risk_sizing.py`：`profit_risk_sizing` 新增 `low_payoff_reasons`，后续新决策会写入触发小单保护的具体原因。
- `scripts/inspect_online_strategy_health.py`：`--entry-only` 的 executed-entry 诊断样本和 `sizing_reason_tag_counts` 支持 `low_payoff:*` 原因聚合，避免后续只看到 `low_payoff_quality=true` 却不知道为什么。
- 本地验证：`pytest tests/test_entry_payoff_quality.py tests/test_trading_service_boundaries.py::test_entry_profit_risk_sizing_records_low_payoff_reason_codes tests/test_inspect_online_strategy_health.py -q` 为 32 passed；Ruff/Black 针对 6 个 touched 文件均通过。
- 线上同步：`python scripts/sync_to_online_server.py --split-services` 上传 `services/entry_payoff_quality.py`、`services/entry_profit_risk_sizing.py`、`scripts/inspect_online_strategy_health.py` 并重启成功，三项服务 active，Dashboard 302。
- 线上复查：部署后立即跑 `--minutes 120 --entry-only`，旧成交样本仍显示 `low_payoff_reasons=[]`，这是预期，因为这些成交在本次字段落库前生成；后续必须等新 entry 决策出现后再用 `low_payoff:*` 判断真实原因分布，不得把旧样本空原因当作修复失败。

---

## 五十、Batch H 补充记录：OKX 51155 不可交易币种重启恢复拦截（2026-06-24）

触发原因：继续观察 Batch H 时，线上出现 `RESOLV/USDT`、`COAI/USDT` 等 entry 候选已经达到可提交探针或小单条件，但 OKX 返回 `sCode=51155 local compliance restrictions`，导致真实可用窗口被交易所合规拒单浪费。此前执行服务会在当前进程内记住不可交易 symbol，但 `TradingService._load_untradable_symbol_blocks()` 重启恢复时只看 `AIDecision.execution_reason`；而线上真实样本的 `execution_reason` 已被翻译成泛化中文，不含 `51155`，原始错误只保存在 `raw_llm_response.execution_result.raw_response.raw_error`。

本次修复范围：
- `services/trading_service.py`：新增 `_decision_execution_error_text()`，只提取持久化执行结果相关字段，包括 `execution_reason`、`execution_result.status`、`execution_result.raw_response.error/raw_error/msg/code/sCode/sMsg/data` 等；不扫描整段 LLM 文本，避免无关内容误触发。
- `services/trading_service.py`：`_load_untradable_symbol_blocks()` 查询新增 `AIDecision.raw_llm_response`，并在 `execution_reason` 或 raw execution result 任一处命中 `51155/local compliance restrictions/can't trade this pair` 时恢复 `remember_untradable_symbol()`。
- `tests/test_trading_service_boundaries.py`：新增回归测试，锁定“泛化 execution_reason + raw 51155”可恢复 blocklist，同时锁定普通 `51008 Insufficient USDT margin` 不会被误当成不可交易 symbol。

安全边界：
- 本批只阻止已知不可交易 symbol 在重启后重复提交 OKX，不改变开仓阈值、entry evidence、候选排序、仓位、杠杆、平仓、ML readiness、模型权重、专家路由或风控 veto。
- `51155` block 只能说明该交易对在当前账户/地区约束下不可交易；不得把它当成策略亏损样本，也不得用它证明候选质量已经改善。
- 普通保证金不足、服务临时异常、价格保护或弱证据跳过不得被永久归为不可交易；必须继续按各自错误类别处理。

本地验证：
- `pytest tests/test_trading_service_boundaries.py::test_trading_service_restores_untradable_symbol_from_raw_execution_error tests/test_trading_service_boundaries.py::test_trading_service_does_not_restore_untradable_block_from_generic_raw_reject tests/test_trading_service_boundaries.py::test_entry_opportunity_gate_blocks_known_untradable_symbol_before_execution tests/test_entry_symbol_blocklist.py -q`：7 passed。
- `ruff check services/trading_service.py tests/test_trading_service_boundaries.py`：no issues。
- `black --check services/trading_service.py tests/test_trading_service_boundaries.py`：通过。
- `git diff --check -- services/trading_service.py tests/test_trading_service_boundaries.py`：通过。

线上同步与复查：
- 单文件部署 `services/trading_service.py` 到 `/data/bb/app/services/trading_service.py`，部署前远端备份为 `/data/bb/app/services/trading_service.py.bak-codex-1782258157`；远端 `py_compile` 通过后替换文件并重启 `bb-paper-trading.service`，服务 active。
- 远端代码复查确认包含 `_decision_execution_error_text`、`AIDecision.raw_llm_response` 查询和 `raw_llm_response.is_not(None)` 恢复条件。
- 线上同环境只读探针显示：最近 30h 有 4 条 51155 样本，`RESOLV/USDT` 与 `COAI/USDT` 的 `execution_reason_has_51155=false`、`raw_has_51155=true`；新恢复逻辑已把 `RESOLV/USDT`、`COAI/USDT` 恢复为执行前 block，`TSLA/USDT` 与 `MAGIC/USDT` 当前未误拦截。
- 部署后 `python scripts/inspect_online_strategy_health.py --minutes 20 --entry-only`：80 decisions、5 entry decisions、3 executed entries、4 orders、4 filled、failed/rejected/rejected_orders 均 0、open_positions 16、fast_loss_close_under_15m 0；交易执行契约 `status=ok`、`contract_violation_count=0`、`weak_evidence_executed_count=0`、`negative_expected_executed_count=0`。
- 同一窗口 ML readiness 为 `ready`，`allow_live_position_influence=true`；但已成交 entry 仍主要为 `evidence_tier=exploration`、`sizing_quality=probe`、`low_payoff_quality=true`、`notional_floor_blocked`，订单名义金额约 11U-17U。

当前结论：
- OKX 51155 这类“明知不可交易还重复提交”的浪费已处理到重启恢复层；后续同类 symbol 应在提交 OKX 前被 blocklist 拦截，而不是再次形成 rejected order。
- 这不等于盈利闭环完成，也不等于小单根因已解决。当前小单主因仍是 evidence tier 偏低、probe/exploration、low payoff quality 和收益质量保护。
- 下一步应继续沿 `low_payoff_reasons`、entry evidence 组件、专家确认、候选质量和已平仓费后收益复盘推进，而不是回到放大仓位/杠杆或降低证据阈值。

后续 AI 防偏要求：
- 看到 OKX rejected 时必须先区分：`51155/local compliance` 是不可交易 symbol，应进入 blocklist；`51008` 等保证金/规则错误不是不可交易 symbol，不能永久封禁。
- 复查 blocklist 必须同时看 `execution_reason` 和 `raw_llm_response.execution_result.raw_response.raw_error`；不得因为页面展示的中文原因不含 `51155` 就认为没有合规拒单。
- 若窗口里 `failed/rejected=0` 且交易执行契约 `ok`，不得把“不赚钱/小单”继续归因为 OKX 下单链路故障；应回到 evidence、profit quality、loss probability、tail risk、ML readiness 和 sizing 原因链。

回滚点：
- 代码层可回滚 `services/trading_service.py` 与 `tests/test_trading_service_boundaries.py`；线上可用备份 `/data/bb/app/services/trading_service.py.bak-codex-1782258157` 回滚并重启 `bb-paper-trading.service`。本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽。

---

## 五十一、Batch H 补充记录：原始 HOLD 的 selected-side 证据修正与健康脚本 JSON 防截断（2026-06-24）

触发原因：继续观察 Batch H 时，`--entry-only` 发现近 60 分钟 9 个 entry 中有 5 个属于 `probe_original_hold_without_independent_support`。进一步核查发现，AI 原始 expected-net 权重被置 0 是正确的，但 selected-side 证据曾混用了 `opportunity_score.expected_net_return_pct` 的聚合值，导致 SUI/USDT、LTC/USDT 等样本的 selected-side net 与 aggregate net 口径不一致。若不修正，后续 AI 容易把“原始 HOLD + probe”误读成 OKX 或 ML 问题，而不是 selected-side 证据口径问题。

本次修复范围：
- `services/entry_direction_metrics.py`：新增 `independent_probe_expert_support()` 与 `original_hold_probe_without_support()`，当 `evidence_profit_probe.triggered=true` 且 `ai_original_action=hold` 时，只有存在独立专家重试支持才允许提高 AI 侧权重。
- `services/entry_direction_metrics.py`：`selected_entry_metrics()` 明确优先使用 selected-side 的 `expected_net_return_pct/profit_quality_ratio/server_profit_expected_return_pct`，缺失时才回退 aggregate；`loss_probability/tail_risk_score` 也带上来源说明，避免 selected-side 与 opposite aggregate 混用。
- entry 诊断增加 selected-side metrics、aggregate metrics 与 opposite-side aggregate 对照，用来解释“原始 HOLD 为什么只能探针或跳过”。
- `tests/test_entry_direction_metrics.py`：新增 3 个回归，锁定 selected-side 优先、原始 HOLD 无独立支持时降权、独立支持存在时才允许进入 probe 评估。
- `scripts/inspect_online_strategy_health.py`：`--entry-only` 改为通过远端 result JSON 文件和 SFTP 拉取结果，避免 SSH stdout 20k 截断导致 `[remote stream truncated]`。
- `tests/test_inspect_online_strategy_health.py`：锁定 result-file 传输、`entry_ai_expected_return_policy_counts`、`executed_entry_sizing_diagnostics.ai_expected_return_policy_counts` 等 compact 字段。

安全边界：
- 本批不提高仓位、不提高杠杆、不降低 evidence/expected-net/profit-quality/ML readiness/OKX/risk gate。
- 原始 HOLD 只能在 selected-side 证据和独立支持足够时作为 probe 观察；不能把 aggregate expected-net 当作 selected-side 证据直接放大。
- `side_expected_net > aggregate_expected_net` 只能说明 selected-side 质量优于聚合口径，仍需同时看 profit quality、loss probability、tail risk 与 evidence tier。
- 结果文件/SFTP 只改变观察脚本传输方式，不改变交易服务行为。

本地验证：
- `pytest tests/test_entry_direction_metrics.py tests/test_entry_priority_policy.py tests/test_stale_entry_candidate_expirer.py tests/test_entry_evidence_probe.py tests/test_entry_payoff_quality.py tests/test_entry_opportunity_scoring.py tests/test_entry_price_guard.py tests/test_entry_loss_cooldown.py tests/test_inspect_online_strategy_health.py -q`：78 passed。
- `ruff check services/entry_direction_metrics.py tests/test_entry_direction_metrics.py scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：no issues。
- `black --check services/entry_direction_metrics.py tests/test_entry_direction_metrics.py scripts/inspect_online_strategy_health.py tests/test_inspect_online_strategy_health.py`：通过。
- `python -m py_compile services/entry_direction_metrics.py scripts/inspect_online_strategy_health.py`：通过。

线上同步与复查：
- 定向部署 `services/entry_direction_metrics.py` 到 `/data/bb/app/services/entry_direction_metrics.py`，远端备份为 `/data/bb/app/services/entry_direction_metrics.py.bak-codex-1782261174`；远端 `py_compile` 通过，`chown bb:bb` 后重启 `bb-paper-trading.service`，服务 active。
- 线上定点复查 UTC `2026-06-24T00:32:54Z`：20m 窗口 CRCL/USDT decision `121819` 已使用 selected-side 证据，避免把 aggregate net 当作当前方向的质量证明。
- 8m 窗口复查 UTC `2026-06-24T00:41:26Z`：37 decisions、3 entry decisions、1 executed entry、3 orders、2 filled、1 failed/rejected、positions_created 1、positions_closed 1、open_positions 21、fast_loss_close_under_15m 0；交易执行契约 `status=ok`，`contract_violation_count=0`、`negative_expected_executed_count=0`、`weak_evidence_executed_count=0`。
- 5m 窗口复查 UTC `2026-06-24T00:41:40Z`：19 decisions、2 entry decisions、1 executed entry，交易执行契约仍为 `ok`。
- 5-8m 窗口内 MET/USDT 为 `standard` market entry，不再是 `probe_original_hold_without_independent_support`；小单原因落到 `low_payoff_reasons=[expected_net_below_min, profit_quality_below_min]`，不是 OKX 故障或原始 HOLD 漏网。
- `inspect_online_strategy_health.py --entry-only` 已改为 result JSON 文件回传，避免 stdout 截断。

当前结论：
- 原始 HOLD 场景下，selected-side metrics 口径已修正；后续不应再把 aggregate expected-net 当作当前方向的放大理由。
- Batch H 仍未证明盈利闭环；market entry 仍可能因 `expected_net_below_min/profit_quality_below_min` 被压成小单或跳过。
- 下一步仍应围绕 selected-side expected net、profit quality、position sizing、market quality、ranker、ML/专家贡献和手续费后平仓收益复盘，而不是放宽仓位或杠杆。

后续 AI 防偏要求：
- 复查原始 HOLD/probe 样本时，必须同时看 `ai_expected_return_weight`、`aggregate_expected_net_return_pct`、`selected_side_quality_gate.source` 与 selected-side expected net。
- 若 `trade_execution_contract.status=ok` 且 failed/rejected 为 0，必须优先查 evidence tier、expected_net_below_min、profit_quality_below_min、loss_probability、tail_risk、ML readiness 和 sizing，不得把问题直接写成 OKX/执行链异常。

回滚点：
- 代码层可回滚 `services/entry_direction_metrics.py` 与 `tests/test_entry_direction_metrics.py`；线上可用 `/data/bb/app/services/entry_direction_metrics.py.bak-codex-1782261174` 回滚并重启 `bb-paper-trading.service`。
- 观察脚本可回滚 `scripts/inspect_online_strategy_health.py` 与 `tests/test_inspect_online_strategy_health.py`；本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易阈值放宽。

---

## 五十二、Batch I 补充记录：动态容量 0 配置归一与超仓小单扩散拦截（2026-06-24）

触发原因：继续追查“不开仓、小单、不赚钱”时，线上 `--entry-only` 短窗显示最近小单不是 OKX 下单失败，也不是原始 HOLD 漏网，而是 `low_payoff_quality=true` 与 `strategy_probe_cap_applied=true` 叠加。持仓复核链路在最近 90 分钟内有 247 次复核、5 次 exit decision 且 5 次均执行，其中 3 次为 `position_quality_capacity_release`，说明释放链路存在；但组合仍有 21 个持仓组。继续核查发现线上 `.env` 中 `MAX_OPEN_POSITIONS_PER_MODEL=0`，旧动态容量逻辑又使用 `base_limit=max(configured_limit, open_group_count + rotation_slots)`，导致超仓时容量上限跟着当前持仓数上移，系统仍可继续开不同交易对的小单或探针单，而不是先释放低质量仓位。

本次修复范围：
- `config/settings.py`：新增 `DEFAULT_MAX_OPEN_POSITIONS_PER_MODEL=20`，`max_open_positions_per_model` 默认值使用该常量；新增 validator，将非法值、空值、`0` 或负数归一为 20。
- `services/dynamic_position_capacity.py`：配置上限通过 `_configured_limit()` 归一；`base_limit` 改为固定使用配置上限，不再随 `open_group_count + rotation_slots` 扩张；当 `open_group_count >= entry_limit` 时，entry limit 不再因 rotation/release 扩张，优先进入 `over_capacity_release_first`。
- `services/trading_service.py`：动态容量 fallback 使用默认 20，不再让 `0` 进入容量判断。
- `services/strategy_learning.py`：策略学习中的持仓上限 fallback 统一使用默认 20，避免 `or 1` 或原始 `0` 把容量解释成极端小值或间接放大。
- `tests/test_settings_capacity.py` 与 `tests/test_position_quality.py`：锁定 `0` 配置归一、超仓不扩张 entry limit、未满仓时仍允许受控 rotation slot。

安全边界：
- 本批不提高仓位、不提高杠杆、不降低开仓阈值、不绕过 evidence/ML readiness/OKX/risk veto，也不把低质量探针直接改成大单。
- `MAX_OPEN_POSITIONS_PER_MODEL=0` 的最终语义是“无效配置，归一为安全默认 20”，不是无限持仓。
- 当 `open_group_count >= entry_limit` 时，短期出现“不开新交易对仓位”是预期容量纪律，不应误判成开仓链路坏了；系统应先释放低质量或超额仓位。
- 该修复只解决“超仓状态仍继续扩散小单”的一个具体根因，不代表系统已经稳定盈利。

本地验证：
- `pytest tests/test_settings_capacity.py tests/test_position_quality.py tests/test_strategy_learning.py tests/test_entry_strategy_mode.py tests/test_analysis_budget.py tests/test_trading_service_boundaries.py tests/test_position_review_priority.py tests/test_position_review_batch.py tests/test_trade_execution_contract.py tests/test_secret_utils.py -q`：246 passed。
- `python -m py_compile config/settings.py services/dynamic_position_capacity.py services/strategy_learning.py services/trading_service.py`：通过。
- `ruff check` 针对 touched files：no issues。
- `black --check` 针对 touched files：通过。
- `git diff --check` 针对 touched files：通过。

线上同步与复查：
- 定向部署并备份：
  - `/data/bb/app/config/settings.py.bak-codex-1782263454`
  - `/data/bb/app/services/dynamic_position_capacity.py.bak-codex-1782263454`
  - `/data/bb/app/services/strategy_learning.py.bak-codex-1782263454`
  - `/data/bb/app/services/trading_service.py.bak-codex-1782263454`
- 重启 `bb-paper-trading.service` 后服务 active/running。
- 线上容量诊断：`settings_max_open_positions_per_model=20`，`settings_zero_probe=20`，`open_position_parts=21`，`base_limit=20`，`target_limit=20`，`effective_limit=20`，`entry_limit=20`，`open_group_count=21`，`low_quality_count=4`，reason 包含 `over_capacity_release_first`。
- 部署后短窗 `inspect_online_strategy_health.py --minutes 10 --entry-only`：`executed_entries=0`，`positions_created=0`，`positions_closed=2`，`open_positions=21`，`failed_orders=0`，`rejected_orders=0`，trade execution contract `status=ok`，无负预期执行、无弱证据执行。

当前结论：
- 之前的动态容量逻辑会在超仓时把“当前已经很多仓”反向变成“允许更多仓”的依据，这是导致小单扩散和组合碎片化的真实根因之一。
- 修复后，系统在 21 个持仓组、上限 20 的状态下应先释放，再允许新的不同交易对 entry；这可能短期减少开仓数量，但目标是阻止低质量小单继续铺开。
- 后续必须观察持仓组降到 20 以下后，新 entry 是否仍然是 `exploration/probe`、`low_payoff_quality`、`expected_net_below_min/profit_quality_below_min`。如果仍然如此，下一根因应回到候选筛选、ranker、证据质量、收益质量、ML/专家贡献和手续费后复盘，而不是改容量或放大仓位。

后续 AI 防偏要求：
- 解释“不开仓”前必须先看 `open_group_count`、`entry_limit` 和 reason；若为 `over_capacity_release_first`，不得把它当成故障或通过降阈值绕过。
- 解释“小单”前必须先看 `executed_entry_sizing_diagnostics`、`sizing_reason_tag_counts`、`low_payoff_reasons`、`evidence_tier` 和 `sizing_quality`，不得直接提高仓位/杠杆。
- 看到 `MAX_OPEN_POSITIONS_PER_MODEL=0` 时必须按安全默认 20 解释，不得还原为无限持仓。
- 若持仓数长期不降，下一步查 `position_quality`、`position_review_priority`、`position_release_decision`、`position_review_batch` 和 release order 执行情况，而不是继续扩 entry。

回滚点：
- 代码层可回滚 `config/settings.py`、`services/dynamic_position_capacity.py`、`services/strategy_learning.py`、`services/trading_service.py` 及对应测试；线上可使用上述 `.bak-codex-1782263454` 备份回滚并重启 `bb-paper-trading.service`。
- 本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易阈值放宽。

---

## 五十三、Batch I 补充记录：market AI 吞吐诊断与预算跳过轮转（2026-06-24）

触发原因：
- 用户指出“分析协作记录里市场分析的交易对经常差不多”，并怀疑这是否是系统不开仓、开仓少、候选不优的原因之一。
- 线上 `--market-symbol-only` 证据显示，全市场扫描和 rank 链路本身存在：`scan_symbol_count=120`、`feature_fetch_requested_count=48`、`rank_selected_count` 最高 8；但 `market_analysis_progress` 显示进入 AI 前已经用掉 43-70 秒，而市场轮次软预算为 27 秒，导致每轮通常只能分析 shortlist 的第 1 个或前 1-2 个交易对。
- 这说明“候选老是差不多”的一个真实根因不是没有全市场筛选，而是 AI 分析吞吐低于 shortlist 覆盖需求；如果后续 AI 不看这个证据，容易走偏到放宽开仓阈值、提高仓位或绕过风控。

本次修复范围：
- `services/trading_service.py` 新增 `_market_analysis_progress_snapshot()` 并随 market decision 落库，记录当前 symbol 在 shortlist 中的 `processed_index`、`ranked_market_symbol_count`、`remaining_after_this_symbol`、进入 AI 前耗时、软预算和预算使用比例。该字段只读，不参与 entry permission、sizing、leverage、ML readiness 或 risk veto。
- `scripts/inspect_online_strategy_health.py` 新增 `market_analysis_progress` 聚合并在 `--market-symbol-only`、summary compact report 中输出，用于判断“ranked shortlist 有多少实际进入 AI”。
- `services/trading_service.py` 新增预算跳过轮转：当上一轮因 `_round_budget_exhausted()` 跳过后续 ranked symbols 时，进程内记录 deferred symbols；下一轮若 deferred symbol 仍在当前 shortlist 内，则从该 symbol 开始遍历，以提升覆盖面。该轮转不改变 rank 分数、不扩大 shortlist、不增加预算、不放宽 evidence/ML/OKX/risk gate。
- `market_candidate_funnel` 新增只读 `market_budget_rotation`，记录轮转是否生效、起始 symbol、deferred 数量和原因，供后续线上复查。
- `tests/test_trading_service_boundaries.py` 与 `tests/test_inspect_online_strategy_health.py` 新增/更新契约测试，锁定 progress、rotation 与 compact report 字段，避免后续诊断被删或被误用成交易放行条件。

安全边界：
- 本批不提高仓位、不提高杠杆、不降低开仓阈值、不绕过 entry evidence、profit quality、loss probability、tail risk、ML readiness、OKX 规则或高风险复核。
- 预算轮转只改变同一 ranked shortlist 的遍历起点，目标是让被预算跳过的候选获得后续分析机会；它不证明这些候选一定更赚钱，也不保证会开仓。
- 如果 deferred symbol 已经不在下一轮 shortlist 中，轮转会安全地 `applied=false`，保持当前 rank 顺序，不强行分析已经失效的旧候选。
- 看到 `market_analysis_progress.count=0` 时不能立刻判定补丁失败；需要确认窗口内是否已经产生部署后的新 market decisions。

本地验证：
- `pytest tests/test_trading_service_boundaries.py::test_market_analysis_progress_snapshot_is_read_only_and_attached tests/test_inspect_online_strategy_health.py -q`：24 passed。
- `pytest tests/test_trading_service_boundaries.py::test_market_budget_deferred_rotation_starts_from_skipped_symbol tests/test_trading_service_boundaries.py::test_market_budget_deferred_rotation_keeps_order_when_no_match tests/test_trading_service_boundaries.py::test_market_budget_deferred_symbols_are_deduped_and_clearable tests/test_trading_service_boundaries.py::test_market_candidate_funnel_snapshot_is_read_only_and_exposes_rank_dedupe_counts tests/test_trading_service_boundaries.py::test_market_analysis_progress_snapshot_is_read_only_and_attached tests/test_inspect_online_strategy_health.py -q`：28 passed。
- `pytest tests/test_market_direct_entry_processor.py tests/test_market_auto_entry_processor.py tests/test_market_queued_entry_processor.py tests/test_entry_capacity.py tests/test_inspect_online_strategy_health.py tests/test_trading_service_boundaries.py -q`：179 passed。
- `python -m py_compile services/trading_service.py scripts/inspect_online_strategy_health.py tests/test_trading_service_boundaries.py tests/test_inspect_online_strategy_health.py`：通过。
- `ruff check` 与 `black --check` 针对 touched files：通过。
- `git diff --check` 针对 touched files：通过。

线上同步与复查：
- 第一次定向部署诊断字段并备份：
  - `/data/bb/app/services/trading_service.py.bak-codex-1782265325`
  - `/data/bb/app/scripts/inspect_online_strategy_health.py.bak-codex-1782265325`
- 第二次定向部署预算轮转并备份：
  - `/data/bb/app/services/trading_service.py.bak-codex-1782266072`
  - `/data/bb/app/scripts/inspect_online_strategy_health.py.bak-codex-1782266072`
- 两次远端 `py_compile` 均通过；重启 `bb-paper-trading.service` 后服务 active，最新 PID `2940250`。
- 线上 `--minutes 5 --market-symbol-only` 复查显示 `market_analysis_progress.count=2`，`budget_used_ratio_before_ai` 中位数约 1.82，证明诊断字段已落库并确认 AI 吞吐超过软预算。
- 后续 `--minutes 8 --market-symbol-only` 显示 `market_unique_symbol_count=5`、`market_top_symbols` 覆盖 `SOL/BTC/CRCL/FIL/LPT`，比此前窗口只集中在少数 symbol 更分散；`market_budget_rotation.applied=false` 的原因是 deferred symbols 不在当前 shortlist，属于安全不命中而非故障。
- 同窗口执行契约仍为 `status=ok`，`failed_orders=0`、`rejected_orders=0`。
- `--minutes 10 --entry-only` 显示 2 笔 executed entries、2 笔 filled、0 失败/拒单；其中 BTC/USDT 是 market entry，`evidence_tier=medium`，`quality_tier=quality_override`，订单名义金额约 258.64U，说明系统已能从 market 分析产生非探针小单级别的执行样本。但这仍只是短窗样本，不能据此宣称稳定盈利。

当前结论：
- “市场分析交易对经常差不多”的根因之一已经被证据定位为 AI 分析吞吐不足，而不是全市场筛选完全没有工作。
- 轮转补丁解决的是覆盖面调度偏斜：当软预算长期只能处理 shortlist 前段时，后续轮次会优先尝试被预算跳过且仍在当前 shortlist 的候选。
- 这不等于“不赚钱”已经解决。后续仍必须继续跟踪已平仓净收益、手续费后表现、fast loss、loss re-entry、medium/normal tier 比例、server_profit 反向贡献和 position release 效果。

后续 AI 防偏要求：
- 解释“候选老是差不多”前必须先跑 `inspect_online_strategy_health.py --minutes <N> --market-symbol-only`，查看 `market_analysis_progress`、`candidate_funnel_window`、`latest_candidate_funnel.market_budget_rotation`、`rank_selected_count`、`market_feature_after_dedupe_count`。
- 若 `rank_selected_count>1` 但 `processed_index_stats.median<=1` 且 `budget_used_ratio_before_ai_stats.median>1`，应归因到 AI 吞吐/调度覆盖问题，不能直接改开仓阈值、仓位、杠杆或 risk veto。
- 若 `market_budget_rotation.applied=false` 且原因是 `deferred symbols no longer match current shortlist`，不应强行复用旧 deferred；说明市场条件变化后旧候选不再入选，应尊重最新 rank。
- 判断是否“不开仓链路坏了”必须同时看 `trade_execution_contract.status`、`failed_orders/rejected_orders`、`market_entry_skip_kind_counts`、`entry_evidence_tier_counts` 和 high risk review。若契约 ok 且拒单为 0，不得把问题写成 OKX 下单故障。
- 出现 BTC/USDT 这类 medium tier、quality override、较大名义金额样本后，也不能立刻放大默认仓位；必须等待足够已平仓样本证明手续费后收益质量和回撤受控。

回滚点：
- 代码层可回滚 `services/trading_service.py`、`scripts/inspect_online_strategy_health.py` 及对应测试；线上可使用 `.bak-codex-1782266072` 备份回滚并重启 `bb-paper-trading.service`。
- 本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易阈值放宽。

---

## 五十四、Batch I 补充记录：market AI 软预算时钟从 full round 改为 AI phase（2026-06-24）

触发原因：
- 上一批诊断和轮转上线后，线上 `--market-symbol-only` 继续显示 `rank_selected_count` 可到 8、全市场扫描与 rank 正常，但 `market_analysis_progress.processed_index_stats.max` 仍长期停在 1。
- 进一步证据显示，`round_elapsed_seconds_before_ai` 在第一个市场 AI 分析前已达到 29-93 秒，而市场 AI 软预算是 27 秒。旧逻辑把 feature fetch、strategy context、position refresh/review 等前置耗时也计入 market AI shortlist 遍历预算，导致第一个 symbol 分析完后，第二个 symbol 立即被 `_round_budget_exhausted(round_start)` 跳过。
- 这说明轮转只能改变“谁排第一”，不能解决“为什么每轮只能分析第一个”。根因是软预算时钟作用域错误，不是 shortlist 不存在，也不是应该放宽开仓阈值。

本次修复范围：
- `services/trading_service.py` 新增 `_market_ai_budget_exhausted(market_ai_started_at)`；市场 AI 循环在 `market_feature_items` 生成后记录 `market_ai_started_at = datetime.now(UTC)`，后续 soft budget 判断改为使用该 AI phase 起点。
- `_market_analysis_progress_snapshot()` 新增 `market_ai_started_at` 参数，并同时写入：
  - `full_round_elapsed_seconds_before_ai`
  - `market_ai_elapsed_seconds_before_symbol`
  - `market_ai_budget_used_ratio_before_symbol`
  - `budget_clock_scope="market_ai_phase"`
- 保留旧字段 `round_elapsed_seconds_before_ai` 与 `budget_used_ratio_before_ai` 的可读性，但预算使用比例现在以 market AI phase 为准；同时诊断文案明确软预算从 market AI phase 开始，不是 full round startup。
- `market_analysis_budget` 与 warnings 同步输出 `market_ai_elapsed_seconds`、`full_round_elapsed_seconds` 和 `budget_clock_scope`，避免后续把完整轮次耗时与 AI 遍历预算混用。
- `scripts/inspect_online_strategy_health.py` 新增上述新字段的聚合；新字段只统计实际存在的样本，避免部署前旧 raw response 缺字段时被当作 0 秒污染窗口判断。
- `tests/test_trading_service_boundaries.py` 新增 `test_market_ai_budget_clock_ignores_pre_ai_round_work`，锁定 full round 已超 27 秒时，market AI phase 仅运行 2 秒不应触发 AI soft budget。
- `tests/test_inspect_online_strategy_health.py` 锁定 compact report 与远端模板包含 `market_ai_elapsed_before_symbol_stats` 和 `budget_clock_scope`。

安全边界：
- 本批只改 market AI shortlist 遍历的软调度预算口径，不提高仓位、不提高杠杆、不降低 evidence/expected-net/profit-quality/ML readiness/OKX/risk gate。
- full round watchdog 仍独立存在，未被 market AI soft budget 替代；本批只避免前置工作耗尽 AI phase 的覆盖预算。
- 该修复提高“同一轮可实际进入 AI 的候选覆盖数”，不等于保证开仓，也不等于证明稳定盈利。
- 看到 `processed_index_stats.max>1` 只能说明吞吐覆盖问题缓解；是否能赚钱仍必须用 filled orders、closed PnL、fast loss、loss re-entry、contract violations 和手续费后收益验证。

本地验证：
- `pytest tests/test_trading_service_boundaries.py -q`：131 passed。
- `pytest tests/test_inspect_online_strategy_health.py -q`：23 passed。
- `pytest tests/test_market_direct_entry_processor.py tests/test_market_auto_entry_processor.py tests/test_market_queued_entry_processor.py tests/test_entry_capacity.py tests/test_inspect_online_strategy_health.py tests/test_trading_service_boundaries.py -q`：180 passed。
- `python -m py_compile services/trading_service.py scripts/inspect_online_strategy_health.py tests/test_trading_service_boundaries.py tests/test_inspect_online_strategy_health.py`：通过。
- `ruff check` 与 `black --check` 针对 touched files：通过。
- `git diff --check` 针对 touched files：通过。

线上同步与复查：
- 定向部署并备份：
  - `/data/bb/app/services/trading_service.py.bak-codex-1782268559`
  - `/data/bb/app/scripts/inspect_online_strategy_health.py.bak-codex-1782268559`
  - 巡检脚本二次口径修正备份：`/data/bb/app/scripts/inspect_online_strategy_health.py.bak-codex-1782268830`
- 远端 `py_compile` 通过；重启 `bb-paper-trading.service` 后 active，PID `2971781`。`bb-dashboard.service` 与 `bb-model-tunnels.service` 复查均 active。
- 部署后新样本 `inspect_online_strategy_health.py --minutes 3 --market-symbol-only`：
  - `market_decision_count=4`
  - `market_unique_symbol_count=4`
  - `processed_index_stats.max=3`
  - `processed_index_stats.median=2`
  - `budget_clock_scope="market_ai_phase"`
  - `full_round_elapsed_before_ai_stats.median=66.233`
  - `market_ai_elapsed_before_symbol_stats.median=15.731`
  - `market_ai_budget_used_ratio_before_symbol_stats.median=0.582639`
- 这证明旧问题“full round 前置耗时已超 27 秒导致每轮只能分析第一个候选”已缓解：同一短窗内 market AI 已处理到 shortlist 第 3 个候选。
- `inspect_online_strategy_health.py --minutes 10 --entry-only`：`trade_execution_contract.status=ok`、`executed_entries=0`、`weak_evidence_executed_count=0`、`negative_expected_executed_count=0`、`fast_loss_count=0`、`fast_loss_without_strong_exit_count=0`。同时暴露 1 笔历史窗口内 `SAHARA/USDT` rejected order，后续根因应查执行数量/OKX 规则链路，不应把本轮吞吐修复误当作盈利闭环完成。

当前结论：
- 本轮解决的是市场 AI 覆盖吞吐的一个真实根因：soft budget 时钟过去错误覆盖了前置阶段。
- 修复后，全市场 scan/rank 后的 ranked shortlist 能在同一轮进入更多 AI 分析，短窗证据已从“最多第 1 个”提升到“最多第 3 个”。
- 这仍不是盈利证明。系统当前仍有 open_positions=20、market_entry_decisions=0、executed_entries=0 的短窗，且有一条 rejected order 需要继续追查。

后续 AI 防偏要求：
- 解释“AI 又只看一个币”时，必须同时看 `full_round_elapsed_before_ai_stats` 与 `market_ai_elapsed_before_symbol_stats`；若 full round 高但 market AI phase 低，不得再把前置耗时当作 AI 遍历预算耗尽。
- 若 `processed_index_stats.max` 已大于 1 但仍不开仓，下一步应查 `market_entry_skip_kind_counts`、entry evidence、expected-net、profit quality、risk/OKX execution，而不是继续调 market AI budget。
- 若出现 rejected/failed order，必须从 `order_execution_result`、OKX rule normalization、planned contracts/base quantity、exchange rejection 证据查起，不得用“提高仓位/杠杆”掩盖执行链问题。
- 不得把 `processed_index_stats.max=3` 宣称为赚钱闭环完成；盈利闭环必须由手续费后已平仓 PnL、回撤、快亏和复开纪律共同证明。

回滚点：
- 代码层可回滚 `services/trading_service.py`、`scripts/inspect_online_strategy_health.py` 及对应测试。
- 线上可使用 `/data/bb/app/services/trading_service.py.bak-codex-1782268559` 与 `/data/bb/app/scripts/inspect_online_strategy_health.py.bak-codex-1782268830` 回滚，并重启 `bb-paper-trading.service`。
- 本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易阈值放宽。

---

## 五十五、Batch I 补充记录：OKX 市价单最大张数预提交拦截（2026-06-24）

触发原因：
- 上一轮 `--entry-only` 复查暴露 `SAHARA/USDT` rejected order。表面看本地订单记录 `quantity=0`，但进一步读取 decision raw response 后确认系统并不是提交 0 张，而是计划提交 `45135` 张、名义金额约 `774U`，OKX 返回 `51202 Market order amount exceeds the maximum amount`。
- 该问题不是 evidence/ML/仓位阈值问题，而是 OKX instrument 规则缺少“单笔市价单最大数量/张数”预提交校验，导致订单已经到交易所后才被拒。

本次修复范围：
- `executor/okx_executor.py` 新增 `_amount_market_max()`，读取 OKX instrument `info.maxMktSz` 及兼容字段，并在 `_entry_order_rule_snapshot()` 中输出：
  - `amount_max_market_contracts`
  - `market_order_within_max_size`
  - 修正后的 `pre_submit_valid`
- 新增 `_entry_market_order_size_rejection_result()`：当入场市价单计划张数超过 OKX 单笔市价单最大张数时，本地返回 `system_pre_submit_market_order_max`，`system_pre_submit_rejection=true`，`okx_rejection=false`，并保留 planned contracts/base quantity、max market contracts 和 rule snapshot。
- 该预检放在杠杆确认后、止盈止损参数与 `create_order()` 之前，避免明知会被 OKX 拒绝的订单继续提交。
- `tests/test_executor_error_safety.py` 新增 max market size 规则快照和 pre-submit 拦截测试，断言超过 `maxMktSz` 时不会调用 `create_order()`。

安全边界：
- 本批不自动拆单、不自动缩小为最大市价张数、不提高仓位、不提高杠杆、不放宽 evidence/expected-net/risk gate。超过交易所单笔市价上限时先明确拦截，避免改变原始交易计划的风险收益结构。
- 本地订单 `quantity=0` 在 rejected 情况下仍表示“没有成交数量”，不是“系统提交了 0 张”；后续解释执行记录时必须查看 raw response 中的 `planned_order_contracts`、`planned_base_quantity`、`execution_blocker`、`okx_rejection`、`system_pre_submit_rejection`。

本地验证：
- `python -m pytest tests/test_executor_error_safety.py -q`：16 passed。
- `pytest tests/test_trade_execution_contract.py tests/test_market_direct_entry_processor.py tests/test_market_auto_entry_processor.py tests/test_market_queued_entry_processor.py tests/test_executor_error_safety.py -q`：41 passed。
- `python -m py_compile executor/okx_executor.py tests/test_executor_error_safety.py`：通过。
- `ruff check` 与 `black --check` 针对 touched files：通过。
- `git diff --check` 针对 touched files：通过。

线上同步与复查：
- 定向部署并备份：`/data/bb/app/executor/okx_executor.py.bak-codex-1782269504`。
- 远端 `py_compile` 通过；重启 `bb-paper-trading.service` 后 active，PID `2985899`。
- 线上 `inspect_online_strategy_health.py --minutes 10 --entry-only`：`trade_execution_contract.status=ok`、`market_entry_decisions=1`、`executed_entries=1`、`orders=1`、`filled_orders=1`、`failed_orders=0`、`rejected_orders=0`、`positions_created=1`、`fast_loss_close_under_15m=0`。
- 同窗口成交样本：`ZRO/USDT` market short，`evidence_tier=small`，order notional 约 `92.92U`，`strategy_probe_cap_applied=true`，说明正常市价单仍可通过，且本批没有破坏执行链。

当前结论：
- `SAHARA/USDT` 那类 `51202 Market order amount exceeds the maximum amount` 的执行失败链路已补预提交规则，后续同类超出 OKX 单笔市价单上限的订单应在本地被解释为 `system_pre_submit_market_order_max`，不再打到 OKX 产生 rejected order。
- 这不代表“小单/盈利”问题完成。当前线上仍有 `open_positions=21`，新成交为小仓探针，后续必须继续观察释放低质量仓位、已平仓 PnL 和手续费后收益。

后续 AI 防偏要求：
- 看到 rejected/quantity=0，不得直接断言“系统提交了 0 张”或“开仓逻辑坏了”；必须读取 decision raw response 和 order raw/diagnostics，区分 `okx_rejection`、`system_pre_submit_rejection`、`planned_order_contracts`、`amount_max_market_contracts`。
- 若后续出现 `system_pre_submit_market_order_max`，不得通过提高杠杆或仓位绕过；应分析该交易对价格/合约面值/OKX maxMktSz 与目标名义金额是否适配，必要时设计受控拆单方案并重新评估风险，而不是静默缩小或放大。
- 若 failed/rejected 为 0 但仍小单，下一步应回到容量释放、strategy probe cap、evidence tier、profit quality 和 closed-PnL 复盘，不再把 OKX 执行链当首要根因。

回滚点：
- 代码层可回滚 `executor/okx_executor.py` 与 `tests/test_executor_error_safety.py`。
- 线上可使用 `/data/bb/app/executor/okx_executor.py.bak-codex-1782269504` 回滚并重启 `bb-paper-trading.service`。
- 本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易阈值放宽。

---

## 五十六、Batch H/I 补充记录：线上重启验证与运行时长口径防偏（2026-06-25）

触发原因：
- 用户指出：此前已说明线上服务器更新并重启，但 Dashboard 页面显示的“运行时长”看起来仍是更新前累计值，怀疑线上没有真正更新或重启成功。
- 只看页面单一“运行时长”会误导后续 AI：顶部运行时长、交易主进程运行时长、服务器主机运行时长和 Dashboard 看板进程运行时长不是同一个口径。若不写入总控，后续 AI 容易把“运行时长未归零”误判为部署失败，或者反过来只看 uptime 不核对真实代码文件与 systemd 状态。

只读核验结果：
- `bb-dashboard.service` 已重启成功：`ExecMainStartTimestamp/ActiveEnterTimestamp=2026-06-24 17:49:25 UTC`，北京时间为 `2026-06-25 01:49:25`；`MainPID=3713971`，命令为 `.venv/bin/python scripts/run_dashboard.py`。
- `bb-paper-trading.service` 交易主进程启动时间为 `2026-06-24 15:52:31 UTC`，北京时间为 `2026-06-24 23:52:31`；`MainPID=3614269`。
- `bb-model-tunnels.service` 启动时间为 `2026-06-24 15:52:27 UTC`；`MainPID=3614250`。
- `/data/bb/app/data/trading_runtime_status.json` 心跳新鲜：核验时 `heartbeat_age_seconds≈1.9`，`running=true`，`paused=false`，`last_round_error/market_last_error/position_last_error` 均为空。
- 交易心跳里的 `uptime_seconds=7476` 来自交易主进程 `started_at=2026-06-24T15:52:58.723240+00:00`，不是 Dashboard 看板进程启动时间。
- 线上代码文件时间已更新：`/data/bb/app/web_dashboard/api/system_audit.py` 为 `2026-06-24 17:44:39 UTC`，`/data/bb/app/services/okx_trade_fact_integrity.py` 为 `2026-06-24 17:49:23 UTC`。

当前结论：
- 这次 Stage 1 审计链的线上代码已更新，`bb-dashboard.service` 已重启并加载新代码；页面运行时长未归零不是部署失败证据。
- Dashboard 顶部 `stat-uptime` 当前主要取 `/api/dashboard/summary` 中的 `uptime_source=split_process_heartbeat` 与交易心跳 `uptime_seconds`，因此只重启 Dashboard 看板服务时该数值不会归零。
- 服务器监控页展示的平台 uptime 可能是主机 uptime，也不会因应用服务重启归零。

后续 AI 防偏要求：
- 判断线上是否更新成功时，必须至少核对四类证据：systemd `MainPID/ActiveEnterTimestamp`、远端目标文件 mtime 或校验、远端 `py_compile`/健康响应、Dashboard/API 实际返回字段；不得只看页面运行时长。
- 解释“运行时长没变”前，必须先区分：
  - Dashboard 看板服务运行时长；
  - 交易主进程运行时长；
  - 模型隧道服务运行时长；
  - 服务器主机运行时长。
- 若只部署/重启 `bb-dashboard.service`，不得期待交易主进程 `trading_runtime_status.json.uptime_seconds` 归零。
- 若部署涉及交易行为或 `services/trading_service.py`，才应重点观察 `bb-paper-trading.service` 的重启时间、交易心跳 `started_at/heartbeat_at`、策略健康和执行契约。
- 后续应把页面展示改为“三类运行时长分离”：看板服务、交易主进程、服务器主机，并在系统巡检或服务器监控中显示各服务 PID、启动时间和 uptime 来源，避免用户和 AI 继续被单一字段误导。

安全边界：
- 本节只记录只读核验结论和后续展示改进要求；未修改代码、未重启服务、未修改数据库、未改变任何开仓/仓位/平仓/杠杆/模型权重/风控阈值。
- 后续若实现“三类运行时长分离”，应作为 Dashboard 可观测性改动处理，不能顺手改交易参数或重启交易服务，除非该批明确需要。

回滚点：
- 本节为文档记录，可直接回滚本节文本；无运行时代码、DB、模型 artifact 或真实交易参数变更。

---

## 五十七、Batch I 二期阶段 1 补充记录：OKX 分拆平仓加权均价审计修正（2026-06-25）

触发原因：
- 二期阶段 1 的 OKX/本地事实口径审计上线后，`okx_trade_fact_integrity` 从最初 10 个严重错配收敛到 1 个 warning：`USAR/USDT` 订单 `2662`、决策 `122960` 显示本地订单价格 `3.86666667`，raw OKX 顶层价格 `3.95`，被标记为 `execution_price_mismatch`。
- 用户此前已多次指出 OKX 后台价格与本地显示不同、USAR 这类盈利单不能被错误口径污染。本批必须查清真实成交事实，不能直接改历史数据，也不能把 USAR 异常收益拿去放大仓位。

只读根因核验：
- 订单 `2662` 是 USAR/USDT 多头分拆平仓，`exchange_order_id=3683484479789965312,3683484609846943744,3683484676586708992`，本地成交数量 `24`，本地价格 `3.8666666667`。
- 决策 raw 中 `split_exit_order=true`，包含 3 个 OKX 子平仓单：
  - 子单 1：10 张，成交价 `3.85`；
  - 子单 2：10 张，成交价 `3.85`；
  - 子单 3：4 张，成交价 `3.95`。
- 子单按成交张数加权均价为 `(10*3.85 + 10*3.85 + 4*3.95) / 24 = 3.8666666667`，与本地订单价、决策 execution_result.price 和 Position `1655.current_price` 一致。
- raw_response 顶层 `average/price=3.95` 只是最后一个子单价格，因为 executor 把 `last_order` 合并到了父级 raw_response；它不是整笔 24 张分拆平仓的均价。
- 因此该 warning 不是本地价格算错，而是审计服务把最后子单价格误当作父级分拆订单均价。

本次修复范围：
- `services/okx_trade_fact_integrity.py`：新增 `_execution_result_payload()`、`_execution_fact_price()`、`_weighted_split_exit_price()`。
- 审计遇到 `split_exit_order=true` 时，优先按 `raw_response.split_chunks[].closed_contracts * price` 计算父级加权均价；若分拆明细缺失，再退回 execution_result 父级 price；普通单仍使用 OKX 单笔 `average/avgPx/price/fillPx` 口径。
- `tests/test_okx_trade_fact_integrity.py` 新增 `test_split_exit_order_uses_weighted_child_fill_price`，用 USAR 同型样本锁定“最后子单价 3.95、父级加权均价 3.86666667”不能误报。

安全边界：
- 本批只修只读审计口径，不修改历史订单、持仓、收益、模型样本、server_profit、影子复盘、开仓阈值、仓位、杠杆或平仓策略。
- 不把 USAR 盈利样本直接抽象为强机会模板；二期阶段 2 仍必须等阶段 1 事实口径稳定后，再做 shadow/审计态强机会识别器。
- 分拆平仓的父级价格必须来自子单成交加权，不得继续读取 raw_response 顶层最后子单价格。

本地验证：
- `pytest tests/test_okx_trade_fact_integrity.py tests/test_system_audit_api.py -q`：38 passed。
- `pytest tests/test_okx_trade_fact_integrity.py::test_split_exit_order_uses_weighted_child_fill_price -q`：1 passed。
- `python -m py_compile services/okx_trade_fact_integrity.py tests/test_okx_trade_fact_integrity.py web_dashboard/api/system_audit.py`：通过。
- `ruff check services/okx_trade_fact_integrity.py tests/test_okx_trade_fact_integrity.py web_dashboard/api/system_audit.py`：no issues。
- `black --check services/okx_trade_fact_integrity.py tests/test_okx_trade_fact_integrity.py web_dashboard/api/system_audit.py`：通过。

线上定向部署与复验：
- 定向上传：
  - `/data/bb/app/services/okx_trade_fact_integrity.py`
  - `/data/bb/app/docs/superpowers/plans/2026-06-22-quant-closed-loop-eradication.md`
- 远端已备份同名文件为 `.bak-codex-<timestamp>`；远端 `py_compile services/okx_trade_fact_integrity.py web_dashboard/api/system_audit.py` 通过。
- 只重启 `bb-dashboard.service`，不重启 `bb-paper-trading.service`；新 Dashboard PID `3731125`，`ActiveEnterTimestamp=2026-06-24 18:08:26 UTC`。
- 线上只读审计：
  - `okx_trade_fact_integrity.status=ok`
  - `checked_orders=189`
  - `checked_positions=110`
  - `issue_count=0`
  - `critical_count=0`
  - `warning_count=0`
- 系统巡检卡：
  - `okx_trade_fact_integrity.status=ok`
  - summary 为“OKX 原始成交、订单和持仓口径在巡检窗口内一致。”
- 使用 Dashboard 真实服务环境复验完整系统巡检：
  - overall `status=warning`
  - `critical=0`
  - `warning=11`
  - `ok=6`
  - `okx_trade_fact_integrity=ok`
  - `model_training=warning`，原因是可选增强数据源未配置，不是模型硬故障；local_ai_tools 为 `ready`，runtime probe 为 `ok`。
- 裸 `DATABASE_URL` 探针曾短暂显示 `model_training critical`，但用 `bb-dashboard.service` 主进程真实环境复验后确认为假阳性。后续不得把未加载服务环境的裸探针结果当成 Dashboard 真实巡检结论。

当前结论：
- 二期阶段 1 的 OKX/本地事实审计已经把 USAR 分拆平仓误报清零；当前窗口内 OKX 原始成交、订单和持仓口径一致。
- 完整系统仍为 warning，不是全绿；剩余 warning 主要是市场数据、策略质量/闭环、模型/专家观察态、特征覆盖、影子错过机会和交易执行契约历史观察项。
- 现在可以继续二期后续阶段，但强机会识别、容量轮动和收益闭环仍必须遵守“先 shadow/只读验证，再逐步启用”的顺序，不能因为 OKX 事实卡片变绿就直接放大仓位。

后续 AI 防偏要求：
- 看到分拆平仓订单时，必须优先读取 `split_chunks` 并按成交数量/张数加权；不得用 raw_response 顶层 `average/price` 代表整笔父级订单。
- 判断 Dashboard 真实巡检状态时，必须使用 `bb-dashboard.service` 等价环境；不得只用裸 `DATABASE_URL` 或 root shell 环境得出模型服务 critical 结论。
- 二期阶段 2 启动前必须确认 `okx_trade_fact_integrity` 保持 ok；若再次出现 symbol/quantity/price/notional critical，应回到阶段 1，不得继续强机会或仓位放大。

回滚点：
- 代码层可回滚 `services/okx_trade_fact_integrity.py` 与 `tests/test_okx_trade_fact_integrity.py`。
- 线上可使用 `/data/bb/app/services/okx_trade_fact_integrity.py.bak-codex-<timestamp>` 回滚并重启 `bb-dashboard.service`。
- 本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽。

---

## 五十八、Batch I 补充记录：线上部署安全与源码/编译产物口径防偏（2026-06-25）

触发原因：
- 用户追问：线上部署是否应该“编译后上传”，当前把代码同步到服务器是否不安全。
- 该问题必须写入总控，否则后续 AI 容易出现两种偏差：一是把 Python `.pyc` 或打包产物误当成安全边界；二是因为“源码在服务器”而错误改动部署方式、密钥位置或重启范围。

当前部署事实：
- 当前后端主体是 Python 服务，线上运行入口为 `.venv/bin/python scripts/run_dashboard.py`、交易服务脚本和模型隧道脚本；Python 运行时需要可加载模块代码。即使生成 `.pyc`、zipapp、wheel 或容器镜像，也不能把业务逻辑真正保密。
- 当前前端主要是 Dashboard 静态 HTML/CSS/JS；它本身就是浏览器可读取资源。若未来引入 TypeScript/Vite/Next 等构建链，前端应按构建产物部署，但这解决的是构建一致性和体积，不是隐藏逻辑。
- 当前真实安全边界不应建立在“看不到源码”上，而应建立在：服务器访问权限、OS 用户权限、systemd 最小权限、密钥外置、日志脱敏、备份回滚、定向同步和线上核验。

后续 AI 防偏要求：
- 不得把“未编译上传”直接判定为不安全。Python `.pyc`、压缩包或 wheel 不是有效保密措施，不能替代权限控制和密钥治理。
- 不得把密钥、OKX API key、模型 API key、数据库密码写入源码、文档、测试样本或前端资源；这些必须继续来自 `.env`、`/etc/bb/bb-runtime.env`、systemd 环境或安全配置层。
- 部署前必须区分改动类型：
  - 只改 Dashboard API/只读巡检：可定向上传相关 Python 文件和文档，优先只重启 `bb-dashboard.service`。
  - 改交易主循环、执行器、风控、仓位、平仓：必须重启 `bb-paper-trading.service` 并做策略健康/执行契约复验。
  - 改模型隧道或本地 AI 工具调用：必须复验 `bb-model-tunnels.service`、模型端点和数据采集/模型训练状态。
  - 改前端静态资源：必须同步静态文件并做页面/API 冒烟；如引入构建链，必须先本地 build，再部署 build 产物。
- 判断“线上更新成功”必须同时看：目标文件 mtime 或校验、远端 `py_compile`/构建结果、systemd `MainPID/ActiveEnterTimestamp`、Dashboard/API 实际返回字段；不得只看页面运行时长。
- 定向部署时必须保留远端 `.bak-codex-<timestamp>` 备份、记录回滚文件、避免上传 `.env`、本地缓存、测试数据库、日志、模型临时产物和无关脏文件。
- 如未来确实需要“整体制品部署”，应做正式发布包或容器镜像：固定 commit、锁依赖、构建产物校验、制品签名/哈希、灰度发布和回滚脚本；不能临时把源码改名或只上传 `.pyc` 来伪装安全。

安全边界：
- 本节是部署安全口径和 AI 防偏约束，不改变运行时代码、数据库、交易参数、开仓/平仓逻辑、仓位、杠杆、模型权重或风控阈值。
- 后续任何涉及部署机制的调整，必须先证明不会遗漏运行时依赖、不会把密钥打进制品、不会破坏定向回滚和线上巡检。

回滚点：
- 本节为文档约束，可回滚本节文本；无运行时代码、DB、模型 artifact 或真实交易参数变更。

---

## 五十九、Batch I 二期阶段 2 补充记录：强机会识别器只读审计接入（2026-06-25）

触发原因：
- 用户指出 USAR/USDT 这类单笔大盈利不能简单当成“以后都放大仓位”的模板，同时系统存在候选集中、错过机会多、旧仓占用和不开仓/小单反复出现的问题。
- 二期阶段 2 的目标是先把“强机会”抽象成可审计的只读识别器，解释哪些 recent entry 决策接近或满足强机会形态；本批不让它直接驱动开仓、仓位、杠杆或风控绕过。

本次修复范围：
- 新增 `services/strong_opportunity.py`：
  - 只读扫描最近 entry 决策；
  - 按 selected-side 口径提取 expected net、profit quality、loss probability、tail risk、effective score、aligned sources、evidence tier 和 high-risk review；
  - 强机会必须同时满足：`expected_net >= 0.8`、`profit_quality >= 1.05`、`loss_probability <= 0.42`、`tail_risk <= 0.72`、`aligned_sources >= 2`、`effective_score >= 0.62`，且无 hard block、shadow only、major/strong opposites、high risk review reject；
  - 输出 `strong_candidates`、`near_misses`、`blocker_counts`、`side_counts`、`evidence_tier_counts` 和阈值说明。
- 接入 `web_dashboard/api/system_audit.py`：
  - 新增 `strong_opportunity` 系统巡检卡；
  - 新增 `/api/strong-opportunity/status` 只读接口；
  - 新增强机会拓扑节点，并把它关联到 `strategy_decision`、`risk_guard` 和 `training_data`；
  - 问题台账中，若强机会卡片为 warning 且 `audit_only=true`、`live_entry_mutation=false`、`live_sizing_mutation=false`、`can_bypass_risk_controls=false`、`can_force_open=false`、`can_apply_live_sizing=false`，则归类为 observing，不归类为未解决故障。
- 新增/更新测试：
  - `tests/test_strong_opportunity.py`：覆盖强机会识别和 near-miss 阻断原因；
  - `tests/test_system_audit_api.py`：覆盖强机会审计和 endpoint 强制只读、问题台账 observing、拓扑节点关联、总巡检聚合计数。

安全边界：
- 本批只新增 shadow/只读审计，不改变真实开仓优先级、仓位 sizing、杠杆、止盈止损、平仓、风控 veto、OKX 下单、ML readiness 或专家权重。
- 即使识别出 strong candidate，也不能绕过 OKX 事实一致性、交易执行契约、收益质量、亏损概率、尾部风险、ML readiness、容量纪律和高风险复核。
- `shadow_memory`、AI 单源乐观 expected net 或 USAR 历史异常收益，不能单独构成强机会。
- 强机会从只读观察升级到 live promotion 前，必须先有独立胜率、手续费后收益、最大亏损、持仓时长、平仓原因和回撤统计，并通过 2h/24h/72h 线上观察。

本地验证：
- `pytest tests/test_strong_opportunity.py tests/test_system_audit_api.py -q`：37 passed。
- `python -m py_compile services/strong_opportunity.py web_dashboard/api/system_audit.py tests/test_strong_opportunity.py tests/test_system_audit_api.py`：通过。
- `ruff check services/strong_opportunity.py web_dashboard/api/system_audit.py tests/test_strong_opportunity.py tests/test_system_audit_api.py`：no issues。
- `black --check services/strong_opportunity.py web_dashboard/api/system_audit.py tests/test_strong_opportunity.py tests/test_system_audit_api.py`：通过。
- `git diff --check -- services/strong_opportunity.py web_dashboard/api/system_audit.py tests/test_strong_opportunity.py tests/test_system_audit_api.py`：通过。

线上定向部署与复验：
- 定向上传：
  - `/data/bb/app/services/strong_opportunity.py`
  - `/data/bb/app/web_dashboard/api/system_audit.py`
  - `/data/bb/app/docs/superpowers/plans/2026-06-22-quant-closed-loop-eradication.md`
- 远端备份后缀：`.bak-codex-1782331817`；远端 `py_compile services/strong_opportunity.py web_dashboard/api/system_audit.py` 通过。
- 只重启 `bb-dashboard.service`，未重启 `bb-paper-trading.service`；新 Dashboard `MainPID=3844379`，`ActiveEnterTimestamp=2026-06-24 20:10:16 UTC`。
- 线上强机会只读接口复验：
  - `audit_only=true`
  - `live_entry_mutation=false`
  - `live_sizing_mutation=false`
  - `can_bypass_risk_controls=false`
  - `can_force_open=false`
  - `can_apply_live_sizing=false`
  - `entry_decisions=9`
  - `strong_candidate_count=0`
  - `near_miss_count=0`
  - 主要阻断原因为 `expected_net_below_strong_threshold`、`profit_quality_below_strong_threshold`、`loss_probability_above_strong_threshold`、`evidence_tier_not_tradeable_strong`。
- 使用 Dashboard 等价环境和 OS 用户 `bb` 复验完整系统巡检：
  - overall `status=warning`
  - `cards=18`
  - `critical=0`
  - `warning=11`
  - `ok=7`
  - `nodes=19`
  - `issue_ledger.fixed=7`
  - `issue_ledger.unresolved=0`
  - `issue_ledger.observing=11`
  - `strong_opportunity.status=warning` 且节点 `state=observing`
  - `strategy_decision` 节点已关联 `strategy_quality`、`strong_opportunity`、`trade_execution_contract`
  - `okx_trade_fact_integrity.status=ok`、`issue_count=0`、`critical_count=0`、`warning_count=0`
- 复验过程中再次确认：裸 root shell 直接运行 Python 会因为 PostgreSQL peer/socket 权限和未正确加载服务环境产生假失败；后续线上巡检必须使用 `bb-dashboard.service` 等价环境或降权 `bb` 用户执行。

当前结论：
- 二期阶段 2 已具备“强机会只读识别 + 系统巡检可见 + 问题台账防误报 + 独立接口”的基础闭环。
- 这还不是“强机会自动放大仓位”完成；下一步必须先上线观察 `strong_candidate_count`、`near_miss_count`、blocker 分布、实际成交强机会样本和手续费后收益，再决定是否进入 canary。

后续 AI 防偏要求：
- 解释“为什么不开仓/小单”时，必须同时看 `strong_opportunity`、`trade_execution_contract`、`shadow_missed_opportunity`、`okx_trade_fact_integrity`、capacity/release 和 selected-side quality，不能只看单个 expected net。
- 若 `strong_opportunity` 为 warning 但只读安全旗标全部为 false/true 的受控组合，应解释为观察项，而不是系统故障或未处理问题。
- 未经明确阶段批准，不得把 `strong_candidates` 直接接入 live entry、live sizing、leverage 或 bypass risk controls。

回滚点：
- 代码层可回滚 `services/strong_opportunity.py`、`web_dashboard/api/system_audit.py`、`tests/test_strong_opportunity.py`、`tests/test_system_audit_api.py`。
- 线上若需回滚，可删除/回滚 `/data/bb/app/services/strong_opportunity.py` 并恢复 `/data/bb/app/web_dashboard/api/system_audit.py.bak-codex-<timestamp>`，再重启 `bb-dashboard.service`。
- 本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽。

---

## 六十、Batch I 二期阶段 4 补充记录：旧仓释放与容量轮动只读审计接入（2026-06-25）

触发原因：
- 用户指出旧仓长期不赚钱、占用容量，同时系统存在不开仓、小单和候选机会被挡的问题，要求不要再用“兜底补丁”绕过，而是查清容量、旧仓释放、拥挤方向阻断和释放决策是否闭环。
- 线上只读诊断显示：当前并不是“满仓卡死”。核验时 open groups 为 7 到 8，entry limit 为 25，当前持仓质量桶均为 high，当前 release candidates 为 0；但最近窗口内出现过 release decisions 未执行，以及 crowded-side blocks。因此本阶段先补“释放链路可见性”，不直接改平仓策略。

本次修复范围：
- 新增 `services/position_capacity_release_audit.py`：
  - 只读读取当前 open positions、最近 AI decisions 和 linked orders；
  - 复用现有 `PositionQualityScorer` 与 `DynamicPositionCapacityPolicy`，不重写仓位质量和容量算法；
  - 输出 open position/group、entry limit、质量桶、当前释放候选、旧盈利轮动候选、release decisions、已成交/未闭环 release decisions、crowded-side blocks；
  - 强制安全旗标：`read_only=true`、`audit_only=true`、`live_exit_mutation=false`、`live_entry_mutation=false`、`live_sizing_mutation=false`、`can_force_close=false`、`can_close_winners=false`、`can_bypass_risk_controls=false`。
- 接入 `web_dashboard/api/system_audit.py`：
  - 新增 `position_capacity_release` 系统巡检卡；
  - 新增 `/api/position-capacity-release/status` 只读接口；
  - 新增容量释放拓扑节点，并把它关联到 `strategy_decision` 与 `risk_guard`；
  - 问题台账中，若该卡片为 warning 且只读安全旗标全部满足，则归为 observing，不归为未解决故障；
  - 修正卡片 details 未透出 `read_only` 的显示口径，避免页面误解为会自动强平。
- 新增/更新测试：
  - `tests/test_position_capacity_release_audit.py`：覆盖未闭环释放决策、已成交释放决策、拥挤方向阻断和容量状态；
  - `tests/test_system_audit_api.py`：覆盖容量释放审计和 endpoint 强制只读、问题台账 observing、拓扑节点关联、总巡检聚合计数。

安全边界：
- 本批只读审计，不创建平仓决策，不调用 OKX 平仓/全平接口，不修改持仓，不调整开仓优先级，不改变 sizing、杠杆、止盈止损、风控 veto、ML readiness 或专家权重。
- 看到旧盈利轮动候选或未闭环 release decision，只能提示“下一步检查执行处理器/订单链接/平仓原因”，不能直接改成强平盈利仓。
- 若后续 `unclosed_release_decision_count` 持续增长，必须先查执行处理器为什么没有把 release decision 转成 close order，不能通过放宽开仓容量或绕过风控解决。

本地验证：
- `pytest tests/test_position_capacity_release_audit.py tests/test_system_audit_api.py -q`：39 passed。
- `python -m py_compile services/position_capacity_release_audit.py web_dashboard/api/system_audit.py tests/test_position_capacity_release_audit.py tests/test_system_audit_api.py`：通过。
- `ruff check services/position_capacity_release_audit.py web_dashboard/api/system_audit.py tests/test_position_capacity_release_audit.py tests/test_system_audit_api.py`：no issues。
- `black --check services/position_capacity_release_audit.py web_dashboard/api/system_audit.py tests/test_position_capacity_release_audit.py tests/test_system_audit_api.py`：通过。

线上定向部署与复验：
- 第一轮定向上传：
  - `/data/bb/app/services/position_capacity_release_audit.py`
  - `/data/bb/app/web_dashboard/api/system_audit.py`
  - 远端备份后缀：`.bak-codex-1782355295`
- 第二轮只修正巡检卡片 `read_only` 透出：
  - `/data/bb/app/web_dashboard/api/system_audit.py`
  - 远端备份后缀：`.bak-codex-1782355830`
- 两轮远端 `py_compile services/position_capacity_release_audit.py web_dashboard/api/system_audit.py` 均通过。
- 只重启 `bb-dashboard.service`，未重启 `bb-paper-trading.service`；最终 Dashboard `MainPID=15743`，`ActiveEnterTimestamp=2026-06-25 02:50:23 UTC`。
- 使用 Dashboard 主进程等价环境和 OS 用户 `bb` 复验：
  - overall `status=warning`
  - `cards=19`
  - `critical=0`
  - `warning=11`
  - `ok=8`
  - `nodes=20`
  - `issue_ledger.fixed=8`
  - `issue_ledger.observing=9`
  - `issue_ledger.unresolved=2`
  - unresolved keys 为 `strategy_quality`、`strategy_closed_loop`
  - `position_capacity_release.status=warning` 且节点 `state=observing`
  - `position_capacity_release` 已关联到 `strategy_decision`
  - `okx_trade_fact_integrity.status=ok`、`issue_count=0`、`critical_count=0`、`warning_count=0`
- 线上容量释放只读接口复验：
  - `read_only=true`
  - `audit_only=true`
  - `live_exit_mutation=false`
  - `live_entry_mutation=false`
  - `live_sizing_mutation=false`
  - `can_force_close=false`
  - `can_close_winners=false`
  - `can_bypass_risk_controls=false`
  - `open_position_count=8`
  - `open_group_count=8`
  - `entry_limit=25`
  - `quality_bucket_counts.high=8`
  - `current_release_candidate_count=0`
  - `old_profit_rotation_candidate_count=1`
  - `release_decision_count=1`
  - `executed_release_decision_count=0`
  - `unclosed_release_decision_count=1`
  - `crowded_block_count=25`

当前结论：
- 二期阶段 4 已具备“旧仓释放/容量轮动只读审计 + 系统巡检可见 + 问题台账防误报 + 独立接口”的基础闭环。
- 当前不是容量满导致系统不开仓；open group 8 明显低于 entry limit 25，当前持仓质量均为 high。
- 仍需继续处理的真实问题是策略质量和策略闭环两个 unresolved：它们与 missed opportunity、强机会为 0、near-miss 增多、收益质量门和策略执行闭环有关，不能用容量放宽解决。

后续 AI 防偏要求：
- 解释“旧仓占用/不开仓”时，必须同时看 `open_group_count`、`entry_limit`、`quality_bucket_counts`、`current_release_candidate_count`、`unclosed_release_decision_count`、`crowded_block_count` 和 `trade_execution_contract`，不得只凭持仓数量判断满仓。
- 若 `position_capacity_release` 为 warning 且只读安全旗标全部满足，应解释为观察项，不得误报为“系统故障未处理”。
- 若要从只读审计升级到真实释放动作，必须先证明 release decision -> close order -> filled order -> position closed 的链路缺在哪里，并补测试；不得直接调用 OKX 全平接口或强平盈利仓作为替代。

回滚点：
- 代码层可回滚 `services/position_capacity_release_audit.py`、`web_dashboard/api/system_audit.py`、`tests/test_position_capacity_release_audit.py`、`tests/test_system_audit_api.py`。
- 线上若需回滚，可删除/回滚 `/data/bb/app/services/position_capacity_release_audit.py` 并恢复 `/data/bb/app/web_dashboard/api/system_audit.py.bak-codex-<timestamp>`，再重启 `bb-dashboard.service`。
- 本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽、无 OKX 平仓/全平调用。
---

## 六十一、Batch I 二期阶段 5 补充记录：ML/server_profit/影子错过机会根因审计接入（2026-06-25）

触发原因：
- 阶段 4 复验后，系统巡检仍有 `strategy_quality`、`strategy_closed_loop` 两个真实 unresolved。当前主要问题不是容量满，也不是简单“线上没更新”，而是 entry 候选存在但高质量候选不足、ML 贡献不稳定、server_profit 多数反向或负向、shadow missed opportunity 很多但不能直接转成可交易证据。
- 用户明确要求不要继续“兜底补丁”，后续必须查根因，避免通过降阈值、放大仓位、硬改 ML readiness 或绕风控来制造成交。

本次修复范围：
- 新增 `services/strategy_signal_root_cause_audit.py`：
  - 只读扫描最近 entry decisions 与 completed shadow backtests；
  - 聚合 `expected_net_breakdown` 组件贡献：`ai`、`local_ml`、`server_profit`、`timeseries`、`shadow_memory`、`fee`、`slippage` 等；
  - 聚合 entry evidence 组件状态：ML ignored/missing/aligned、server_profit opposite/ignored_negative_expected/aligned、timeseries、shadow_memory 等；
  - 输出 ML readiness 摘要、server_profit selected-side expected return 分布、shadow missed opportunity 完成/错过/方向/币种分布；
  - 给出机器可读根因码：`ml_not_contributing`、`server_profit_negative_or_opposite`、`high_quality_entry_gap`、`candidate_symbol_concentration`、`shadow_missed_not_convertible`、`positive_ev_still_below_evidence_quality`、`weak_evidence_dominates`、`no_entry_candidates`。
- 接入 `web_dashboard/api/system_audit.py`：
  - 新增 `strategy_signal_root_cause` 系统巡检卡；
  - 新增策略信号根因拓扑节点，并把它关联到 `strategy_decision` 与 `strategy_closed_loop`；
  - 问题台账中，若该卡为 warning 且只读安全旗标满足，则归为 observing，不归为 unresolved；
  - 卡片 details 强制透出安全旗标，避免后续 AI 把诊断卡当成策略开关。
- 新增/更新测试：
  - `tests/test_strategy_signal_root_cause_audit.py`：覆盖 ML 不贡献、server_profit 反向、shadow missed opportunity 未转化、高质量候选缺口、候选集中等根因聚合；
  - `tests/test_system_audit_api.py`：覆盖新卡强制只读、问题台账 observing、拓扑节点关联、总巡检聚合计数。

安全边界：
- 本批只读审计，不修改开仓阈值、不提高仓位、不提高杠杆、不调用 OKX、不改持仓、不改平仓、不改 ML readiness、不改模型权重、不绕过风控。
- `strategy_signal_root_cause` 只能解释“为什么不开仓/为什么小单/为什么高质量候选不足”，不能直接驱动开仓、调仓、放大仓位或硬改模型状态。
- 看到 `shadow_missed_not_convertible` 时，不能把错过机会直接变成开仓；必须通过同币种同方向重复证据、selected-side expected net、profit quality、loss probability、tail risk、ML readiness 和执行契约共同确认。
- 看到 `server_profit_negative_or_opposite` 时，不能简单忽略 server_profit；必须先查 OKX 事实一致性、手续费/滑点、方向标签、训练样本污染和 selected-side 口径。
- 看到 `ml_not_contributing` 时，不能硬改 ready；必须先修训练质量、样本标注、dirty sample、PR-AUC、top return 和模型新鲜度。

本地验证：
- `pytest tests/test_strategy_signal_root_cause_audit.py tests/test_system_audit_api.py -q`：41 passed。
- `ruff check services/strategy_signal_root_cause_audit.py web_dashboard/api/system_audit.py tests/test_strategy_signal_root_cause_audit.py tests/test_system_audit_api.py`：All checks passed。
- `black --check services/strategy_signal_root_cause_audit.py web_dashboard/api/system_audit.py tests/test_strategy_signal_root_cause_audit.py tests/test_system_audit_api.py`：通过。
- `python -m py_compile services/strategy_signal_root_cause_audit.py web_dashboard/api/system_audit.py tests/test_strategy_signal_root_cause_audit.py tests/test_system_audit_api.py`：通过。
- `git diff --check` 覆盖本批 touched files：通过。

线上同步与复查：
- 已定向部署 `services/strategy_signal_root_cause_audit.py`、`web_dashboard/api/system_audit.py`、`tests/test_strategy_signal_root_cause_audit.py`、`tests/test_system_audit_api.py` 与本总控文档到 `/data/bb/app`；最终远端备份后缀为 `.bak-codex-1782395134`。
- 远端 `py_compile services/strategy_signal_root_cause_audit.py web_dashboard/api/system_audit.py tests/test_strategy_signal_root_cause_audit.py tests/test_system_audit_api.py` 通过。
- 本批只改 Dashboard 只读审计链，因此只重启 `bb-dashboard.service`；最终 Dashboard `MainPID=538740`，`ActiveEnterTimestamp=Thu 2026-06-25 13:45:38 UTC`（北京时间 `2026-06-25 21:45:38`）。
- 使用 Dashboard 主进程环境并降权 OS 用户 `bb` 复验 `collect_system_audit_status(record_history=False, source="codex-stage5-verify")`：overall `warning`、critical 0、cards 20、warning 11、ok 9、nodes 21。
- 新卡 `strategy_signal_root_cause` 已上线：card status `warning`、node state `observing`、owner `services/strategy_signal_root_cause_audit.py`，且 `strategy_decision`、`strategy_closed_loop` 节点均关联该卡。
- 新卡安全旗标复验：`audit_only=true`、`read_only=true`、`can_force_open=false`、`can_override_thresholds=false`、`can_change_ml_readiness=false`、`can_bypass_risk_controls=false`。
- 新卡当前真实样本：entry decisions 22、market entry decisions 8、high quality entry 1、unique entry symbols 10、ML usable rate 0.0、server_profit negative/opposite 9、shadow missed count 117；当前根因码为 `ml_not_contributing`、`weak_evidence_dominates`。
- 复验过程中，Dashboard 重启冷启动后首轮完整聚合曾短暂返回该卡“巡检模块执行失败”；单卡同环境调试与随后重复完整聚合均正常，最终以重复完整聚合结果为准。
- 远端目标测试未跑：线上 venv 返回 `/data/bb/app/.venv/bin/python: No module named pytest`。这不是运行时失败，但后续若要求远端 pytest，需要先安装测试依赖或用专门测试环境。

当前结论：
- 二期阶段 5 已完成本地代码接入、局部质量门、本地测试、线上定向部署和 Dashboard 等价环境复验。
- 该阶段不会直接解决“不赚钱/不开仓/小单”。它解决的是：系统巡检必须能准确说明卡在 ML、server_profit、shadow missed、候选集中，还是 evidence/expected-net 质量，而不是让后续 AI 继续猜。
- 以本次线上窗口看，最新阻塞不是“最终候选只剩 5 个交易对”：当前 entry unique symbols 为 10、market entry decisions 为 8；更直接的根因为 ML 不贡献和弱证据占主导。
- 下一步必须继续治理 `strategy_closed_loop`、ML readiness、弱证据向高质量候选转化、server_profit/OKX 事实质量和真实已平仓手续费后 PnL；不得把 Stage 5 诊断卡当成盈利闭环完成。

后续 AI 防偏要求：
- 解释“现在又不开仓”时，必须先看 `strategy_signal_root_cause.root_causes`，再结合 `strategy_quality`、`strategy_closed_loop`、`trade_execution_contract`、`strong_opportunity`、`position_capacity_release`，不得直接调阈值。
- 若 `strategy_signal_root_cause` 为 warning 但安全旗标全为只读，应解释为“观察项/根因诊断卡”，不是新故障。
- 若高质量候选仍为 0，必须沿 expected-net 组件、evidence 组件、ML readiness、server_profit、shadow missed conversion、profit quality、loss probability、tail risk 逐项定位；不得用单一大盈利样本推导“直接放大仓位”。
- 阶段 5 虽已通过线上等价环境复验，但只是诊断闭环上线；不得宣称“不赚钱/不开仓/小单”已经被根治。

回滚点：
- 代码层可回滚 `services/strategy_signal_root_cause_audit.py`、`web_dashboard/api/system_audit.py`、`tests/test_strategy_signal_root_cause_audit.py`、`tests/test_system_audit_api.py`。
- 本批无 DB 迁移、无历史覆盖、无模型 artifact 替换、无真实交易参数放宽、无 OKX 下单/平仓调用。

---

## 六十二、Batch I 二期阶段 4/5 补充记录：容量释放审计误报收口与平仓决策终态修复（2026-06-26）

触发原因：
- 线上 `position_capacity_release` 只读报告在 24 小时窗口内显示大量 `unclosed_release_decision_count` 和 `crowded_block_count`，但逐条核对后发现旧口径把三类不同事实混在一起：
  - `LAB/USDT` 因 OKX 返回不可交易/交割合约相关拒绝后进入不可交易平仓冷却；
  - `LINK/USDT`、`AAVE/USDT` 等低质量小仓触发释放候选，但扣费后预计净亏且无硬风险，系统按保护规则不执行；
  - `SPK/USDT` 等已成交或强信号 override 的 entry raw 中带有 `crowded_side_cap` 元信息，却被字符串匹配误计为拥挤阻断。
- 另有少量历史平仓决策显示 `execution_reason=订单已成交` 但没有 `decision_id -> Order` 直连，属于成交回报与订单链接缺口；还有 1 条 `LAB/USDT` 历史 close decision 留下空执行原因。
- 以上问题会让后续 AI 误以为“平仓链路全部失败/拥挤方向一直阻断”，从而偏离真实根因。

本次修复范围：
- `services/position_capacity_release_audit.py`
  - 将释放决策执行状态拆分为：
    - `executed`
    - `protected_not_executed`
    - `exchange_blocked`
    - `reported_executed_without_link`
    - `stale_skipped`
    - `pending_unclosed`
  - 新增 `release_execution_state_counts`、`release_execution_block_counts`、`protected_release_decision_count`、`exchange_blocked_release_decision_count`、`execution_link_gap_release_decision_count`、`stale_release_decision_count`。
  - `unclosed_release_decision_count` 只保留真正“应该有终态但没有订单/保护/过期/成交回报解释”的 `pending_unclosed`。
  - 拥挤阻断审计改为结构化读取 `crowded_side_cap.mode`，只统计 entry 动作中 `crowded_block` 或 `hard_ceiling`，不再把 `crowded_strong_override`、已成交 entry、close decision 或历史 raw 元信息误算成阻断。
- `web_dashboard/api/system_audit.py`
  - Dashboard 巡检卡 details 透出上述新分类和列表；
  - 所有新增列表继续强制 `can_force_close=false`、`can_close_winners=false`、`can_bypass_risk_controls=false`。
- `services/decision_final_state_ensurer.py`
  - 修复 close_long/close_short 决策在恢复器拿不到原因时直接返回的问题；
  - 未来若平仓裁决没有生成本地平仓委托、也没有 OKX 成功/失败回报，会写入明确终态原因，不再留下空 `execution_reason`。
- 历史数据单行校正：
  - 仅更新线上 `AIDecision.id=131840` 的空 `execution_reason`；
  - 更新前已备份到 `/data/bb/app/data/codex_backups/decision_131840_before_execution_reason_repair_20260625T221132Z.json`；
  - 未修改持仓、订单、盈亏、OKX 原始数据、模型数据或交易参数。

本地验证：
- `pytest tests/test_position_capacity_release_audit.py tests/test_system_audit_api.py -q`：46 passed。
- `pytest tests/test_decision_final_state_ensurer.py tests/test_position_capacity_release_audit.py tests/test_system_audit_api.py -q`：50 passed。
- 全量测试：`pytest -q`：1517 passed。
- `ruff check services/decision_final_state_ensurer.py services/position_capacity_release_audit.py web_dashboard/api/system_audit.py tests/test_decision_final_state_ensurer.py tests/test_position_capacity_release_audit.py tests/test_system_audit_api.py`：no issues。
- `black --check services/decision_final_state_ensurer.py services/position_capacity_release_audit.py web_dashboard/api/system_audit.py tests/test_decision_final_state_ensurer.py tests/test_position_capacity_release_audit.py tests/test_system_audit_api.py`：通过。

Git 与线上部署：
- 本地提交：`dd1b942 fix: classify capacity release audit states`。
- 第一次定向上线只上传：
  - `services/position_capacity_release_audit.py`
  - `web_dashboard/api/system_audit.py`
  - 只重启 `bb-dashboard.service`。
- 第二次完整 split-services 上线只新增上传：
  - `services/decision_final_state_ensurer.py`
  - 重启 `bb-paper-trading.service`、`bb-dashboard.service`、`bb-model-tunnels.service`。
- 远端哈希复验全部与本地一致：
  - `services/position_capacity_release_audit.py`：`44978a7057b54255fb28d0a8cdb6ba10ae3de9dc105c24b90af7f19aedc1f3d4`
  - `web_dashboard/api/system_audit.py`：`00082214dad5a66b5af9b11a70bd335fb6adda821e45f1ed067acae35f0b7d4e`
  - `services/decision_final_state_ensurer.py`：`17ab5663479a5abb939f41509edaa7a5a52c6fe9d1d60bc1aefce5d7ee245dba`
- 线上服务复验：
  - `bb-paper-trading.service=active`，`MainPID=900377`，`ActiveEnterTimestamp=Thu 2026-06-25 22:10:43 UTC`
  - `bb-dashboard.service=active`，`MainPID=900381`，`ActiveEnterTimestamp=Thu 2026-06-25 22:10:44 UTC`
  - `bb-model-tunnels.service=active`

线上结果：
- 修复前线上容量释放报告约为：
  - `release_decision_count=158`
  - `executed_release_decision_count=0`
  - `unclosed_release_decision_count=158`
  - `crowded_block_count=164`
- 修复、部署和单行历史校正后：
  - `open_position_count=5`
  - `open_group_count=5`
  - `quality_bucket_counts.high=4`
  - `quality_bucket_counts.release_candidate=1`
  - `current_release_candidate_count=1`
  - `old_profit_rotation_candidate_count=0`
  - `release_decision_count=143`
  - `protected_release_decision_count=15`
  - `exchange_blocked_release_decision_count=124`
  - `execution_link_gap_release_decision_count=2`
  - `stale_release_decision_count=2`
  - `unclosed_release_decision_count=0`
  - `crowded_block_count=0`
- 当前唯一实时释放候选为 `AAVE/USDT` 小额 long：
  - `position_quality.bucket=release_candidate`
  - reason 为 `stale_probe_capital_inefficient`
  - 最近释放决策被保护性跳过，原因是扣费后预计净亏且未触发硬止损、止盈、严重趋势失效或预测下行风险。

当前结论：
- “大量未闭环 release decision”和“拥挤方向阻断很多”已证明主要是审计误报和历史链接/终态分类不清；本批已把它们拆成可审计类别，并让真正未闭环计数清零。
- 当前不是容量满导致不开仓：`open_group_count=5`，`entry_limit=25`。
- 当前仍存在 1 个低质量小仓释放候选，但系统选择不为腾容量制造扣费后净亏；这是策略保护行为，不是平仓链路失败。
- `execution_link_gap_release_decision_count=2` 说明历史成交回报与本地订单链接仍有数据台账缺口，后续若继续治理脏数据，应沿订单/决策链接修复，不应把它解释成“OKX 没平掉”。
- 本批仍不等于“不赚钱/不开仓/小单”完成；下一阶段继续治理策略质量、ML readiness、server_profit、弱证据转高质量候选和真实手续费后 PnL。

后续 AI 防偏要求：
- 解释容量释放时必须先看 `release_execution_state_counts`，不得只看 `release_decision_count`。
- `protected_not_executed` 表示系统避免制造扣费后净亏或无硬风险平仓；不得把它说成平仓失败。
- `exchange_blocked` 表示 OKX/交易对状态阻断或冷却；不得通过重复提交或强行全平掩盖，应先确认交易所状态、合约生命周期、API 权限和市场列表。
- `reported_executed_without_link` 表示成交回报与本地订单链接缺口；后续应修链接/数据台账，不得重复平仓。
- `stale_skipped` 表示旧裁决过期或本轮未进入下单，不能当作当前可执行信号。
- 只有 `pending_unclosed` 才是“真正未闭环”的核心告警；若它再次大于 0，必须追查生成/执行/终态写入链路。

回滚点：
- 代码层可回滚 `services/position_capacity_release_audit.py`、`web_dashboard/api/system_audit.py`、`services/decision_final_state_ensurer.py`、`tests/test_position_capacity_release_audit.py`、`tests/test_system_audit_api.py`、`tests/test_decision_final_state_ensurer.py`。
- 历史数据仅改了 `AIDecision.id=131840.execution_reason`，可用备份 JSON 恢复；不涉及订单、持仓、盈亏或 OKX 数据。

---

## 六十三、Batch I 二期阶段 5 补充记录：策略闭环台账历史/样本观察分层修正（2026-06-26）

触发原因：
- 线上 Dashboard 等价环境复验时，完整系统巡检已经没有新增弱证据执行、快亏平、shadow-only 执行、执行无订单等当前硬错误，但问题台账仍把 `strategy_closed_loop` 放在 `unresolved`。
- 逐项核对后确认，`strategy_closed_loop` 的 warning 来源是 24 小时历史窗口中的 `historical_ml_not_effective=true` 与 `insufficient_effectiveness_samples=true`；当前运行窗口内：
  - `current_weak_executed=false`
  - `current_no_high_quality_entries=false`
  - `current_fast_loss_cluster=false`
  - `current_ml_not_effective=false`
  - `shadow_only_executed=false`
  - `executed_without_order=false`
- 这类状态表示“收益/ML 有效性尚未被新样本证明，仍需观察”，不等同于当前执行链硬故障；若继续把它放在 `unresolved`，后续 AI 容易重复改执行链、容量释放或阈值，偏离真实根因。

本次修复范围：
- `web_dashboard/api/system_audit.py`
  - 新增 `strategy_closed_loop` 当前诊断键与观察诊断键分组。
  - 当前硬错误/当前质量问题仍保持 `unresolved`，包括弱证据执行、高质量候选持续为 0、当前快亏平集群、当前 ML 不参与、shadow-only 执行、执行无订单。
  - 仅当当前诊断为干净，且 warning 只来自历史遗留、历史 ML 不参与或收益样本不足时，`strategy_closed_loop` 进入 `observing`。
  - 保留通用 `current_runtime_window.historical_legacy_issues` 语义，避免破坏 `trade_execution_contract` 等已有“历史遗留当前未复现”台账逻辑。
- `tests/test_system_audit_api.py`
  - 新增“历史/样本不足 warning 进入 observing”的用例。
  - 新增“当前窗口真实质量问题仍 unresolved”的用例。
  - 保留并复验既有 `strategy_quality` 随 `strategy_closed_loop` 历史观察进入 observing 的用例。

本地验证：
- `pytest tests/test_system_audit_api.py -q`：41 passed。
- 全量测试 `pytest -q`：1519 passed。
- `ruff check web_dashboard/api/system_audit.py tests/test_system_audit_api.py`：All checks passed。
- `black --check web_dashboard/api/system_audit.py tests/test_system_audit_api.py`：通过。
- `git diff --check`：通过。

线上部署与复验：
- 使用 `scripts/sync_to_online_server.py --skip-restart` 上传源码，仅上传：
  - `/data/bb/app/web_dashboard/api/system_audit.py`
- 远端 `py_compile web_dashboard/api/system_audit.py` 通过。
- 本次只涉及 Dashboard 巡检/台账展示逻辑，未改交易主循环、OKX 下单、持仓同步、风控、仓位或杠杆；因此仅重启 `bb-dashboard.service`，未重启 `bb-paper-trading.service`。
- 远端 hash 已与本地一致：
  - `web_dashboard/api/system_audit.py`：`17ecdfc3a8f77a83017e5033af6ee578725b9be12f96ab3a11c87698973ef2fe`
- Dashboard 服务复验：
  - `bb-dashboard.service=active`
  - `MainPID=923896`
  - `ActiveEnterTimestamp=Thu 2026-06-25 22:36:53 UTC`
  - 本机 Dashboard HTTP 返回 `302`
- 使用 Dashboard 主进程环境并降权 OS 用户 `bb` 复验 `collect_system_audit_status(record_history=False, source="codex_verify")`：
  - overall `status=warning`
  - `cards=20`
  - `critical=0`
  - `warning=11`
  - `ok=9`
  - `nodes=21`
  - `issue_ledger.fixed=9`
  - `issue_ledger.unresolved=0`
  - `issue_ledger.observing=11`
  - `unresolved_keys=[]`
  - `observing_keys` 包括 `strategy_closed_loop`、`strategy_signal_root_cause`、`model_training`、`shadow_missed_opportunity`、`strong_opportunity`、`position_capacity_release` 等。

当前真实结论：
- `strategy_closed_loop` 已从“未修复硬项”调整为“历史/样本观察项”；这修的是系统巡检台账的分层准确性，不是宣称策略已经会赚钱。
- 当前仍未完成的是收益闭环证明：已平仓手续费后正收益样本不足、ML 仍处于学习观察/不贡献状态，`strategy_signal_root_cause` 当前根因为 `ml_not_contributing`。
- 当前不能因为 `unresolved=0` 就放宽阈值、放大仓位、提高杠杆、硬改 ML readiness 或让 shadow missed opportunity 直接驱动实盘开仓。
- 后续继续二期时，应优先推进：
  1. ML readiness 与训练样本质量；
  2. server_profit/OKX 事实口径与 selected-side 收益质量；
  3. 弱证据向高质量候选转化；
  4. 真实已平仓手续费后 PnL 观察；
  5. 脏数据/乱码代码治理和本地工作区清理。

后续 AI 防偏要求：
- 看到 `strategy_closed_loop.status=warning` 但 `issue_ledger.unresolved=0` 时，必须解释为“当前硬执行错误未复现，但盈利/ML 有效性仍在观察”，不得解释为“策略闭环完成”。
- 若 `strategy_closed_loop` 后续再次进入 `unresolved`，必须先看 `diagnostics` 中哪个 `current_*` 或 hard execution 键为 true，再决定修执行链、策略质量、ML 或数据口径；不得直接调开仓阈值。
- 若 warning 只来自 `historical_ml_not_effective` 或 `insufficient_effectiveness_samples`，应继续观察并治理 ML/收益样本，不得重复改 OKX 平仓、容量释放或订单同步链路。
- 任何仓位放大、阈值放宽、杠杆提高或 ML readiness live promotion，仍必须等待 2h/24h/72h 或至少 20 笔真实已平仓手续费后样本验证。

回滚点：
- 代码层可回滚 `web_dashboard/api/system_audit.py` 与 `tests/test_system_audit_api.py`。
- 本批无 DB 迁移、无历史数据覆盖、无模型 artifact 替换、无真实交易参数放宽、无 OKX 下单/平仓调用。

---

## 六十四、Batch I 二期阶段 5 补充记录：ML 不贡献根因细化与当前 ML 观察分桶（2026-06-26）

触发原因：
- 继续按总控推进二期时，线上 `strategy_signal_root_cause` 仍显示 `ml_not_contributing`，但这个根因太笼统，后续 AI 容易误判成“ML 没接上”或“readiness 阈值太严”，然后走偏到硬改 ready、降阈值或放大仓位。
- 线上只读诊断确认：
  - ML 样本量足够：`sample_count=5922`、`test_count=1481`。
  - 数据版本一致：`training_data_version=2026-06-23.v3`。
  - 脏样本比例可控：`dirty_sample_ratio=0.0263`，低于 `max_dirty_sample_ratio=0.08`。
  - PR-AUC 不是当前 blocker：`long_pr_auc≈0.5269`、`short_pr_auc≈0.5442`，均高于 `min_pr_auc=0.52`。
  - 真正 blocker 是高分桶收益仍为负：`top_long_avg_return_pct≈-0.0595`、`top_short_avg_return_pct≈-0.0291`，低于 `min_top_return_pct=0.05`。
- 线上 dry-run 训练窗口评估 `/data/bb/app/.venv/bin/python scripts/evaluate_ml_training_windows.py --limit 20000` 也显示 `ready_variants=[]`，推荐结论为 `No variant passed readiness; do not enable local ML live influence.`。

本次修复范围：
- `services/strategy_signal_root_cause_audit.py`
  - `ml.readiness` 额外透出 `blocking_reasons`、`thresholds`、bottom/top return 指标。
  - 新增根因码 `ml_top_return_not_profitable`。
  - 当 readiness blocker 包含 `long_top_return_below_threshold` 或 `short_top_return_below_threshold` 时，根因卡会输出：
    - `blocking_reason_codes`
    - `readiness_state`
    - `allow_live_position_influence`
    - `top_long_avg_return_pct`
    - `top_short_avg_return_pct`
    - `bottom_long_avg_return_pct`
    - `bottom_short_avg_return_pct`
    - `required_min_top_return_pct`
  - 新增 next action：必须继续观察，直到高分桶费后收益为正；先查标签、训练窗口、脏样本和收益口径，不得硬改 ready。
- `web_dashboard/api/system_audit.py`
  - `strategy_closed_loop` 中 `current_ml_not_effective` 从“当前硬 unresolved”移到“观察诊断”。
  - 若当前仅 ML 不贡献，而弱证据执行、快亏、高质量候选缺口、shadow-only 执行、执行无订单均未出现，则 `strategy_closed_loop` 进入 observing。
  - 若当前真的出现弱证据执行、高质量候选持续为 0、快亏平集群、shadow-only 执行或执行无订单，仍进入 unresolved。
  - 摘要从笼统“当前运行窗口仍存在弱证据执行、高质量候选不足、ML弱参与或快亏平风险”细化为“当前运行窗口 ML 仍未有效参与；执行硬错误暂未复现，需继续治理 ML readiness 与收益样本。”
- 测试：
  - `tests/test_strategy_signal_root_cause_audit.py` 新增 `ml_top_return_not_profitable` 根因与 next action 覆盖。
  - `tests/test_system_audit_api.py` 新增“当前仅 ML 不贡献时 strategy_closed_loop 进入 observing”的覆盖。

本地验证：
- `pytest tests/test_system_audit_api.py tests/test_strategy_signal_root_cause_audit.py -q`：45 passed。
- 全量测试 `pytest -q`：1521 passed。
- `ruff check web_dashboard/api/system_audit.py tests/test_system_audit_api.py services/strategy_signal_root_cause_audit.py tests/test_strategy_signal_root_cause_audit.py`：All checks passed。
- `black --check web_dashboard/api/system_audit.py tests/test_system_audit_api.py services/strategy_signal_root_cause_audit.py tests/test_strategy_signal_root_cause_audit.py`：通过。
- `git diff --check`：通过。

线上部署与复验：
- 使用 `scripts/sync_to_online_server.py --skip-restart` 分两次上传：
  - `/data/bb/app/services/strategy_signal_root_cause_audit.py`
  - `/data/bb/app/web_dashboard/api/system_audit.py`
- 远端 `py_compile web_dashboard/api/system_audit.py services/strategy_signal_root_cause_audit.py` 通过。
- 远端 hash 与本地一致：
  - `services/strategy_signal_root_cause_audit.py`：`a10741caa005cfd1e625173fef977a1c17337a9530b0d05e77dee5c671374c37`
  - `web_dashboard/api/system_audit.py`：`2b572bb2dd0ba0f28d7599a2c4dce7c650e73ec869d4b7a4f2a4f05e7931da98`
- 仅重启 `bb-dashboard.service`，未重启交易主进程；最终 Dashboard：
  - `bb-dashboard.service=active`
  - `MainPID=942124`
  - `ActiveEnterTimestamp=Thu 2026-06-25 22:59:00 UTC`
  - Dashboard HTTP 返回 `302`
- 线上 `strategy_signal_root_cause` 单卡复验：
  - `audit_only=true`
  - `read_only=true`
  - `live_entry_mutation=false`
  - `live_sizing_mutation=false`
  - `live_leverage_mutation=false`
  - `can_force_open=false`
  - `can_override_thresholds=false`
  - `can_change_ml_readiness=false`
  - `can_bypass_risk_controls=false`
  - 根因码：`ml_not_contributing`、`ml_top_return_not_profitable`
  - `top_long_avg_return_pct=-0.05951014678283073`
  - `top_short_avg_return_pct=-0.029080449850134617`
  - `required_min_top_return_pct=0.05`
- 线上完整系统巡检第二次稳定复验：
  - overall `status=warning`
  - `cards=20`
  - `critical=0`
  - `warning=11`
  - `ok=9`
  - `issue_ledger.fixed=9`
  - `issue_ledger.unresolved=0`
  - `issue_ledger.observing=11`
  - `unresolved_keys=[]`
  - observing 包括 `strategy_closed_loop` 与 `strategy_signal_root_cause`

当前真实结论：
- 本批没有让 ML 参与实盘，也没有让系统更激进；它把“ML 不贡献”从笼统根因细化为“高分桶收益不达标”。
- 当前 local ML 不能 live influence 是正确保护：样本量和 PR-AUC 虽然够，但模型最高分组仍未证明能筛出费后正收益。
- “现在不开仓/小单/不赚钱”的一部分原因是 ML 不贡献，但不能通过硬改 readiness 解决；必须先让训练样本的高分桶收益变正，并用真实已平仓费后收益验证。
- 本批也确认了候选交易对并非只剩 5 个：最近 120 分钟市场分析 42 个交易对、市场 entry 5 个且各不重复；真正卡点仍是收益质量、ML 高分桶收益、候选过滤和执行所确认。

后续 AI 防偏要求：
- 看到 `ml_top_return_not_profitable` 时，必须解释为“模型最高分组仍不赚钱”，不得解释为“阈值太严”。
- 不得用 `sample_count` 足够或 PR-AUC 达标单独证明 ML 可用；必须同时满足高分桶费后收益为正、top bucket 优于 bottom bucket、模型新鲜度、数据版本和脏样本比例。
- 任何 ML live promotion 都必须先跑 dry-run 训练窗口评估，并确认至少一个窗口通过 readiness；若 `ready_variants=[]`，禁止启用。
- 后续若要真正修复 ML，需要优先检查收益标签、shadow backtest 口径、训练窗口选择、持仓/平仓脏数据、OKX 事实同步和费后收益计算；不得通过降门槛或扩大仓位制造“看起来更会交易”。
- 看到 `strategy_closed_loop` observing 且 `current_ml_not_effective=true` 时，应解释为“当前执行硬错误未复现，但 ML/收益样本仍未闭环”，不得重复改 OKX 平仓、容量释放或订单同步链路。

回滚点：
- 代码层可回滚 `services/strategy_signal_root_cause_audit.py`、`web_dashboard/api/system_audit.py`、`tests/test_strategy_signal_root_cause_audit.py`、`tests/test_system_audit_api.py`。
- 本批无 DB 迁移、无历史数据覆盖、无模型 artifact 替换、无真实交易参数放宽、无 OKX 下单/平仓调用。

---

## 六十五、Batch I 二期阶段 1/工程基线补充记录：OKX 历史对账 dry-run 超时根因收口（2026-06-26）

触发原因：
- 二期未完成台账中 `okx_reconciliation` 仍有 dry-run `TimeoutError` 观察项，用户多次指出 OKX/本地口径、历史脏数据、订单关联和收益样本不能继续靠页面提示绕过去。
- 线上单卡复核确认：单独执行 OKX 历史对账可以完成，但完整系统巡检并发执行时，OKX 对账仍可能被 8 秒内部预算打断，导致页面继续显示 dry-run 超时。
- 根因不是 OKX API，也不是交易所仓位接口；当前历史对账脚本在巡检中扫描 14 天所有 filled OKX 订单，再逐笔进入 `plan_missing_closed_position()`，普通开仓成交也会先进入候选扫描，完整巡检并发 DB 压力下容易超时。

本次修复范围：
- `scripts/repair_missing_closed_positions_from_orders.py`
  - 新增 `ReconciliationScanReport` 和 `collect_missing_closed_position_scan()`。
  - SQL 入口只扫描有 `AIDecision.action in (close_long, close_short)` 且方向与平仓动作匹配的 filled OKX 平仓单。
  - 输出 `candidate_order_count`、`scanned_order_count`、`truncated`、`max_close_orders`、`duration_seconds`，让后续排查不再只看到 `TimeoutError`。
  - 保留原 `collect_missing_closed_position_plans()` 兼容调用方。
- `web_dashboard/api/system_audit.py`
  - `okx_reconciliation` 卡片改读扫描报告，展示候选平仓单、已扫描平仓单、缺失闭仓和耗时。
  - `okx_reconciliation` 加入优先串行巡检组，先于普通并发诊断执行，避免完整巡检下被其它 DB 重任务挤超时。
  - 若未来出现截断扫描，卡片会保持 warning，不会把“只扫了一部分”误报为完整 ok。
- `tests/test_order_position_reconciliation.py`
  - 新增回归：大量普通 entry filled 订单存在时，历史对账候选只包含真实 close order。
- `tests/test_system_audit_api.py`
  - 更新 OKX 对账缓存/超时测试，覆盖新扫描报告字段。

安全边界：
- 本批只改只读审计和 dry-run 扫描入口，不写数据库，不补历史仓位，不修改订单、持仓、盈亏、训练样本、模型 artifact、开仓阈值、仓位、杠杆、止盈止损、平仓逻辑或 OKX 下单/平仓接口。
- `okx_reconciliation=ok` 只表示 14 天窗口没有由本地 OKX 成交订单反推出的缺失闭仓，不代表所有历史脏样本、历史执行记录和训练样本污染已经彻底清理。
- 后续若再次看到 OKX 对账超时，必须先看 `candidate_close_order_count`、`scanned_close_order_count`、`duration_seconds` 和巡检调度分组，不得直接补历史数据或把超时当 OK。

本地验证：
- `pytest tests/test_order_position_reconciliation.py tests/test_system_audit_api.py -q`：47 passed。
- 全量测试 `pytest -q`：1522 passed。
- `ruff check scripts/repair_missing_closed_positions_from_orders.py web_dashboard/api/system_audit.py tests/test_order_position_reconciliation.py tests/test_system_audit_api.py`：All checks passed。
- `black --check scripts/repair_missing_closed_positions_from_orders.py web_dashboard/api/system_audit.py tests/test_order_position_reconciliation.py tests/test_system_audit_api.py`：通过。
- `git diff --check`：通过。

线上部署与复验：
- 已使用 `scripts/sync_to_online_server.py --skip-restart` 上传：
  - `/data/bb/app/scripts/repair_missing_closed_positions_from_orders.py`
  - `/data/bb/app/web_dashboard/api/system_audit.py`
- 远端 `py_compile scripts/repair_missing_closed_positions_from_orders.py web_dashboard/api/system_audit.py` 通过。
- 仅重启 `bb-dashboard.service`，未重启 `bb-paper-trading.service`；最终 Dashboard：
  - `bb-dashboard.service=active`
  - `MainPID=968920`
  - `ActiveEnterTimestamp=Thu 2026-06-25 23:28:31 UTC`
- 使用 Dashboard 主进程等价环境、OS 用户 `bb` 只读复验：
  - 单卡：`okx_reconciliation.status=ok`
  - `candidate_close_order_count=247`
  - `scanned_close_order_count=247`
  - `missing_closed_positions=0`
  - `truncated=false`
  - 单卡耗时约 `3.27s`
  - 完整系统巡检中该卡耗时约 `4.45s`
  - 完整系统巡检：overall `warning`、cards `20`、critical `0`、warning `10`、ok `10`
  - issue ledger：`fixed=10`、`unresolved=0`、`observing=10`

当前真实结论：
- OKX 历史对账 dry-run 超时链路已从“泛化 TimeoutError 观察项”收口为“可解释、可计数、完整巡检可通过”的只读审计。
- 当前 14 天窗口没有缺失闭仓；这降低了历史缺仓继续污染收益判断的风险。
- 但二期 OKX/本地历史脏数据闭环还没有全部完成：历史执行记录、历史持仓、历史收益样本仍需继续按订单 ID、OKX `ordId`、OKX `fillId` 做隔离/清单/回滚式治理。
- 本批不解决“不赚钱、不开仓、小单、ML top return 仍负、server_profit 反向或强机会 canary”问题；这些仍按第十七点六推荐顺序继续推进。

后续 AI 防偏要求：
- 不得因为 `okx_reconciliation=ok` 就宣称历史脏数据彻底清完；只能说“当前 14 天缺失闭仓 dry-run 为 0，且超时已收口”。
- 不得把 OKX 对账卡变绿当作放大仓位、降低阈值、硬启 ML 或启用强机会 live sizing 的依据。
- 继续治理历史脏数据时，优先用订单 ID / OKX `ordId` / OKX `fillId` 关联；交易对 alias 只能作为辅助。

回滚点：
- 代码层可回滚 `scripts/repair_missing_closed_positions_from_orders.py`、`web_dashboard/api/system_audit.py`、`tests/test_order_position_reconciliation.py`、`tests/test_system_audit_api.py` 与本节文档。
- 本批无 DB 迁移、无历史数据覆盖、无模型 artifact 替换、无真实交易参数放宽、无 OKX 下单/平仓调用。

---

## 六十六、Batch I 二期阶段 2 补充记录：OKX 51155 不可交易交易对执行前阻断刷新（2026-06-26）

触发原因：
- 线上最近窗口反复出现 `RESOLV/USDT` 开仓被 OKX 返回 `51155 local compliance restrictions`，用户指出“明知道不符合 OKX，为什么还提交到开仓这一步再失败”。
- 代码里已有 `EntrySymbolBlocklistPolicy`、`remember_untradable_symbol()` 和启动时 `_load_untradable_symbol_blocks()`，但历史拒单恢复只在 `TradingService.initialize()` 执行一次，且只扫最近 300 条错误决策。
- 交易主进程长时间运行、仅重启 Dashboard、或最近 300 条被大量分析/hold 记录挤掉时，已知不可交易交易对可能再次进入执行链并提交 OKX。

本次修复范围：
- `services/trading_service.py`
  - 新增 `ENTRY_SYMBOL_BLOCK_REFRESH_SECONDS=60.0` 与 `_entry_symbol_blocks_refreshed_at`。
  - 新增 `_refresh_entry_symbol_blocks_if_stale()`，启动、每轮 `run_once()` 和真正 entry execution policy 评估前都会刷新最近不可交易/临时阻断事实。
  - `ExecutionService` 的 entry policy provider 从直接调用 `entry_execution_pipeline.evaluate` 改为 `evaluate_entry_execution_policy()`，确保提交 OKX 前一定先恢复最近持久拒单事实。
  - `_load_untradable_symbol_blocks()` 从“最近 300 条”改为“最近 24 小时错误决策窗口，最多 2000 条”，避免真实 51155 被普通分析记录挤出恢复窗口。
- `tests/test_trading_service_boundaries.py`
  - 新增回归测试：数据库已有 `RESOLV/USDT` 51155 拒单时，新的 entry 在执行策略阶段必须被 `entry_opportunity_gate` 阻断，不能再进入 OKX 提交。

安全边界：
- 本批不放宽开仓阈值、不放大仓位、不提高杠杆、不改变 ML readiness、不改 OKX 下单参数、不改止盈止损、不覆盖历史订单/持仓/收益。
- 本批只让“已经被 OKX 明确拒绝、且属于不可交易/合规限制/交割合约等不可提交原因”的交易对，在下一次 entry 执行前被本地阻断。
- `51155` 阻断不是盈利能力优化，不能解释为“不赚钱/小单/不开仓已经解决”；它只减少重复失败订单和无意义 OKX 提交。

本地验证：
- `pytest tests/test_entry_symbol_blocklist.py tests/test_trading_service_boundaries.py -q`：147 passed。
- `pytest tests/test_execution_result_classifier.py tests/test_system_audit_api.py tests/test_order_position_reconciliation.py -q`：61 passed。
- 合并相关测试：208 passed。
- 全量测试 `pytest -q`：1523 passed。
- `ruff check services/trading_service.py tests/test_trading_service_boundaries.py`：All checks passed。
- `black --check services/trading_service.py tests/test_trading_service_boundaries.py`：通过。
- `git diff --check`：通过。

线上部署与复验：
- 使用 `python scripts/sync_to_online_server.py --split-services` 上传 `services/trading_service.py` 并重启 split services。
- 三项服务均 active，Dashboard 返回 `302`。
- 远端 `py_compile services/trading_service.py` 通过。
- 交易主进程已重启，确认不是只重启 Dashboard：
  - `bb-paper-trading.service=active`
  - `MainPID=993899`
  - `ActiveEnterTimestamp=Thu 2026-06-25 23:57:33 UTC`
  - `bb-dashboard.service=active`
  - `MainPID=993904`
- 线上真实库只读探针确认：
  - `RESOLV/USDT`、`RESOLV-USDT`、`RESOLV-USDT-SWAP` 均能从最近 51155 决策恢复为 active block。
  - 该探针同时发现 `LAB/USDT` 也有大量历史不可交易拒单，会被同一机制阻断。
- 重启后窗口检查：
  - `post_restart_51155_count=0`
  - `post_restart_order_status_counts={}`
- 120 分钟策略健康脚本：
  - `trade_execution_contract.status=ok`
  - `contract_violation_count=0`
  - `weak_evidence_executed_count=0`
  - `negative_expected_executed_count=0`
  - `fast_loss_without_strong_exit_count=0`
  - `local_ml_readiness=degraded`，阻塞仍为 `long_top_return_below_threshold` 与 `short_top_return_below_threshold`
- 系统巡检必须使用 Dashboard/交易服务等价环境验证；裸 `sudo -u bb` 环境没有 `/etc/bb/bb-runtime.env` 会误报 local AI tools 401。
  - 服务等价环境复验：overall `warning`、cards `20`、critical `0`、warning `9`、ok `11`
  - issue ledger：`fixed=11`、`unresolved=0`、`observing=9`
  - `model_training=warning`，原因是可选增强数据源未配置，不是 local AI tools 硬故障。

当前真实结论：
- 已知不可交易交易对的重复 OKX 提交链路已收口到执行前阻断，并且交易主进程已加载新代码。
- 本批降低的是失败订单/重复拒单风险，不会让系统自动多开仓，也不会让小单自动变大。
- 当前仍未完成的核心问题仍是收益闭环：ML 高分桶收益为负、部分候选收益质量不足、强机会可复制识别和历史脏数据/收益样本治理仍需继续。

后续 AI 防偏要求：
- 后续看到 `51155`、`local compliance restrictions`、`cannot trade this pair`、`contract under delivery` 等交易所明确不可交易错误时，必须先检查 blocklist 是否已恢复和是否在 entry policy 前生效，不得重复提交 OKX。
- 不得把这类本地阻断显示成“OKX 已执行”或“OKX 同步平仓”；执行来源和状态必须如实标为本地策略阻断/跳过。
- 验证系统巡检时必须加载线上服务等价环境；裸命令缺少 runtime env 时出现 local AI tools 401，不能当作真实线上故障。
- 若后续 51155 在重启后再次新增，优先检查交易主进程 PID、`_load_untradable_symbol_blocks()` 查询窗口、`AIDecision.raw_llm_response/execution_reason` 是否写入了 OKX 原始错误；不得用延长冷却时间代替根因排查。

回滚点：
- 代码层可回滚 `services/trading_service.py` 与 `tests/test_trading_service_boundaries.py`。
- 线上回滚后必须重启 `bb-paper-trading.service`，否则交易主进程仍会保留旧 provider。
- 本批无 DB 迁移、无历史数据覆盖、无模型 artifact 替换、无真实交易参数放宽。

---

## 六十七、Batch I 二期阶段 2 补充记录：OKX swap 成交量/名义额口径修正，解除候选误杀（2026-06-26）

触发原因：
- 线上候选池仍然反复集中，最新漏斗里 `analysis_notional_below_floor` 频繁出现；进一步核对发现低价合约（例如 `PEPE-USDT-SWAP`）被算成极低名义额。
- OKX swap ticker 的 `vol24h` 是合约张数，不是基础币成交量；`volCcy24h` 才是基础币数量。旧链路把 WebSocket 的 `vol24h` 写入 `volume_24h`，ranker 再用 `current_price * volume_24h` 计算 USDT 名义额。
- 实测例子：`PEPE-USDT-SWAP` 的 `last=0.000002355`、`vol24h=5357584.8`、`volCcy24h=53575848000000`；旧算法得到 `12.617U`，正确基础币名义额约 `126171122.04U`。这会把真实高流动性低价合约误判为名义额不足。

本次修复范围：
- 新增 `data_feed/okx_ticker_volume.py`，统一抽取 OKX swap ticker 的 `volume_24h_contracts`、`volume_24h_base`、`volume_24h_quote`、`notional_24h_usdt`、`volume_24h_source`。
- `data_feed/okx_ws_client.py`：WebSocket ticker 不再把 `vol24h` 当基础币成交量；`volume_24h` 兼容保持为基础币成交量，额外保留合约张数和名义额。
- `services/data_service.py` 与 `data_feed/feature_vector.py`：FeatureVector 增加显式成交量/名义额字段；DB `market_tickers.volume_24h` 保持兼容，不做迁移，额外诊断字段进入 `raw_data` 和实时 market state。
- `services/entry_feature_ranker.py`：候选流动性评分、tradable/analysis 名义额门、诊断输出统一优先读取 `notional_24h_usdt`，仅旧数据缺失时才退回 `price * volume_24h`。
- `data_feed/okx_rest_client.py`、`data_feed/okx_sdk_client.py`、`web_dashboard/api/dashboard.py`：REST/SDK/Dashboard 公共 ticker 解析同步保留 base/contracts/notional，避免不同页面口径不一致。

安全边界：
- 本批不放宽 `analysis_notional`、`analysis_volume_ratio`、收益质量、ML readiness、仓位、杠杆、止盈止损、OKX 下单或平仓规则。
- 本批不修改历史订单/持仓/收益数据，不做 DB 迁移，不替换模型 artifact。
- 这次只修正“名义额计算单位错误导致候选误杀”，不能解释为“不赚钱/小单/不开仓全部解决”。

本地验证：
- 相关测试：`91 passed`。
- 全量测试：`pytest -q`，`1529 passed`。
- `ruff check data_feed services web_dashboard tests`：通过。
- `black --check data_feed services web_dashboard tests`：通过。
- `git diff --check`：通过。

线上部署与复验：
- 使用 `scripts/sync_to_online_server.py --split-services` 上传并重启 split services；上传变更文件 8 个：`data_feed/feature_vector.py`、`data_feed/okx_rest_client.py`、`data_feed/okx_sdk_client.py`、`data_feed/okx_ticker_volume.py`、`data_feed/okx_ws_client.py`、`services/data_service.py`、`services/entry_feature_ranker.py`、`web_dashboard/api/dashboard.py`。
- 远端 `py_compile` 通过。
- 服务状态：`bb-paper-trading.service=active MainPID=1027761 ActiveEnterTimestamp=Fri 2026-06-26 00:44:21 UTC`；`bb-dashboard.service=active MainPID=1027765 ActiveEnterTimestamp=Fri 2026-06-26 00:44:21 UTC`；`bb-model-tunnels.service=active MainPID=1027729 ActiveEnterTimestamp=Fri 2026-06-26 00:44:17 UTC`。
- 重启后 10 分钟只读策略健康：`trade_execution_contract.status=ok`；`orders=0`、`failed_orders=0`、`rejected_orders=0`；`market_decisions=8`、`market_unique_symbol_count=6`。
- 最新候选漏斗：`rank_selected_count=2`、`rank_underfilled=true`；`rank_filtered_out_reason_counts` 中 `analysis_notional_below_floor=2`，主因已变为 `analysis_volume_ratio_below_floor=26`。
- 最新选中样本：`ETH/USDT notional_24h=10372469947.47`、`BNB/USDT notional_24h=78290768.8`，说明名义额口径已回到真实量级。
- `local_ml_readiness` 仍为 `degraded`，阻塞仍是 `long_top_return_below_threshold` 与 `short_top_return_below_threshold`。

当前真实结论：
- “低价合约被合约张数误算成极低 USDT 名义额”的根因已修复，候选池不应再因为 `vol24h` 单位错误被系统性误杀。
- 这次修复后，最新漏斗的主瓶颈已经从“名义额误杀”转为“量比不足/后续收益质量/ML degraded”。因此系统仍可能不开仓，这不是本批代码未上线，而是剩余策略质量闭环仍未完成。
- 后续如果仍看到候选集中，不得再先调低名义额阈值；必须先看 `notional_24h_usdt`、`volume_24h_source`、`analysis_volume_ratio_below_floor`、ML top return、server_profit selected-side 与 shadow missed opportunity 的证据链。

后续 AI 防偏要求：
- 看到 `analysis_notional_below_floor` 时，必须先确认该样本是否有 `notional_24h_usdt`，以及来源是否为 OKX `volCcy24h`/base volume；不得直接放宽名义额阈值。
- 看到低价币 notional 异常小（例如小于几十 U）时，必须先怀疑成交量单位，不得解释为“币本身没流动性”。
- 看到 `rank_underfilled=true` 时，必须区分是名义额不足、量比不足、异常影线、ADX、波动上限还是市场 AI 时间预算，不得笼统说“候选少/不开仓”。
- 本批不允许作为放大仓位、提高杠杆、硬启 ML readiness、降低收益门、降低量比门或绕过 OKX 风控的依据。

回滚点：
- 代码层可回滚 `data_feed/okx_ticker_volume.py`、`data_feed/okx_ws_client.py`、`data_feed/feature_vector.py`、`services/data_service.py`、`services/entry_feature_ranker.py`、`data_feed/okx_rest_client.py`、`data_feed/okx_sdk_client.py`、`web_dashboard/api/dashboard.py` 及对应测试。
- 线上回滚后必须重启 `bb-paper-trading.service` 与 `bb-dashboard.service`，否则交易主进程会保留旧/新混合行情解析逻辑。

---

## 六十八、Batch I 二期阶段 2 补充记录：候选量比与指标缺失口径修正，避免默认指标伪候选和 1h 量比误杀（2026-06-26）

触发原因：
- 线上候选漏斗在 OKX swap 名义额修正后仍反复 `rank_underfilled=true`，最新 30-120 分钟窗口主因转为 `analysis_volume_ratio_below_floor`。
- 只读线上探针确认：候选池不是只剩 5 个交易对，最近 30 分钟市场分析已覆盖 21 个交易对、120 分钟覆盖 42 个交易对；真正收窄发生在 ranker 过滤后。
- 探针进一步发现两个根因：
  - 部分被选中的样本呈现 `volume_ratio=1.0`、`ADX=20`、`volatility_20=0` 的默认值形态，而线上 DB K 线已经滞后数小时。这说明指标快照缺失后，FeatureVector 默认值把无指标候选伪装成可分析候选。
  - `volume_ratio` 原先沿用趋势周期优先级，通常来自 `1h`。入场候选筛选需要的是当前活跃度，直接拿 1h 量比会把短周期正在活跃、但 1h 量比低或当前 1h 蜡烛未完成的交易对误杀。

本次修复范围：
- `data_feed/feature_vector.py`：新增 `indicator_snapshot_available`、`volume_ratio_timeframe`、`entry_activity_volume_ratio`、`entry_activity_volume_timeframe`，并在 LLM 上下文输出入场活跃量比来源。
- `services/data_service.py`：K 线指标计算前丢弃当前未完成蜡烛；趋势量比保留原 `volume_ratio`，短周期活跃度写入 `entry_activity_volume_ratio`；真实指标存在时才写 `indicator_snapshot_available=true`。
- `services/entry_feature_ranker.py`：候选评分与 tradable/analysis 量比门统一使用 `entry_activity_volume_ratio`；`indicator_snapshot_available=false` 的 FeatureVector 归因 `missing_indicator_snapshot`，不得进入 hard/secondary candidate，也不得 fallback 消耗市场分析。
- `scripts/inspect_online_strategy_health.py`：market symbol compact 输出保留 `volume_ratio_source`、趋势量比和短周期活跃量比字段，线上排查不再只看到一个模糊 `volume_ratio`。
- `tests/test_data_service_security.py`、`tests/test_entry_feature_ranker.py`、`tests/test_inspect_online_strategy_health.py`：覆盖未完成 K 线丢弃、短周期活跃量比用于候选过滤、指标缺失不能 fallback、诊断保留量比来源。
- `tests/test_strong_opportunity.py`：修复固定日期导致的 24 小时 lookback 脆弱测试，改为相对当前时间。

安全边界：
- 本批不降低 `analysis_volume_ratio`、`analysis_notional`、收益质量、ML readiness、仓位、杠杆、止盈止损、OKX 下单或平仓规则。
- 本批不改训练样本、不改模型 artifact、不写历史订单/持仓/收益数据、不做 DB 迁移。
- `entry_activity_volume_ratio` 只改变“候选是否值得花 AI 分析预算”的活跃度口径；它不是执行许可，也不能绕过证据、收益质量、风控、仓位、杠杆、OKX 合约规则。
- `missing_indicator_snapshot` 出现时，后续必须优先检查 K 线缓存、OKX fetch、feature batch timeout、服务运行环境和数据源刷新；不得把默认指标当成真实技术形态。

本地验证：
- 定向测试：`pytest tests/test_entry_feature_ranker.py tests/test_data_service_security.py tests/test_inspect_online_strategy_health.py -q`，48 passed。
- 相关边界测试：`pytest tests/test_trading_service_boundaries.py tests/test_crypto_feature_coverage.py tests/test_trading_params.py -q`，165 passed。
- 相关辅助测试：`pytest tests/test_entry_candidate_filter.py tests/test_market_hold_penalty.py tests/test_entry_probe_market_quality.py tests/test_market_decision_risk_assessment.py -q`，22 passed。
- 全量测试：`pytest -q`，1532 passed。
- `ruff check .`：通过。
- `black --check .`：通过。

当前真实结论：
- 本批解决的是“候选量比口径错误”和“指标缺失默认值伪候选”问题。它应减少无真实技术指标的市场分析消耗，并减少 1h 量比对短周期活跃候选的误杀。
- 这不是“强行多开仓”的改法。后续是否开仓仍取决于证据、预期净收益、收益质量、server_profit、ML readiness、执行契约和 OKX 风控。
- 如果部署后候选仍 underfilled，下一步必须看 `rank_filtered_out_reason_counts` 中 `missing_indicator_snapshot`、`analysis_volume_ratio_below_floor`、`analysis_notional_below_floor` 的新比例，以及 `volume_ratio_source` 是否已经从 `entry_activity_volume_ratio` 生效。
- 当前仍未完成的问题包括：ML top return 为负、server_profit selected-side 质量、shadow missed opportunity、强机会 canary、历史脏数据/乱码代码治理、以及不开仓/小单/不赚钱的收益闭环。

后续 AI 防偏要求：
- 看到 `volume_ratio=1.0`、`ADX=20`、`volatility_20=0` 时，必须先检查 `indicator_snapshot_available` 和 timeframe，不得把默认形态解释成“技术指标健康”。
- 看到 `analysis_volume_ratio_below_floor` 时，必须同时输出 `volume_ratio_source`、`trend_volume_ratio_timeframe`、`entry_activity_volume_timeframe`；不得直接降阈值。
- 看到候选交易对重复时，必须区分 scan 覆盖、feature valid、rank filter、recent dedupe、market AI budget 和执行 gate，不能笼统说“全市场没扫到”。
- 任何后续“多开一点、仓位大一点”的动作，都必须建立在真实强机会识别、收益质量和已成交费后收益验证上，不能用本批候选口径修复当理由。

回滚点：
- 代码层可回滚 `data_feed/feature_vector.py`、`services/data_service.py`、`services/entry_feature_ranker.py`、`scripts/inspect_online_strategy_health.py` 与对应测试。
- 线上回滚后必须重启 `bb-paper-trading.service` 与 `bb-dashboard.service`，否则交易主进程仍可能保留旧 FeatureVector/ranker 逻辑。

线上复验补记（2026-06-26 09:49 北京时间）：

- 本轮追加修正了 `scripts/inspect_online_strategy_health.py` 的远端采样模板 compact 层。原因是本地 compact 已输出 `volume_ratio_source` 等字段，但线上 `--market-symbol-only` 实际走的是远端模板；若只补本地层，后续仍会看到模糊 `volume_ratio`，容易误判为没有上线或继续走向降阈值。
- 本地最终验证：
  - `pytest tests/test_entry_feature_ranker.py tests/test_inspect_online_strategy_health.py -q`：34 passed。
  - `pytest -q`：1533 passed。
  - `ruff check .`：通过。
  - `black --check .`：通过。
  - `git diff --check`：通过。
- 线上部署：
  - 第一次同步上传 `scripts/inspect_online_strategy_health.py`、`services/entry_feature_ranker.py` 并重启 split services。
  - 第二次同步仅上传 `scripts/inspect_online_strategy_health.py`，补齐远端模板 compact 层，并重启 split services。
  - 最终服务状态：`bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service` 均 active，Dashboard 返回 `302`。
  - 最终远端进程：`bb-paper-trading.service MainPID=1069302 ActiveEnterTimestamp=Fri 2026-06-26 01:44:01 UTC`；后续第二次脚本同步也已重启三项服务。
  - 远端 `py_compile services/entry_feature_ranker.py scripts/inspect_online_strategy_health.py` 通过。
- 线上 5 分钟只读复验：
  - `trade_execution_contract.status=ok`。
  - `orders=0`、`failed_orders=0`、`rejected_orders=0`、`fast_loss_close_under_15m=0`。
  - 最新候选漏斗已显示：`volume_ratio_source=entry_activity_volume_ratio`、`trend_volume_ratio_timeframe=1h`、`entry_activity_volume_timeframe=15m/1h`。
  - 最新漏斗样本示例：`ADA/USDT volume_ratio=0.44 trend_volume_ratio=0.12 entry_activity_volume_ratio=0.4403 entry_activity_volume_timeframe=15m`。
  - 当前 `rank_filtered_out_reason_counts` 仍以 `analysis_volume_ratio_below_floor` 为主；`missing_indicator_snapshot` 在短窗口中已可被单独计数。
- 当前真实结论：
  - 候选诊断字段已经在线上打通，后续看到 `analysis_volume_ratio_below_floor` 时必须同时看量比来源和时间框架。
  - 本批仍不代表“不赚钱、不开仓、小单、ML degraded、server_profit 反向、历史脏数据、乱码代码”全部完成；这些仍按二期未完成闭环继续推进。
  - 不得用本批结果作为降量比阈值、放大仓位、提高杠杆、硬启 ML readiness 或绕过 OKX/风控的依据。

---

## 六十九、Batch I 二期阶段 3 补充记录：ML 自动训练 artifact 晋级门，阻止 degraded 候选替换线上模型（2026-06-26）

触发原因：
- 继续二期收益闭环时，线上 6 小时根因审计显示 `ml_not_contributing` 与 `ml_top_return_not_profitable` 仍是核心阻塞；当前线上 `local_ml_readiness=degraded`，`allow_live_position_influence=false`。
- 生产等价 dry-run 训练评估显示，当前候选训练窗口没有任何 variant 通过 readiness；主要指标包括 long/short top bucket fee 后收益为负、long PR-AUC 不达标、top return 不优于 bottom bucket 等。
- 复查 `MLSignalService.maybe_auto_train()` 发现自动训练会直接调用 `train_from_frame()` 的默认持久化路径：先写 `data/ml_signal/winrate_model.joblib` 和 metadata，再热加载。也就是说，即便新训练结果 degraded，也可能先替换线上 artifact。

本次修复范围：
- `services/ml_signal_service.py`：自动训练改为两阶段晋级：
  - 第一阶段只调用 `train_from_frame(..., persist_artifact=False)` 生成候选元数据，不写模型文件、不热加载。
  - 立即使用现有 `_influence_policy()` 与 `build_ml_readiness_report()` 评估候选 readiness。
  - 候选 `allow_live_position_influence=false` 时返回 `candidate_readiness_rejected`，保留上一版线上 artifact，并输出候选 metrics、readiness blockers、训练窗口组成和质量 totals。
  - 只有候选 readiness 通过时，才第二次调用 `train_from_frame(..., persist_artifact=True)` 写 artifact，并热加载。
- `services/ml_signal_service.py`：`training_policy` 增加 `promotion_requires_readiness=true`、`candidate_artifact_persisted=false`、`persist_artifact_only_when_readiness_allows_live_influence=true`，避免后续 AI 把“训练已跑”误读成“模型可上线影响交易”。
- `services/ml_signal_service.py`：候选被拒后增加进程内冷却判断；在未达到重新评估间隔或新增样本门槛前，自动训练不会每 30 分钟反复跑同一批 degraded 候选。
- `tests/test_ml_signal_training_quality.py`：新增两条契约测试，锁定 degraded 候选不得持久化、不得热加载、必须返回 readiness 阻塞原因；ready 候选必须先 dry-run 再 persist。
- `tests/test_trading_service_boundaries.py`：更新旧自动训练边界测试，明确 `force=True` 只强制评估候选，不代表绕过 readiness 强制写 artifact。

安全边界：
- 本批不降低 ML readiness 阈值，不硬启 `allow_live_position_influence`，不改变开仓阈值、收益质量门、server_profit、仓位、杠杆、止盈止损、OKX 下单或平仓规则。
- 本批不替换线上模型 artifact；只有未来候选模型真实通过现有 readiness 才允许自动晋级。
- 这不会直接让系统多开仓、放大仓位或立刻盈利；它解决的是“不要把不赚钱/不达标的候选模型写上线继续污染判断”。
- 如果后续用户看到 ML 仍 degraded，这是正确受控状态，不得为了页面好看改成 ready。

本地验证：
- `pytest tests/test_ml_signal_training_quality.py::test_ml_signal_auto_train_rejects_degraded_candidate_without_persisting tests/test_ml_signal_training_quality.py::test_ml_signal_auto_train_promotes_ready_candidate_only_after_dry_run tests/test_trading_service_boundaries.py::test_ml_signal_auto_train_quarantines_before_training -q`：3 passed。
- `pytest tests/test_ml_signal_training_quality.py tests/test_trading_service_boundaries.py::test_ml_signal_auto_train_uses_completed_cursor_for_new_samples tests/test_trading_service_boundaries.py::test_ml_signal_auto_train_quarantines_before_training -q`：19 passed。
- 全量测试 `pytest -q`：1536 passed。
- `ruff check services/ml_signal_service.py tests/test_ml_signal_training_quality.py tests/test_trading_service_boundaries.py`：通过。
- `ruff check .`：通过。
- `black --check services/ml_signal_service.py tests/test_ml_signal_training_quality.py tests/test_trading_service_boundaries.py`：通过。
- `black --check .`：通过。
- `git diff --check`：通过。

线上部署与复验：
- 使用 `python scripts/sync_to_online_server.py --split-services` 上传 `docs/superpowers/plans/2026-06-22-quant-closed-loop-eradication.md` 与 `services/ml_signal_service.py`，并重启 split services。
- 三项服务均 active，Dashboard 返回 `302`。
- 远端 `py_compile services/ml_signal_service.py` 通过。
- 远端服务状态：
  - `bb-paper-trading.service active MainPID=1112733 ActiveEnterTimestamp=Fri 2026-06-26 02:48:26 UTC`
  - `bb-dashboard.service active MainPID=1112737 ActiveEnterTimestamp=Fri 2026-06-26 02:48:26 UTC`
  - `bb-model-tunnels.service active MainPID=1112656 ActiveEnterTimestamp=Fri 2026-06-26 02:48:19 UTC`
- 远端无副作用桩验证通过：degraded candidate 返回 `candidate_readiness_rejected`，`train_persist_calls=[false]`，`artifact_persisted=false`，`candidate_artifact_persisted=false`，`ensure_load_calls=[]`，阻塞码包含 `long_top_return_below_threshold`、`short_top_return_below_threshold`。
- 15 分钟线上策略健康：
  - `trade_execution_contract.status=ok`
  - `orders=0`、`failed_orders=0`、`rejected_orders=0`、`fast_loss_close_under_15m=0`
  - `local_ml_readiness.status=degraded`、`allow_live_position_influence=false`
- 120 分钟线上策略健康：
  - `trade_execution_contract.status=ok`
  - `orders=1`、`filled_orders=1`、`failed_orders=0`、`rejected_orders=0`、`fast_loss_close_under_15m=0`
  - `contract_violation_count=0`、`weak_evidence_executed_count=0`、`negative_expected_executed_count=0`、`fast_loss_without_strong_exit_count=0`
  - `local_ml_readiness.status=degraded`，阻塞仍包括 long PR-AUC、long/short top return 和 top-vs-bottom 分层问题。

当前真实结论：
- 这轮修复把 ML 自动训练从“训练即替换”改成“候选先评估，ready 才晋级”，防止 degraded 模型反复覆盖线上 artifact。
- 当前线上历史 dry-run 已证明候选模型没有通过 readiness，因此正确行为是拒绝晋级，而不是放宽门槛或强启 ML。
- 下一步仍要继续做收益闭环：分析为什么训练样本/特征无法让高分桶稳定盈利，继续处理 server_profit、shadow missed opportunity、强机会识别、历史脏数据/乱码代码和真实费后收益归因。

后续 AI 防偏要求：
- 后续看到 `candidate_readiness_rejected` 时，必须解释为“候选模型未准入，线上 artifact 保持不变”，不得说成训练失败或系统故障。
- 后续不得把 `force=True` 当作强制写模型；它只能强制跑候选评估，不能绕过 readiness。
- 任何“让 ML 参与实盘影响”的动作，都必须以 `candidate_readiness.allow_live_position_influence=true` 且本地/线上验证通过为前提。
- 如果自动训练反复被拒，必须回到样本选择、标签、特征、side-aware 分桶、费后收益和 shadow 质量，而不是降低 readiness 阈值。

回滚点：
- 代码层可回滚 `services/ml_signal_service.py`、`tests/test_ml_signal_training_quality.py` 与 `tests/test_trading_service_boundaries.py`。
- 线上回滚后必须重启 `bb-paper-trading.service`，否则交易主进程仍会保留新/旧 MLSignalService 逻辑。
- 本批无 DB 迁移、无历史数据覆盖、无交易参数放宽、无模型 artifact 替换。
