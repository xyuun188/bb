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

- 新增记录不能再写入 `鏈/璇/銆/�/?` 这类乱码；
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
| A 乱码与文本治理 | 本地完成，真实库 dry-run 通过 | 只改文本治理写入边界、dry-run 审计脚本和系统巡检节点；不影响开仓/仓位/平仓/杠杆/模型权重/专家路由 | `pytest` 35 passed；脚本 `--help` 通过；Ruff/Black 局部通过 | 临时启动 DB 隧道后 dry-run 扫描 822 条：疑似记录 0、疑似字段 0、未修改数据；隧道已关闭 | 未提交 | 回滚本批改动文件即可；无 DB 迁移、无历史覆盖 | 还未部署线上服务；后续部署后需在系统巡检页面复查 `runtime_text_integrity` 节点 |
| B 训练数据治理 | 本地完成，真实库 dry-run 通过 | 只改训练样本质量评估、shadow 隔离 dry-run、ML/Local AI 训练读入口与数据采集治理快照；不删除原始数据，不影响开仓/仓位/平仓/杠杆/模型权重 | `pytest` 42 passed；脚本 `--help` 通过；Ruff/Black 局部通过 | 临时 DB 隧道 dry-run 扫描 200 条 completed shadow：quarantined 0、top_reasons 空，未修改数据；隧道已关闭 | 未提交 | 回滚本批 touched files 即可；无 DB 迁移、无历史覆盖；如规则误伤可回退 `DATA_QUALITY_VERSION` v3 相关改动并重训 | 尚未部署线上服务；Batch C 需基于治理报告进入 ML readiness 状态机；真实交易行为改善仍需后续批次 |
| C ML readiness | 本地完成，只读状态检查通过 | 只改 ML readiness 状态机、训练元数据指标、预测/状态接口 gating 与 dashboard 展示；不硬改 `ready`，不改开仓/仓位/平仓/杠杆/专家路由/真实模型权重 | `pytest` 16 passed；新增预测层 gating 回归测试；Ruff/Black 局部通过 | 只读检查本地模型：`status=learning_only`、`allow_live_position_influence=false`，阻塞项为样本/测试数不足、PR-AUC 缺失、训练数据版本旧、模型过旧；未修改真实交易状态 | 未提交 | 回滚本批 touched files 即可；无 DB 迁移、无历史覆盖；如 readiness 误判可回退 `services/ml_readiness.py` 与 ML status/predict 集成 | 尚未部署线上服务；旧模型因 v1 数据质量与 PR-AUC 缺失被保守锁为 learning_only，需后续重训/线上观察后才可能进入 ready |
| C2 模型/专家体检 | 本地完成，真实库只读验证通过 | 只做模型/专家参与、采纳、收益、耗时、JSON 错误、未返回率体检与状态建议；不直接调整真实权重、不开启/禁用线上模型、不替换主链路 | `pytest` 29 passed；Ruff/Black 局部通过；新增只读完整报告接口 `/model-expert-health/status` 与系统巡检卡 | 临时 DB 隧道只读调用 C2 巡检卡成功：7 个组件、全部 `shadow_only` 复核建议、`live_weight_mutation=false`；未修改数据；隧道已关闭 | 未提交 | 回滚 `services/model_expert_health.py`、系统巡检接入与测试即可；无 DB 迁移、无权重/路由写入 | 尚未部署线上服务；当前真实库体检建议只是观察/复核，后续 C3 需建立基线竞赛后才能把贡献用于权重调整 |
| C3 模型/专家竞赛 | 本地完成，真实库只读验证通过 | 建立只读竞赛报告、baseline 对比、shadow/sim/live 分层状态与权重建议来源；不直接提高/降低真实权重，不让新模型上线，不替换主链路 | `pytest` 18 passed；Ruff/Black 局部通过；新增只读完整报告接口 `/model-expert-competition/status` 与系统巡检卡 | 临时 DB 隧道只读调用 C3 巡检卡成功：`baseline_missing`、shadow 样本 1192、`can_apply_live_weight=false`、未修改数据；隧道已关闭 | 未提交 | 回滚 `services/model_expert_competition.py`、系统巡检接入与测试即可；无 DB 迁移、无权重/路由写入 | 当前真实库缺少可用 baseline outcome，竞赛报告只能阻塞真实权重判断；后续需补足 executed outcome 或模拟 A/B 账本后才可作为 C5 路由依据 |
| C4 数字货币特征补强 | 本地完成，真实库只读验证通过 | 新增数字货币特征覆盖/新鲜度治理报告、数据采集页特征覆盖面板、系统巡检卡与只读接口；只做特征可用性审计，不改开仓/仓位/平仓/杠杆/模型权重/专家路由，不让缺失特征驱动实盘 | `pytest` 75 passed；Ruff/Black 局部通过；新增 `/crypto-feature-coverage/status` 与数据采集页 `feature_coverage` 只读返回 | 临时 DB 隧道只读调用 C4 报告和巡检卡成功：17 类特征、缺失 5、过期 0、已中性阻断 5、观测币种 56、`audit_only=true`、`live_signal_mutation=false`、`can_missing_features_drive_live_entry=false`、缺失策略 `neutral_blocked`；未修改数据；隧道已关闭 | 未提交 | 回滚 `services/crypto_feature_coverage.py`、系统巡检/数据采集接入、前端展示与测试即可；无 DB 迁移、无历史覆盖、无实盘参数改动 | 当前真实库仍缺少部分 C4 特征源，报告会阻断其真实开仓影响；后续如要让事件日历/清算/板块联动等进入可用状态，需先补真实采集来源和时间戳证据 |
| C5 动态路由 | 本地完成，真实库只读验证通过（尚未部署产生线上路由样本） | 新增模型/专家动态路由影子策略、决策 raw 追踪、只读路由报告、系统巡检卡与 `/model-dynamic-routing/status`；初始只生成 shadow/canary 计划，不改变真实专家调用、不跳过风控、不替换主链路 | `pytest` 43 passed（C5 路由、系统巡检、batch expert/错误安全/多样性回归）；Ruff/Black 局部通过 | 临时 DB 隧道只读调用 C5 报告和巡检卡成功：当前历史 route_plan 0（新代码未部署，尚无 `dynamic_model_routing` 样本）、`audit_only=true`、`live_route_mutation=false`、`can_apply_live_route=false`、巡检卡 warning；未修改数据；隧道已关闭 | 未提交 | 回滚 `services/model_dynamic_routing.py`、`ai_brain/ensemble_coordinator.py` 动态路由 raw 写入、系统巡检接入与测试即可；无 DB 迁移、无权重/路由写入 | 真实线上尚未产生动态路由样本，因此“减少重复专家调用”只在影子计划层可验证；必须部署后观察 route_plan、弱证据执行和快亏平指标，再决定是否进入 canary，不能直接 live 替换 |
| D 影子错过机会闭环 | 本地完成，真实库只读验证通过 | 新增 shadow missed opportunity 闭环服务、策略学习接入、系统巡检卡、只读接口与页面展示；只把错过机会作为同币种同方向重复证据后的学习/probe 候选，不强制开仓、不绕过风控、不改杠杆/仓位/模型权重/专家路由 | `pytest` 109 passed（D 服务、策略学习、系统巡检、Dashboard 契约）；`pytest` 66 passed（entry evidence/probe/opportunity/memory/data collection 回归）；Ruff/Black 局部通过 | 临时 DB 隧道只读调用 D 默认报告和巡检卡成功：默认窗口 `24h/200`、主键窗口查询、服务约 8.019s、巡检卡约 6.729s；completed 192、missed 80、adopted 1、probe 2、blocked 27、weak_evidence_executed 0；`audit_only=true`、`live_entry_mutation=false`、`can_bypass_risk_controls=false`、`weak_evidence_execution_allowed=false`、`global_missed_count_can_drive_entries=false`；隧道已关闭 | 未提交 | 回滚 `services/shadow_missed_opportunity_closed_loop.py`、`services/strategy_learning.py`、系统巡检/前端接入与测试即可；无 DB 迁移、无历史覆盖、无真实交易参数改动 | 当前 D 只完成“错过机会可被保守利用”的闭环，真实收益改善仍需部署后观察；巡检卡 warning 是因为存在被阻塞原因，不是弱证据执行；在线报告为性能安全使用 24h/200 近端窗口，不代表全历史统计 |
| E 开仓/仓位/平仓重构 | 本地完成，真实库只读发现历史遗留 critical，需部署后观察 | 新增交易执行契约审计、只读报告和系统巡检/API 接入；补成交确认写入最终 execution reason；审计只回查订单 decision_id 与快亏附近平仓决策，不放宽开仓、不绕过风控、不改杠杆/仓位/模型权重/专家路由、不写真实库 | `pytest tests/test_trade_execution_contract.py tests/test_system_audit_api.py -q` 29 passed；`pytest tests/test_trading_service_boundaries.py tests/test_entry_evidence_policy.py tests/test_entry_evidence_probe.py tests/test_entry_opportunity_scoring.py tests/test_position_quality.py tests/test_new_pair_loss_pause.py tests/test_trade_execution_contract.py -q` 185 passed；Ruff/Black 局部通过 | 临时 DB 隧道只读调用 E 报告和巡检卡成功：服务约 5.895s、巡检卡约 4.538s；`audit_only=true`、`live_entry_mutation=false`、`live_exit_mutation=false`、`can_bypass_risk_controls=false`；补充订单决策 28、补充 exit 决策 7；executed_entry 13、weak_evidence_executed 4、missing_entry_explanation 1、fast_loss 8、fast_loss_without_strong_exit 0、contract_violation 5；隧道已关闭 | 未提交 | 回滚 `services/trade_execution_contract.py`、`services/execution_service.py`、系统巡检接入与测试即可；无 DB 迁移、无历史覆盖、无真实交易参数放宽 | 历史真库仍有 4 条弱证据执行和 1 条缺开仓解释，不能标记线上全绿；当前本地 `EntryPolicy` 已阻断 `weak_conflict_probe/degraded_missing_probe`，成交确认后会写 reason，但需部署后观察新增样本必须为 0，未清零前不得做任何开仓放大 |
| F 系统巡检升级 | 本地完成，真实库只读验证通过但整体仍 warning | 新增巡检 `owner_path/state/state_label` 契约、issue ledger 状态分层、section timeout、失败 fallback 保留原始 section key、模型训练未配置观察态与可见文本乱码源头修复；不做展示型假指标，不改开仓/仓位/平仓/杠杆/模型权重/专家路由 | `pytest tests/test_system_audit_api.py -q` 23 passed；`pytest tests/test_system_audit_api.py tests/test_text_integrity_runtime.py tests/test_runtime_text_integrity_audit.py -q` 30 passed；Ruff/Black 局部通过 | 临时 DB 隧道只读调用完整巡检成功：总耗时约 33.424s；status=`warning`；cards 16、critical 0、warning 12、ok 4、findings 10、nodes 18；issue ledger fixed 4、unresolved 11、observing 1；`visible_text_encoding` ok、offender_count 0；无缺失 `owner_path/state/state_label`，无泛化 `audit_section_*` root cause；隧道已关闭 | 未提交 | 回滚 `web_dashboard/api/system_audit.py`、`services/text_integrity.py` 与对应测试即可；无 DB 迁移、无巡检历史写入、无真实交易参数改动 | 当前线上整体仍为 warning，说明还有 11 个未解决项和 1 个观察项；F 只提升定位和防偏能力，不代表策略收益已改善，也不能覆盖 E 的历史弱证据执行风险；后续批次必须继续读 issue ledger 而不是只看页面摘要 |
| G 质量债治理 | 本地完成，待 H 统一部署/线上观察 | 按总控先固化 bug/security 门，再分目录清理 Ruff/Black；只处理 Ruff/Black 报告中的文件，未全仓一键格式化，未修改开仓/仓位/平仓/杠杆/模型权重/专家路由 | `ruff check . --select F,B,E9,S608,E722` 0 issues；`ruff check .` 0 issues；`black --check scripts services web_dashboard/api tests ai_brain config db executor` 通过；`pytest tests/test_dashboard_auth_accounts.py tests/test_dashboard_security.py tests/test_secret_utils.py tests/test_model_runtime_policy.py tests/test_server_monitor_probe.py tests/test_batch_expert_json_stability.py -q` 76 passed；`pytest tests/test_trading_service_boundaries.py tests/test_market_auto_entry_processor.py tests/test_memory_feedback.py tests/test_order_position_reconciliation.py -q` 134 passed；`pytest tests/test_system_audit_api.py tests/test_text_integrity_runtime.py tests/test_runtime_text_integrity_audit.py -q` 30 passed；关键脚本/模块 `py_compile` 通过 | 本批为静态质量与格式治理，未连接真实库、未写线上数据；线上验证并入 H 部署后 systemd/Dashboard/系统巡检/策略健康观察 | 未提交 | 回滚本批 Ruff/Black touched files 即可；无 DB 迁移、无历史覆盖、无真实交易参数改动 | G 已清零当前 Ruff/Black 门，但它只降低质量债和后续误改风险，不代表策略收益改善；后续新增文件必须继续进质量门，H 部署前仍需密钥扫描和线上巡检 |
| H 线上部署与持续评估 | 已部署线上，进入 2h/24h/72h 持续观察 | 按 split-services 流程同步源码并重启 `bb-model-tunnels.service`、`bb-paper-trading.service`、`bb-dashboard.service`；部署前记录本地 HEAD `375f410dd311f98d20871491dbb05c5430e8cc9e` 与分支 `codex/profit-attribution-state-machine`；本批不放宽开仓、不改杠杆、不绕过风控，所有交易行为改动仍受前序契约审计约束 | 部署前最终闸门：`pytest -q` 1296 passed；`ruff check .` 0 issues；`black --check scripts services web_dashboard/api tests ai_brain config db executor core` 通过；`security_secret_scan.py --fail-on high .` 扫描 514 files OK；`git diff --check` 通过；`node --check web_dashboard/static/js/dashboard.js` 通过 | `sync_to_online_server.py --split-services` 上传 60 个变更文件并重启成功：model tunnels ok，三项 systemd 均 active，Dashboard `302` 健康响应；线上模型服务 `qwen3-14b-trade`、`deepseek-r1-14b-risk`、`local-ai-tools` active，废弃服务 inactive；120 分钟策略健康：240 decisions 全 hold、entry/orders/failed_orders/fast_loss_close 均 0、open_positions 2；真实库只读文本审计扫描 200 条 0 疑似；真实库系统巡检 30.512s 返回 warning、critical 0、cards 16、warning 12、ok 4、owner/state 契约完整；交易执行契约仍为历史遗留 5 violations（4 weak evidence、1 missing explanation），无 fast_loss_without_strong_exit | 本轮统一提交（见 Git 历史） | 首要回滚点为部署前 commit `375f410dd311f98d20871491dbb05c5430e8cc9e`，可回滚线上 `/data/bb/app` 到该代码点并重启三项服务；本批无 DB 迁移、无历史覆盖、无真实交易参数放宽 | 线上仍为 warning，不是全绿：历史 E 契约 violation 未消失，策略近 120 分钟仍 0 开仓，错过机会样本仍多；不得据此做开仓放大，必须继续观察新增弱证据执行、失败订单、快亏平、route_plan 样本和收益闭环，2h/24h/72h 观察未达标时回到对应批次定位 |

---

这版核心就是：**不再围绕现有死框架修补，而是建立一个模型/专家/策略持续竞赛、淘汰、替换、增强的系统，最终以最懂赚钱、最懂数字货币投资的组合为准。**

新增防偏内容只服务一个目的：让后续 AI 按这个总控执行时，不会偷换目标、不乱放宽交易、不硬改状态、不造假指标、不跳过验证。
