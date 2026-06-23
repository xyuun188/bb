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

这版核心就是：**不再围绕现有死框架修补，而是建立一个模型/专家/策略持续竞赛、淘汰、替换、增强的系统，最终以最懂赚钱、最懂数字货币投资的组合为准。**

新增防偏内容只服务一个目的：让后续 AI 按这个总控执行时，不会偷换目标、不乱放宽交易、不硬改状态、不造假指标、不跳过验证。
