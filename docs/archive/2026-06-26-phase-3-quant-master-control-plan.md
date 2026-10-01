# Quant Rebuild Phase-3 Master Control Plan

> Status: historical Phase 3 foundation plan and implementation log.
> For new Profit-First v3 execution, use `2026-06-29-profit-first-v3-authoritative-master-plan.md` as the single entrypoint. This file remains authoritative only for preserved Phase 3 facts, safety boundaries, infrastructure history, and completed implementation checkpoints that do not conflict with the new authoritative plan.

> **总控定位：** 这份文档是“量化系统三期开发总控文档”，用于把“更赚钱、更高效、更智能、且不浪费贵服务器资源”的目标落成可执行任务。
> 本文档不直接改真实交易行为；任何影响真实开仓、仓位、平仓、模型权重、专家路由、风险阈值的改动，都必须先经过 `shadow -> canary -> live`。
> 本文档默认前提：保留现有系统主框架、固定专家槽、`local_ai_tools` 契约和现网可回退能力，重点升级内部引擎与主控逻辑。
> 新大模型服务器本次按“停旧占用、释放 GPU、旧数据原地隔离保留、全量资源用于新方案”处理：旧模型、旧缓存、旧运行时、旧试验数据原地保留但不得参与三期；旧服务、旧容器、旧进程必须停止并禁止继续占用 GPU/端口/调度资源；三期新增模型、缓存、训练、日志、运行时统一放在 `/data/BB`。
> 本文档同时吸收并继承旧总控文档《2026-06-22-quant-closed-loop-eradication.md》的核心约束，防止后续改版走偏、只做表面补丁或为了推进进度牺牲真实收益口径。

---

## 零、旧总控继承摘要

### 0.1 这份三期总控继承哪些旧原则

从旧总控文档继承并继续生效的核心原则如下：

- `赚钱优先`
  - 最终看手续费后净收益、盈亏比、最大回撤、快亏快平比例、小仓质量、错失机会减少。
- `证据优先`
  - 模型、专家、策略、训练、路由都必须能被回放、对比、验证，不能凭感觉上线。
- `动态组合`
  - 不同市场状态下调用不同模型、不同专家、不同风控策略，不能死流程。
- `能进能退`
  - 模型可晋级，也必须可降权、可退役、可回退。
- `不迷信大模型`
  - 大模型负责解释、仲裁、复核；数值收益预测必须依赖结构化模型和真实交易数据。
- `不盲目微调`
  - 数据没治理干净、标签不稳定之前，不做大规模微调。
- `不简单放宽开仓`
  - 任何放宽都必须经过净收益、损失概率、尾部风险、模型一致性、OKX 规则验证。
- `不让 AI 自行扩题`
  - AI 只能解决当前批次明确范围，不得顺手改无关策略或数据口径。
- `不以测试通过代替真实有效`
  - 本地测试只是底线，交易行为改动必须有线上巡检、回放、影子验证和实盘观察。

### 0.2 这份三期总控继承哪些旧红线

以下情况即使代码改了，也不能算完成：

- 只改页面文案，不改真实链路
- 只把 `learning_only` 硬改成 `ready`
- 只放宽阈值让系统乱开仓
- 只清源码乱码，不清数据库、缓存、训练入口污染
- 只做本地测试，不做线上核验
- 没有 Git、巡检、回退点、上线观察
- 没有说明本批是否影响真实交易
- 没有处理 OKX 对账和历史脏数据污染

### 0.3 这份三期总控继承哪些批前纪律

每个开发批次在进入实现前，仍然必须回答：

- 本批只解决什么问题
- 本批明确不解决什么问题
- 本批会不会影响真实交易
- 本批会触碰哪些链路
- 本批最小验收标准是什么

每批仍然必须遵守：

- `Preflight`
- `RED`
- `GREEN`
- `VERIFY`
- `REPORT`

### 0.4 三期开发在旧总控基础上的新增重点

本次三期开发相比旧总控，新增并抬升为一等事项的只有四类：

- 新贵模型服务器停旧占用、释放 GPU、三期数据根目录隔离
- 旧服务器白名单迁移与退役
- OKX 对账链路根治
- 历史脏数据治理与训练重建

---

## 一、改版目标

- 把系统从“通用大模型主导交易判断”升级为“专业量化模型主导、LLM 辅助仲裁”的结构。
- 把贵服务器的资源优先用于时序预测、收益预测、风险控制、滚动训练、walk-forward 验证，而不是长期只托一个超大聊天模型。
- 保留现有可运行框架，降低重构期间的生产中断风险。
- 建立模型晋级、回退、评估、训练、主控调度的闭环。
- 解决这些核心问题：
  - 开仓质量不稳定；
  - 小赚大亏、利润回吐、快亏快平；
  - 通用大模型延迟和输出不稳定影响主链路；
  - 服务器资源使用不聚焦；
  - 训练、影子评估、生产切换没有形成闭环；
  - 改完一处容易把另一处搞坏。

### 1.1 最高目标：Alpha Factory 自成长赚钱闭环

三期最终目标不是堆更多模型、更多页面或更多功能，而是把系统改造成一个围绕真实净收益自我进化的 `Alpha Factory`：

- `发现机会`
  - 从 OKX 原生行情、成交、盘口、资金费、波动、强弱轮动、异常插针、市场状态中生成候选机会。
- `验证机会`
  - 用时序模型、收益模型、本地 ML、方向竞赛、滑点/手续费模型和风控规则共同验证，不让单一 LLM 直接决定交易。
- `控制损失`
  - 每个候选必须先证明手续费后期望收益、亏损概率、尾部风险、流动性和执行成本可接受。
- `小仓学习`
  - 新策略、新模型、新币种、新市场状态只能从 shadow / tiny paper / canary 逐级晋级。
- `结果回灌`
  - 每笔交易都必须记录进场原因、拒单原因、模型贡献、执行偏差、真实收益、最大不利波动、最大有利波动、退出原因。
- `自动淘汰`
  - 不能持续证明正贡献的模型、策略、币种、方向、市场状态组合必须自动降权、暂停或退役。
- `放大优势`
  - 只有在 walk-forward、paper 观察、canary 结果中证明净收益和回撤达标的组合，才允许提高调用优先级、仓位上限或进入 live 候选。

三期所有开发任务都必须回答一个问题：它是否提高真实交易的期望收益、降低回撤、减少误判、改善执行质量、增强复盘学习。不能回答的任务，不进入主线。

### 1.2 赚钱闭环验收指标

后续阶段验收不以“模型装好了”或“接口通了”为最终完成标准，而以以下证据为准：

- `净收益`
  - 扣除手续费、滑点、资金费后的 paper/canary 净收益为正，且能按币种、方向、策略、市场状态拆分。
- `回撤`
  - 最大回撤、连续亏损、快亏快平、小赚大亏比例在阈值内。
- `预测质量`
  - 方向命中、收益分桶校准、亏损概率校准、时序预测误差、收益/风险排序能力必须有报告。
- `执行质量`
  - 预期价格与 OKX 实际成交价格、滑点、拒单、部分成交、平仓失败、资金费影响必须可回放。
- `模型贡献`
  - 每个模型和专家必须能证明自己对最终净收益、错失机会、风控拦截有正贡献；否则降权或退役。
- `数据干净`
  - 训练样本只能来自 Phase 3 干净视图；旧数据和备份只可审计，不得绕过白名单进入训练。

### 1.3 LLM 专家组新定位

- `BB-FinQuant-Expert-14B` 的 5 个专家角色是同一个 LLM 来源下的多视角分析，不得被系统当成 5 个独立模型投票。
- 同源 LLM 角色共享同一个 `provider_model` 时，交易放行必须按来源去重，最多算一个 LLM 证据源。
- 普通开仓必须叠加独立量化证据：
  - 服务器盈利模型；
  - Chronos / TimesFM 时序模型；
  - 本地 ML；
  - 方向竞赛；
  - OKX 原生交易事实和执行质量证据。
- `Qwen3-32B` 最终交易员只做确认、降级或否决，不得绕过主控层直接放大仓位。
- `DeepSeek-R1-14B` 高风险复核拥有独立否决权，但不进入普通交易对快路径。
- LLM 可以提出新策略假设、解释异常、总结亏损模式，但新策略必须经过离线回测、walk-forward、shadow、paper、canary 后才能影响 live。

### 1.4 最优组合方案：Profit-First Alpha Factory v2

本次三期后续开发以 `Profit-First Alpha Factory v2` 为唯一主线。目标不是让系统看起来更像 AI，而是让每个模块都围绕“真实数字货币市场中更稳定地赚到钱”服务。

- `最高目标函数`
  - 以手续费、滑点、资金费、拒单、部分成交、平仓失败全部扣除后的真实净收益为第一目标。
  - 以最大回撤、连续亏损、尾部亏损、快亏快平、小赚大亏比例作为硬约束。
  - 以可复盘、可训练、可淘汰、可晋级作为系统成长约束。
- `核心赚钱引擎`
  - OKX 原生事实层负责行情、订单、成交、仓位、资金费、账户权益和执行偏差，不再让 CCXT 抽象交易对成为事实来源。
  - Chronos-2 / TimesFM 负责短周期时序方向、路径风险和 challenger 对照。
  - CatBoost / LightGBM / XGBoost / 本地 ML 负责手续费后收益质量、亏损概率和分桶校准。
  - 方向竞赛、策略学习、收益归因负责证明“哪个模型、哪个策略、哪个币种、哪个市场状态真的在赚钱”。
- `LLM 正确用法`
  - LLM 不做主交易引擎，不靠 5 个同源专家互相投票制造虚假共识。
  - 5 个专家角色必须固定成不同任务切片：方向结构、盈利质量、短线时序、持仓退出、异常风险。
  - 同一个 `BB-FinQuant-Expert-14B` 无论扮演几个角色，放行新开仓时最多只算一个 LLM 来源。
  - LLM 的主要价值是解释异常、识别事件风险、总结失败模式、提出可测试假设、辅助复盘，而不是替代量化证据。
- `专家协作协议`
  - 专家之间不是多数投票，而是“分工证据板”：每个专家只回答自己领域的问题。
  - 新开仓至少需要一个 LLM 来源加独立量化证据，或多个真正独立来源加量化证据；同源专家一致不能单独放行。
  - 风险专家只负责硬风险和执行安全，不得用泛泛谨慎长期压制所有机会。
  - 持仓/退出专家优先保护真实落袋收益，减少利润回吐和亏损扩大。
- `自我成长闭环`
  - 每个候选、拒单、试单、开仓、加仓、减仓、平仓都必须进入干净 Phase 3 事实表。
  - 每个模型输出必须记录 `model_name`、`model_version`、`route_mode`、`latency_ms`、`expected_return`、`actual_return`、`decision_impact`。
  - paper/canary 结果按币种、方向、策略、市场状态、模型来源拆分归因。
  - 贡献为正的组合才允许提高优先级或进入更高阶段；贡献为负或证据不足的组合自动降权、暂停或退役。
- `资源使用原则`
  - 8 张 RTX 5090 不以“显存占满”为目标，而以“线上推理 + 影子挑战 + 训练评估 + walk-forward”同时产出收益证据为目标。
  - GPU 0-1 服务最终决策仲裁，GPU 2 服务专家池，GPU 3 服务高风险复核，GPU 4-5 服务专业时序主模型与 challenger，GPU 6-7 服务训练、回测、walk-forward 和超参搜索。
  - 如果某个常驻模型不能产生可量化正贡献，必须让出 GPU 给训练、评估或更有效的 challenger。
- `验收边界`
  - 任何功能只有在能提升净收益、降低回撤、减少误判、改善执行质量或增强训练闭环时，才算三期主线有效完成。
  - 任何“模型已部署”“接口已连通”“专家能回答”都不是最终完成证据，只能算基础设施状态。
  - paper 启动前必须先保证 OKX 原生事实、模型服务、训练视图、影子评估、Go/No-Go 与 Stage Handoff 不存在硬阻塞。

### 1.5 Profit-First 主控 v3：从功能闭环升级为盈利闭环

本节作为三期后续开发的新主控补丁，优先级高于“继续补页面功能”或“继续堆模型”。如果系统仍然出现开仓小单、长期不开仓、平仓亏损、净收益不起色，必须按本节重新验收，不得用“接口已完成”“模型已部署”“页面有数据显示”代替盈利结果。

#### 1.5.1 当前未达最终目标的核心判断

- `系统功能很多，但主控仍偏防守`
  - 当前链路已有收益期望、证据分、尾部风险、ML、时序、拥挤方向、回撤模式、策略学习画像、持仓优先预算等多层门禁。
  - 多层门禁叠加后容易形成“看起来很安全，但长期不开仓或只开极小仓”的结构性结果。
- `小仓探针没有自动升级成盈利仓位`
  - 小仓学习是必要的，但如果强机会也长期被 probe/cap/recovery 限制压住，就会出现方向对但赚不到钱。
  - 系统必须能把强机会从 `tiny/probe` 自动晋级到 `meaningful size`，否则贵模型服务器和复杂主控都无法转化为收益。
- `开仓和平仓不是同一个收益目标函数`
  - 开仓侧关注期望收益和风险，平仓侧更多是触发式止损/保护。
  - 后续必须让每次开仓都携带退出计划，并用真实退出结果反向校准开仓质量。
- `策略学习还没有成为真正主控`
  - balanced_probe、loss_release、winner_hold 等画像不能只用于解释当前状态。
  - 哪个画像、币种、方向、市场状态最近真实净收益为正，必须自动提高优先级；持续亏损必须自动降权、停用或回到 shadow。
- `模型资源还没有完全转化为交易优势`
  - LLM、时序模型、本地 ML 不能只是“多个意见来源”。
  - 每个模型都必须对候选收益、亏损概率、方向、退出建议给出可结算预测，并用 OKX 真实结果计算贡献。

#### 1.5.2 新主控目标函数

后续所有开仓、仓位、平仓、训练、模型晋级都必须围绕同一个目标函数：

- `第一目标`
  - 扣除手续费、滑点、资金费、拒单、部分成交、平仓失败后的 OKX 真实净收益。
- `硬约束`
  - 最大回撤、连续亏损、尾部亏损、快亏快平、小赚大亏、长期不开仓、无意义小单比例。
- `成长约束`
  - 每个模型、策略、币种、方向、市场状态组合都必须可复盘、可训练、可降权、可晋级、可退役。
- `禁止事项`
  - 禁止只靠放宽阈值解决不开仓。
  - 禁止只靠小仓探针制造交易数量。
  - 禁止让本地估算收益覆盖 OKX 真实收益。
  - 禁止让同源 LLM 多角色投票冒充独立证据。

#### 1.5.3 统一交易评分

每个候选机会必须生成统一交易评分，字段至少包括：

- `expected_net_return_pct`
- `loss_probability`
- `tail_loss_risk`
- `expected_hold_minutes`
- `fee_slippage_funding_adjusted_return`
- `recommended_position_size_pct`
- `recommended_leverage`
- `entry_reason`
- `no_entry_reason`
- `exit_plan`
- `model_contribution_sources`

验收规则：

- 缺少统一评分的候选，只能进入 shadow，不得进入真实 paper 开仓。
- expected net、亏损概率、尾部风险、手续费/滑点后收益口径不一致时，不得放大仓位。
- 统一评分必须进入 Phase 3 干净事实表，后续用 OKX 真实结果结算预测质量。

#### 1.5.4 仓位从固定小仓升级为分级晋级

仓位不再只按“探针/普通/恢复”粗粒度控制，而按机会质量分级：

- `Shadow only`
  - 证据不足、字段不全、模型冲突严重、OKX 事实不同步。
- `Tiny probe`
  - 新币种、新策略、新市场状态、低样本组合，只允许 0.5%-2%。
- `Validated probe`
  - 正期望、低尾部风险、至少一个独立量化证据支持，可用 2%-4%。
- `Meaningful entry`
  - expected net、profit quality、loss probability、时序/ML 至少两源支持，可用 5%-8%。
- `High conviction`
  - 多源独立验证、历史同桶正贡献、执行质量达标、高风险复核通过，可用 8%-12%。

验收规则：

- 小仓比例必须可统计：`tiny_probe_count / executed_entry_count` 长期过高时，必须触发“小仓无效”告警。
- 强机会长期无法升级到 `meaningful entry` 时，视为主控失败，不得继续只解释为“系统谨慎”。
- 仓位放大必须先经过 shadow/paper/canary 证据，不能直接影响 live。

#### 1.5.5 开仓必须绑定退出计划

每次开仓必须同步生成退出计划，不允许“先开仓，后面再临时判断怎么平”：

- `initial_stop_loss`
- `take_profit_zone`
- `trailing_profit_rule`
- `max_hold_minutes`
- `profit_drawdown_line`
- `partial_close_plan`
- `invalidation_conditions`
- `reentry_block_after_loss`

验收规则：

- 没有退出计划的开仓不得真实提交。
- 平仓必须记录触发原因：止损、利润回吐、趋势反转、持仓超时、风险事件、资金费、模型反向、手动平仓、OKX 同步修复。
- 亏损平仓必须回写归因：开仓错、进场晚、止损太近、持仓太短、趋势反转、模型误判、滑点/手续费、平仓过急。

#### 1.5.6 不开仓诊断必须成为主控输入

每次未开仓必须分类，而不是只显示“观望”：

- `profit_expectancy`
- `evidence_gate`
- `risk_or_precheck`
- `model_disagreement`
- `position_capacity`
- `market_budget`
- `okx_sync_or_exchange`
- `data_quality`
- `strategy_learning_pause`

验收规则：

- 每 24 小时必须输出不开仓诊断：到底是市场没机会，还是系统过度保守。
- 如果 positive expected net 候选长期存在但没有开仓，必须触发“错失机会/过度保守”复盘。
- 如果没有 positive expected net 候选，必须反查特征、模型、市场扫描范围和时序预测是否失效。

#### 1.5.7 亏损平仓和小赚大亏专项闭环

亏损平仓不再只是执行结果，而是训练和策略淘汰的强信号：

- `fast_loss`
- `profit_giveback`
- `small_win_large_loss`
- `late_entry`
- `premature_exit`
- `over_hold_loser`
- `under_hold_winner`

验收规则：

- 快亏快平比例超过阈值时，暂停对应币种/方向/策略组合进入 meaningful size。
- 小赚大亏比例恶化时，优先升级退出计划和持仓管理，而不是简单降低所有仓位。
- 盈利仓过早平仓和亏损仓拖太久必须分开统计，不能混成一个“平仓质量”指标。

#### 1.5.8 模型/策略排行榜和自动升降级

每个模型、专家、策略画像、币种、方向、市场状态组合都必须有排行榜：

- `net_pnl_after_cost`
- `win_rate`
- `profit_factor`
- `avg_win`
- `avg_loss`
- `max_drawdown`
- `fast_loss_rate`
- `small_win_large_loss_rate`
- `missed_opportunity_delta`
- `decision_latency_ms`
- `rejection_or_execution_error_rate`

升降级规则：

- 连续正贡献且回撤达标：提高调用优先级、允许更高仓位档。
- 贡献不显著：保持 shadow 或 tiny probe。
- 连续负贡献：降权、暂停、退回 shadow。
- 数据不足：不得晋级，只能继续采样。

#### 1.5.9 贵模型服务器资源使用新验收

8 卡服务器不再按“装了几个模型”验收，而按“是否产出交易优势证据”验收：

- GPU 0-1：最终仲裁必须输出可结算决策影响。
- GPU 2：专家池必须按同源去重，不能用 5 个角色制造虚假共识。
- GPU 3：高风险复核必须统计误拦、误放和放行后收益。
- GPU 4-5：Chronos/TimesFM 必须做时序 challenger 对照，并按真实收益结算贡献。
- GPU 6-7：训练、walk-forward、超参搜索必须定期产出晋级/退役报告。

如果某个常驻模型 7 天内无法产生可量化贡献证据，必须进入降级审计：释放 GPU、改为按需调用、替换 challenger 或转为离线评估。

#### 1.5.10 v3 完成口径

只有同时满足以下条件，才允许说 Profit-First 主控 v3 达到阶段目标：

- OKX 账户权益、订单、成交、持仓、历史仓位、费用和 PnL 与后台一致。
- 每次开仓都有统一交易评分和退出计划。
- 小仓、不开仓、亏损平仓都有分类诊断和 24 小时报表。
- 强机会能从 probe 自动晋级到 meaningful size，且有回退开关。
- 策略画像能按真实净收益自动升降级。
- 模型贡献能按真实 OKX 结果结算。
- paper/canary 净收益、回撤、快亏快平、小赚大亏指标达到验收线。

---

## 二、现状判断

### 2.1 现有架构可继续利用

- 固定专家槽已存在，可继续保留：
  - `trend_expert`
  - `momentum_expert`
  - `sentiment_expert`
  - `position_expert`
  - `risk_expert`
  - `decision_maker`
- `local_ai_tools` 已经具备独立契约：
  - `/profit/predict`
  - `/timeseries/deep/predict`
  - `/sentiment/deep/analyze`
  - `/exit/advise`
- 本地 ML、影子样本、交易样本、训练入口、自动训练触发都已存在。
- 当前系统并不缺“接模型的地方”，真正缺的是“更专业的内部引擎”和“更强的主控层”。

### 2.2 现有问题不适合靠继续堆大模型解决

- 通用 reasoning 模型在主链路里会带来延迟、JSON 截断、超时回退、证据不稳定问题。
- 8 卡长期托一个大模型，不等于更赚钱，只等于资源占用更高。
- 时序预测、收益判断、平仓纪律，本质上更适合专业结构化模型和时序模型。
- 现有本地 ML 不是没用，但应从“主角”转成“基线、回退、校准层”。

### 2.3 必须补进总控的四个现实问题

- `OKX 后台与本地持仓/平仓/收益记录长期对不上`
- `历史训练数据和收益样本存在脏数据污染`
- `旧模型服务器不再继续使用，必须做白名单迁移`
- `新贵模型服务器现在存在旧系统/旧模型/旧缓存残留和 GPU 占用，必须先停旧占用并把三期新数据隔离到 /data/BB`

这些问题如果不先纳入总控范围，后续再强的新模型也会被脏账本、脏样本和脏环境拖垮。

---

## 三、目标架构

### 3.1 四层结构

- `核心量化层`
  - 由 `local_ai_tools v2` 承担核心数值预测。
  - 负责时序、收益、风险、平仓建议。
- `专家仲裁层`
  - 保留多专家框架。
  - 主要负责解释、冲突仲裁、上下文理解、事件辅助判断。
- `风险控制层`
  - 负责仓位纪律、风险闸门、高风险复核、异常市场保护模式。
- `训练研究层`
  - 负责滚动训练、walk-forward、challenger 比赛、模型晋级、自动降权。

### 3.2 模型职责重分配

- `timeseries/deep/predict`
  - 主模型：`Chronos-2`
  - challenger：`TimesFM 2.5`
  - fallback：`Granite TTM`
- `profit/predict`
  - 主模型：`CatBoost + LightGBM + XGBoost`
  - 目标：`expected_return_pct`、`loss_probability`、`profit_quality_score`
- `sentiment/deep/analyze`
  - 主模型：`FinGPT / FinBERT 类`
  - 定位：辅助证据，不单独决定开仓
- `exit/advise`
  - 线上：规则 + 树模型
  - 影子：`FinRL` 或 RL/策略学习 challenger
- `LLM experts`
  - `trend/momentum/sentiment/position/risk`：快而稳的小中型模型
  - `decision_maker`：更强但仍受超时和权重约束的仲裁模型
  - `high_risk_review`：独立高风险复核模型

### 3.3 策略调度权归属

- 策略调度权归属 `主控层`，不得交给单一 LLM 独裁。
- `local_ai_tools v2` 负责计算：
  - 方向概率
  - 手续费后期望收益
  - 亏损概率
  - 收益质量
  - 平仓建议
- `LLM experts` 负责补充解释、异常上下文、事件风险和专家证据，不直接拥有下单权。
- `decision_maker` 只负责冲突整合、复杂上下文仲裁、高价值候选复议，不负责替代量化模型做全部判断。
- `high_risk_review` 只负责高风险候选独立复核，不进入普通交易对快路径。
- `risk gate` 拥有最终否决权；任何模型给出放行建议，都不能绕过硬风控。
- `trading_service / entry_direction_competition / model_dynamic_routing / strategy_learning` 共同组成主控调度链路。

### 3.4 交易判断分层调用

- 快路径：
  - 行情特征、本地 ML、`timeseries`、`profit`、基础 risk gate。
- 普通候选：
  - 在快路径基础上调用专家池 LLM。
- 高价值或证据冲突候选：
  - 在普通候选基础上调用 `decision_maker`。
- 高风险候选：
  - 大仓位、高波动、新闻冲击、连续亏损后重开、模型分歧过大时调用 `high_risk_review`。
- 不允许每个交易对都全量调用所有 LLM。
- 不允许为了“模型更全”牺牲普通交易对扫描时效。

---

## 四、服务器资源使用方案

### 4.1 目标原则

- 不追求“显存全部占满”。
- 追求“线上推理 + 离线进化 + 影子挑战”三者同时产生收益。
- 留足训练和评估资源，比单纯托大模型更值钱。

### 4.2 推荐 GPU 分工

- `GPU 0-1`
  - `decision_maker` 主仲裁模型
- `GPU 2`
  - 专家池主模型
- `GPU 3`
  - `high_risk_review`
- `GPU 4`
  - `Chronos-2` 线上主时序
- `GPU 5`
  - `TimesFM 2.5` shadow challenger
- `GPU 6-7`
  - 训练 / walk-forward / 超参搜索 / challenger 评估 / regime 重训

### 4.2.1 LLM 槽位最终口径

- 服务器资源分配以“角色槽位”为准，不以某个具体模型名永久锁死。
- `GPU 0-1` 是主 LLM 槽位：
  - 服务对象：`decision_maker`
  - 候选模型：`Qwen / GLM / DeepSeek` 中通过交易任务评测者
- `GPU 2` 是专家池 LLM 槽位：
  - 服务对象：`trend/momentum/sentiment/position/risk` 专家池
  - 要求：低延迟、稳定 JSON、可并发
- `GPU 3` 是高风险复核 LLM 槽位：
  - 服务对象：`high_risk_review`
  - 要求：独立超时、独立熔断、独立日志
- `GPU 4-5` 不作为常驻 LLM 聊天模型槽位，优先保留给时序模型与 challenger。
- `GPU 6-7` 不作为常驻 LLM 聊天模型槽位，优先保留给训练、回测和模型竞争。
- 可以安装多个 LLM 候选，但线上常驻模型必须按槽位、按评测结果启用。
- 未通过 shadow/canary 评测的 LLM 候选只能作为冷备用、离线评估或 challenger，不得直接进入 live 调度。

### 4.2.2 耗时控制规则

- 普通交易对扫描不得依赖全量 LLM 调用。
- 普通候选优先走快路径，只有进入候选池后才调用专家池。
- `decision_maker` 只处理少数冲突候选或高价值候选。
- `high_risk_review` 只处理高风险候选。
- 每个模型槽位必须记录：
  - `route_mode`
  - `model_name`
  - `model_version`
  - `latency_ms`
  - `timeout`
  - `fallback_reason`
  - `decision_impact`
- 任一新增模型导致普通扫描延迟明显上升时，必须先退回 shadow 或降低调用频率。

### 4.3 资源使用红线

- 不允许为了“更像 AI”而牺牲训练和评估资源。
- 不允许把全部 GPU 长期锁死给单一聊天模型。
- 不允许高风险复核模型进入所有请求的普通快路径。
- 不允许把候选 LLM 的安装清单当作最终交易架构。
- 不允许因为有空闲显存就把训练/评估 GPU 改成常驻聊天模型。

---

## 四点五、新贵模型服务器资源释放与三期隔离原则

### 4.5.1 总体原则

- 新大模型服务器按“新方案专用服务器”处理。
- 旧数据不再作为默认清理对象：旧模型、旧缓存、旧中间产物、旧测试环境可以原地保留，作为隔离历史数据处理。
- 旧服务、旧容器、旧进程、旧端口映射必须停止；旧数据不得继续占用 GPU、端口、定时任务、模型路由或训练资源。
- 不做“在旧残留上修修补补”的兼容方案；三期只使用新的隔离根目录。
- 三期新增模型、缓存、下载、训练、评估、日志、运行时、manifest 统一放在 `/data/BB`，不得写入系统盘或旧目录。
- 阶段 0 的目标不是删除历史数据，而是让历史数据“断电、隔离、不可被三期引用”。

### 4.5.2 必须停止/隔离的内容

- 历史 vLLM 进程、Open WebUI、SearXNG、通用聊天/工具容器。
- 历史 systemd 服务、端口映射、无主进程和自动重启策略。
- 历史 HuggingFace / ModelScope / vLLM 缓存不得被三期环境变量引用。
- 历史 `/data/trade_ai`、`/data/vols`、`/data/autodl-tmp` 等目录不得作为三期模型或缓存根目录。
- 历史 Python/conda/venv 试验环境不得作为三期服务运行环境。
- 旧 Qwen3.5-122B 等历史模型不得继续常驻占用 8x RTX 5090。

### 4.5.2.1 不删除历史数据的边界

- 不因阶段 0 自动删除 `/data/vols`、`/data/autodl-tmp`、旧模型目录、旧缓存目录或旧日志。
- 如未来确需删除，必须另开清理门禁，列出目录、体量、保留白名单、回滚边界和确认口令。
- 当前阶段只允许停止旧占用、取消旧自动重启、记录旧数据位置和体量。

### 4.5.3 重置后的目标状态

- `/data/BB` 是唯一三期新增数据根。
- `/data/BB/models` 存放三期模型。
- `/data/BB/cache` 存放 HuggingFace / ModelScope / pip / uv / torch / vLLM 缓存。
- `/data/BB/training` 存放训练、walk-forward、clean export。
- `/data/BB/runtime` 存放运行时 pid/socket/vLLM 配置。
- `/data/BB/logs` 存放三期服务、下载、训练日志。
- `/data/BB/manifests` 存放 reset/迁移/部署证据。
- 统一的新模型部署清单、训练/评估/日志约定、迁移白名单。
- 旧目录可以存在，但三期服务不得引用。

---

## 四点六、旧模型服务器迁移原则

### 4.6.1 必迁资产

- 原始交易底账导出
  - 订单
  - 成交
  - 持仓
  - 平仓
  - 反思样本
  - shadow/backtest 样本
- 训练评估报告
- 当前有效模型版本元信息
- tunnel / 服务端口 / 服务名口径
- 必要的安全配置项来源说明

### 4.6.2 不迁移资产

- 历史聊天模型缓存
- 旧 vLLM 临时目录
- 旧测试模型
- 无法证明有效性的 bundle
- 历史试验脚本
- 历史脏日志

### 4.6.3 迁移原则

- 迁移“底账与已验收资产”，不迁移“垃圾与包袱”。
- 迁移以白名单执行，不做整盘拷贝。
- 迁移后旧服务器进入退役状态，不再继续承担生产角色。

---

## 五、实施清单版

## 5.0 阶段 0：服务器资源释放、资产迁移、账本冻结

### 目标

- 在动模型和主控之前，把运行环境、历史账本、迁移口径先理顺。

### 任务

- 新贵模型服务器停旧占用并释放 GPU
- 三期新增数据根目录固定为 `/data/BB`
- 旧数据原地保留但不参与三期模型、缓存、训练和运行时
- 旧模型服务器迁移白名单导出
- 交易底账冻结快照
- 当前训练产物版本冻结
- 历史系统服务、端口、模型、环境盘点

### 产出

- `新服务器资源释放与隔离报告`
- `/data/BB` 三期工作区 manifest
- `旧服务器迁移白名单`
- `账本冻结快照`
- `训练产物冻结清单`

### 验收

- 新服务器旧 vLLM / WebUI / 通用容器 / 无主进程不再占用 GPU、端口或自动重启
- 三期新增数据只写入 `/data/BB`
- 旧数据可以原地保留，但不得被三期环境变量、服务配置、模型路由或训练脚本引用
- 旧服务器不再承担后续新方案生产职责
- 迁移范围可以逐条审计

---

## 5.1 阶段 A：基线冻结

### 目标

- 在改版前明确旧系统的真实表现。
- 固化“谁在赚钱、谁在拖后腿、哪些场景最差”。

### 任务

- 导出最近 `30-60` 天交易基线：
  - 开仓率
  - 胜率
  - 净收益
  - 盈亏比
  - 最大回撤
  - 平均持仓时长
  - 快亏快平比例
  - 利润回吐比例
- 固化三类证据来源贡献：
  - `ml_signal`
  - `local_ai_tools`
  - `LLM experts`
- 固化按币种、方向、周期、市场状态分桶表现。

### 产出

- `旧系统基线报告`
- `旧模型证据贡献报告`
- `旧系统回退快照`

### 验收

- 能明确回答“当前最赚钱的路径是什么”
- 能明确回答“当前最亏钱的路径是什么”
- 能明确回答“当前谁真正影响了实盘决策”

---

## 5.2 阶段 B：`local_ai_tools v2` 影子层上线

### 目标

- 不改现网调用契约，先把核心量化引擎升级到可对比状态。

### 任务

- 保留现有接口：
  - `/profit/predict`
  - `/timeseries/deep/predict`
  - `/sentiment/deep/analyze`
  - `/exit/advise`
- 给每个接口增加：
  - `primary_model`
  - `challenger_model`
  - `model_version`
  - `route_mode`
  - `shadow_payload`
  - `feature_coverage`
  - `fallback_reason`
- `timeseries/deep/predict` 先做三层输出：
  - live 使用旧逻辑
  - shadow 记录新逻辑
  - fallback 保持现有轻量逻辑

### 产出

- `local_ai_tools v2 shadow service`
- `时序预测误差对比看板`
- `多模型时序挑战记录`

### 验收

- 新旧结果能并排记录
- 新模型超时不影响旧逻辑
- 影子记录可按币种、市场状态查询

---

## 5.3 阶段 C：收益预测器升级

### 目标

- 把“会不会赚钱”从启发式和轻量树模型提升为收益优先集成模型。

### 任务

- 将 `/profit/predict` 的核心目标统一为：
  - `expected_return_pct`
  - `adjusted_expected_return_pct`
  - `loss_probability`
  - `profit_quality_score`
- 引入：
  - 手续费后净收益
  - 尾部损失惩罚
  - 流动性惩罚
  - 币种/方向 profile 惩罚
- 保留旧本地 ML 作为：
  - baseline
  - fallback
  - 校准层

### 产出

- `profit v2 ensemble`
- `收益质量评分卡`
- `symbol/side profile 风险画像`

### 验收

- 低质量单过滤能力提升
- 不靠抬高阈值硬性减少开仓
- 收益质量优于旧版基线

---

## 5.4 阶段 D：平仓引擎升级

### 目标

- 优先解决“小赚大亏、盈利回吐、亏损仓拖太久”。

### 任务

- 重做 `/exit/advise`，明确四类动作：
  - `hold`
  - `trail_profit`
  - `protect_profit`
  - `reduce_or_close`
- 增加四类判断维度：
  - 历史 profile 压力
  - 当前 pnl 结构
  - 持仓时长结构
  - 市场短线修复证据
- RL challenger 仅进入 shadow，不得直接 live 控仓。

### 产出

- `exit advisor v2`
- `平仓动作审计视图`
- `利润回吐治理报表`

### 验收

- 平均利润回吐下降
- 亏损仓超时持有下降
- 不出现明显过早止盈副作用

---

## 5.5 阶段 E：专家层权重重排

### 目标

- 保留专家框架，但把 LLM 从主脑降级成仲裁层。

### 任务

- 提高 `timeseries/profit/risk` 证据权重。
- 降低空泛文本推理对最终方向的直接影响。
- `decision_maker` 只负责：
  - 冲突整合
  - 复杂上下文解释
  - 高风险候选复议
- `risk_expert` 保持强制参与。

### 产出

- `新证据加权逻辑`
- `专家职责收缩说明`
- `冲突仲裁规则表`

### 验收

- 主方向不再主要由单次文本推理决定
- experts 分歧时能解释为什么最后放行或拦截
- reasoning 模型不会拖慢普通快路径

---

## 5.6 阶段 F：高风险复核独立化

### 目标

- 把“少数但昂贵的错误”单独拿出来处理。

### 任务

- 明确必须进入高风险复核的场景：
  - 大仓位
  - 高波动
  - 新闻冲击
  - 连续亏损后重开同向
  - 模型证据冲突大
- 高风险复核必须具备：
  - 独立模型
  - 独立超时
  - 独立熔断
  - 独立日志

### 产出

- `high_risk_review 独立复核通道`
- `高风险开仓审计视图`

### 验收

- 高风险单未通过复核不得放行
- 普通请求不受高风险复核拖累
- 复核失败原因可审计

---

## 5.7 阶段 G：训练与评估流水线升级

### 目标

- 把贵服务器从“模型推理机”升级成“模型工厂”。

### 任务

- 拆分三类训练：
  - `正式训练`
  - `shadow 训练`
  - `walk-forward 验证`
- 增加分桶评估：
  - 币种
  - 方向
  - 时间周期
  - 市场状态
  - 交易模式
- 建立模型晋级状态：
  - `shadow`
  - `canary`
  - `live`
  - `degraded`
  - `retired`

### 产出

- `训练评估总控脚本`
- `模型晋级规则`
- `challenger 比赛流水线`

### 验收

- 每次训练都有版本和质量报告
- 未通过晋级门槛的模型不得上线
- 可追溯任意 live 模型来自哪次训练

---

## 5.8 阶段 H：OKX 对账链路根治

### 目标

- 彻底解决本地持仓/平仓/收益和 OKX 后台不一致的问题。

### 根因定义

- 历史 symbol alias 不一致
- close fill 查询超时或缺失
- 拆分平仓/部分平仓处理不稳
- order / fill / position 关联主键不强
- 历史补账使用估算逻辑污染 closed position
- 费用分摊与净收益重算不一致

### 任务

- 强制以 `exchange_order_id / ordId / fillId` 作为对账一等主键
- symbol alias 只保留为辅助匹配
- 重建部分平仓和拆分平仓聚合逻辑
- 重建 entry fee / close fee / realized pnl 分摊逻辑
- 把“估算补账”降级为最后兜底
- 增加每日 dry-run 对账和审计报告

### 产出

- `OKX 对账根因报告`
- `持仓/平仓/收益重建脚本`
- `每日 dry-run 对账报表`

### 验收

- 最近窗口内本地与 OKX 后台持仓/平仓/收益一致
- 误差来源可解释
- 不再出现历史补账重复污染训练的情况

---

## 5.9 阶段 I：历史脏数据治理与训练重建

### 目标

- 彻底区分“审计底账”和“可训练数据”，避免脏样本继续污染模型。

### 数据处理原则

- 原始底账保留，不物理删除
- 训练视图重建，不继续沿用旧脏产物
- 可疑样本默认隔离，不默认放行

### 数据分层

- `audit_only`
- `repairable`
- `trainable`
- `quarantined`
- `retired_artifact_source`

### 任务

- 对历史订单、成交、持仓、收益样本做只读审计
- 标注无法强关联到 OKX 真值的样本
- 清理旧训练 artifact 与旧 bundle
- 重建 clean training view
- 重建本地 ML、`local_ai_tools`、向量索引

### 产出

- `历史脏数据治理报告`
- `训练样本白名单视图`
- `旧 artifact 退役清单`
- `新训练起点版本`

### 验收

- 无法确认真值的旧样本不再参与训练
- 原始底账仍可审计
- 新模型训练只消费 clean training view

---

## 5.10 阶段 J：生产切换顺序

### 切换顺序

1. `timeseries/deep/predict`
2. `profit/predict`
3. `exit/advise`
4. 专家权重重排
5. 高风险复核收紧

### 切换原则

- 先 `shadow`
- 再 `canary`
- 最后 `live`

### 强制要求

- 每一步都有回退开关
- 每一步都有对比报表
- 每一步都不能破坏现有契约

---

## 5.11 阶段 K：Profit-First 主控 v3 收口

### 目标

- 直接解决当前仍然存在的开仓小单、很久不开仓、平仓亏损、不赚钱问题。
- 把三期从“功能很多、规则很多、模型很多”收口为“每个交易动作都围绕 OKX 真实净收益优化”。
- 把策略学习从事后解释升级为主控输入：赚钱的组合自动晋级，亏钱的组合自动降级。

### 核心任务

- `统一交易评分`
  - 为每个候选生成统一评分：预期净收益、亏损概率、尾部风险、预计持仓时间、手续费/滑点/资金费后收益、建议仓位、建议杠杆、退出计划。
  - 缺少统一评分的候选只能进 shadow，不得真实 paper 开仓。
- `分级仓位晋级`
  - 把仓位分为 `shadow_only / tiny_probe / validated_probe / meaningful_entry / high_conviction`。
  - 强机会必须能从 tiny/probe 晋级到 meaningful size，不能长期被小仓 cap 压住。
- `开仓绑定退出计划`
  - 开仓前必须生成止损、止盈区、移动止盈、最大持仓时间、利润回吐线、分批平仓、失效条件和亏损后重开限制。
  - 平仓后必须反写退出计划是否有效。
- `不开仓诊断`
  - 把每次未开仓分类为收益不足、证据不足、风控/预检、模型分歧、持仓容量、市场预算、OKX/交易所、数据质量、策略学习暂停。
  - 每 24 小时输出“市场没机会 vs 系统过度保守”判断。
- `亏损平仓专项闭环`
  - 快亏快平、利润回吐、小赚大亏、进场过晚、过早平仓、亏损仓拖太久必须单独统计。
  - 亏损归因必须反向影响币种、方向、策略画像、模型权重和仓位档。
- `模型/策略排行榜`
  - 按模型、策略画像、币种、方向、市场状态统计净收益、回撤、快亏率、小赚大亏率、错失机会、延迟和执行错误。
  - 榜单结果必须驱动自动升降级。
- `服务器资源贡献验收`
  - 8 卡服务器每个常驻模型必须产出可量化贡献证据。
  - 7 天内没有正贡献证据的常驻模型进入降级审计，释放 GPU 给训练、walk-forward、challenger 或按需调用。

### 文件级实施重点

- `services/trading_params.py`
  - 新增 Profit-First v3 参数组：仓位档、晋级阈值、不开仓诊断阈值、亏损平仓归因阈值。
- `services/entry_opportunity_scoring.py`
  - 输出统一交易评分，不再只给分数和部分收益字段。
- `services/entry_profit_risk_sizing.py`
  - 把小仓 cap 改成分级仓位晋级；强机会允许 meaningful size。
- `services/entry_opportunity_gate.py`
  - 缺统一评分或缺退出计划时只能 shadow；禁止直接真实开仓。
- `services/trading_service.py`
  - 开仓前绑定退出计划；平仓后回写归因；每轮输出不开仓诊断。
- `services/strategy_learning.py`
  - 策略画像按真实净收益、回撤、快亏率、小赚大亏率自动升降级。
- `services/model_contribution_performance.py`
  - 模型贡献从“是否参与”升级为“对 OKX 真实净收益的边际贡献”。
- `services/profit_attribution.py`
  - 增加开仓质量、退出质量、错失机会、过度保守归因。
- `services/phase3_go_no_go.py`
  - 增加 Profit-First v3 go/no-go 条件：统一评分、退出计划、小仓比例、不开仓诊断、模型贡献榜。
- `web_dashboard/api/dashboard.py`
  - 增加 v3 主控诊断 API：小仓原因、不开仓原因、亏损平仓原因、策略/模型排行榜。
- `web_dashboard/static/js/dashboard.js`
  - 页面展示从“功能状态”升级为“赚钱闭环状态”。
- `scripts/inspect_online_strategy_health.py`
  - 增加 v3 只读体检摘要：开仓频率、仓位档分布、不开仓卡点、亏损平仓归因、模型贡献。

### 验收

- 最近窗口内每笔真实 paper 开仓都有统一交易评分和退出计划。
- 小仓比例、不开仓原因、亏损平仓原因可以按 24 小时和 7 天窗口查看。
- 强机会能自动晋级到 meaningful size；如果没有晋级，页面必须显示明确阻断原因。
- 策略画像和模型榜单能解释“谁赚钱、谁亏钱、谁被降权、谁被晋级”。
- paper/canary 的净收益、回撤、快亏快平、小赚大亏、错失机会指标进入 go/no-go。
- 未达到本阶段验收前，不得宣称三期已经实现“更赚钱、更智能、更高效”的最终目标。

---

## 六、文件级实施清单

## 6.1 第一批重点文件

- `scripts/deploy_local_ai_tools_service.py`
  - 核心引擎改造主文件
- `services/order_position_reconciliation.py`
  - 订单/成交/持仓对账主链路
- `scripts/repair_missing_closed_positions_from_orders.py`
  - 历史缺失 closed position 修复
- `scripts/repair_okx_native_full_close_fills.py`
  - 原生 close fill 对账和收益修复
- `services/exchange_close_fill_finder.py`
  - OKX close fill 发现逻辑
- `services/sync_service.py`
  - 同步与对账容错逻辑
- `services/shadow_training_quarantine.py`
  - 影子样本脏数据隔离
- `services/training_data_quality.py`
  - clean training view 治理
- `services/local_ai_tools_client.py`
  - 影子结果、模型版本、容灾回退元数据透传
- `services/ml_signal_service.py`
  - 基线模型、fallback、校准层保留
- `services/entry_direction_competition.py`
  - 证据加权重排
- `services/trading_service.py`
  - 训练触发、主流程接线、观测埋点
- `scripts/train_local_ai_tools_models.py`
  - 训练、验证、晋级
- `ai_brain/ensemble_coordinator.py`
  - 最终仲裁逻辑收缩
- `services/model_dynamic_routing.py`
  - 后续主控升级基础
- `services/entry_high_risk_review.py`
  - 高风险复核独立化
- `web_dashboard/api/data_collection.py`
  - 训练触发与数据治理入口

## 6.2 改造优先级

- `P0`
  - 新服务器重置
  - 旧服务器迁移白名单
  - OKX 对账链路根治
  - 历史脏数据审计与 clean training view
  - `timeseries/deep/predict`
  - `profit/predict`
  - 影子记录和评估
- `P1`
  - `exit/advise`
  - 专家权重重排
  - 高风险复核独立化
- `P2`
  - 自动训练、walk-forward、challenger 晋级
  - 主控升级

---

## 七、二期主控优化改版计划

## 7.1 二期总目标

- 把主控从“多个模块都参与”升级成“有统一决策层级和动态调度能力”的智能总控。
- 让系统学会：
  - 按市场状态切换策略
  - 按自身近期表现收紧或放宽风险
  - 按 challenger 比赛结果动态降权或晋级模型

---

## 7.2 二期模块一：主控分层

### 目标

- 把主流程拆成四级：
  - `机会发现`
  - `收益评估`
  - `风险闸门`
  - `执行批准`

### 结果

- 任何单子必须逐级晋级，不能被单一专家直接放行。
- 每一级都能解释“为什么过、为什么不过”。

---

## 7.3 二期模块二：动态路由 live 化

### 目标

- 把当前 shadow-only 的动态路由升级成可控 live routing。

### 原则

- 初期只允许对非核心专家做 live route mutation。
- `risk_expert` 永远不可被动态绕过。
- 缺 baseline、缺 feature coverage、缺 shadow 胜率时，不得 live 化。

### 结果

- 高质量候选走更全专家组合
- 低质量候选减少非核心专家计算成本
- 高风险场景自动强制恢复安全专家

---

## 7.4 二期模块三：市场状态驱动主控

### 目标

- 主控不再固定规则处理全部行情。

### 市场状态

- 趋势行情
- 震荡行情
- 新闻驱动行情
- 流动性恶化
- 快亏保护模式

### 结果

- 趋势时更敢做
- 震荡时更谨慎
- 新闻冲击时更依赖事件和风控
- 流动性恶化时优先保命

---

## 7.5 二期模块四：资金曲线反馈主控

### 目标

- 让主控不仅看市场，也看系统自己最近是否在犯错。

### 触发因素

- 连续亏损
- 快亏快平增加
- 利润回吐增加
- 同向重复误判
- 高风险复核拦截率升高

### 响应动作

- 降仓
- 提高收益门槛
- 收紧同向再开
- 强制进入高风险复核
- 提高 risk gate 敏感度

---

## 7.6 二期模块五：模型竞争闭环

### 目标

- 把训练侧结果真正反馈到主控，而不是只做报告。

### 规则

- 某 challenger 在特定场景长期优于 live 模型，允许晋级。
- 某 live 模型持续漂移、降质、误判，自动降权或退回 shadow。
- 晋级和降级都必须有统计证据，不得主观拍脑袋。

### 结果

- 系统持续进化
- 不需要每次都靠手工换模型

---

## 7.7 二期模块六：主控异常治理

### 强制拦截条件

- 证据冲突过大
- 收益预测不达标
- 风险惩罚过高
- 数据覆盖不全
- 模型超时或异常降级
- 当前处于保护模式

### 结果

- 主控宁可少做，不得硬猜
- 异常市场里优先保住资金曲线

---

## 7.8 二期模块七：主控可观测性

### 目标

- Dashboard 必须能回答：
  - 本轮卡在哪一级
  - 是谁放行的
  - 是谁被降权了
  - 哪个 gate 拦了
  - 为什么不给开仓
  - 为什么高风险复核没通过

### 结果

- 改版后能查问题
- 出现副作用时能快速定位

---

## 八、验收指标

### 8.1 一期验收

- 开仓质量提升
- 利润回吐下降
- 快亏快平下降
- 低质量单减少
- 主链路延迟不明显上升
- 新旧模型对比数据完整

### 8.2 二期验收

- 主控能按市场状态切换策略
- 主控能按资金曲线自我保护
- challenger 能规范晋级或退役
- Dashboard 能完整解释主控行为

### 8.3 节点级验证规则

#### `timeseries/deep/predict`

- 必须同时记录 live / shadow / challenger 输出。
- 必须按币种、方向、周期、市场状态统计预测误差。
- 必须证明新模型超时不会阻塞旧路径。
- 未优于旧基线前，只允许 shadow 或 canary。

#### `profit/predict`

- 必须输出手续费后 `expected_return_pct`、`loss_probability`、`profit_quality_score`。
- 必须和旧本地 ML、旧启发式做同窗口对比。
- 必须证明低质量单过滤能力提升，且不是单纯靠少开仓达成。
- 未通过收益质量对比前，不得提高 live 权重。

#### `exit/advise`

- 必须按 `hold / trail_profit / protect_profit / reduce_or_close` 记录动作。
- 必须统计利润回吐、亏损仓持有时长、过早止盈副作用。
- RL 或 challenger 平仓模型只能先进入 shadow。
- 任何 live 平仓行为变更必须有回退开关。

#### `LLM experts`

- 必须记录每个专家的输入、输出、耗时、是否参与最终决策。
- 必须证明专家输出只是证据，不是直接下单指令。
- 必须统计 JSON 失败、超时、fallback 和被主控忽略次数。

#### `decision_maker`

- 必须只在冲突、高价值或需要仲裁的候选中调用。
- 必须记录仲裁前后证据变化和最终影响。
- 必须证明普通交易对扫描不依赖 `decision_maker` 全量参与。
- 若延迟或错误率升高，必须自动退回专家池 + 结构化模型路径。

#### `high_risk_review`

- 必须只在高风险候选中触发。
- 必须具备独立超时、熔断、日志和失败原因。
- 必须统计拦截率、误拦率、放行后收益和放行后最大回撤。
- 高风险复核失败时默认不放行，除非已有明确人工/策略白名单。

#### `主控调度`

- 必须记录每一轮卡在哪一级：
  - `opportunity_discovery`
  - `profit_evaluation`
  - `risk_gate`
  - `execution_approval`
- 必须记录当前使用的市场状态、资金曲线状态、模型路由和风控模式。
- 必须证明任一单子不能绕过主控层和 risk gate 直接下单。
- 主控 live 化前必须完成 shadow 回放和 canary 验证。

#### `模型晋级/退役`

- 必须有同窗口、同样本、同交易成本口径对比。
- 必须按币种、方向、市场状态分桶验证。
- 必须保留旧模型回退路径。
- 任一 challenger 晋级 live 前，必须可追溯训练数据版本、评估报告和上线时间。

---

## 九、回退策略

- `timeseries v2` 回退到旧时序逻辑
- `profit v2` 回退到旧本地 ML + 旧启发式
- `exit v2` 回退到旧 profile/rule 逻辑
- 专家权重回退到当前固定配置
- 动态路由回退到 audit-only / shadow-only
- 高风险复核回退到当前现网配置

任何阶段只要出现以下情况之一，必须立即停止继续推进并回退：

- 收益质量显著恶化
- 高风险误开明显增加
- 主链路延迟显著上升
- 回退后仍无法恢复现网稳定

---

## 十、建议排期

### 第零波：`1-2` 周

- 阶段 0：新服务器停旧占用、释放 GPU、`/data/BB` 隔离
- 阶段 0：旧服务器迁移白名单和底账快照
- 阶段 H：OKX 对账链路根治启动
- 阶段 I：历史脏数据审计启动

### 第一波：`4-6` 周

- 阶段 A：基线冻结
- 阶段 B：`local_ai_tools v2 shadow`
- 阶段 C：收益预测器升级
- 阶段 D：平仓引擎升级
- 阶段 E：专家权重重排

### 第二波：`3-5` 周

- 阶段 F：高风险复核独立化
- 阶段 G：训练与评估流水线升级
- 阶段 H：OKX 对账链路收口
- 阶段 I：训练重建和 artifact 重置
- 二期主控分层
- 二期动态路由 live 化
- 二期市场状态驱动
- 二期资金曲线反馈
- 二期模型竞争闭环
- 二期主控可观测性

### 第三波：`3-4` 周

- 阶段 K：Profit-First 主控 v3 收口
- 统一交易评分和退出计划接入主链路
- 小仓、不开仓、亏损平仓专项诊断上线
- 分级仓位晋级和强机会 meaningful size 验证
- 策略画像、模型、币种、方向、市场状态排行榜驱动自动升降级
- 8 卡模型服务器按真实贡献做资源再分配审计
- 以 paper/canary 净收益、回撤、快亏快平、小赚大亏、错失机会减少作为 v3 验收

> 阶段 K 依赖阶段 H / I 的 OKX 真账和 clean training view。若 OKX 账户、订单、持仓、历史仓位、PnL 仍未对齐，K 只能进入 shadow/audit，不能宣称已经实现最终赚钱闭环。

---

## 十一、最终结论

这次改版的核心不是“再装几个大模型”，而是把系统重构为：

- 干净的新服务器负责跑新方案
- 旧服务器只做白名单迁移后退役
- 干净账本负责真实收益口径
- 干净训练视图负责模型学习
- 专业模型负责算
- 主控负责放行
- LLM 负责解释和复核
- 训练平台负责持续进化

如果执行到位，最终得到的不会是一台“跑大模型的服务器”，而是一台真正服务量化收益闭环的“交易模型工厂”。

---

## 十二、执行任务表

> 本章是本项目唯一有效的执行任务表。
> 以后不再单独维护第二份“任务表文档”，避免总纲、排期、文件清单、专项治理分叉。

### 12.1 执行原则

- 所有影响真实交易行为的改动，必须走 `shadow -> canary -> live`。
- 优先处理最影响赚钱能力和最容易继续埋雷的前置项：
  - 新服务器停旧占用、释放 GPU、`/data/BB` 隔离
  - 旧服务器迁移白名单
  - OKX 对账根治
  - 历史脏数据治理
  - `timeseries`
  - `profit`
  - `exit`
- 不允许一开始同时大改主控、前端、所有模型和所有部署。
- 每一批必须同时具备：
  - 基线
  - 测试
  - 回退点
  - 观测口径

### 12.2 优先级

#### P0

- 新贵模型服务器停旧占用、释放 GPU、三期数据根目录隔离
- 旧服务器迁移白名单导出
- OKX 对账链路根治启动
- 历史脏数据审计与 `clean training view`
- 基线冻结
- `timeseries/deep/predict` 影子升级
- `profit/predict` 收益预测器升级
- `local_ai_tools_client` 元数据透传
- 训练与影子评估入口打通

#### P1

- `exit/advise` 升级
- 专家证据权重重排
- 高风险复核独立化
- 训练流水线分层
- 历史训练产物退役与重建

#### P2

- 主控四级分层
- 动态路由 live 化
- 市场状态驱动
- 资金曲线反馈
- challenger 晋级/降级闭环
- 主控可观测性

### 12.3 文件级任务

#### `scripts/deploy_local_ai_tools_service.py`

- 重构 `/timeseries/deep/predict`
- 重构 `/profit/predict`
- 升级 `/exit/advise`
- 升级 `/sentiment/deep/analyze`
- 统一输出：
  - `primary_model`
  - `challenger_model`
  - `model_version`
  - `route_mode`
  - `fallback_reason`
  - `feature_coverage`

#### `services/local_ai_tools_client.py`

- 透传新元数据
- 兼容新旧结果结构
- 保持超时、熔断、fallback 稳定

#### `services/ml_signal_service.py`

- 保留为 baseline / fallback / calibrator
- 增加版本和状态标记
- 不再承担唯一主收益判断

#### `services/order_position_reconciliation.py`

- 强化 `exchange_order_id / ordId / fillId` 主键链路
- 降低 symbol alias 对主链路的影响
- 明确部分平仓 / 全平聚合逻辑

#### `scripts/repair_missing_closed_positions_from_orders.py`

- 增加白名单 apply 限制
- 输出可修 / 不可修 / 需人工核对分类

#### `scripts/repair_okx_native_full_close_fills.py`

- 重算 `closed_at / close_price / realized_pnl / fee`
- 标记历史估算修复样本
- 输出训练污染风险标签

#### `services/exchange_close_fill_finder.py`

- 提升 close fill 查询稳定性
- 输出超时、缺失、弱匹配原因

#### `services/sync_service.py`

- 减少 lookup unavailable 场景下的误补账/误闭仓
- 明确延迟确认和待补记状态

#### `services/shadow_training_quarantine.py`

- 把无法确认真实收益的样本自动隔离
- 增加污染原因标签

#### `services/training_data_quality.py`

- 建立 `audit_only / repairable / trainable / quarantined` 视图
- 所有训练只消费 `clean training view`

#### `scripts/train_local_ai_tools_models.py`

- 拆分 `formal / shadow / walk_forward`
- 输出训练版本、质量报告、晋级建议

#### `services/trading_service.py`

- 接入新 `local_ai_tools` 结果
- 记录新旧模型证据
- 接入训练版本状态
- 二期接入主控分层结果

#### `services/entry_direction_competition.py`

- 提高结构化证据权重
- 降低空泛 LLM 文本直接影响

#### `ai_brain/ensemble_coordinator.py`

- 收缩 `decision_maker` 职责
- 二期增加四级主控输出接口

#### `services/entry_high_risk_review.py`

- 抽离高风险场景定义
- 强化独立超时、熔断、原因输出

#### `services/model_dynamic_routing.py`

- 从 shadow advisory 升级到可控 live route
- `risk_expert` 永远强制参与

#### `web_dashboard/api/data_collection.py`

- 增加 shadow / walk-forward 触发
- 返回训练状态、版本和质量结果

#### `web_dashboard/api/system_health.py`

- 展示主模型/challenger/fallback 健康状态
- 增加高风险复核健康卡

#### `web_dashboard/api/system_audit.py`

- 增加主控四级 gate 审计
- 增加 challenger 晋级/降级审计

### 12.4 周排期

#### 第 0 周

- 新贵模型服务器停旧占用、释放 GPU、`/data/BB` 工作区就绪
- 旧服务器迁移白名单定稿
- 底账和训练产物快照冻结

#### 第 1 周

- OKX 对账根治审计启动
- 历史 closed position / 收益修复审计
- 历史脏数据分层规则定稿
- 基线和对比字段冻结

#### 第 2 周

- `timeseries/deep/predict` 影子升级

#### 第 3 周

- `profit/predict` 升级

#### 第 4 周

- 训练与影子评估打通
- 旧 artifact 退役路径确认

#### 第 5 周

- `exit/advise` 升级

#### 第 6 周

- 专家权重重排
- 高风险复核独立化

#### 第 7 周

- 一期观测收口
- 系统健康与审计展示补齐

#### 第 8-12 周

- 二期主控分层
- 动态路由 live 化
- 市场状态驱动
- 资金曲线反馈
- challenger 闭环
- 主控可观测性

### 12.5 最佳执行顺序

1. 新服务器停旧占用、释放 GPU、`/data/BB` 隔离
2. 旧服务器迁移白名单
3. OKX 对账链路根治
4. 历史脏数据审计与 `clean training view`
5. 基线冻结
6. `timeseries/deep/predict`
7. `profit/predict`
8. 训练与影子评估
9. `exit/advise`
10. 专家权重
11. 高风险复核
12. 二期主控

### 12.6 每批次强制验收项

- 有基线对比
- 有测试
- 有影子数据
- 有回退点
- 有健康检查
- 有上线后观测项

任何一批如果不能回答下面问题，都不得进入下一批：

- 改了什么
- 不改什么
- 会影响真实交易吗
- 如何回退
- 怎么证明比旧系统更好

---

## 13. Phase 3 Implementation Ledger

### 13.1 Completed In Current Batch

- High-risk review audit is now a first-class system audit card:
  - `services/high_risk_review_audit.py` summarizes recent `raw_llm_response.high_risk_review`.
  - `web_dashboard/api/system_audit.py` exposes `high_risk_review_audit`.
  - The card is read-only and cannot set live entry mutation, force open, or bypass risk controls.
  - Warning-level blocked high-risk reviews are classified as `observing`; only executed entries without required approval remain hard unresolved/critical.
- Phase 3 model governance is now surfaced in audit and dashboard:
  - local AI training status carries `training_mode`, `model_stage`, `evaluation_policy`, `promotion_flow`, `live_mutation`, and `promotion_recommendation`.
  - dynamic routing exposes `promotion_gate` with canary/live readiness counts.
  - live route mutation remains disabled unless explicitly enabled outside audit.
- OKX reconciliation now reports root-cause classes instead of only a generic mismatch count:
  - `root_cause_summary.status` distinguishes `clean`, `dirty`, and `incomplete`.
  - Counts are split into `repairable_count`, `manual_review_count`, `skipped_candidate_count`, and `unscanned_candidate_count`.
  - `training_data_policy` records that dirty or unclassified trade facts must be quarantined, not deleted.
  - Dashboard OKX audit details now render root causes and training rebuild requirements.
- Training data governance keeps the current red line:
  - raw historical rows are preserved for audit.
  - dirty, manually repaired, or untrusted trade facts are excluded from training.
  - `okx_reconciliation=ok` must not be interpreted as "all historical dirty data is clean"; it only proves the current audited window/check passed.

### 13.2 Verification Completed

- Focused high-risk/system audit regression: `47 passed`.
- OKX/system audit/dashboard focused regression: `101 passed`.
- Training data governance focused regression: `34 passed`.
- Phase 3 touched regression suite: `224 passed`.
- Python compile check passed for `web_dashboard/api/system_audit.py`.

### 13.3 Current Safety Boundary

- No live trading behavior was enabled by this batch.
- No automatic historical data deletion was added.
- No OKX repair apply path is triggered from system audit or dashboard status views.
- High-risk review audit and OKX root-cause reporting are observability/governance layers unless a later batch explicitly enables a guarded apply flow.

### 13.4 Strategy Scheduler Observability Completed

- Added read-only scheduler diagnostics to `services/strategy_signal_root_cause_audit.py`:
  - summarizes `raw_llm_response.strategy_mode` and `strategy_learning_context` across all recent decisions, including hold/no-trade decisions.
  - reports strategy/posture/risk-mode/profile/cache-status distributions.
  - reports active learning guards: context timeout, entry pause, execution guard, release pressure, health guard, drawdown clamp, and market-regime soft bias.
  - reports dynamic position capacity constraints, entry-limit pressure, and capacity reason-code distributions.
- The strategy signal root-cause card now explains who shaped the scheduler posture:
  - `EntryStrategyModeContextPolicy`
  - `StrategyLearningService.apply_to_strategy_context`
  - `DynamicPositionCapacityPolicy`
- Added scheduler root-cause codes:
  - `strategy_learning_context_timeout`
  - `strategy_learning_entry_pause_active`
  - `dynamic_capacity_constrained`
  - `drawdown_clamp_active`
  - `market_regime_soft_bias_active`
- Dashboard system audit details now render scheduler distributions, flags, capacity reason codes, top scheduler reasons, root causes, and latest scheduler samples.
- Safety boundary:
  - this batch is diagnosis only.
  - it does not change live entries, thresholds, sizing, leverage, model readiness, OKX repair, or historical data.
  - scheduler samples and report details are forced to `can_force_open=false`, `can_override_thresholds=false`, and `can_bypass_risk_controls=false`.

Verification:

- Focused scheduler/system/dashboard regression: `107 passed`.
- Expanded Phase 3 governance regression: `138 passed`.

### 13.5 Clean Training View Cursor Completed

- Local AI tools trade-sample cursor now uses the clean training view:
  - `_completed_trade_sample_count()` counts only samples that survive `annotate_training_payload`.
  - untrusted OKX trade facts, manual/test trades, repaired historical facts, and other excluded samples no longer inflate the completed-trade cursor.
  - auto-training, dashboard-triggered training, and CLI training now report `trade_sample_cursor_policy=clean_training_view_only`.
- Raw history is still preserved for audit:
  - responses include `raw_trade_sample_count`, `trainable_trade_sample_count`, and `quarantined_trade_sample_count`.
  - no historical row deletion was added.
  - dirty facts are excluded from model training and promotion decisions, not erased.
- `LocalAIToolsClient.train()` now forwards the clean-view trade cursor policy and raw/trainable/quarantined trade counts to the model server.

Verification:

- Focused clean training view regression: `68 passed`.
- Expanded Phase 3 training/governance/system regression: `183 passed`.

Safety boundary:

- No live trading behavior changed.
- No OKX repair apply path changed.
- This only prevents dirty historical trade facts from making training and promotion readiness look better than the clean data actually supports.

### 13.6 OKX Realtime Position Mismatch Root Causes Completed

- `position_price_integrity` now distinguishes three live mismatch classes:
  - `mark_price_mismatch` / `okx_upl_mismatch` / `mark_price_recomputed_pnl_mismatch` for matched local and OKX positions whose price or unrealized PnL differs.
  - `local_open_position_missing_on_okx` for local open positions that do not exist in the OKX position snapshot.
  - `okx_open_position_missing_locally` for OKX open positions that do not exist in local open positions.
- The audit details now include:
  - `root_cause_summary`
  - `mismatch_count`
  - `split_count`
  - `local_only_count`
  - `exchange_only_count`
  - `root_cause_counts`
  - sampled `splits`, `local_only_positions`, and `exchange_only_positions`
  - OKX mark/entry/quantity/contracts/contract-size/raw-symbol fields for diagnosis.
- Safety boundary:
  - this remains read-only.
  - `live_repair_mutation=false`.
  - no OKX sync, repair apply, order mutation, close mutation, or DB mutation is triggered by the audit.
  - untrusted position facts remain excluded from training until OKX/local match is restored.

Verification:

- Focused OKX realtime position audit regression: `55 passed`.
- Expanded OKX/system/dashboard/training/scheduler regression: `143 passed`.

### 13.7 Historical OKX Repair Apply Guard Completed

- `scripts/repair_okx_history_position_reconciliation.py` now follows the Phase 3 historical-data safety rule:
  - default execution remains dry-run.
  - `--apply` now requires `--position-id` or `--exchange-order-id`.
  - unfiltered bulk apply is rejected before any DB write.
  - apply prints `apply_policy=apply_requires_position_id_or_exchange_order_id`.
- Before applying, the script writes a JSONL backup of the affected position/order rows under:
  - `/data/bb/app/data/codex_backups/okx-history-position-reconciliation`
- `collect_repairs()` now accepts the same precise filters used by the CLI so dry-run and apply inspect the same bounded candidate set.

Verification:

- Focused repair-script safety regression: `11 passed`.
- Expanded OKX repair/system/training regression: `89 passed`.

Safety boundary:

- No repair is executed by dashboard/system audit.
- No unfiltered historical DB mutation is allowed by this script.
- Historical repair output remains excluded from training through clean training view governance.

### 13.8 OKX Position Mismatch Dashboard Detail Completed

- Dashboard system audit details now render `position_price_integrity` root causes:
  - position mismatch root-cause counts.
  - price/PnL split samples.
  - local-only open positions.
  - OKX-only open positions.
  - OKX raw symbol, contracts, and contract-size fields where available.
- This closes the visibility gap where the backend could detect mismatches but the UI still showed only generic detail fields.

Verification:

- Dashboard/system audit contract regression: `105 passed`.

Safety boundary:

- UI-only visibility change.
- No sync, repair, order, close, DB write, strategy threshold, sizing, or leverage mutation.

### 13.9 OKX Authoritative Current Sync Runtime Visibility Completed

- Current open-position synchronization now treats OKX private API position facts as the operational authority:
  - `services/trading_service.py` starts an automatic `okx_authoritative_sync` loop with a bounded timeout.
  - the loop calls the existing `OkxSyncService.reconcile_positions()` path, so current local open positions/prices/protection state are refreshed from OKX instead of drifting silently.
  - this targets live/current mismatch symptoms such as local SPK/HOME state disagreeing with OKX, while historical row repair remains separately controlled.
- Runtime heartbeat now records `okx_authoritative_sync`:
  - `status`: `pending` / `ok` / `warning` / `stale`.
  - last start/success/failure timestamps.
  - last duration, result count, success/failure counts, and last error.
  - stale threshold is based on the configured sync cadence, with a hard minimum visibility window.
- Dashboard and system health now surface the same runtime state:
  - `web_dashboard/api/dashboard.py` passes heartbeat sync status through split-process stats.
  - `web_dashboard/api/system_health.py` downgrades the trading-service self-check to warning when automatic OKX sync is failed or stale.
  - `web_dashboard/static/index.html` and `web_dashboard/static/js/dashboard.js` show an `OKX auto sync` line in the main status card.
- Historical repair policy remains unchanged:
  - no dashboard/system-audit route can apply historical fixes.
  - HOME/USDT-like dirty history must go through dry-run, precise allowlist, backup, then targeted apply.
  - training uses only OKX-backed clean facts; dirty or repaired history stays quarantined unless explicitly trusted by the clean training view.

Verification:

- Dashboard/system audit contract regression: `106 passed`.
- OKX/trading authoritative sync regression: `231 passed`.
- Python compile check passed for `services/trading_service.py`, `web_dashboard/api/dashboard.py`, and `web_dashboard/api/system_health.py`.

Safety boundary:

- No live entry threshold, sizing, leverage, model routing, or strategy scheduler behavior changed.
- Automatic sync only uses the existing current-position reconciliation boundary.
- Historical deletion, unfiltered repair apply, and training inclusion of dirty facts remain forbidden.

### 13.10 OKX Historical Position-Link Apply Guard Completed

- `scripts/repair_missing_position_links_from_okx_fills.py` now follows the same Phase 3 historical repair boundary:
  - default execution remains dry-run.
  - `--apply` requires at least one explicit `--position-id`.
  - unfiltered apply is rejected before OKX calls or DB writes.
  - output includes `apply_policy=apply_requires_position_id`.
- This script may backfill missing `entry_exchange_order_id`, `close_exchange_order_id`, `okx_inst_id`, or missing local order rows only after a position-scoped dry-run review.
- This closes a bulk-mutation gap in the HOME/SPK-style historical repair workflow: OKX-backed evidence can be used to repair links, but not through an accidental broad apply.
- Apply now writes a `TradeReflection` marker with `source=okx_position_link_repair`:
  - the clean training view recognizes this as historical repair provenance.
  - repaired positions remain quarantined from training by default even when OKX order links become complete.
  - this prevents a repaired HOME/SPK-like historical row from re-entering local AI tools training just because the link fields were backfilled.

Verification:

- Historical position-link repair guard regression: `9 passed`.
- Position-link repair + clean training quarantine regression: `24 passed`.

Safety boundary:

- No dashboard/system-audit path can call this apply flow.
- No unfiltered historical position-link or missing-order-row mutation is allowed.
- Repaired historical facts remain excluded from training unless the clean training view later classifies them as trusted OKX-backed facts.

### 13.11 OKX Exit Position Match Diagnostics Completed

- OKX close execution now requires the exchange position snapshot to match both:
  - the decision symbol / OKX `instId`.
  - the requested close direction (`long` for `CLOSE_LONG`, `short` for `CLOSE_SHORT`).
- This prevents account-wide or fallback OKX position snapshots from using another symbol's same-side position as the close target.
- `executor/okx_executor.py` now attaches `okx_exit_position_mismatch` diagnostics when a close is rejected as `no_position`:
  - source path: `pre_submit_position_lookup`, `exchange_no_position_rejection`, or `native_reduce_no_position_rejection`.
  - decision symbol, normalized symbol, expected OKX `instId`, OKX request symbol, target position side, and close order side.
  - candidate OKX positions with raw symbol, normalized symbol, side, contracts, quantity, contract size, mark price, entry price, UPL, and reason (`symbol_mismatch`, `side_mismatch`, `zero_contracts`, or `matches`).
- `services/execution_service.py` now also writes `okx_exit_position_mismatch_summary` into the persisted `execution_result` snapshot:
  - dashboards, execution traces, and audits can read the close-failure root cause without digging through large raw OKX payloads.
  - the detailed raw diagnostic is still retained under `execution_result.raw_response.okx_exit_position_mismatch`.
- `core/symbols.py` no longer treats a generic order `id` such as `exit-1` as an OKX instrument id:
  - market payload `id` remains handled by `symbol_from_okx_market`.
  - order payload symbol extraction now prefers `info.instId`, `instId`, explicit OKX fields, then `symbol`.
  - this reduces backend trade-pair drift where an order id could be rendered as a fake symbol like `EXIT/1`.

Verification:

- OKX executor and position-market fallback regression: `30 passed`.
- Core symbol, executor, order-log, and position-persistence regression: `48 passed`.
- Execution-service mismatch-summary regression: `2 passed`.
- Expanded local OKX/execution/training regression before deploy: `102 passed`.
- Online deployment:
  - `python scripts/sync_to_online_server.py --split-services` uploaded 34 changed files to `/data/bb/app`.
  - `bb-model-tunnels.service`, `bb-paper-trading.service`, and `bb-dashboard.service` restarted successfully and are active.
  - Remote `py_compile` passed for `core/symbols.py`, `executor/okx_executor.py`, `services/execution_service.py`, `services/trading_service.py`, `services/okx_authoritative_sync.py`, `web_dashboard/api/dashboard.py`, and `web_dashboard/api/system_health.py`.
  - Dashboard port `8002` returned `302`; unauthenticated `/api/status` returned `401` as expected with auth enabled.
  - Recent `bb-paper-trading.service` journal check after restart showed no `Traceback`, `ERROR`, or `Exception`.

Safety boundary:

- No strategy threshold, sizing, leverage, model routing, or scheduler behavior changed.
- The only live execution behavior change is safer close matching: symbol and side must both match before submitting a close.
- If OKX does not return a usable symbol/instId for a close candidate, the system rejects with diagnostics instead of risking a wrong-symbol close.

### 13.12 OKX Current Sync Result Classification Completed

- Current-position OKX authoritative sync results are now machine-readable, not just free-text notes:
  - `snapshot_update`: local open-position quantity/price/protection was refreshed from OKX.
  - `reopened_local_position`: OKX still has the position, so a mistakenly closed local row was reopened.
  - `created_missing_local_position`: OKX has an open position and a matching exchange-backed entry order, so the missing local position was restored.
  - `closed_from_okx_close_fill`: OKX no longer has the position and a real close fill was found, so the local position was closed from OKX evidence.
  - `quantity_reduction_closed_slice`: OKX reports a smaller still-open quantity and a matching close fill, so the reduced slice was recorded as closed history.
  - `close_fill_lookup_unavailable`, `active_order_snapshot_unavailable`, and `missing_exchange_position_without_close_fill` stay open and set `requires_attention=true` where needed instead of inventing a local close.
  - `active_exchange_order_present` records that OKX has an in-flight entry/exit order and local close is intentionally deferred.
- Runtime heartbeat now includes richer `okx_authoritative_sync` diagnostics:
  - `last_result_kinds`.
  - `last_requires_attention_count`.
  - compact `last_samples` with kind, symbol, side, exchange order id, attention flag, and bounded note.
- Execution/Agent open-position context now performs a short OKX authoritative refresh before returning positions:
  - `open_positions_context_for_execution()` calls `reconcile_positions("execution open positions context refresh")` with bounded timeout and non-global timeout-error recording.
  - this reduces stale local-position decisions before close/strategy dispatch and directly targets symptoms like SPK/HOME local-vs-OKX drift.
- SPK root-cause follow-up found the live mismatch was caused by OKX net-mode position parsing:
  - OKX returned `posSide=net` with signed `pos=-200` for `SPK-USDT-SWAP`.
  - the previous parser only accepted explicit `long` / `short`, so it dropped the real OKX short position and treated SPK as local-only.
  - `parse_exchange_position_snapshot()` now infers `short` from negative net `pos` and `long` from positive net `pos`.
  - `services/sync_service.py` now uses the same parsed exchange-position key for reconciliation and open-position context merging, preventing net-mode positions from being filtered out or duplicated in Agent context.

Verification:

- Trading-service boundary regression: `146 passed`.
- Dashboard/system audit contract regression: `106 passed`.
- Python compile check passed for `services/sync_service.py` and `services/trading_service.py`.
- Exchange net-position parser + OKX/sync/executor regression: `182 passed`.
- Expanded OKX/sync/dashboard regression after net-mode key fix: `290 passed`.
- Online deployment and verification:
  - `python scripts/sync_to_online_server.py --split-services` deployed `services/exchange_position_state.py` and then `services/sync_service.py`.
  - `bb-model-tunnels.service`, `bb-paper-trading.service`, and `bb-dashboard.service` were active; dashboard returned `302`.
  - remote `py_compile` passed for the changed sync/parser/trading files.
  - recent `bb-paper-trading.service` journal check showed no `Traceback`, `ERROR`, or `Exception`.
  - online runtime heartbeat after the fix showed `okx_authoritative_sync.status=ok`, `last_result_count=0`, `last_requires_attention_count=0`, and empty `last_samples`, confirming the SPK local-vs-OKX mismatch was cleared by the parser/key fix.

Safety boundary:

- No strategy threshold, sizing, leverage, model routing, GPU/model allocation, or scheduler scoring behavior changed.
- No historical deletion or broad repair apply was added.
- If OKX state is unavailable or no real close fill is found, local positions stay open and the sync result is flagged for attention instead of fabricating a close.

### 13.13 OKX Net-Mode Position Audit Evidence Completed

- `parse_exchange_position_snapshot()` now preserves OKX net-mode evidence in the parsed snapshot:
  - raw OKX `posSide`.
  - raw signed OKX `pos`.
  - signed position size used for side inference.
  - side inference source (`okx_net_signed_pos`, `ccxt_side`, `okx_pos_side`, or `unresolved`).
- Net-mode side inference now prioritizes the signed OKX position value:
  - `posSide=net`, `pos<0` => `short`.
  - `posSide=net`, `pos>0` => `long`.
  - this remains correct even when CCXT exposes `contracts` as a positive absolute value.
- `position_price_integrity` system audit now exposes:
  - `okx_pos_side_counts`.
  - `okx_side_inference_counts`.
  - per-row `okx_pos_side`, `okx_raw_pos`, `okx_signed_position_size`, `okx_side_inference`, and raw CCXT side evidence.
- Dashboard audit details now render:
  - OKX position mode counts.
  - OKX side inference counts.
  - raw `posSide`, raw signed `pos`, and inference source on price/PnL split samples and OKX-only open positions.
- This closes the SPK/HOME-style observability gap where the backend could parse or reject net-mode positions but the dashboard could not show the raw OKX evidence behind the inferred side.

Verification:

- Position parser, system audit, and dashboard contract regression: `115 passed`.
- OKX authoritative sync, trading boundary, executor safety, system audit, dashboard contract regression: `290 passed`.
- Python compile check passed for `services/exchange_position_state.py` and `web_dashboard/api/system_audit.py`.
- `git diff --check` passed.

Safety boundary:

- Read-only audit/visibility change plus safer net-mode evidence preservation.
- No order submission, close execution, repair apply, historical deletion, strategy threshold, sizing, leverage, model routing, GPU/model allocation, or scheduler scoring behavior changed.

### 13.14 OKX Current-Mode Price/UPL Refresh Completed

- `OkxSyncService.refresh_position_prices()` now uses OKX position snapshots by execution mode:
  - current active mode OKX executor is queried first.
  - paper OKX executor remains a fallback/source for paper positions.
  - each parsed OKX snapshot is keyed by `(execution_mode, symbol, side)`.
  - local open-position price/UPL updates only consume a snapshot with the same `execution_mode`.
- This directly targets recurring dashboard-vs-OKX drift:
  - live positions no longer depend on paper-only OKX snapshots during price refresh.
  - live OKX marks/UPL cannot overwrite paper positions.
  - paper OKX marks/UPL cannot overwrite live positions.
  - feature-vector prices are still used only when the matching OKX snapshot is unavailable.
- Net-mode support from `13.12` and `13.13` is preserved for this price-refresh path because snapshots still go through `parse_exchange_position_snapshot()`.

Verification:

- Focused position price refresh regression: `4 passed`.
- OKX authoritative sync + trading boundary regression: `150 passed`.
- Expanded OKX/sync/executor/system-audit/dashboard regression: `291 passed`.
- Python compile check passed for `services/sync_service.py` and `tests/test_trading_service_boundaries.py`.
- `git diff --check` passed.

Safety boundary:

- No order submission, close execution, repair apply, historical deletion, strategy threshold, sizing, leverage, model routing, GPU/model allocation, or scheduler scoring behavior changed.
- This changes only the source selection for persisted open-position current price and unrealized PnL refresh.
- If a matching OKX snapshot is unavailable, the existing feature-vector/local fallback remains in place.

### 13.15 Historical Trade Fact Audit Completed

- Added a read-only historical closed-position fact audit:
  - file: `services/historical_trade_fact_audit.py`.
  - scans closed positions over the configured lookback window.
  - classifies facts as `trainable` only when OKX-backed trade links are complete and no repair provenance exists.
  - quarantines missing entry order link, missing close order link, manual/synthetic close markers, and historical repair provenance.
- The audit reports the operator-facing cleanup boundary:
  - raw history is preserved.
  - cleanup mode is `quarantine_not_delete`.
  - training policy is `clean_training_view_only`.
  - dashboard/system audit cannot delete history or apply repair.
  - repair still requires existing dry-run, allowlist, and backup-required scripts.
- `model_training` system audit details now include `historical_trade_fact_audit`, so the dashboard can answer:
  - how many historical closed trade facts are trainable.
  - how many are quarantined from training.
  - why facts are quarantined.
  - which symbols/reasons are most affected.
  - whether a row is only repairable, not directly deletable.

Verification:

- Historical trade fact audit + trust policy regression: `12 passed`.
- Training data governance/system audit/data collection regression: `89 passed`.
- Expanded OKX/sync/executor/system-audit/dashboard/historical-audit regression: `292 passed`.
- Python compile check passed for `services/historical_trade_fact_audit.py`, `web_dashboard/api/system_audit.py`, and related tests.
- `git diff --check` passed.

Safety boundary:

- Read-only audit only.
- No DB mutation, historical deletion, repair apply, order execution, close execution, strategy threshold, sizing, leverage, model routing, GPU/model allocation, or scheduler scoring behavior changed.
- Dirty historical rows are excluded from training by classification; they are not erased.

### 13.16 Phase 3 Server Resource-Release/Migration Gate Completed

- Added a read-only Phase 3 model-server resource-release and migration readiness gate:
  - file: `services/phase3_server_migration_audit.py`.
  - verifies the new expensive model server has Phase 3 resource-release evidence.
  - blocks Phase 3 go-live if legacy model services, old vLLM/WebUI/container processes, old 32B/122B/DeepSeek runtime processes, or legacy download/runtime processes are still occupying GPU, ports, or routing resources.
  - old model directories, caches, bundles, logs, and experiments may remain on disk as isolated historical data, but Phase 3 services must not reference them.
  - verifies Phase 3 new model/cache/training/runtime/log data is rooted under `/data/BB`.
  - verifies old-server migration is whitelist-only, not whole-disk copy.
  - allowed migration categories are limited to secure-setting references, clean training export manifests, approved Phase 3 deploy manifests, regenerated runtime secrets, and operator reset evidence.
- `web_dashboard/api/system_audit.py` now exposes the new `phase3_server_migration` card and `server_migration` topology node.
- Dashboard audit details now render:
  - go-live blocked state.
  - resource-release marker status.
  - whitelist migration manifest status.
  - blocker/warning rows.
  - forbidden legacy services/processes/resource usage.
  - approved whitelist migration policy.
- This closes the planning gap where "new server must release legacy resource usage, keep Phase 3 isolated under `/data/BB`, and keep all old server migration whitelist-only" existed in the document but was not machine-checkable.

Verification:

- Phase 3 server migration audit regression: `3 passed`.
- System audit API regression: `50 passed`.
- Dashboard UI contract regression: `57 passed`.
- Python compile check passed for `services/phase3_server_migration_audit.py` and `web_dashboard/api/system_audit.py`.
- Dashboard JavaScript syntax check passed for `web_dashboard/static/js/dashboard.js`.

Safety boundary:

- Read-only audit only.
- No remote deletion, no SSH mutation, no service restart, no DB mutation, no historical deletion, no repair apply, no model routing change, no GPU allocation change, and no live trading behavior change.
- If the gate cannot verify the server, it blocks Phase 3 model-server go-live instead of assuming readiness.

### 13.17 Retired: Phase 3 Paper Cold-Start Watermark

- This historical deletion path was removed on 2026-07-24 because it deleted paper orders, positions and decisions and then hid older OKX fills behind a marker.
- It is replaced by `scripts/reset_training_derived_state.py`, which removes only derived training state and preserves the exchange-backed fact ledger.
- Online cold-start state verified after marker reconstruction:
  - `bb-paper-trading.service=inactive`.
  - OKX paper gate: `open_position_count=0`, `open_order_count=0`.
  - core paper/training/cache tables: `orders=0`, `positions=0`, `ai_decisions=0`, `strategy_learning_events=0`, `trade_reflections=0`, `shadow_backtests=0`, `strategy_profile_snapshots=0`, `expert_memories=0`, `risk_events=0`, `model_performance_snapshots=0`, `market_klines=0`, `market_tickers=0`, `news_articles=0`, `social_posts=0`.
  - preserved tables: `dashboard_users=1`, `secure_settings=6`, `secure_setting_audit=6`, `virtual_accounts=1`.
  - virtual account reset: `ensemble_trader` balance and PnL reset to `4000 / 0`.
  - OKX authoritative sync: `status=ok`, `cold_start_watermark_applied=true`, `okx_position_count=0`, `okx_fill_order_count=0`, `issue_count=0`, `fetch_errors=[]`.

Verification:

- The old cold-start and watermark regressions were deleted with the implementation.
- New derived-state reset regression verifies that raw trade facts and audit events remain intact.

Safety boundary:

- The replacement requires `bb-paper-trading.service`, `bb-model-tunnels.service`, and `bb-dashboard.service` to be stopped before apply.
- No marker can suppress or hide historical OKX facts.

### 13.18 OKX Native Facts Layer Started

- Added a reusable read-only OKX-native facts layer:
  - file: `services/okx_native_facts.py`.
  - groups fills by OKX `ordId`.
  - preserves `instId`, `tradeId`, `posSide`, raw fill rows, contracts, average fill price, fee, PnL, and exchange timestamp.
  - derives app display symbols from OKX `instId`, not from CCXT aliases.
- `services/okx_authoritative_sync.py` now pulls recent fills through this native layer instead of maintaining its own separate `privateGetTradeFillsHistory` parser.
- `services/exchange_close_fill_finder.py` now uses the same native layer for OKX fills-history close evidence.
  - Later Phase 3 hardening removed the old CCXT `closed_orders` / `my_trades` close-evidence paths from this service.
  - A close fact is trusted only when backed by OKX native fills-history fields such as `instId`, `ordId`, `tradeId`, `side`, `fillSz`, `fillPx`, `fee`, `fillPnl`, and `ts`.
- This is the first implementation step toward the Phase H direction:
  - core truth source becomes OKX native fields.
  - CCXT can still provide request/signing/retry plumbing, but it is no longer the symbol/fill truth source for this path.
  - SPK/HOME-style alias drift is reduced because `SPK-USDT-SWAP` remains the authoritative instrument id even when a CCXT market alias differs.

Verification:

- OKX native facts + authoritative sync + close-fill finder regression: `13 passed`.
- Python compile check passed for `services/okx_native_facts.py`, `services/okx_authoritative_sync.py`, and `services/exchange_close_fill_finder.py`.

Safety boundary:

- Read-only fact extraction and reconciliation support only.
- No order submission, close execution, repair apply, historical deletion, strategy threshold, sizing, leverage, model routing, GPU/model allocation, or scheduler scoring behavior changed.
- Full removal of CCXT is not done in this step; this step narrows CCXT's role and creates the native facts foundation needed for safe replacement of the core trading chain.

### 13.19 OKX Native Current-State Strict Chain Completed

- `OKXExecutor.get_positions_strict()` now reads current positions through OKX native `privateGetAccountPositions`:
  - request scope is OKX `instId`, e.g. `SPK-USDT-SWAP`.
  - returned rows preserve raw `instId`, `posSide`, signed `pos`, `ctVal`, mark price, average price, leverage, notional, and PnL in `info`.
  - `posSide=net` is converted from signed `pos`: negative means `short`, positive means `long`.
  - the old CCXT `fetch_positions` symbol-specific lookup and account-wide fallback are no longer used for strict current-position truth.
- `OKXExecutor.get_open_orders_strict()` now reads current pending orders through OKX native `privateGetTradeOrdersPending`:
  - request scope is OKX `instId`.
  - returned rows preserve raw `ordId`, `clOrdId`, `instId`, `side`, `posSide`, `ordType`, `state`, `reduceOnly`, size, filled size, and timestamps in `info`.
  - the old CCXT `fetch_open_orders` path is no longer used for strict pending-order truth.
- Core pre-trade guards now consume the strict native pending-order path:
  - active entry-order detection.
  - active exit-order detection.
  - leverage retry open-order cleanup candidate detection.
- Native API failure is now fail-closed for core trading/reconciliation truth:
  - missing/unavailable native positions API raises.
  - missing/unavailable native pending-orders API raises.
  - the system must not silently convert native current-state failure into `[]` and then continue as if OKX has no positions or no orders.
- Non-strict wrapper methods may still return `[]` for dashboard/diagnostic compatibility, but those wrappers are not authoritative and must not be used as the source of truth for trading, closing, reconciliation, or training labels.
- This step is specifically enabled by the paper cold-start reset:
  - historical compatibility is no longer allowed to justify CCXT current-state fallback in core execution.
  - from the reset marker forward, new trading facts must be clean OKX-native facts.

Verification:

- OKX native facts, executor native current-state, OKX authoritative sync, close-fill finder, and executor safety regression: `49 passed`.
- Python compile check passed for `services/okx_native_facts.py`, `executor/okx_executor.py`, `services/okx_authoritative_sync.py`, and `services/exchange_close_fill_finder.py`.

Safety boundary:

- Trading service remains stopped unless explicitly restarted after deployment gates pass.
- This change does not submit orders by itself, does not clear live data, does not change strategy thresholds, sizing, leverage target policy, model routing, GPU allocation, or scheduler scoring.
- It makes core current-state reads stricter: if OKX native current state cannot be read, the correct behavior is to block and surface the fault instead of proceeding with stale local data or CCXT alias-derived empty snapshots.

### 13.20 OKX Native Protection Algo Orders Completed

- `OKXExecutor.get_position_protection_orders()` now reads TP/SL protection orders through OKX native `privateGetTradeOrdersAlgoPending`:
  - uses OKX `orders-algo-pending` instead of CCXT `fetch_open_orders(... ordType=...)`.
  - queries `conditional`, `oco`, `trigger`, and `move_order_stop` algo order types.
  - scopes symbol-specific reads by OKX `instId`.
  - preserves raw `algoId`, `algoClOrdId`, `instId`, `side`, `posSide`, `ordType`, `state`, `tpTriggerPx`, `slTriggerPx`, trigger price fields, and timestamps.
- Protection-map consumers continue to receive the existing normalized structure:
  - `symbol`.
  - `position_side`.
  - `close_side`.
  - `order_type`.
  - `take_profit_price`.
  - `stop_loss_price`.
  - `trigger_price`.
  - `algo_id`.
  - `updated_at_ms`.
- Native algo API failure is explicit:
  - missing/unavailable OKX native algo pending-orders API raises at the executor/native-facts boundary.
  - the provider may use its existing short-lived cache or diagnostic empty result only after logging the failure; it must not treat a failed native read as proof that no protection exists.
- This removes another CCXT current-state dependency from the trading safety chain:
  - pending orders are native via `privateGetTradeOrdersPending`.
  - positions are native via `privateGetAccountPositions`.
  - protection algo orders are native via `privateGetTradeOrdersAlgoPending`.

Verification:

- OKX native facts, exchange protection map, and executor safety regression: `44 passed`.
- Python compile check passed for `services/okx_native_facts.py`, `executor/okx_executor.py`, and `services/exchange_position_state.py`.
- `git diff --check` passed.

Safety boundary:

- No order submission, close execution, repair apply, historical deletion, strategy threshold, sizing, leverage target policy, model routing, GPU allocation, or scheduler scoring behavior changed.
- This change improves protection-order visibility only; it does not create or cancel TP/SL algo orders by itself.

### 13.21 Strategy Quality Audit Uses Trusted Trade Facts Completed

- `strategy_quality` system-audit fast-loss detection now uses trusted closed trade facts only:
  - closed positions still load from the recent audit window.
  - fast-loss samples are calculated only after `closed_position_trade_fact_trusted()` passes.
  - dirty or repaired historical closed rows can no longer keep the strategy-quality card in warning state by themselves.
- The audit details now expose the quarantine boundary:
  - `closed_position_count`.
  - `trusted_closed_position_count`.
  - `quarantined_closed_position_count`.
  - `trade_fact_policy=strategy_quality_fast_loss_uses_trusted_closed_facts_only`.
- This aligns dashboard diagnostics with the Phase 3 clean-training policy:
  - raw history is preserved.
  - trusted OKX-backed facts drive learning/quality signals.
  - quarantined facts remain visible through audit counts instead of silently influencing quality warnings.

Verification:

- System audit + trade fact trust regression: `57 passed`.
- Python compile check passed for `web_dashboard/api/system_audit.py` and `services/trade_fact_trust.py`.
- `git diff --check` passed.

Safety boundary:

- Read-only diagnostic classification only.
- No order submission, close execution, repair apply, historical deletion, strategy threshold, sizing, leverage target policy, model routing, GPU allocation, scheduler scoring, or model training behavior changed.

### 13.22 Execution Allocation Uses Native/Trusted Facts Completed

- `ExecutionAllocationService` now aligns with Phase 3 fact-governance rules:
  - open-position exchange validation requires `get_positions_strict()` so OKX native current positions are used.
  - non-strict `get_positions()` is not an acceptable execution-allocation truth source after Phase 3 cold start.
  - closed-position realized PnL is counted only when `closed_position_trade_fact_trusted()` passes.
- This prevents dirty historical closed rows from affecting:
  - realized profit/loss.
  - today realized profit/loss.
  - total PnL used by execution allocation.
  - daily equity baseline inputs.
  - strategy posture and risk context that consume allocation state.
- Added regression coverage proving:
  - strict exchange-position reads are required.
  - untrusted closed facts missing OKX links or marked manual-close are excluded from allocation PnL.

Verification:

- Execution allocation + trade fact trust regression: `9 passed`.
- Expanded execution allocation, trading boundary, strategy learning, model contribution, profit attribution, and trade fact trust regression: `218 passed`.
- Python compile check passed for `services/execution_allocation_service.py`, `services/trading_service.py`, `services/strategy_learning.py`, `services/model_contribution_performance.py`, and `services/profit_attribution.py`.
- `git diff --check` passed.

Safety boundary:

- No order submission, close execution, repair apply, historical deletion, strategy threshold, sizing, leverage target policy, model routing, GPU allocation, scheduler scoring, or model training behavior changed directly.
- This changes the allocation/strategy context input to ignore untrusted historical realized PnL, which is intentional after the Phase 3 cold-start and data-governance reset.

### 13.23 Strategy Scheduler Trusted-Fact Input Audit Completed

- Audited the major strategy-scheduler performance inputs that can affect posture, direction bias, loss pause, model contribution, and risk context:
  - `DailyPerformanceService`.
  - `DailySidePerformanceService`.
  - `SymbolSidePerformanceService`.
  - `ModelContributionPerformanceService`.
  - `NewPairLossPausePolicy`.
  - `StrategyLearningService`.
  - `ProfitAttributionService`.
  - `ExecutionAllocationService`.
- Confirmed or enforced the rule:
  - closed historical rows may be displayed/audited.
  - closed rows must pass `closed_position_trade_fact_trusted()` before influencing training, strategy learning, allocation PnL, side/symbol performance, model contribution, loss pause, or profit attribution.
- The only clear scheduler-input gap found in this pass was `ExecutionAllocationService`, which was fixed in `13.22`.
- `strategy_quality` dashboard diagnostics were already tightened in `13.21` so old dirty fast-loss rows do not keep the quality card warning by themselves.

Verification:

- Comprehensive focused regression across OKX native facts, OKX executor, authoritative sync, close-fill finder, exchange protection map, executor safety, execution allocation, trading boundaries, strategy learning, model contribution, profit attribution, trade fact trust, and system audit: `330 passed`.
- Python compile check passed for all touched OKX, allocation, strategy, attribution, and audit modules.
- `git diff --check` passed.

Safety boundary:

- Audit and trust-filter hardening only.
- No order submission, close execution, repair apply, historical deletion, strategy threshold, sizing, leverage target policy, model routing, GPU allocation, scheduler scoring formula, or model training artifact changed.

### 13.24 OKX Native Truth Chain No-Fallback Hardening Completed

- Closed-position reconciliation now treats OKX native fills-history as the only close-fill fact source:
  - `ExchangeCloseFillFinder` no longer reads CCXT `fetch_closed_orders`.
  - `ExchangeCloseFillFinder` no longer reads CCXT `fetch_my_trades`.
  - close-fill candidates must come from `OkxNativeFactsClient.fetch_fill_groups(..., strict=True)`.
  - native fills query failure propagates to sync as `lookup_unavailable` instead of being converted into an empty "no close fill" fact.
- Active position context now uses OKX-native current positions as the trading truth source:
  - `OkxSyncService.get_open_positions_context()` returns no positions if the active OKX strict snapshot is unavailable.
  - when OKX returns a position, quantity, side, entry price, mark price, contracts, contract size, and unrealized PnL come from the OKX snapshot.
  - local DB rows may only add non-exchange metadata such as model name, stop loss, take profit, and local created time.
  - local `okx_inst_id` / `okx_pos_id` are not injected into OKX snapshots, preventing old IDs from polluting current context.
- Dashboard and system-audit position views now use strict OKX-native snapshots:
  - `position_price_integrity` calls `get_positions_strict()` instead of soft `get_positions()`.
  - dashboard open-position symbol discovery calls `get_positions_strict()`.
  - dashboard mark-price map calls `get_positions_strict()`.
  - strict-read failure is surfaced as an unavailable strict read or stale-cache display, not as a trusted empty OKX state.

Verification:

- OKX native facts, close-fill finder, and open-position context regression: `18 passed`.
- Dashboard fallback logging, position-price integrity, and trade-execution contract regression: `13 passed`.
- Python compile check passed for `services/okx_native_facts.py`, `services/exchange_close_fill_finder.py`, `services/sync_service.py`, `web_dashboard/api/dashboard.py`, and `web_dashboard/api/system_audit.py`.

Safety boundary:

- No order submission, close execution, historical deletion, repair apply, strategy threshold, sizing, leverage target policy, model routing, GPU allocation, scheduler scoring formula, or model training artifact changed by this hardening pass.
- This intentionally changes truth-source selection for reconciliation, position context, and dashboard OKX snapshots so OKX API/query failures cannot be mistaken for valid empty state and local stale DB rows cannot override OKX current positions.

### 13.25 OKX Strict Position Inputs For Sync And Allocation Completed

- `OkxAuthoritativeSyncService` now requires executor support for `get_positions_strict()`:
  - current-position reconciliation no longer falls back to soft `get_positions()`.
  - missing strict support raises `OKX authoritative sync requires get_positions_strict`.
  - system audit surfaces the failure as OKX authoritative sync unavailable instead of treating soft/local state as valid exchange truth.
- `ExecutionAllocationService` now requires strict OKX current positions for open-position allocation metrics:
  - missing strict support raises `execution allocation requires get_positions_strict`.
  - if the strict OKX snapshot is unavailable, local open positions do not contribute to `used_margin` or `unrealized_pnl`.
  - trusted closed trade facts still contribute realized PnL through `closed_position_trade_fact_trusted()`.
- This prevents allocation, daily risk PnL, and strategy context from being affected by stale local open positions when OKX current-state truth is unavailable.

Verification:

- OKX authoritative sync and execution allocation regression: `8 passed`.
- Python compile check passed for `services/okx_authoritative_sync.py` and `services/execution_allocation_service.py`.

Safety boundary:

- No order submission, close execution, historical deletion, repair apply, strategy threshold, sizing, leverage target policy, model routing, GPU allocation, scheduler scoring formula, or model training artifact changed by this hardening pass.
- This intentionally changes open-position allocation metrics to fail closed when OKX strict current positions are unavailable.

### 13.26 Strict Executor Contract And Position Cache Hardening Completed

- `AbstractExecutor.get_positions_strict()` no longer defaults to soft `get_positions()`:
  - executors without an authoritative strict current-position implementation now raise `NotImplementedError`.
  - this prevents future core callers from accidentally treating paper/local/CCXT-soft reads as strict OKX truth.
- `AbstractExecutor.get_open_orders_strict()` no longer defaults to soft `get_open_orders()`:
  - executors without authoritative strict pending-order support now raise `NotImplementedError`.
  - this keeps active-order guards aligned with the OKX-native pending-order policy.
- `PositionTracker.sync_from_executor()` now refreshes only from `get_positions_strict()`:
  - it no longer calls soft `executor.get_positions()`.
  - if strict sync fails, the model's in-memory position cache is cleared and logged with `stale_positions_cleared=True`.
  - stale in-memory positions cannot continue affecting exposure/PnL/stop-trigger helpers after OKX strict truth is unavailable.

Verification:

- Strict executor contract and position-tracker regression: `3 passed`.
- Combined OKX/native sync/allocation/dashboard/trade-boundary regression including strict contract tests: `105 passed`.
- Python compile check passed for `executor/base_executor.py`, `executor/position_tracker.py`, and `tests/test_executor_strict_contracts.py`.

Safety boundary:

- No order submission, close execution, historical deletion, repair apply, strategy threshold, sizing, leverage target policy, model routing, GPU allocation, scheduler scoring formula, or model training artifact changed by this hardening pass.
- This intentionally changes strict interface semantics: callers that require authoritative exchange truth must fail closed unless the executor implements a real strict OKX-native method.

### 13.27 OKX Auto Sync Entry Blocker Completed

- Trading-service runtime now turns automatic OKX authoritative-sync health into a hard new-entry gate:
  - if `okx_authoritative_sync.status` is `warning` or `stale`, new-symbol analysis is paused before spending model/scheduler resources.
  - if the latest sync result has `last_requires_attention_count > 0`, new entries are paused until the current-state difference is reconciled.
  - the execution-facing policy gate also blocks `decision.is_entry` before submit with blocker `okx_authoritative_sync_unhealthy`.
- The gate intentionally applies only to new entries:
  - exits, close decisions, stop-loss/take-profit enforcement, and position review are not blocked by this new gate.
  - this prevents the system from expanding OKX/backend mismatch while still allowing risk reduction and cleanup.
- Runtime diagnostics remain machine-readable through `okx_authoritative_sync` so dashboard/system-health can show whether the pause is caused by stale sync, sync failure, or current-state differences requiring review.

Verification:

- OKX auto-sync entry blocker regression: `4 passed`.
- Trading boundary, execution-service, market auto-entry, system audit, and dashboard fallback regression: `228 passed`.
- OKX authoritative sync, OKX native facts, close-fill finder, execution allocation, and strict executor contract regression: `27 passed`.
- Python compile check passed for `services/trading_service.py` and `tests/test_trading_service_boundaries.py`.
- `git diff --check` passed.

Safety boundary:

- No order submission, close execution, historical deletion, repair apply, strategy threshold, sizing, leverage target policy, model routing, GPU allocation, scheduler scoring formula, or model training artifact changed directly.
- This intentionally changes only new-entry availability when OKX authoritative current-state sync is unhealthy. It is a fail-closed safety gate for opening risk, not a blocker for exits or position-risk reduction.

### 13.28 OKX Runtime Reconciliation Visibility Completed

- System audit now joins the read-only OKX fact audit with split-process trading runtime heartbeat:
  - loads `trading_runtime_status.json` without touching or starting the trading engine.
  - derives `runtime_okx_entry_gate` from runtime `okx_authoritative_sync`.
  - exposes whether new entries are currently blocked by `okx_authoritative_sync_unhealthy`.
  - surfaces the exact block reason, heartbeat age, sync status, latest result kind counts, and recent sync samples.
- Dashboard OKX audit details now show:
  - `Entry gate`: `blocked` / `open` / `unknown`.
  - `Runtime OKX entry gate` table with running state, sync status, heartbeat age, blocker, and reason.
  - `Runtime OKX sync result kinds`.
  - `Runtime OKX sync samples`, including whether each sample requires attention.
- This closes the visibility gap where OKX/local mismatch could already trigger a hard new-entry block, but the operator still had to infer the reason from logs or raw runtime JSON.

Verification:

- Runtime OKX entry-gate focused regression: `2 passed`.
- System audit API and dashboard static contract regression: `112 passed`.
- OKX authoritative sync, trading boundary, and dashboard fallback regression: `168 passed`.
- Python compile check passed for `web_dashboard/api/system_audit.py`, `tests/test_system_audit_api.py`, and `tests/test_dashboard_main_ui_contract.py`.
- `git diff --check` passed.

Safety boundary:

- Read-only observability and audit-detail rendering only.
- No order submission, close execution, historical deletion, repair apply, strategy threshold, sizing, leverage target policy, model routing, GPU allocation, scheduler scoring formula, model training artifact, or trading-service startup behavior changed.

### 13.29 OKX Daily Dry-Run Report Script Completed

- Added `scripts/run_okx_daily_reconciliation_report.py` as a standalone, timer-friendly OKX reconciliation report generator:
  - runs only read-only OKX-related audit cards.
  - clears the short in-process OKX reconciliation cache by default so the daily report is a fresh dry-run.
  - includes `okx_reconciliation`, `okx_trade_fact_integrity`, `position_price_integrity`, and `trade_execution_contract`.
  - writes a dated JSON artifact plus `latest.json` under `data/okx_daily_reconciliation_reports/`.
  - supports `--stdout-only`, `--output-dir`, `--json-indent`, and `--allow-cache`.
  - returns exit code `0` for `ok`, `1` for `warning`, and `2` for `critical`.
- Report payload includes:
  - `dry_run=true`.
  - `mutates_database=false`.
  - `live_order_mutation=false`.
  - `repair_apply_enabled=false`.
  - issue ledger grouped as fixed/unresolved/observing.
  - explicit training policy: `do_not_train_dirty_or_unclassified_okx_facts`.
- This converts the Phase H requirement "每日 dry-run 对账报表" from a dashboard-only/manual observation into a schedulable artifact that can be checked by systemd timer, cron, or external monitoring.

Verification:

- Daily OKX report script regression: `3 passed`.
- Script, system audit API, and dashboard static contract regression: `115 passed`.
- OKX authoritative sync, OKX native facts, trade execution contract, and dashboard fallback regression: `34 passed`.
- Python compile check passed for `scripts/run_okx_daily_reconciliation_report.py`, `web_dashboard/api/system_audit.py`, and related tests.
- `git diff --check` passed.
- Local dry-run command `python scripts/run_okx_daily_reconciliation_report.py --stdout-only --json-indent 0` produced valid JSON and returned warning exit code when local environment could not connect to the online DB/OKX source, which is the intended timer-visible failure mode.

Safety boundary:

- Read-only report generation only.
- No DB mutation, no repair apply, no historical deletion, no order submission, no close execution, no trading-service startup, no strategy threshold/sizing/leverage/model-routing/GPU/model-training change.

### 13.30 OKX Daily Reconciliation Systemd Timer Completed

- Added `scripts/install_okx_daily_reconciliation_timer.py` to install a repeatable online timer:
  - service: `bb-okx-daily-reconciliation.service`.
  - timer: `bb-okx-daily-reconciliation.timer`.
  - default schedule: `*-*-* 00:10:00`.
  - runs as `User=bb` / `Group=bb`.
  - uses `WorkingDirectory=/data/bb/app`.
  - loads `/data/bb/app/.env` and `/etc/bb/bb-runtime.env`.
  - runs `scripts/run_okx_daily_reconciliation_report.py --json-indent 0`.
  - ensures `data/okx_daily_reconciliation_reports` exists and is owned by `bb:bb`.
- This fixes the operational gap found during online validation:
  - running the report manually as `root` can hit PostgreSQL peer-auth role errors.
  - root-created report directories can block the `bb` service user from writing future reports.
  - the timer installer makes the intended user/env/working-directory explicit so daily reports match the dashboard service environment.
- The report script now also returns structured JSON with `artifact_error.code=artifact_write_failed` if artifact writing fails, instead of leaking a traceback into timer output.

Verification:

- OKX daily report + timer regression: `6 passed`.
- Script, system audit API, and dashboard static contract regression: `118 passed`.
- OKX authoritative sync, OKX native facts, trade execution contract, and dashboard fallback regression: `34 passed`.
- Python compile check passed for the new report/timer scripts and related tests.
- `git diff --check` passed.
- Online validation before timer install:
  - root-run report showed `role "root" does not exist`, proving user/env must be fixed.
  - after `chown -R bb:bb`, running with the `bb` environment produced clean `latest.json`.
  - latest online report status was `ok`: 4 cards ok, 0 warning, 0 critical; 14-day missing closed positions = 0; OKX fact issue_count = 0; position price integrity ok; trade execution contract ok.
- Online timer installation and verification:
  - `bb-okx-daily-reconciliation.timer` is `enabled` and `active`.
  - next scheduled run: `2026-06-27 00:11:57 UTC`.
  - immediate oneshot run succeeded with `status=ok`.
  - `/data/bb/app/data/okx_daily_reconciliation_reports/latest.json` is owned by `bb:bb`.
  - latest report ledger: `fixed=4`, `unresolved=0`, `observing=0`, `total=4`.
  - `bb-paper-trading.service` remained `inactive`.
  - `bb-dashboard.service` remained `active`.

Safety boundary:

- Timer/report automation only.
- Does not start `bb-paper-trading.service`.
- No DB mutation, no repair apply, no historical deletion, no order submission, no close execution, no strategy threshold/sizing/leverage/model-routing/GPU/model-training change.

### 13.31 Phase I Artifact Retirement Audit Completed

- Added `services/artifact_retirement_audit.py` as a read-only Phase I artifact governance layer:
  - scans local model artifact roots under `data/ml_signal`, `data/local_ai_tools`, and `data/models`.
  - classifies artifacts as `phase3_compatible`, `retired_legacy`, `missing_manifest`, or `untrusted`.
  - requires explicit Phase 3 evidence before an artifact can influence live:
    - `artifact_policy_id=phase3_clean_training_artifact_v1`.
    - `training_policy=clean_training_view_only`.
    - `promotion_flow=shadow_to_canary_to_live`.
    - `live_mutation=false`.
    - persisted artifact evidence for model binaries.
  - preserves every discovered artifact and sets `can_delete_artifacts=false`.
- Connected the artifact retirement report to the `model_training` system-audit card:
  - exposes `artifact_retirement_audit` in card details.
  - shows retired/untrusted artifact count in evidence.
  - marks retired or untrusted artifacts as a warning/observing rebuild gate, not a trading-chain hard failure.
  - requires rebuilding from the Phase 3 clean training view before live influence.
- Local read-only audit result:
  - `data/ml_signal/winrate_model.joblib` classified as `retired_legacy`.
  - `data/ml_signal/winrate_model_metadata.json` classified as `retired_legacy`.
  - both files are preserved and cannot influence live under the Phase 3 artifact policy.

Verification:

- Artifact retirement audit regression: `3 passed`.
- Model-training system audit regression with artifact rebuild gate: `9 passed`.
- Full system audit API regression: `58 passed`.
- Python compile check passed for `services/artifact_retirement_audit.py`, `web_dashboard/api/system_audit.py`, and related tests.
- `git diff --check` passed.

Safety boundary:

- Read-only artifact audit and dashboard/system-audit visibility only.
- No artifact deletion, no DB mutation, no repair apply, no historical deletion, no order submission, no close execution, no trading-service startup, no strategy threshold/sizing/leverage/model-routing/GPU allocation change.
- This does not train or replace models yet; it blocks legacy/untrusted artifact promotion until Phase 3 clean training rebuild is completed.

### 13.32 Phase I Phase-3 Artifact Policy Metadata Completed

- Added Phase 3 artifact identity metadata to future local ML training outputs:
  - `artifact_policy_id=phase3_clean_training_artifact_v1`.
  - `phase=phase3_model_factory`.
  - `training_policy=clean_training_view_only`.
  - `trade_sample_cursor_policy=clean_training_view_only`.
  - `training_mode=walk_forward`.
  - `model_stage=shadow`.
  - `evaluation_policy.promotion_flow=shadow_to_canary_to_live`.
  - `evaluation_policy.live_mutation=false`.
- Added the same artifact identity contract to the generated `local_ai_tools` model bundle metadata:
  - `/train` now persists the Phase 3 artifact policy, clean training policy, promotion flow, live-mutation=false, and `artifact_persisted=true`.
  - missing policy fields are filled with Phase 3 defaults before metadata is written.
- This completes the other half of 13.31:
  - old artifacts are detected and retired.
  - future rebuilt artifacts can prove they were produced by the Phase 3 clean training path.

Verification:

- Local ML training quality regression: included in focused `36 passed`.
- Local AI tools deploy-service contract regression: included in focused `36 passed`.
- Artifact retirement audit regression: included in focused `36 passed`.
- End-to-end temporary artifact check: a synthetic Phase 3 artifact with the new metadata was classified as `phase3_compatible`, `retired_or_untrusted_count=0`.
- Python compile check passed for `services/ml_signal_service.py`, `scripts/deploy_local_ai_tools_service.py`, and related tests.
- `git diff --check` passed.

Safety boundary:

- Metadata contract only.
- No live training run, no artifact replacement, no artifact deletion, no DB mutation, no repair apply, no historical deletion, no order submission, no close execution, no trading-service startup, no strategy threshold/sizing/leverage/model-routing/GPU allocation change.
- Current old artifacts remain retired until a separate Phase I rebuild step explicitly trains and promotes new artifacts through shadow/canary gates.

### 13.33 Phase I Manual ML Rebuild Preflight Gate Completed

- Hardened `scripts/train_ml_signal_model.py` so manual ML rebuilds are safe by default:
  - default `run_training()` mode is now Phase 3 preflight only.
  - preflight does not quarantine rows and does not write model artifacts.
  - `--dry-run` remains accepted as a deprecated alias for the default preflight behavior.
  - artifact writing requires both `--persist-artifact` and `--confirm-phase3-rebuild`.
  - calling `run_training(persist_artifact=True)` without confirmation raises a hard error.
- This prevents manual scripts from bypassing the Phase I clean-view + artifact-policy rebuild flow.
- Confirmed rebuilds still run quarantine first and then call `train_from_frame(..., persist_artifact=True)`, so the formal path remains available when Phase 3 gates explicitly allow it.

Verification:

- ML signal training quality regression: included in focused `92 passed`.
- Local AI tools deploy-service contract regression: included in focused `92 passed`.
- Artifact retirement audit and model-training system audit regression: included in focused `92 passed`.
- Python compile check passed for `scripts/train_ml_signal_model.py`, `services/ml_signal_service.py`, `scripts/deploy_local_ai_tools_service.py`, and related tests.

Safety boundary:

- Training entrypoint gate only.
- No live training run, no artifact replacement, no artifact deletion, no DB mutation, no repair apply, no historical deletion, no order submission, no close execution, no trading-service startup, no strategy threshold/sizing/leverage/model-routing/GPU allocation change.
- This makes manual rebuild safer; it does not promote any model by itself.

### 13.34 Phase I Local AI Tools Rebuild Preflight Gate Completed

- Hardened the generated `local_ai_tools` training service and callers so Phase 3 local quant bundle rebuilds are safe by default:
  - `/train` defaults to preflight-only and returns `reason=phase3_preflight_no_artifact_write` without writing `local_quant_models.joblib`.
  - artifact persistence now requires both `persist_artifact=true` and `confirm_phase3_rebuild=true`.
  - an unconfirmed persist request returns `reason=phase3_rebuild_confirmation_required`.
  - metadata now records `artifact_persisted`, `preflight_only`, `persist_artifact_requested`, and `confirm_phase3_rebuild`.
- Hardened the client and CLI entrypoints:
  - `LocalAIToolsClient.train()` defaults to `persist_artifact=false` and `confirm_phase3_rebuild=false`.
  - `scripts/train_local_ai_tools_models.py` defaults to Phase 3 preflight, skips quarantine writes in preflight, and requires `--confirm-phase3-rebuild` with `--persist-artifact`.
  - default evaluation policy includes `phase=phase3_model_factory`.
- This closes the gap left after 13.33:
  - manual ML rebuilds are preflight-gated.
  - local AI tools rebuilds are now preflight-gated too.
  - future artifact creation is still possible, but only through an explicit Phase 3 rebuild command.

Verification:

- Focused local AI tools regression: `54 passed`.
- Expanded Phase 3 training/governance/system/data-collection regression: `148 passed`.
- Python compile check passed for `scripts/deploy_local_ai_tools_service.py`, `services/local_ai_tools_client.py`, `scripts/train_local_ai_tools_models.py`, and related tests.
- `git diff --check` passed.

Safety boundary:

- Training entrypoint gate only.
- No live training run, no artifact replacement, no artifact deletion, no DB mutation, no repair apply, no historical deletion, no order submission, no close execution, no trading-service startup, no strategy threshold/sizing/leverage/model-routing/GPU allocation change.
- This prevents accidental model bundle replacement while Phase 3 clean training, OKX native fact governance, and new model-server deployment are still being completed.

### 13.35 Phase I Confirmed Rebuild Readiness Gate Completed

- Added `services/phase3_rebuild_readiness.py` as the read-only total gate before any confirmed Phase 3 artifact rebuild can write files:
  - aggregates `local_ai_tools` clean-view sample counts.
  - checks clean training governance status and contamination risk.
  - checks historical trade-fact audit status.
  - checks artifact retirement audit status.
  - checks model runtime probe status.
  - checks Phase 3 promotion flow and `live_mutation=false`.
  - enforces the double-confirmation rule: `--persist-artifact` plus `--confirm-phase3-rebuild`.
- Connected the report into the `model_training` system-audit card as `phase3_rebuild_readiness`.
- The gate intentionally distinguishes two separate permissions:
  - `can_run_confirmed_rebuild`: clean enough to run the explicit rebuild command.
  - `can_persist_artifact`: the current request has the required explicit write confirmation.
- The gate never grants live trading influence:
  - target artifacts are rebuilt to `shadow`.
  - `live_mutation=false`.
  - promotion still requires later `shadow -> canary -> live` validation.
- Retired legacy artifacts are treated as a rebuild warning/trigger, not as permission to delete history or bypass clean training.

Verification:

- Focused Phase 3 rebuild readiness and model-training audit regression: `5 passed`.
- Expanded Phase 3 training/governance/system/data-collection regression: `153 passed`.
- Python compile check passed for `services/phase3_rebuild_readiness.py`, `web_dashboard/api/system_audit.py`, and related tests.
- `git diff --check` passed.

Safety boundary:

- Read-only readiness gate only.
- No live training run, no artifact replacement, no artifact deletion, no DB mutation, no repair apply, no historical deletion, no order submission, no close execution, no trading-service startup, no strategy threshold/sizing/leverage/model-routing/GPU allocation change.
- This prepares the formal rebuild workflow but does not execute it.

### 13.36 Phase I Unified Rebuild Preflight Script Completed

- Added `scripts/run_phase3_rebuild_preflight.py` as the unified read-only preflight entrypoint for Phase 3 model rebuilds:
  - collects clean training-view sample counts for `ml_signal` and `local_ai_tools`.
  - runs historical trade-fact audit and artifact retirement audit.
  - optionally runs the runtime model probe.
  - calls `Phase3RebuildReadinessService`.
  - outputs one JSON report with `readiness`, `training_summary`, `quality_report`, `governance_report`, audit reports, and follow-up commands.
- The script explicitly prints the two operator command families but does not execute them:
  - preflight commands.
  - confirmed rebuild commands using `--persist-artifact --confirm-phase3-rebuild`.
- The script is hardened for operations:
  - database/audit collection failures now return structured `blocked` JSON in `collection_errors`.
  - it can return non-zero with `--fail-on-blocked` for timers/automation.
  - it always reports `read_only=true`, `mutates_database=false`, `writes_artifacts=false`, and `starts_trading_service=false`.
- This creates the bridge from “readiness gate exists” to “operator has a safe one-command preflight report” without letting a script silently train or overwrite artifacts.

Verification:

- Focused rebuild preflight/readiness regression: `7 passed`.
- Expanded Phase 3 training/governance/system/data-collection regression: `154 passed`.
- Local smoke with disconnected PostgreSQL returned structured `blocked` JSON and `collection_errors` instead of a traceback.
- Python compile check passed for `scripts/run_phase3_rebuild_preflight.py`, `services/phase3_rebuild_readiness.py`, and related tests.
- `git diff --check` passed.

Safety boundary:

- Read-only preflight/report script only.
- No training execution, no artifact replacement, no DB mutation, no repair apply, no historical deletion, no order submission, no close execution, no trading-service startup, no strategy threshold/sizing/leverage/model-routing/GPU allocation change.
- Confirmed rebuild commands remain suggestions until an operator explicitly runs them after blockers are clear.

### 13.37 OKX Native Position/Ticker Truth Read Path Completed

- Hardened OKX position reads so the legacy-compatible `OKXExecutor.get_positions()` now delegates to `get_positions_strict()`:
  - current position truth is read from OKX native `privateGetAccountPositions`.
  - the non-strict compatibility wrapper may return `[]` on read failure, but it no longer calls `ccxt.fetch_positions`.
  - tests now fail if a future change reintroduces CCXT position reads in the OKX executor path.
- Hardened REST public ticker reads:
  - `OKXRestClient.fetch_ticker()` now calls OKX native `publicGetMarketTicker` with `instId`.
  - `OKXRestClient.fetch_tickers()` now calls OKX native `publicGetMarketTickers` and filters by native `instId`.
  - market discovery now reuses the same native ticker path instead of `_ccxt_call("fetch_tickers")`.
  - returned ticker shape remains compatible with dashboard/data-service callers while exposing native `id=*-USDT-SWAP`.
- Hardened pending-order test fixtures to match the Phase 3 contract:
  - fake OKX clients provide `privateGetTradeOrdersPending` and `privateGetAccountPositions`.
  - fake legacy `fetch_open_orders` / `fetch_positions` now assert if used by core guards.

Verification:

- Focused OKX native/current-state/reconciliation/REST ticker/dashboard/data-service regression: `90 passed`.
- Intermediate focused suites passed: `22 passed`, `32 passed`, `62 passed`.
- Python compile check passed for `executor/okx_executor.py`, `data_feed/okx_rest_client.py`, `tests/test_okx_executor_position_market_fallback.py`, `tests/test_okx_pending_orders.py`, and `tests/test_okx_rest_client_symbols.py`.
- Residual source search found no core `ccxt.fetch_positions`, `_ccxt_call("fetch_ticker")`, or `_ccxt_call("fetch_tickers")` truth-source path in executor/data-feed/services/dashboard.
- `git diff --check` passed.

Safety boundary:

- Read/write code hardening only.
- No DB mutation, no repair apply, no historical deletion, no artifact replacement, no model-routing/sizing/leverage threshold change, no order submission, no close execution, no trading-service startup.
- Online deployment must use precise upload for the changed files and must keep `bb-paper-trading.service` inactive unless the user explicitly approves a go-live/restart gate.

### 13.38 OKX Native Execution Ticker Path Completed

- Hardened execution sizing so `OKXExecutor.place_order()` no longer calls CCXT `fetch_ticker`:
  - execution price now comes from OKX native `publicGetMarketTicker` using `instId`.
  - entries fail closed with `execution_blocker=okx_native_ticker_unavailable` if native ticker is unavailable or has no positive last price.
  - entries rejected by the native ticker gate do not submit an OKX order.
  - exits may still continue when ticker is unavailable, but only by using the OKX-native position snapshot mark/avg price.
- Reduced OKX read-window inconsistency during exits:
  - if the exit path already reads OKX-native positions to recover mark price, the later close-position validation reuses the same snapshot instead of issuing another immediate position read.
  - this lowers the chance of "price read saw a position, close validation saw a transient different state" inside one execution attempt.
- Hardened tests:
  - execution test doubles now provide `publicGetMarketTicker`.
  - legacy `fetch_ticker` in execution-sizing test doubles raises if called, proving the executor cannot silently fall back to CCXT ticker.
  - a new entry pre-submit test verifies native ticker outage rejects before order submission.

Verification:

- Focused execution native-ticker regression: `43 passed`.
- Expanded OKX native/current-state/reconciliation/REST ticker/dashboard/data-service/execution regression: `91 passed`.
- Python compile check passed for `executor/okx_executor.py`, `tests/test_okx_pending_orders.py`, and `tests/test_executor_error_safety.py`.
- Residual source search found no core `ccxt.fetch_ticker`, `ccxt.fetch_positions`, `_ccxt_call("fetch_ticker")`, `_ccxt_call("fetch_tickers")`, or CCXT open-order truth-source path in executor/data-feed/services/dashboard.
- `git diff --check` passed.

Safety boundary:

- Execution read-path hardening only.
- No DB mutation, no repair apply, no historical deletion, no artifact replacement, no model-routing/sizing/leverage threshold change, no order submission, no close execution, no trading-service startup.
- If native ticker is unavailable during future go-live, entry is blocked instead of using stale/abstract/fallback prices.

### 13.39 OKX REST Compatibility Read Path Nativeized

- Hardened `data_feed.OKXRestClient` compatibility methods so future callers cannot accidentally reintroduce CCXT truth-source reads:
  - `fetch_positions()` now calls OKX native `privateGetAccountPositions` and filters by native `instId`.
  - `fetch_open_orders()` now calls OKX native `privateGetTradeOrdersPending` and filters/deduplicates by native `ordId`.
  - `fetch_ticker()` and `fetch_tickers()` were already nativeized in 13.37 and remain based on `publicGetMarketTicker(s)`.
- The client still returns the existing CCXT-like dict shape expected by dashboard/data-service callers:
  - positions expose `symbol`, `side`, `contracts`, `contractSize`, mark/entry price, UPL, leverage, and raw `info`.
  - pending orders expose `id`, `clientOrderId`, `symbol`, `side`, `type`, `status`, amount/filled/remaining, reduce-only state, timestamps, and raw `info`.
- This closes a latent fallback gap:
  - even if future code uses `OKXRestClient.fetch_positions()` or `fetch_open_orders()`, it will receive OKX-native facts rather than CCXT-normalized alias facts.

Verification:

- Focused OKX REST/native execution regression: `49 passed`.
- Expanded OKX native/current-state/reconciliation/REST ticker/dashboard/data-service/execution regression: `93 passed`.
- Python compile check passed for `data_feed/okx_rest_client.py` and `tests/test_okx_rest_client_symbols.py`.
- Residual source search found no core `_ccxt_call("fetch_positions")`, `_ccxt_call("fetch_open_orders")`, `_ccxt_call("fetch_ticker")`, `_ccxt_call("fetch_tickers")`, `ccxt.fetch_positions`, `ccxt.fetch_ticker`, or CCXT open-order truth-source path in executor/data-feed/services/dashboard.
- `git diff --check` passed.

Safety boundary:

- Read-path compatibility hardening only.
- No DB mutation, no repair apply, no historical deletion, no artifact replacement, no model-routing/sizing/leverage threshold change, no order submission, no close execution, no trading-service startup.
- CCXT may still be used as signed request transport for OKX native endpoints; it is not used as the position/order/ticker truth source in these core read paths.

### 13.40 OKX Daily Reconciliation Artifact Self-Description Completed

- Hardened `scripts/run_okx_daily_reconciliation_report.py` report writing:
  - `write_report()` now inserts `artifacts.report_path` and `artifacts.latest_path` into the report before writing the dated JSON and `latest.json`.
  - `latest.json` is now self-describing, so dashboard/automation/operator checks can read the latest report and know the exact dated artifact path without parsing stdout or systemd journal text.
- This fixes a timer observability gap found during online validation:
  - systemd stdout showed artifact paths.
  - the persisted `latest.json` did not include `artifacts` because paths were added after the file write.

Verification:

- Focused OKX daily report/timer regression: `6 passed`.
- Python compile check passed for `scripts/run_okx_daily_reconciliation_report.py` and `tests/test_okx_daily_reconciliation_report.py`.
- Online timer natural run after the native read-path deployment completed successfully:
  - `bb-paper-trading.service=inactive`.
  - `bb-okx-daily-reconciliation.timer=active`.
  - latest report status `ok`.
  - summary `cards=4`, `critical=0`, `warning=0`, `ok=4`.
  - issue ledger `fixed=4`, `observing=0`, `unresolved=0`.

Safety boundary:

- Report metadata hardening only.
- No DB mutation, no repair apply, no historical deletion, no artifact/model replacement, no order submission, no close execution, no trading-service startup.
- Manual online re-run is allowed only through the read-only oneshot report service/script.

### 13.41 Runtime OKX Entry Gate Stale/Inactive Heartbeat Corrected

- Corrected a misleading audit state found after 13.40 online validation:
  - `runtime_okx_entry_gate` previously judged "OKX runtime sync healthy for new entries" from the last `okx_authoritative_sync.status` alone.
  - when `bb-paper-trading.service` was intentionally stopped, an old heartbeat could still carry `okx_authoritative_sync.status=ok`, making the daily report look fully green.
  - the gate now first checks trading runtime availability, `running`, and heartbeat freshness before looking at OKX sync details.
- New runtime gate states:
  - missing heartbeat: `status=runtime_unavailable`, `blocker=runtime_heartbeat_unavailable`, `entry_blocked=true`.
  - stopped runtime: `status=runtime_inactive`, `blocker=trading_runtime_inactive`, `entry_blocked=true`.
  - stale heartbeat: `status=runtime_heartbeat_stale`, `blocker=trading_runtime_heartbeat_stale`, `entry_blocked=true`.
  - fresh runtime but unhealthy OKX sync remains `blocker=okx_authoritative_sync_unhealthy`.
- Dashboard details now separate:
  - `Runtime status`: whether the runtime itself is usable for entry decisions.
  - `Sync status`: the last OKX authoritative sync result.
  - this prevents operators from confusing an intentional trading-service stop with an OKX position/order mismatch.
- Issue ledger classification:
  - runtime-only entry blocks caused by inactive/stale/unavailable heartbeat are classified as `observing` when there are no OKX/local data-integrity issues.
  - real OKX/local mismatches remain `warning/unresolved` and still require investigation.

Verification:

- Runtime gate focused regression: `4 passed`.
- System audit + OKX daily reconciliation regression: `62 passed`.
- Python compile check passed for `web_dashboard/api/system_audit.py`, `tests/test_system_audit_api.py`, `scripts/run_okx_daily_reconciliation_report.py`, and `tests/test_okx_daily_reconciliation_report.py`.
- JavaScript syntax check passed for `web_dashboard/static/js/dashboard.js`.

Safety boundary:

- Audit/dashboard/report semantics only.
- No DB mutation, no repair apply, no historical deletion, no artifact/model replacement, no order submission, no close execution, no trading-service startup.
- During the current paused-trading period, a non-green runtime entry gate is expected and correct; it means new entries remain blocked until the trading runtime is explicitly restarted and publishes a fresh OKX sync heartbeat.

### 13.42 OKX Daily Reconciliation Observing-Only Exit Code Corrected

- Follow-up from 13.41 online validation:
  - the new runtime gate correctly changed the report to `warning` when the trading runtime heartbeat was stale.
  - issue ledger correctly showed `fixed=3`, `observing=1`, `unresolved=0`.
  - however, the systemd oneshot exited non-zero for any `warning`, causing an observing-only paused-trading report to appear as a failed timer run.
- Added `exit_code_for_report()` to distinguish warning classes:
  - `critical` report exits `2`.
  - `warning` with any `unresolved` issue exits `1`.
  - `warning` with only `observing` items exits `0`.
  - `ok` exits `0`.
- This preserves alerting for real OKX/local data problems while keeping intentional paused-trading observation from looking like timer failure.

Verification:

- OKX daily reconciliation + system audit regression: `64 passed`.
- Python compile check passed for `scripts/run_okx_daily_reconciliation_report.py`, `tests/test_okx_daily_reconciliation_report.py`, `web_dashboard/api/system_audit.py`, and `tests/test_system_audit_api.py`.
- `git diff --check` passed.

Safety boundary:

- Report process exit-code semantics only.
- No DB mutation, no repair apply, no historical deletion, no artifact/model replacement, no order submission, no close execution, no trading-service startup.
- The report JSON still says `status=warning` for paused/stale runtime; only systemd success/failure semantics are adjusted so the timer means "ran successfully" instead of "all cards green".

### 13.43 Trading Runtime Stop Writes Inactive Heartbeat Completed

- Moved the inactive-runtime signal closer to the source:
  - `TradingService.stop()` now writes a final split-process runtime heartbeat immediately after setting `_running=false`.
  - dashboard/system audit/daily reconciliation no longer need to wait for heartbeat expiry to discover an intentional graceful stop.
  - the heartbeat preserves OKX sync status details but marks the runtime itself as inactive, so entry gates can distinguish "runtime stopped" from "OKX data mismatch".
- This complements 13.41:
  - 13.41 made stale/old heartbeats fail closed at audit time.
  - 13.43 ensures future graceful stops publish an explicit inactive heartbeat first.

Verification:

- Focused trading runtime stop/sync regression: `4 passed`.
- Expanded split-process runtime/dashboard/system-audit/daily-report regression: `243 passed`.
- Python compile check passed for `services/trading_service.py`, `tests/test_trading_service_boundaries.py`, `web_dashboard/api/system_audit.py`, and `scripts/run_okx_daily_reconciliation_report.py`.
- JavaScript syntax check passed for `web_dashboard/static/js/dashboard.js`.
- `git diff --check` passed.

Safety boundary:

- Runtime observability only.
- No DB mutation, no repair apply, no historical deletion, no artifact/model replacement, no order submission, no close execution, no trading-service startup.
- Online deployment must upload the changed file without restarting `bb-paper-trading.service`; the new final heartbeat will take effect the next time the service is explicitly started and then gracefully stopped.

### 13.44 Phase 3 Rebuild Preflight Report Artifact Completed

- Hardened the Phase 3 model-rebuild preflight so it leaves durable evidence for automation and operator handoff:
  - `scripts/run_phase3_rebuild_preflight.py` now writes dated JSON reports plus `latest.json` under `data/phase3_rebuild_preflight_reports` by default.
  - `report_artifacts.report_path` and `report_artifacts.latest_path` are embedded into the JSON before writing.
  - `--stdout-only` remains available for pure terminal inspection with no report file writes.
- Clarified artifact semantics:
  - `writes_artifacts=false` still means no model artifacts are written.
  - `report_artifacts` are audit/preflight report JSON files only; they do not train models, mutate DB rows, or promote anything.
- This gives Phase I/Stage I a stable readiness evidence file before any future confirmed rebuild on the new model server.

Verification:

- Phase 3 rebuild preflight/readiness regression: `10 passed`.
- Python compile check passed for `scripts/run_phase3_rebuild_preflight.py`, `tests/test_phase3_rebuild_preflight.py`, and `services/phase3_rebuild_readiness.py`.
- `git diff --check` passed.

Safety boundary:

- Report artifact writing only.
- No DB mutation, no repair apply, no historical deletion, no model artifact write, no model training, no order submission, no close execution, no trading-service startup.
- Confirmed rebuild commands remain suggestions until an operator explicitly runs them with `--persist-artifact --confirm-phase3-rebuild` after blockers are clear.

### 13.45 Phase 3 Rebuild Preflight Systemd Timer Completed

- Added a repeatable online timer installer for Phase 3 rebuild readiness:
  - file: `scripts/install_phase3_rebuild_preflight_timer.py`.
  - service: `bb-phase3-rebuild-preflight.service`.
  - timer: `bb-phase3-rebuild-preflight.timer`.
  - default schedule: daily `00:40` with `Persistent=true` and `RandomizedDelaySec=300`.
- The service mirrors the safe OKX daily report pattern:
  - runs as `User=bb` / `Group=bb`.
  - uses `WorkingDirectory=/data/bb/app`.
  - loads `EnvironmentFile=-/data/bb/app/.env` and `EnvironmentFile=/etc/bb/bb-runtime.env` through systemd instead of hand-sourcing `.env`.
  - executes `scripts/run_phase3_rebuild_preflight.py --json-indent 0`.
- This fixes the operational problem found during manual online validation:
  - shell-sourcing `.env` can misparse app env values and make PostgreSQL fall back to peer auth.
  - systemd `EnvironmentFile` is the correct online execution mode and avoids false `Peer authentication failed` preflight blockers.

Verification:

- Phase 3 rebuild preflight/readiness/timer regression: `12 passed`.
- Python compile check passed for `scripts/run_phase3_rebuild_preflight.py`, `scripts/install_phase3_rebuild_preflight_timer.py`, `tests/test_phase3_rebuild_preflight.py`, `tests/test_phase3_rebuild_preflight_timer.py`, and `services/phase3_rebuild_readiness.py`.
- `git diff --check` passed.
- Online deployment and run-now verification completed:
  - `bb-phase3-rebuild-preflight.timer=active`.
  - `bb-phase3-rebuild-preflight.service` exited `status=0/SUCCESS`.
  - latest report path: `/data/bb/app/data/phase3_rebuild_preflight_reports/latest.json`.
  - latest report `status=blocked`, `readiness.status=blocked`.
  - blockers: `shadow_sample_floor_not_met`, `trade_sample_floor_not_met`.
  - warnings: `legacy_or_untrusted_artifacts_retired_before_rebuild`, `preflight_only_no_artifact_write_requested`.
  - `collection_errors={}`; this confirms systemd environment loading fixed the false peer-auth failure seen in manual shell-source validation.
  - `writes_artifacts=false`, `mutates_database=false`, `starts_trading_service=false`.
  - `bb-paper-trading.service=inactive`, `bb-dashboard.service=active`, `bb-okx-daily-reconciliation.timer=active`.

Safety boundary:

- Timer/service installation for a read-only report only.
- No DB mutation, no repair apply, no historical deletion, no model artifact write, no model training, no order submission, no close execution, no trading-service startup.
- The timer is allowed to produce JSON readiness reports; it must not run confirmed rebuild commands.

### 13.46 Phase 3 Quant Model-Server Readiness Gate Completed

- Added a dedicated read-only readiness gate for the new quant-only model server:
  - service: `services/phase3_model_server_readiness.py`.
  - report script: `scripts/run_phase3_model_server_readiness_audit.py`.
  - dashboard/system audit card: `phase3_model_server_readiness`.
  - system audit node: `model_server_readiness`.
- The gate separates three states that must not be mixed:
  - server resource/migration boundary: handled by `phase3_server_migration`.
  - model artifact/CUDA/GPU readiness: handled by `phase3_model_server_readiness.artifact_ready`.
  - serving endpoint/shadow-routing readiness: handled by `phase3_model_server_readiness.runtime_ready`.
- The gate is fail-closed and does not use fallback truth:
  - missing download/validation manifest blocks the gate.
  - CUDA unavailable, tiny CUDA tensor failure, or fewer than 8 GPUs blocks the gate.
  - every required model slot must be present in the validation manifest and pass validation.
  - LLM roles must remain shadow/candidate-only until service and promotion gates pass.
  - model-serving service/endpoint absence is reported as `artifact_ready_service_pending`, not as `ready`.
- Required Phase 3 artifact slots currently checked:
  - `timeseries_primary`.
  - `timeseries_challenger`.
  - `sentiment_primary`.
  - `llm_decision_maker`.
  - `llm_expert_pool`.
  - `llm_high_risk_review`.
- Real new-server read-only verification result:
  - host label from probe: `gpu-ser01`.
  - status: `artifact_ready_service_pending`.
  - `artifact_ready=true`.
  - `runtime_ready=false`.
  - `phase3_model_service_go_live_blocked=true`.
  - GPU count: `8`.
  - required slots ready: `6/6`.
  - active model services: `0`.
  - active endpoints: `0`.
  - blockers: `[]`.
  - warnings: `model_service_manifest_missing`, `model_services_not_running`, `model_endpoints_unavailable`, `gpu_runtime_idle`.
- Local report artifact generated:
  - `data/phase3_model_server_readiness_reports/latest.json`.
- Online deployment/update completed:
  - uploaded `services/phase3_model_server_readiness.py`, `scripts/run_phase3_model_server_readiness_audit.py`, `web_dashboard/api/system_audit.py`, related tests, and this master-control document.
  - appended the new model server SSH host key to `/data/bb/app/.ssh/known_hosts`.
  - updated the encrypted online model-server SSH settings from the old server to the new server `61.133.218.214:62001` with password masked in output.
  - restarted only `bb-dashboard.service` to clear old monitor cache.
  - `bb-paper-trading.service` remained `inactive`.
  - online report artifact generated: `/data/bb/app/data/phase3_model_server_readiness_reports/latest.json`.

Verification:

- Phase 3 model-server readiness regression: `6 passed`.
- System audit integration focused regression: `4 passed`.
- Combined local readiness/system-audit regression: `65 passed`.
- Python compile check passed for `services/phase3_model_server_readiness.py`, `scripts/run_phase3_model_server_readiness_audit.py`, `web_dashboard/api/system_audit.py`, `tests/test_phase3_model_server_readiness.py`, and `tests/test_system_audit_api.py`.
- Real new-server read-only probe completed with no blockers and correctly reported service/runtime pending.
- Online verification after encrypted settings cutover:
  - `status=artifact_ready_service_pending`.
  - `artifact_ready=true`.
  - `runtime_ready=false`.
  - `gpu_count=8`.
  - required slots ready: `6/6`.
  - active model services: `0`.
  - active endpoints: `0`.
  - blockers: `[]`.
  - warnings: `model_service_manifest_missing`, `model_services_not_running`, `model_endpoints_unavailable`, `gpu_runtime_idle`.
  - `bb-dashboard.service=active`, dashboard HTTP `302`, and recent dashboard logs had no Traceback/Error/Exception/Failed.
- `git diff --check` passed.

Safety boundary:

- Read-only model-server audit/reporting only.
- No remote deletion, no DB mutation, no repair apply, no model artifact write, no model training, no model-serving service startup, no platform tunnel switch, no model routing change, no order submission, no close execution, no trading-service startup.
- The next implementation step is audited Phase 3 model-service installation under `/data/BB`, followed by shadow-only platform tunnel/routing verification.

### 13.47 Phase 3 Model-Server Readiness Timer Completed

- Added the online systemd timer installer for recurring model-server readiness evidence:
  - file: `scripts/install_phase3_model_server_readiness_timer.py`.
  - service: `bb-phase3-model-server-readiness.service`.
  - timer: `bb-phase3-model-server-readiness.timer`.
  - default schedule: daily `00:55` with `Persistent=true` and `RandomizedDelaySec=300`.
- The service mirrors the Phase 3 preflight and OKX daily report safety pattern:
  - runs as `User=bb` / `Group=bb`.
  - uses `WorkingDirectory=/data/bb/app`.
  - loads `EnvironmentFile=-/data/bb/app/.env`.
  - loads `EnvironmentFile=/etc/bb/bb-runtime.env` through systemd, so encrypted model-server settings and DB access match dashboard runtime.
  - executes `scripts/run_phase3_model_server_readiness_audit.py --json-indent 0`.
  - writes only report artifacts under `data/phase3_model_server_readiness_reports`.
- Added regression coverage:
  - `tests/test_phase3_model_server_readiness_timer.py`.
  - asserts the service uses `bb`, runtime env, the readiness report script, and does not reference `bb-paper-trading.service`.
  - dry-run mode must not connect to the remote server.

Verification:

- Phase 3 model-server readiness + timer regression: `8 passed`.
- Python compile check passed for `scripts/install_phase3_model_server_readiness_timer.py` and `tests/test_phase3_model_server_readiness_timer.py`.
- `git diff --check` passed.
- Online timer installation with `--run-now` completed:
  - `bb-phase3-model-server-readiness.timer=active`.
  - oneshot service exited `status=0/SUCCESS`.
  - latest report: `/data/bb/app/data/phase3_model_server_readiness_reports/latest.json`.
  - latest report status: `artifact_ready_service_pending`.
  - `artifact_ready=true`, `runtime_ready=false`, `gpu_count=8`, required slots `6/6`.
  - `bb-paper-trading.service=inactive`.
  - `bb-dashboard.service=active`.
  - dashboard HTTP `302`.

Safety boundary:

- Timer/service installation for a read-only model-server readiness report only.
- No DB mutation, no repair apply, no historical deletion, no model artifact write, no model training, no model-serving service startup, no platform tunnel switch, no model routing change, no order submission, no close execution, no trading-service startup.
- The next implementation step remains audited Phase 3 model-service installation under `/data/BB`, then shadow-only platform tunnel/routing verification.

### 13.48 Phase 3 Shadow Model Services Started and Readiness Gate Corrected

- Installed/started the first Phase 3 shadow-only LLM service layer on the new quant-only model server under `/data/BB`:
  - installer: `scripts/deploy_phase3_model_server_services.py`.
  - service manifest: `/data/BB/manifests/phase3_model_service_manifest.json`.
  - initial bootstrap finding: `bb-phase3-llm-decision.service` used GPU `0`, port `8000`, served model `qwen3-14b-trade`.
  - corrected target contract: `bb-phase3-llm-decision.service` must use GPU `0-1`, port `8000`, served model `qwen3-32b-trade`.
  - `bb-phase3-llm-expert.service`: GPU `2`, port `8003`, served model `qwen3-14b-expert-pool`.
  - `bb-phase3-llm-risk-review.service`: GPU `3`, port `8002`, served model `deepseek-r1-14b-risk`.
- Runtime policy is still shadow-only:
  - each vLLM service binds `127.0.0.1` only.
  - manifest keeps `shadow_only=true`, `live_routing_enabled=false`, and `can_start_trading=false`.
  - no service references `bb-paper-trading.service`.
  - no strategy, allocation, close, order, or live model-routing path was switched to these endpoints.
- Corrected two readiness-audit false negatives found during real model-server validation:
  - `core/remote_ssh.py` now honors explicit large `max_output_chars` requests above the default `20000` limit, capped by `MAX_REMOTE_OUTPUT_TEXT_LIMIT=200000`, so large JSON probes are not silently truncated.
  - `services/phase3_model_server_readiness.py` now parses `systemctl list-units` active/running columns by whitespace instead of fixed single-space substrings, so aligned systemd output is recognized correctly.
- Real new-server validation after service startup:
  - status: `ready`.
  - `artifact_ready=true`.
  - `runtime_ready=true`.
  - required slots ready: `6/6`.
  - manifest services ready: `3/3`.
  - active model services: `3`.
  - active endpoints: `3`.
  - GPU runtime processes: `3`.
  - blockers: `[]`.
  - warnings: `[]`.
- Important interpretation:
  - this means the model server's artifact/runtime gate is ready for the next shadow integration step.
  - it does not mean live trading or live LLM routing is approved.
  - overall Phase 3 production go-live still requires platform-side tunnel/routing verification, shadow-result evaluation, promotion policy checks, and explicit operator approval.

Verification:

- Phase 3 model-server readiness/deploy/SSH-output regression: `14 passed`.
- Python compile check passed for `core/remote_ssh.py`, `services/phase3_model_server_readiness.py`, `scripts/deploy_phase3_model_server_services.py`, `tests/test_phase3_model_server_readiness.py`, and `tests/test_remote_ssh_output_limit.py`.
- Real model-server `/v1/models` probes returned the expected served-model names on ports `8000`, `8002`, and `8003`.
- Online deployment/update completed:
  - uploaded `core/remote_ssh.py`, `services/phase3_model_server_readiness.py`, `scripts/deploy_phase3_model_server_services.py`, related tests, and this master-control document.
  - online Python compile check passed for the uploaded files.
  - online focused pytest was skipped because `/data/bb/app/.venv` does not have `pytest`; local focused regression is the verification source for tests.
  - restarted only `bb-dashboard.service` to load the corrected readiness audit code.
  - triggered `bb-phase3-model-server-readiness.service` once.
  - latest online report: `/data/bb/app/data/phase3_model_server_readiness_reports/latest.json`.
  - latest online report `status=ready`, `artifact_ready=true`, `runtime_ready=true`, required slots `6/6`, manifest services `3/3`, active endpoints `3`, blockers `0`, warnings `0`.
  - `bb-dashboard.service=active`, `bb-phase3-model-server-readiness.timer=active`, `bb-paper-trading.service=inactive`.

Safety boundary:

- Shadow-only model-serving service installation/startup on the quant model server.
- No DB mutation, no repair apply, no historical deletion, no model training, no platform tunnel switch, no model routing change, no order submission, no close execution, no trading-service startup.
- `bb-paper-trading.service` must remain inactive until a separate go-live gate is passed and the operator explicitly approves trading restart.

### 13.49 Phase 3 Platform Tunnel Cutover to New Quant Server Completed

- Found and fixed a critical tunnel drift:
  - online `bb-model-tunnels.service` was still a long-running SSH process connected to the old model-server session.
  - ports `18000` and `18002` returned the expected model names, but their model roots were `/data/trade_models/...`, proving they were not serving the new Phase 3 `/data/BB` runtime.
  - this was a real source of future confusion: names looked correct, but traffic still went to the old layout.
- Updated the platform tunnel contract to the Phase 3 quant-only topology:
  - `18000 -> model server 127.0.0.1:8000 -> qwen3-32b-trade`.
  - `18001 -> model server 127.0.0.1:8101 -> phase3_quant_api health/inventory`.
  - `18002 -> model server 127.0.0.1:8002 -> deepseek-r1-14b-risk`.
  - `18003 -> model server 127.0.0.1:8003 -> qwen3-14b-expert-pool`.
- Updated platform runtime generation:
  - `AI_MODELS` now routes `decision_maker` to `18000`.
  - expert slots route to `18003` expert pool by default.
  - high-risk review remains independent on `18002`.
  - legacy local-ai-tools is disabled by default in generated runtime env.
  - copying the old `/data/trade_ai/local_ai_tools.env` key now requires an explicit legacy flag.
- Updated monitoring and dashboard contracts:
  - remote monitor probe checks all three vLLM endpoints `8000/8002/8003`.
  - remote monitor checks `8101 /health` as `phase3_quant_api`, not old local-ai-tools `8001`.
  - platform self-check expects `phase3_quant_api` on `18001` and `qwen3-14b-expert-pool` on `18003`.
  - dashboard no longer advertises old public `21840/21841/21842` model endpoints.
- Updated maintenance scripts:
  - `scripts/check_server_model_status.py` now checks `/data/BB`, Phase 3 service names, and `8101`.
  - `scripts/inspect_server_ai_services.py` now inspects `/data/BB` scripts/manifests and Phase 3 services.

Verification:

- Local tunnel/monitor/dashboard regression: `116 passed`.
- Python compile check passed for `scripts/start_online_model_tunnels.py`, `scripts/sync_to_online_server.py`, `scripts/check_server_model_status.py`, `scripts/inspect_server_ai_services.py`, `core/server_monitor_probe.py`, `services/server_monitor_status.py`, `web_dashboard/api/system_health.py`, and `web_dashboard/api/system_audit.py`.
- Online deployment/update completed:
  - uploaded tunnel, monitor, dashboard, maintenance-script, and focused-test changes.
  - online Python compile check passed for uploaded runtime files.
  - restarted `bb-model-tunnels.service` to force a fresh SSH connection to the new encrypted model-server settings.
  - restarted only `bb-dashboard.service` to load updated monitor/dashboard code.
  - online tunnel probe passed:
    - initial bootstrap probe returned `qwen3-14b-trade` rooted under `/data/BB/models/llm_decision_maker`; this is superseded by the corrected `qwen3-32b-trade` decision-maker contract in 13.50.
    - `18002` returns `deepseek-r1-14b-risk` rooted under `/data/BB/models/llm_high_risk_review`.
    - `18003` returns `qwen3-14b-expert-pool` rooted under `/data/BB/models/llm_expert_pool`.
    - `18001` returns `phase3_quant_api` rooted under `/data/BB`.
  - `bb-model-tunnels.service=active`, `bb-dashboard.service=active`, `bb-paper-trading.service=inactive`.

Safety boundary:

- Platform tunnel and monitoring cutover only.
- No DB mutation, no repair apply, no historical deletion, no model training, no order submission, no close execution, no trading-service startup.
- Runtime model endpoints are now reachable from the platform, but live trading and live model-routing still require the next shadow/canary evaluation gates and explicit operator approval.

### 13.50 Phase 3 LLM Slot Diversity Correction Completed

User correction:

- The Phase 3 GPU allocation plan is not changed:
  - GPU `0-1`: final decision maker.
  - GPU `2`: expert-pool main LLM.
  - GPU `3`: high-risk review.
  - GPU `4-5`: time-series primary/challenger.
  - GPU `6-7`: training, walk-forward, hyperparameter search, challenger evaluation.
- The mistake was the bootstrap model assignment, not the GPU topology:
  - `llm_decision_maker` and `llm_expert_pool` were both backed by `Qwen/Qwen3-14B-AWQ`.
  - This wastes an expensive GPU and creates false model diversity.
  - Duplicate 14B Qwen slots may only be tolerated as a temporary shadow bootstrap while wiring is tested; they must not pass canary/live/final promotion gates.

Corrected model contract:

- `llm_decision_maker`:
  - service: `bb-phase3-llm-decision.service`.
  - port: `8000`, platform tunnel `18000`.
  - model: `qwen3-32b-trade`.
  - artifact path: `/data/BB/models/llm_decision_maker/Qwen--Qwen3-32B-AWQ`.
  - GPU: `0,1`.
  - purpose: stronger final trade synthesis, conflict arbitration, and action/no-action confirmation.
- `llm_expert_pool`:
  - service: `bb-phase3-llm-expert.service`.
  - port: `8003`, platform tunnel `18003`.
  - model: `qwen3-14b-expert-pool`.
  - artifact path: `/data/BB/models/llm_expert_pool/Qwen--Qwen3-14B-AWQ`.
  - GPU: `2`.
  - purpose: faster specialist prompt pool for trend, profit-quality, short-timeseries, position-exit, and anomaly-risk opinions.
- `llm_high_risk_review`:
  - service: `bb-phase3-llm-risk-review.service`.
  - port: `8002`, platform tunnel `18002`.
  - model: `deepseek-r1-14b-risk`.
  - GPU: `3`.
  - purpose: independent high-risk veto and exception review.

Machine-checkable gates added/updated:

- `services/phase3_model_server_readiness.py` now blocks canary/live readiness with `llm_role_diversity_missing` if `llm_decision_maker` and `llm_expert_pool` use the same base model and no audited adapter/fine-tune/specialization evidence exists.
- Deployment contract updated in `scripts/deploy_phase3_model_server_services.py`:
  - decision service now serves `qwen3-32b-trade` from the 32B artifact and uses GPUs `0,1`.
  - expert service remains `qwen3-14b-expert-pool` on GPU `2`.
- Platform runtime generation updated in `scripts/sync_to_online_server.py`:
  - `decision_maker -> http://127.0.0.1:18000/v1 -> qwen3-32b-trade`.
  - expert slots stay on `http://127.0.0.1:18003/v1 -> qwen3-14b-expert-pool`.
- Self-check/UI contracts updated:
  - `web_dashboard/api/system_health.py` expects `qwen3-32b-trade` on `18000`.
  - dashboard display maps `qwen3-32b-trade` to platform loopback `18000`.

Required next operational action before model-service promotion:

- Ensure `/data/BB/models/llm_decision_maker/Qwen--Qwen3-32B-AWQ` exists and passes validation on the new model server.
- Reinstall/restart only Phase 3 model services after artifact validation.
- Re-run `phase3_model_server_readiness` and shadow LLM probe.
- Keep `bb-paper-trading.service` inactive until all shadow/canary gates pass and operator explicitly approves trading restart.

### 13.51 Phase 3 Optimal Quant-Model Direction Locked

Correction after model-strategy review:

- `qwen3-32b-trade + qwen3-14b-expert-pool` is the immediate corrected deployment, not the final definition of "best possible".
- Generic finance LLMs such as FinGPT/InvestLM/FinMA-style models are useful for financial-text understanding, sentiment/RAG, and fine-tuning recipes, but they are not automatically better at OKX crypto perpetual short-horizon entry/exit decisions.
- The strongest Phase 3 direction is therefore not "replace Qwen with a generic finance chat model"; it is:
  - keep a stronger general reasoning LLM for final trade synthesis.
  - keep an independent reasoning LLM for high-risk veto.
  - build the expert-pool slot into a BB-specific quant expert using clean OKX trade facts, failed-order root causes, position-management outcomes, time-series forecasts, news/sentiment, and strategy-learning data.
  - keep professional time-series foundation/forecast models for actual short-horizon sequence prediction.

Final target architecture:

- GPU `0-1`: `qwen3-32b-trade` decision maker.
- GPU `2`: `BB-FinQuant-Expert-14B` expert pool.
  - initial carrier/base: `Qwen/Qwen3-14B-AWQ`.
  - final promotion requires audited BB quant specialization evidence such as LoRA/fine-tune manifest, RAG corpus manifest, or training artifact.
  - a plain base `Qwen3-14B-AWQ` expert pool may run shadow only and must be reported as `finquant_expert_specialization_pending`.
- GPU `3`: `deepseek-r1-14b-risk` high-risk independent review/veto.
- GPU `4-5`: Chronos/TimesFM-style time-series primary and challenger.
- GPU `6-7`: training, walk-forward, hyperparameter search, challenger evaluation, and expert-pool specialization.

Machine gates:

- `services/phase3_model_server_readiness.py` blocks duplicate decision/expert base models with `llm_role_diversity_missing`.
- The same readiness gate now emits `finquant_expert_specialization_pending` when GPU `2` is still only a base expert model with no BB quant specialization evidence.
- This warning is acceptable for shadow integration but not acceptable as the final "expensive server fully utilized" state.

Operational execution started:

- stopped the old duplicate `bb-phase3-llm-decision.service`.
- removed `/data/BB/models/llm_decision_maker/Qwen--Qwen3-14B-AWQ`.
- started background download of `Qwen/Qwen3-32B-AWQ` into `/data/BB/models/llm_decision_maker/Qwen--Qwen3-32B-AWQ`.
- download log: `/data/BB/logs/downloads/download_phase3_decision_32b.log`.
- `bb-paper-trading.service` remained inactive.

### 13.52 Phase 3 32B Decision Slot, Inventory Truth, and Shadow Probe Closed

Completed correction:

- The duplicate 14B decision/expert mistake is now fully corrected on both model server and platform:
  - `llm_decision_maker`: `Qwen/Qwen3-32B-AWQ`, served as `qwen3-32b-trade`, GPU `0,1`, platform tunnel `18000`.
  - `llm_expert_pool`: `Qwen/Qwen3-14B-AWQ`, served as `qwen3-14b-expert-pool`, GPU `2`, platform tunnel `18003`.
  - `llm_high_risk_review`: `deepseek-r1-14b-risk`, GPU `3`, platform tunnel `18002`.
- Removed the old duplicate decision artifact:
  - `/data/BB/models/llm_decision_maker/Qwen--Qwen3-14B-AWQ`.
- Downloaded and validated the new decision artifact:
  - `/data/BB/models/llm_decision_maker/Qwen--Qwen3-32B-AWQ`.
- Fixed the inventory truth drift that caused `8101 /health` to still report old 14B:
  - canonical manifests under `/data/BB/manifests` were already 32B.
  - `phase3_quant_api` was reading `/data/BB/reports/inventory/*latest.json`, which still contained old 14B decision rows.
  - added `scripts/sync_phase3_model_inventory.py` to synchronize `/data/BB/reports/inventory` from canonical manifests.
  - updated `scripts/download_phase3_decision_model.py` so future 32B downloads update both canonical manifests and report inventory manifests.
- Added a machine-checkable gate:
  - `services/phase3_model_server_readiness.py` now blocks `llm_candidate_policy_mismatch` when policy `llm_candidates` disagree with validated slot models.

Live platform status after correction:

- Model server readiness:
  - status: `ready`.
  - `artifact_ready=true`.
  - `runtime_ready=true`.
  - blockers: `[]`.
  - required slots ready: `6/6`.
  - manifest services ready: `3/3`.
  - active endpoints: `3`.
  - GPU count: `8`.
- Platform tunnel probes:
  - `18000 -> qwen3-32b-trade`, root `/data/BB/models/llm_decision_maker/Qwen--Qwen3-32B-AWQ`.
  - `18001 -> phase3_quant_api`, health now reports `llm_decision_maker=Qwen/Qwen3-32B-AWQ`.
  - `18002 -> deepseek-r1-14b-risk`.
  - `18003 -> qwen3-14b-expert-pool`.
- Online shadow LLM probe:
  - `decision_maker`: ok, JSON available, no thinking tag, latency about `339ms`.
  - `expert_pool`: ok, JSON available, no thinking tag, latency about `494ms`.
  - `high_risk_review`: ok, JSON available, DeepSeek raw output may include `<think>`, latency about `2293ms`.
  - probe status: `ready`, `3/3`.
- High-risk review contract correction:
  - `scripts/run_phase3_shadow_llm_probe.py` now gives DeepSeek the same token headroom as the real high-risk review path.
  - `services/high_risk_review_service.py` records `raw_has_think_tag` and `reasoning_stripped` in attempt metadata.
  - final execution contract remains strict: high-risk review must yield extractable JSON; otherwise it blocks required high-risk entries.

Verification:

- Focused local regression: `84 passed`.
- Earlier model-server maintenance/readiness regression: `60 passed`.
- High-risk review and shadow probe regression: `24 passed`.
- Python compile checks passed for changed model inventory, shadow probe, readiness, and high-risk review files.
- Online Python compile checks passed for uploaded files.
- Online platform code uploaded without restarting `bb-paper-trading.service`.
- `bb-paper-trading.service` remained `inactive`.
- `bb-dashboard.service` and `bb-model-tunnels.service` remained `active`.

Remaining planned warning:

- `finquant_expert_specialization_pending` remains expected.
- This is not a duplicate-model bug. It means GPU `2` is currently a base `Qwen3-14B-AWQ` expert-pool carrier and has not yet been specialized into `BB-FinQuant-Expert-14B`.
- The next model-server work should focus on BB-specific quant expert specialization using clean OKX trade facts, outcome labels, failed-order root causes, position management outcomes, and strategy-learning data.

Safety boundary:

- No order submission.
- No close execution.
- No trading-service startup.
- No live model-routing promotion.
- No DB mutation.
- No historical deletion.
- Current state is shadow-ready infrastructure only.

### 13.53 OKX Daily Reconciliation Operational Gates Completed

Completed hardening:

- `scripts/run_okx_daily_reconciliation_report.py` now emits machine-readable operational gates in every daily OKX reconciliation report:
  - `can_open_new_entries`
  - `can_refresh_training`
  - `requires_attention`
  - `operational_gates.entry_blocked`
  - `operational_gates.training_blocked`
  - `operational_gates.entry_blockers`
  - `operational_gates.training_blockers`
  - `operational_gates.attention_items`
- This turns OKX/backoffice mismatch visibility into explicit execution and training decisions:
  - runtime-only heartbeat/inactive blocks new entries but does not block clean training refresh.
  - real OKX/native fact mismatch, missing position links, failed OKX pull, unresolved audit card, or manual-review data issue blocks both new entries and training refresh.
  - observing-only warnings remain timer-success eligible so systemd does not page on intentional paused trading.
- The report remains read-only:
  - no DB mutation.
  - no historical repair apply.
  - no order submit/close.
  - no trading service startup.

Verification:

- Daily report focused regression: `7 passed`.
- System-audit OKX/ledger focused regression: `5 passed, 54 deselected`.
- Python compile check passed for `scripts/run_okx_daily_reconciliation_report.py`.

Safety boundary:

- No trading behavior changed.
- No model routing changed.
- No historical deletion or repair apply path was added.
- `bb-paper-trading.service` must remain inactive until the operator explicitly approves a restart/go-live gate.

### 13.54 OKX Daily Report Gates Surfaced In Dashboard Completed

Completed hardening:

- System audit now reads `data/okx_daily_reconciliation_reports/latest.json` as a read-only latest-report artifact.
- `okx_trade_fact_integrity` details now include `daily_reconciliation_report` with:
  - report freshness and age.
  - `can_open_new_entries`.
  - `can_refresh_training`.
  - `requires_attention`.
  - entry/training blockers.
  - issue-ledger summary.
- Dashboard OKX audit details now render the latest daily report gates, entry blockers, and training blockers.
- Stale or missing latest report is treated as a warning signal for observability, not as permission to repair or trade.

Verification:

- OKX/system-audit focused regression: `8 passed, 52 deselected`.
- OKX daily report + dashboard contract focused regression: `10 passed, 55 deselected`.
- Python compile check passed for `web_dashboard/api/system_audit.py` and `scripts/run_okx_daily_reconciliation_report.py`.

Online validation:

- `bb-okx-daily-reconciliation.service` executed successfully under the real `bb` systemd environment.
- Latest online report showed:
  - `status=warning`.
  - `issue_ledger.unresolved=0`.
  - `can_open_new_entries=false`.
  - `can_refresh_training=true`.
  - blocker is only `trading_runtime_heartbeat_stale`, expected while trading service is intentionally stopped.
- `bb-paper-trading.service` remained `inactive`.

Safety boundary:

- Dashboard/system-audit visibility only.
- No order submission.
- No close execution.
- No trading-service startup.
- No DB mutation.
- No historical deletion or repair apply path.

### 13.55 Training Refresh Blocked By OKX Daily Gate Completed

Completed hardening:

- Dashboard training-governance refresh now checks the latest OKX daily reconciliation report before any training refresh work starts.
- If the latest report is missing, stale, requires attention, or has `can_refresh_training=false`, the endpoint returns `status=blocked` and does not run:
  - dirty-shadow quarantine.
  - local ML retraining.
  - local AI tools retraining.
  - vector-memory reindex.
- If the latest report allows clean-view training refresh, the existing clean training view flow continues unchanged and records `okx_daily_reconciliation_gate` in the response.

Verification:

- Training governance refresh focused regression: `3 passed, 14 deselected`.
- OKX/system-audit/daily-report focused regression: `12 passed, 55 deselected`.
- Python compile check passed for `web_dashboard/api/data_collection.py`, `web_dashboard/api/system_audit.py`, and `scripts/run_okx_daily_reconciliation_report.py`.

Safety boundary:

- No DB repair or deletion.
- No order submission or close execution.
- No trading-service startup.
- This only prevents retraining/reindexing from running when OKX daily reconciliation does not allow clean-view training refresh.

### 13.56 OKX Daily Training Gate Unified Across Refresh, Auto-Train, and CLI Completed

Completed hardening:

- Added `services/okx_training_gate.py` as the shared read-only source for `data/okx_daily_reconciliation_reports/latest.json`.
- The same gate now protects:
  - dashboard training-governance refresh.
  - trading-service local AI tools auto-training.
  - `scripts/train_local_ai_tools_models.py`.
  - `scripts/train_ml_signal_model.py` artifact persistence.
- Gate behavior:
  - missing/stale latest report blocks training refresh.
  - `requires_attention=true` blocks training refresh.
  - `can_refresh_training=false` blocks training refresh.
  - runtime-only entry blockers can still allow clean-view training if the daily report says training is allowed.

Verification:

- Data collection API regression: `17 passed`.
- OKX training gate / auto-train / CLI focused regression: `8 passed, 183 deselected`.
- Earlier combined focused regression: `11 passed, 197 deselected`.
- Python compile check passed for `services/okx_training_gate.py`, `web_dashboard/api/data_collection.py`, `services/trading_service.py`, `scripts/train_local_ai_tools_models.py`, and `scripts/train_ml_signal_model.py`.

Safety boundary:

- No order submission.
- No close execution.
- No trading-service startup.
- No historical deletion or repair apply path.
- No artifact persistence unless existing explicit confirmation gates and OKX daily training gate both pass.

### 13.57 OKX Native Order Confirmation Chain Completed

Completed hardening:

- Market-order confirmation now reads OKX native order detail through `privateGetTradeOrder` using `instId + ordId`.
- Entry and exit refresh paths no longer call `ccxt.fetch_order`.
- If native order detail is unavailable after submit, the system does not upgrade a CCXT-reported filled/closed order into a local filled fact; it keeps the order in tracking state until OKX native order/position/fill sync confirms it.
- Final `ExecutionResult.order_id` and `exchange_order_id` are pinned to OKX `ordId` / `clOrdId`.
- `OKXRestClient.fetch_order()` now also uses OKX native `privateGetTradeOrder`, so future data-service callers do not reintroduce CCXT symbol-alias order facts.
- Added regression coverage for the SPK/SAHARA class of alias pollution:
  - CCXT create-order response can report the wrong alias symbol.
  - OKX native order detail returns `SPK-USDT-SWAP`.
  - final execution result must persist/report `SPK/USDT`.

Verification:

- OKX pending/execution safety/REST native-order focused regression: `45 passed`.
- Python compile check passed for `executor/okx_executor.py`, `data_feed/okx_rest_client.py`, `tests/test_okx_pending_orders.py`, `tests/test_executor_error_safety.py`, and `tests/test_okx_rest_client_symbols.py`.
- Source search found no `ccxt.fetch_order` and no `_ccxt_call("fetch_order"` in project code.

Safety boundary:

- No order submission during verification.
- No close execution.
- No trading-service startup.
- No DB mutation.
- No historical deletion or repair apply path.
- This only hardens how submitted/queried order facts are confirmed and normalized.

### 13.58 OKX Native Cancel Order Chain Completed

Completed hardening:

- All core OKX cancel paths now use OKX native `privatePostTradeCancelOrder` with `instId + ordId`.
- Removed core `ccxt.cancel_order` usage from:
  - stale exit-order replacement.
  - public executor `cancel_order()`.
  - leverage-retry stale entry-order cleanup.
- `OKXRestClient.cancel_order()` now also calls `privatePostTradeCancelOrder`.
- Native cancel responses are checked through top-level `code` and per-row `sCode`; non-zero OKX native cancel responses fail closed and are logged as cancel errors.
- Tests explicitly forbid the old CCXT `cancel_order` path and verify native cancel parameters for:
  - direct executor cancel.
  - leverage-retry cleanup.
  - REST client cancel.

Verification:

- OKX pending/execution safety/REST focused regression: `48 passed`.
- Python compile check passed for `executor/okx_executor.py`, `data_feed/okx_rest_client.py`, `tests/test_okx_pending_orders.py`, `tests/test_executor_error_safety.py`, and `tests/test_okx_rest_client_symbols.py`.
- Source search found no `ccxt.cancel_order` and no `_ccxt_call("cancel_order"` in executor/data-feed/services paths.

Safety boundary:

- No order submission during verification.
- No close execution.
- No trading-service startup.
- No DB mutation.
- No historical deletion or repair apply path.
- This only hardens cancellation of already-existing OKX orders so replacement/cleanup cannot be routed through CCXT symbol aliases.

### 13.59 OKX Native Trade-Fact Persistence Gate Completed

Completed hardening:

- Added strict OKX-instId parsing mode to `core/symbols.py` via `okx_inst_id_from_payload(..., include_fallback=False)`.
- Order logging now resolves persisted `orders.symbol` from real OKX `instId` before looking at `canonical_exchange_symbol` or display symbol.
- Position execution persistence now resolves persisted `positions.symbol` from real OKX `instId` before looking at `canonical_exchange_symbol` or display symbol.
- This closes the final write-boundary gap where execution might be OKX-native but a stale/wrong display alias could still override the stored trade fact.
- Added SPK/SAHARA regression tests for both order rows and position rows:
  - raw payload contains `info.instId=SPK-USDT-SWAP`.
  - raw payload also contains wrong alias/display values such as `SAHARA/USDT`.
  - persisted symbol must be `SPK/USDT`.

Verification:

- Core symbol + order-log + position-persistence + OKX execution focused regression: `69 passed`.
- Python compile check passed for `core/symbols.py`, `services/trade_order_log_service.py`, and `services/position_execution_persistence.py`.
- Diff check passed for the touched persistence and symbol files.

Safety boundary:

- No order submission during verification.
- No close execution.
- No trading-service startup.
- No DB mutation.
- No historical deletion or repair apply path.
- This only changes how already-confirmed execution facts are normalized before local persistence.

### 13.60 OKX Native Sync-Service Write Boundary Completed

Completed hardening:

- `services/sync_service.py` now normalizes auto-sync write payloads through OKX `instId` before persisting or reporting symbols.
- Missing local position creation from OKX current positions now stores `positions.symbol` from OKX `instId`, not from a stale display alias.
- Partial quantity-reduction close-history creation now stores:
  - closed position `symbol` from close fill `instId`.
  - generated close order `symbol` from close fill `instId`.
  - reconciliation result/log symbol from the same OKX-native symbol.
- This prevents the continuous OKX reconciliation loop from reintroducing SPK/SAHARA-style alias pollution after the execution and persistence gates have already cleaned it.

Verification:

- Sync-service focused regression: `4 passed`.
- Core symbol + persistence + OKX execution + sync boundary regression: `73 passed`.
- Python compile check passed for `services/sync_service.py` and `tests/test_trading_service_boundaries.py`.
- Diff check passed for sync-service touched files.

Safety boundary:

- No order submission during verification.
- No close execution.
- No trading-service startup.
- No DB mutation.
- No historical deletion or repair apply path.
- This only changes how authoritative OKX sync facts are normalized when the existing sync boundary writes local records.

### 13.61 OKX Native Historical Repair Evidence Gate Completed

Completed hardening:

- Historical repair paths that backfill position/order links now carry OKX native `instId` as a first-class evidence field.
- `scripts/repair_okx_native_full_close_fills.py` keeps raw OKX `instId` inside each fill group and rejects fill groups that do not include a native instrument id.
- `scripts/repair_missing_position_links_from_okx_fills.py` now:
  - builds plans from OKX fill `instId`, not from local display symbol.
  - queries fills by existing `position.okx_inst_id` when present.
  - refuses apply when a plan has no native `okx_inst_id`.
  - refuses apply when plan `okx_inst_id` conflicts with existing `position.okx_inst_id`.
  - writes missing local order rows with symbol derived from OKX `instId`.
- `scripts/repair_okx_position_fact_links.py` now respects existing `position.okx_inst_id` when looking for matching local orders, preventing SPK/SAHARA-style symbol/time/price matches from linking the wrong order.
- `services/order_position_reconciliation.py` now extracts native `instId` from entry/close decision execution payloads where available and rejects missing-closed-position repair plans when entry and close native instruments conflict.
- Added regressions for:
  - missing native `instId` plans being skipped.
  - conflicting native `instId` plans being skipped.
  - existing `position.okx_inst_id=SPK-USDT-SWAP` preventing a SAHARA order from being linked.
  - entry/close order-pair repair being rejected when native `instId` evidence conflicts.

Verification:

- Historical repair focused regression: `16 passed`.
- OKX/persistence/sync expanded focused regression: `187 passed`.
- Python compile check passed for `services/order_position_reconciliation.py`, `scripts/repair_missing_position_links_from_okx_fills.py`, `scripts/repair_okx_position_fact_links.py`, and `scripts/repair_okx_native_full_close_fills.py`.
- Diff check produced only existing Windows line-ending warnings and no whitespace-error output.

Safety boundary:

- No order submission during verification.
- No close execution.
- No trading-service startup.
- No DB mutation during verification.
- No historical deletion.
- No repair apply was executed.
- This only tightens evidence gates for explicit historical repair entrypoints so old dirty rows cannot be repaired or linked through symbol alias, time, price, and quantity alone.

### 13.62 OKX Authoritative Audit Position Key Nativeized

Completed hardening:

- `services/okx_authoritative_sync.py` now compares local open positions to OKX open positions using local `position.okx_inst_id` first.
- If a local position has `okx_inst_id=SPK-USDT-SWAP`, the read-only authoritative audit key is `SPK/USDT + side` even if the legacy local `positions.symbol` is polluted as `SAHARA/USDT`.
- OKX fill pull symbol collection now also uses local `position.okx_inst_id` before local display symbol, so bounded fill-history audits are requested against the native instrument when available.
- This prevents system audit / OKX daily report / training gate from falsely reporting both:
  - `okx_open_position_missing_locally`
  - `local_open_position_missing_on_okx`
  when the only mismatch is a dirty local display symbol and the OKX-native `okx_inst_id` is already correct.
- Added regression coverage where local `symbol=SAHARA/USDT` but `okx_inst_id=SPK-USDT-SWAP`, while OKX reports `SPK-USDT-SWAP`; the authoritative audit must not report open-position missing issues.

Verification:

- OKX authoritative sync focused regression: `4 passed`.
- OKX authoritative sync + daily reconciliation + system audit + trade-fact integrity + trading-boundary regression: `239 passed`.
- Python compile check passed for `services/okx_authoritative_sync.py` and `tests/test_okx_authoritative_sync.py`.

Safety boundary:

- Read-only audit behavior only.
- No order submission.
- No close execution.
- No trading-service startup.
- No DB mutation.
- No historical deletion.
- No repair apply was executed.
- This does not hide real OKX/local position mismatch; it only prevents a known dirty local display symbol from overriding a valid OKX-native `okx_inst_id`.

### 13.63 OKX Authoritative Audit Order Fill Pull Nativeized

Completed hardening:

- `services/okx_authoritative_sync.py` now loads the local order's linked `AIDecision.raw_llm_response` for read-only audit context.
- When an order has native OKX `instId` evidence in its execution payload, the authoritative audit uses that `instId` to:
  - collect OKX fill-history query symbols.
  - display/report local order audit symbol.
  - avoid classifying the order as missing from bounded OKX fills when only the local `orders.symbol` display value is dirty.
- The extraction checks:
  - top-level decision raw payload.
  - `execution_result`.
  - `execution_result.raw_response`.
- Added regression where local `orders.symbol=SAHARA/USDT`, but the execution payload contains `info.instId=SPK-USDT-SWAP` and OKX fills return `SPK-USDT-SWAP`; the authoritative audit must not emit:
  - `local_order_not_found_in_recent_okx_fills`
  - `okx_fill_missing_local_order`

Verification:

- OKX authoritative sync focused regression: `5 passed`.
- OKX authoritative sync + daily reconciliation + system audit + trade-fact integrity + trading-boundary regression: `240 passed`.
- Python compile check passed for `services/okx_authoritative_sync.py` and `tests/test_okx_authoritative_sync.py`.

Safety boundary:

- Read-only audit behavior only.
- No order submission.
- No close execution.
- No trading-service startup.
- No DB mutation.
- No historical deletion.
- No repair apply was executed.
- This does not trust a display alias. It only uses already-stored OKX-native execution evidence to ask OKX for the correct instrument's fills.

### 13.64 OKX Trade-Fact Integrity Symbol Alignment Nativeized

Completed hardening:

- `services/okx_trade_fact_integrity.py` now audits order-position symbol alignment using OKX-native evidence first:
  - order side: `execution_result.raw_response.info.instId` / `instId` / `okx_inst_id` / `okx_symbol`.
  - position side: `position.okx_inst_id`.
  - local display `orders.symbol` / `positions.symbol` is only used when no native evidence exists.
- `_related_positions_for_order()` now includes the order's OKX-native symbol in candidate matching and compares candidate positions through `position.okx_inst_id` when present.
- This prevents old dirty display symbols such as `SAHARA/USDT` from creating false `order_position_symbol_mismatch` when both OKX-native facts prove the real contract is `SPK-USDT-SWAP`.
- Genuine native conflicts remain critical:
  - order raw `instId=SPK-USDT-SWAP`.
  - position `okx_inst_id=SAHARA-USDT-SWAP`.
  - audit still emits `order_position_symbol_mismatch`.
- Existing dirty local display symbols remain visible through `symbol_alias_mismatch` / `position_okx_inst_id_symbol_mismatch`; this change only stops display aliases from overriding native order-position truth.

Verification:

- Trade-fact integrity focused regression: `13 passed`.
- OKX authoritative sync + daily reconciliation + system audit + trade-fact integrity + trading-boundary regression: `242 passed`.
- Python compile check passed for `services/okx_trade_fact_integrity.py` and `tests/test_okx_trade_fact_integrity.py`.
- Online precise deployment uploaded only:
  - `services/okx_trade_fact_integrity.py`
  - `tests/test_okx_trade_fact_integrity.py`
- Online Python compile check passed.
- Online service verification:
  - `bb-dashboard.service=active`.
  - Dashboard HTTP `302`.
  - `bb-paper-trading.service=inactive`.

Safety boundary:

- Read-only audit behavior only.
- No order submission.
- No close execution.
- No trading-service startup.
- No DB mutation.
- No historical deletion.
- No repair apply was executed.
- This completes the next OKX-native audit-layer closure after 13.62 and 13.63, so order, position, fill-pull, and trade-fact integrity checks all prefer OKX-native instrument evidence over local display aliases.

### 13.65 Local AI Tools V2 Shadow Contract Closed

Completed hardening:

- `scripts/deploy_local_ai_tools_service.py` now gives every generated local quant-tool response a Phase 3 shadow contract:
  - `primary_model`.
  - `challenger_model`.
  - `model_version`.
  - `route_mode`.
  - `fallback_reason`.
  - `feature_coverage`.
  - `promotion_flow=shadow_to_canary_to_live`.
  - `live_mutation=false`.
  - `shadow_payload`.
- The generated `/profit/predict` contract now exposes Phase 3 profit targets at the top level:
  - `expected_return_pct`.
  - `adjusted_expected_return_pct`.
  - `loss_probability`.
  - `profit_quality_score`.
  - side-level loss probabilities for long/short comparison.
- The generated `/exit/advise` contract is now constrained to the Phase 3 four-action vocabulary:
  - `hold`.
  - `trail_profit`.
  - `protect_profit`.
  - `reduce_or_close`.
- `no_matching_open_position` is now represented as `action=hold` plus `no_matching_position=true`, so downstream control logic does not treat a non-action diagnostic as a trading action.
- The final exit-advice response now also passes through the same `with_model_metadata()` gate as profit/timeseries/sentiment responses.
- The deploy path is now Phase 3-native instead of legacy local-ai-tools:
  - service: `bb-phase3-quant-api.service`.
  - root: `/data/BB`.
  - app dir: `/data/BB/services/phase3_quant_api`.
  - model dir: `/data/BB/models/local_ai_tools`.
  - env file: `/data/BB/env/phase3.env`.
  - port: `127.0.0.1:8101`.
  - health service identity: `phase3_quant_api`.
  - `live_mutation=false` remains enforced.
- Platform key sync now prefers `/data/BB/env/phase3.env` and only falls back to the legacy `/data/trade_ai/local_ai_tools.env` for explicit legacy compatibility.
- Platform runtime now enables the Phase 3 local AI tools client against tunnel `http://127.0.0.1:18001`, so the quant API can participate in paper/shadow decision context without enabling live trading.

Verification:

- Local AI tools deploy-service + client regression: `44 passed`.
- Local AI tools + entry direction + model routing + training + data collection + system audit regression: `149 passed`.
- Phase 3 deploy contract + maintenance + tunnel/self-check regression: `195 passed`.
- Python compile check passed for `scripts/deploy_local_ai_tools_service.py` and `tests/test_local_ai_tools_deploy_service.py`.
- Online model-server deployment:
  - `bb-phase3-quant-api.service=active`.
  - 8101 was reclaimed from the old inventory-only process and is now systemd-owned by the Phase 3 quant API.
  - `/health` returns `service=phase3_quant_api`, `root=/data/BB`, `downloaded_model_count=8`, `validated_model_count=8`, `validation_all_ok=true`, `live_mutation=false`.
  - `/profit/predict` returns `shadow_payload`, `promotion_flow=shadow_to_canary_to_live`, `adjusted_expected_return_pct`, `loss_probability`, `profit_quality_score`, and `live_mutation=false`.
- Online platform deployment:
  - precise files uploaded only for this batch.
  - `/etc/bb/bb-runtime.env` now has `LOCAL_AI_TOOLS_ENABLED=true` and `LOCAL_AI_TOOLS_API_BASE=http://127.0.0.1:18001`.
  - `bb-dashboard.service=active`.
  - `bb-model-tunnels.service=active`.
  - `bb-paper-trading.service=inactive`.
  - dashboard HTTP probe returned `302`.
  - platform tunnel `18001` returned the Phase 3 quant API health/profit contract.

Safety boundary:

- No live trading behavior was enabled.
- No model was promoted to canary or live.
- No order submission.
- No close execution.
- No trading-service startup.
- No DB mutation.
- No artifact rebuild or replacement.
- This completes the output-contract part of stages B/C/D and prepares the later model-factory work; it does not by itself prove the new models are profitable or ready for live routing.

### 13.66 Specialist Quant Model Shadow Chain Visibility Completed

Completed hardening:

- `phase3_quant_api /health` now exposes `specialist_model_chains` for the professional quant slots:
  - time-series primary: `amazon/chronos-2`.
  - time-series challenger: `google/timesfm-2.5-200m-transformers`.
  - time-series fallback: `ibm-granite/granite-timeseries-ttm-r2`.
  - sentiment primary: `ProsusAI/finbert`.
  - sentiment challenger: `yiyanghkust/finbert-tone`.
- `/timeseries/deep/predict` now returns specialist shadow metadata:
  - `specialist_primary_model`.
  - `specialist_challenger_model`.
  - `specialist_artifacts_ready`.
  - `specialist_inference_active=false`.
  - `specialist_model_chain`.
  - `professional_model_shadow`.
- `/sentiment/deep/analyze` now returns the same specialist shadow metadata for the FinBERT sentiment chain.
- `shadow_payload` now also carries the specialist chain fields, so downstream audit can distinguish:
  - model artifacts downloaded and validated.
  - baseline response still in use.
  - real specialist inference not yet promoted.
  - no live mutation.

Verification:

- Local focused regression: `67 passed`.
- Model-server deployment:
  - `bb-phase3-quant-api.service=active`.
  - `/health` returns `specialist_model_chains` with Chronos/TimesFM/FinBERT artifacts ready.
- Online platform tunnel verification through `18001`:
  - `/timeseries/deep/predict` returns `timeseries_primary=amazon/chronos-2`, `timeseries_challenger=google/timesfm-2.5-200m-transformers`, `specialist_artifacts_ready=true`, `specialist_inference_active=false`, `live_mutation=false`.
  - `/sentiment/deep/analyze` returns `sentiment_primary=ProsusAI/finbert`, `sentiment_challenger=yiyanghkust/finbert-tone`, `specialist_artifacts_ready=true`, `specialist_inference_active=false`, `live_mutation=false`.
  - `bb-dashboard.service=active`.
  - `bb-model-tunnels.service=active`.
  - `bb-paper-trading.service=inactive`.
  - dashboard HTTP probe returned `302`.

Safety boundary:

- This does not claim Chronos/TimesFM/FinBERT are live predictors yet.
- This does not promote any model to canary or live.
- This does not start trading.
- This closes the visibility gap where professional models were installed but not visible in per-endpoint output contracts.

### 13.67 Specialist Adapter Preflight Hard Gate Completed

Completed hardening:

- Added `phase3_quant_api /specialists/preflight`.
- The preflight separates four different states that must not be confused:
  - specialist artifacts downloaded and validated.
  - required runtime imports available.
  - adapter implementation ready.
  - walk-forward gate passed.
- The endpoint currently reports the professional model files and base imports are ready, but real specialist shadow inference is still blocked by:
  - `specialist_adapter_not_implemented`.
  - `walk_forward_required`.
- `/models/status` also carries `specialist_adapter_preflight`, so Dashboard/system audit can consume the same hard gate.

Verification:

- Local focused regression: `69 passed`.
- Model-server deployment:
  - `bb-phase3-quant-api.service=active`.
  - `/specialists/preflight` returns `policy=phase3_specialist_adapter_preflight`, `stage=preflight_only`, `live_mutation=false`.
- Online platform tunnel verification through `18001`:
  - `all_artifacts_ready=true`.
  - `all_required_imports_ready=true`.
  - `any_shadow_inference_ready=false`.
  - `blocked_reasons=[specialist_adapter_not_implemented, walk_forward_required]`.
  - `adapter_count=5`.
  - `bb-dashboard.service=active`.
  - `bb-model-tunnels.service=active`.
  - `bb-paper-trading.service=inactive`.
  - dashboard HTTP probe returned `302`.

Safety boundary:

- This prevents the system from treating installed Chronos/TimesFM/FinBERT artifacts as live predictors.
- The next implementation step is actual specialist adapters in shadow-only mode, followed by walk-forward reports before any canary/live promotion.

### 13.68 FinBERT Sentiment Shadow Adapter Started

Completed hardening:

- Implemented a conservative FinBERT sentiment specialist adapter inside `phase3_quant_api`:
  - local-file-only model loading.
  - capped text input.
  - CPU-safe transformer classification path.
  - cached tokenizer/model instances.
  - no order mutation and no live route mutation.
- `/sentiment/deep/analyze` now runs real FinBERT primary shadow inference when request text/headlines are present.
- The endpoint now returns:
  - `model=finbert-shadow-ensemble-v1`.
  - `status=specialist_shadow_inference`.
  - `specialist_inference_active=true`.
  - `professional_model_shadow.actual_inference=true`.
  - `fallback_reason=specialist_sentiment_shadow_only`.
  - `live_mutation=false`.
- Specialist preflight now correctly distinguishes:
  - FinBERT sentiment adapter code ready for shadow inference.
  - time-series adapters still not implemented.
  - walk-forward still required before promotion.

Verification:

- Local focused regression: `70 passed`.
- Model-server deployment:
  - `bb-phase3-quant-api.service=active`.
- Online platform tunnel verification through `18001`:
  - `/sentiment/deep/analyze` returned `status=specialist_shadow_inference`.
  - primary `ProsusAI/finbert` inference succeeded and produced a positive signed score from supplied headlines.
  - `specialist_inference_active=true`.
  - `shadow_payload.specialist_inference_active=true`.
  - `live_mutation=false`.

Known follow-up:

- `yiyanghkust/finbert-tone` challenger artifact is present but tokenizer instantiation failed on the current local copy. It remains non-promoted and must be repaired or replaced before challenger comparison can be considered complete.
- Chronos/TimesFM time-series adapters remain blocked by `specialist_adapter_not_implemented` and `walk_forward_required`.

### 13.69 FinBERT Challenger Compatibility Fixed

Completed hardening:

- Fixed the local `yiyanghkust/finbert-tone` challenger adapter without mutating the raw model artifact:
  - if `AutoTokenizer` cannot infer tokenizer metadata but `vocab.txt` exists, use explicit `BertTokenizer`.
  - if `AutoModelForSequenceClassification` rejects config because `model_type` is absent, load an in-memory `BertConfig` from the local config and instantiate `BertForSequenceClassification`.
- This keeps the raw artifact auditable while allowing the shadow adapter to run.
- The sentiment chain now supports primary + challenger shadow comparison:
  - primary: `ProsusAI/finbert`.
  - challenger: `yiyanghkust/finbert-tone`.
  - disagreement score is emitted in `professional_model_shadow`.

Verification:

- Local focused regression: `72 passed`.
- Model-server deployment:
  - `bb-phase3-quant-api.service=active`.
- Online platform tunnel verification through `18001`:
  - `/sentiment/deep/analyze` returned `status=specialist_shadow_inference`.
  - `primary_available=true`.
  - `challenger_available=true`.
  - `specialist_inference_active=true`.
  - `disagreement=0.929076` on the test headline pair.
  - `live_mutation=false`.
  - `bb-dashboard.service=active`.
  - `bb-model-tunnels.service=active`.
  - `bb-paper-trading.service=inactive`.
  - dashboard HTTP probe returned `302`.

Safety boundary:

- Sentiment specialist output is still shadow-only.
- No sentiment model is promoted to canary/live.
- Time-series specialist adapters remain the next major missing piece.

### 13.70 TimesFM Time-Series Shadow Adapter Started

Completed hardening:

- Implemented a real TimesFM challenger adapter inside `phase3_quant_api`:
  - local-file-only `AutoModelForTimeSeriesPrediction` loading.
  - close-sequence normalization from `close_sequence`, `recent_closes`, or `closes`.
  - safe synthetic sequence only when the request does not provide enough closes.
  - cached model instance.
  - no order mutation and no live route mutation.
- `/timeseries/deep/predict` now runs TimesFM as a professional shadow model when a close sequence is present.
- The endpoint keeps the existing baseline response as the executable response and emits TimesFM only as shadow evidence:
  - `specialist_inference_active=true`.
  - `professional_model_shadow.actual_inference=true`.
  - `professional_model_shadow.baseline_response=true`.
  - `timesfm_shadow_expected_return_pct`.
  - `timesfm_shadow_side`.
  - `timesfm_shadow_confidence`.
  - `fallback_reason=specialist_timeseries_shadow_only`.
  - `live_mutation=false`.
- Specialist preflight now marks `timeseries_challenger` adapter code ready, while keeping:
  - `timeseries_primary` Chronos-2 blocked until a safe native adapter exists.
  - `timeseries_fallback` Granite TTM blocked until supported by the runtime.
  - `walk_forward_required` as the hard gate before any canary/live promotion.

Verification:

- Local focused regression:
  - `tests/test_local_ai_tools_deploy_service.py`: `29 passed`.
  - `tests/test_entry_direction_competition.py tests/test_local_ai_tools_client.py`: `34 passed`.
  - `scripts/deploy_local_ai_tools_service.py` py_compile passed.
- Model-server deployment:
  - `bb-phase3-quant-api.service=active`.
  - model validation still reports `8` ready model artifacts under `/data/BB`.
- Online platform tunnel verification through `18001`:
  - `/timeseries/deep/predict` returned `specialist_inference_active=true`.
  - `fallback_reason=specialist_timeseries_shadow_only`.
  - `timesfm_shadow_expected_return_pct=0.024106` on the test close sequence.
  - `timesfm_shadow_side=long`.
  - `professional_model_shadow.actual_inference=true`.
  - `professional_model_shadow.baseline_response=true`.
  - `live_mutation=false`.
  - `bb-dashboard.service=active`.
  - `bb-model-tunnels.service=active`.
  - `bb-paper-trading.service=inactive`.

Safety boundary:

- TimesFM specialist output is real inference but remains shadow-only.
- It is not written into the executable `expected_return_pct` contract before promotion.
- Chronos-2 is not fake-wired through T5 because the current runtime probe showed unsafe missing/unexpected weight loading.
- No time-series specialist model is promoted to canary/live until walk-forward evidence is generated and accepted by the Phase 3 promotion gate.

### 13.71 Specialist Shadow Evidence Training Loop Closed

Completed hardening:

- Shadow backtest creation now captures a compact, auditable `feature_snapshot.local_ai_tools_shadow` block when local AI tools participate in a market decision.
- The captured block is field-whitelisted:
  - keeps TimesFM/FinBERT specialist inference flags.
  - keeps shadow side, expected return, confidence, promotion flow, and live mutation flags.
  - keeps compact `professional_model_shadow.shadow_result`.
  - drops raw large model outputs and unrelated payloads.
- Local AI tools training export now preserves that compact shadow evidence in `shadow_samples[].features.local_ai_tools_shadow`.
- Training data quality now reports `specialist_shadow_models`:
  - sample coverage by tool.
  - actual specialist inference count.
  - direction count and direction hit count.
  - direction hit rate.
  - average shadow expected return.
- Phase 3 promotion recommendation now includes a `specialist_shadow_gate`:
  - specialist shadow sample floor is required before canary/live readiness.
  - low direction hit rate blocks promotion.
  - live mutation remains disabled unless the operator-controlled gate explicitly allows it.

Verification:

- Local focused regression:
  - `tests/test_shadow_backtest_service.py`.
  - `tests/test_train_local_ai_tools_models.py`.
  - `tests/test_training_data_quality.py`.
  - `tests/test_model_promotion_policy.py`.
  - `tests/test_local_ai_tools_deploy_service.py`.
  - `tests/test_local_ai_tools_client.py`.
  - Result: `93 passed`.
- Local py_compile passed for:
  - `services/shadow_backtest_service.py`.
  - `services/trading_service.py`.
  - `scripts/train_local_ai_tools_models.py`.
  - `services/training_data_quality.py`.
  - `services/model_promotion_policy.py`.
  - `scripts/deploy_local_ai_tools_service.py`.
- Online platform deployment:
  - precisely uploaded platform runtime files and this master document.
  - remote py_compile passed.
  - `bb-dashboard.service=active`.
  - `bb-model-tunnels.service=active`.
  - `bb-paper-trading.service=inactive`.
  - dashboard HTTP probe returned `200`.

Safety boundary:

- This does not start trading.
- This does not promote any model.
- It only makes specialist shadow outputs measurable against future realized returns so the system can prove whether the expensive model server improves trading quality before any canary/live route.

### 13.72 Specialist Shadow Challenger Evaluation Report Added

Completed hardening:

- Added a read-only specialist shadow evaluation service:
  - `services/specialist_shadow_evaluation.py`.
  - `scripts/run_specialist_shadow_evaluation.py`.
- The report reads completed `ShadowBacktest` rows and evaluates compact `feature_snapshot.local_ai_tools_shadow`.
- Metrics now include:
  - completed shadow sample count.
  - eligible specialist-shadow sample count.
  - model/tool sample count.
  - actual specialist inference count.
  - direction count.
  - direction hit count/rate.
  - average realized return for the model-selected side.
  - average expected return.
  - false signal count.
  - worst and best realized return.
  - top symbols.
  - promotion blockers.
- Promotion readiness remains conservative:
  - minimum specialist shadow samples required.
  - direction hit-rate floor required.
  - average realized return floor required.
  - false-signal loss floor enforced.
- The script writes:
  - timestamped report JSON.
  - `specialist_shadow_evaluation_latest.json`.

Verification:

- Local focused regression:
  - `tests/test_specialist_shadow_evaluation.py`.
  - `tests/test_training_data_quality.py`.
  - `tests/test_model_promotion_policy.py`.
  - Result: `24 passed`.
- Local py_compile passed for:
  - `services/specialist_shadow_evaluation.py`.
  - `scripts/run_specialist_shadow_evaluation.py`.
  - `services/training_data_quality.py`.
  - `services/model_promotion_policy.py`.
- Online deployment and read-only execution:
  - uploaded specialist evaluation service and CLI script.
  - remote py_compile passed.
  - generated `/data/bb/app/reports/phase3/specialist_shadow_evaluation_latest.json`.
  - current report returned `completed_count=0`, `eligible_shadow_count=0`, `model_count=0`, which is expected after cold-start reset/no resumed paper trading.
  - `bb-dashboard.service=active`.
  - `bb-model-tunnels.service=active`.
  - `bb-paper-trading.service=inactive`.

Safety boundary:

- This report is read-only.
- It does not start trading.
- It does not promote TimesFM, FinBERT, or any other model.
- After paper trading is explicitly restarted later, this report becomes the evidence source for deciding whether professional specialist models are improving realized returns.

### 13.73 Specialist Shadow Evaluation Timer Installed

Completed hardening:

- Added online systemd timer installer:
  - `scripts/install_specialist_shadow_evaluation_timer.py`.
- Installed read-only online units:
  - `bb-specialist-shadow-evaluation.service`.
  - `bb-specialist-shadow-evaluation.timer`.
- Timer policy:
  - runs every `30` minutes.
  - executes `scripts/run_specialist_shadow_evaluation.py`.
  - writes `/data/bb/app/reports/phase3/specialist_shadow_evaluation_latest.json`.
  - never starts trading.
  - never promotes models.
  - never writes database rows.

Verification:

- Local focused regression:
  - `tests/test_specialist_shadow_evaluation.py`.
  - `tests/test_specialist_shadow_evaluation_timer.py`.
  - Result: `6 passed`.
- Local py_compile passed for:
  - `services/specialist_shadow_evaluation.py`.
  - `scripts/run_specialist_shadow_evaluation.py`.
  - `scripts/install_specialist_shadow_evaluation_timer.py`.
- Online installation:
  - timer installed and active.
  - first oneshot execution exited `SUCCESS`.
  - latest report exists at `/data/bb/app/reports/phase3/specialist_shadow_evaluation_latest.json`.
  - current cold-start report has `completed_count=0`, `eligible_shadow_count=0`, `model_count=0`.
  - next timer run scheduled in about 30 minutes.
  - `bb-dashboard.service=active`.
  - `bb-model-tunnels.service=active`.
  - `bb-paper-trading.service=inactive`.

Safety boundary:

- This is monitoring/evaluation automation only.
- It prepares the proof loop for when paper trading is explicitly resumed later.
- It does not alter live decisions, paper decisions, OKX state, model routing, or training artifacts.

### 13.74 Specialist Shadow Evaluation Visible In System Audit

Completed hardening:

- `web_dashboard/api/system_audit.py` now reads the latest specialist shadow evaluation report.
- The `model_training` audit card now exposes:
  - report availability.
  - generated timestamp.
  - report path.
  - completed shadow count.
  - eligible specialist-shadow count.
  - model count.
  - promotion-ready count.
  - blocked count.
  - top model rows and promotion blockers.
  - `live_mutation=false`.
  - `promotion_flow=shadow_to_canary_to_live`.
- Missing report is surfaced as a warning/observing condition instead of being hidden.

Verification:

- Local focused regression:
  - `tests/test_system_audit_api.py -k model_training`: `10 passed`.
- Local py_compile:
  - `web_dashboard/api/system_audit.py` passed.
- Online deployment:
  - uploaded `web_dashboard/api/system_audit.py`.
  - remote py_compile passed.
  - restarted only `bb-dashboard.service`.
  - `bb-dashboard.service=active`.
  - `bb-specialist-shadow-evaluation.timer=active`.
  - `bb-paper-trading.service=inactive`.
  - dashboard HTTP probe returned `200`.
  - latest specialist report was read successfully with `completed_count=0`, `eligible_shadow_count=0`, `live_mutation=false`.

Safety boundary:

- This is visibility only.
- It does not start trading.
- It does not promote or demote any model.
- It makes professional model evaluation evidence visible before future paper/live decisions can depend on it.

### 13.75 Phase 3 Paper Resume Hard-Gate Preflight Added

Completed hardening:

- Added a read-only paper-resume hard gate:
  - `services/phase3_paper_resume_preflight.py`.
  - `scripts/run_phase3_paper_resume_preflight.py`.
- The preflight combines the required resume checks into one machine-readable go/no-go report:
  - OKX native authoritative current-state sync.
  - OKX/local trade-fact integrity.
  - Phase 3 quant model-server runtime readiness.
  - Platform tunnel and Phase 3 quant API health.
  - `/profit/predict`, `/timeseries/deep/predict`, `/sentiment/deep/analyze`, `/exit/advise` child endpoint health.
  - Dashboard/model-tunnel/paper service states.
  - Specialist shadow evaluation report availability and freshness.
- `can_resume_paper=false` if any hard blocker exists:
  - OKX native pull unavailable.
  - OKX/local differences still unresolved.
  - critical trade-fact integrity issue.
  - Phase 3 model-server runtime not ready.
  - Phase 3 quant API or child endpoints unavailable.
  - `bb-paper-trading.service` already active during preflight.
  - specialist shadow report missing, stale, or no longer shadow-only.
- Platform runtime status now actually probes the Phase 3 quant API child endpoints instead of only trusting `/health`.
- `web_dashboard/api/system_audit.py` now exposes a new card:
  - `phase3_paper_resume_preflight`.
  - It shows `can_resume_paper`, blocker count, warning count, OKX issue count, model runtime readiness, and quant API availability.

Verification:

- Local focused regression:
  - `tests/test_phase3_paper_resume_preflight.py`.
  - `tests/test_system_audit_api.py -k "phase3_paper_resume_preflight or phase3_model_server_readiness"`.
  - Result: `10 passed`.
- Expanded related regression:
  - `tests/test_phase3_paper_resume_preflight.py`.
  - `tests/test_system_audit_api.py`.
  - `tests/test_phase3_model_server_readiness.py`.
  - `tests/test_server_monitor_probe.py`.
  - `tests/test_specialist_shadow_evaluation.py`.
  - `tests/test_training_data_quality.py`.
  - `tests/test_model_promotion_policy.py`.
  - Result: `117 passed`.
- Local py_compile passed for:
  - `services/phase3_paper_resume_preflight.py`.
  - `scripts/run_phase3_paper_resume_preflight.py`.
  - `services/server_monitor_status.py`.
  - `web_dashboard/api/system_audit.py`.
  - `tests/test_phase3_paper_resume_preflight.py`.
  - `tests/test_system_audit_api.py`.

Safety boundary:

- This preflight is read-only.
- It does not start `bb-paper-trading.service`.
- It does not submit, close, or cancel orders.
- It does not write database rows.
- It does not promote models or change model routing.
- Even when `can_resume_paper=true`, an explicit operator action is still required to start paper trading.

### 13.76 Controlled Paper Start Entrypoint Added

Completed hardening:

- Added a controlled paper start entrypoint:
  - `scripts/start_phase3_paper_with_preflight.py`.
- The script defaults to report-only mode.
- It will not call `systemctl start bb-paper-trading.service` unless all conditions are true:
  - Phase 3 paper-resume preflight returns `can_resume_paper=true`.
  - operator passes `--start-service`.
  - operator passes `--confirm-resume-paper CONFIRM_PHASE3_PAPER_RESUME`.
- If the preflight is blocked, the script exits with a structured blocked report and does not call `systemctl`.
- If confirmation is missing or wrong, the script exits with `resume_confirmation_missing` and does not call `systemctl`.
- Even when it starts paper, it verifies `systemctl is-active bb-paper-trading.service` and records command results.

Verification:

- Local focused regression:
  - `tests/test_start_phase3_paper_with_preflight.py`.
  - `tests/test_phase3_paper_resume_preflight.py`.
  - Result: `11 passed`.
- Local py_compile passed for:
  - `scripts/start_phase3_paper_with_preflight.py`.
  - `tests/test_start_phase3_paper_with_preflight.py`.

Safety boundary:

- This batch did not start paper trading.
- The new script only gives future operators a safe start path.
- Default mode remains read-only/report-only.
- Live trading remains disabled and untouched.

### 13.77 Post-Resume Paper Observation Window Added

Completed hardening:

- Added a read-only post-resume observation report:
  - `services/phase3_paper_resume_observation.py`.
  - `scripts/run_phase3_paper_resume_observation.py`.
- The observation report tracks:
  - whether `bb-paper-trading.service` is active.
  - latest paper-resume preflight state.
  - OKX native authoritative sync health and issue count.
  - Phase 3 quant API and child endpoint health.
  - new shadow sample count in the observation window.
  - completed shadow outcome count in the observation window.
  - created/filled order count.
  - open position count.
  - trade reflection count.
  - specialist shadow eligible sample count.
- Status semantics:
  - `waiting_for_resume`: paper is still stopped; this is an observation baseline, not a failure.
  - `warming_up`: paper is active but sample floors are not met yet.
  - `healthy`: OKX/quant API/specialist report are healthy and post-resume sample floors pass.
  - `critical`: OKX differences, quant API outage, or unsafe live mutation appeared.
- `web_dashboard/api/system_audit.py` now exposes:
  - `phase3_paper_resume_observation`.
  - Waiting/warming states are classified as observing when the card is read-only and cannot start trading, submit orders, or change routing.

Verification:

- Local focused regression:
  - `tests/test_phase3_paper_resume_observation.py`.
  - `tests/test_phase3_paper_resume_preflight.py`.
  - `tests/test_start_phase3_paper_with_preflight.py`.
  - `tests/test_system_audit_api.py -k "phase3_paper_resume_observation or phase3_paper_resume_preflight or phase3_model_server_readiness or paper_start"`.
  - Result: `22 passed`.
- Local py_compile passed for:
  - `services/phase3_paper_resume_observation.py`.
  - `scripts/run_phase3_paper_resume_observation.py`.
  - `web_dashboard/api/system_audit.py`.
  - `tests/test_phase3_paper_resume_observation.py`.
  - `tests/test_system_audit_api.py`.

Safety boundary:

- This is read-only observation.
- It does not start paper trading.
- It does not submit, close, or cancel orders.
- It does not write database rows.
- It does not promote models or change model routing.
- It prevents “paper resumed” from being treated as success until OKX remains clean and new shadow/specialist evidence actually accumulates.

### 13.78 Phase 3 Paper Observation Timer Added

Completed hardening:

- Added a repeatable online systemd timer installer for the paper-resume observation window:
  - `scripts/install_phase3_paper_resume_observation_timer.py`.
  - service: `bb-phase3-paper-resume-observation.service`.
  - timer: `bb-phase3-paper-resume-observation.timer`.
- The timer runs the existing read-only observation report every 30 minutes by default:
  - `scripts/run_phase3_paper_resume_observation.py --json-indent 0`.
  - writes durable reports under `data/phase3_paper_resume_observation_reports`.
  - loads the same systemd environment as the dashboard/runtime through `/etc/bb/bb-runtime.env`.
- The installer records the safety contract explicitly:
  - `read_only=true`.
  - `starts_trading_service=false`.
  - `submits_orders=false`.
- `--run-now` starts only the observation oneshot, then probes `bb-paper-trading.service` state for evidence.
- It never starts `bb-paper-trading.service`, never submits orders, and never changes model routing.

Verification:

- Local focused regression:
  - `tests/test_phase3_paper_resume_observation_timer.py`.
  - Result: `3 passed`.
- Local py_compile passed for:
  - `scripts/install_phase3_paper_resume_observation_timer.py`.
  - `tests/test_phase3_paper_resume_observation_timer.py`.
- Online deployment and verification:
  - uploaded the observation service/script, timer installer, timer test, and this master-control document.
  - remote py_compile passed.
  - installed `bb-phase3-paper-resume-observation.timer` with `--run-now`.
  - `bb-phase3-paper-resume-observation.timer=active`.
  - latest report exists at `/data/bb/app/data/phase3_paper_resume_observation_reports/latest.json`.
  - latest report: `status=waiting_for_resume`, `paper_active=false`, `can_use_for_promotion=false`.
  - latest report: `starts_trading_service=false`, `submits_orders=false`, `changes_model_routing=false`.
  - `bb-paper-trading.service=inactive`, `bb-dashboard.service=active`, `bb-model-tunnels.service=active`.

Safety boundary:

- This timer is observation-only.
- It turns paper-resume status into continuous evidence instead of manual one-off checks.
- Waiting/warming/critical states remain visible and do not automatically trigger resume, repair, promotion, or routing changes.

### 13.79 Paper Observation Promotion Gate Closed

Completed hardening:

- Phase 3 model promotion recommendations now require the paper-resume observation gate by default:
  - `services/model_promotion_policy.py`.
  - `paper_observation_gate.required=true`.
  - if the latest paper observation is missing, waiting, warming, critical, unsafe, or not `can_use_for_promotion=true`, canary/live readiness is blocked.
- Added a single read-only loader for the latest observation evidence:
  - `load_latest_paper_observation_report()`.
  - primary path: `settings.data_dir/phase3_paper_resume_observation_reports/latest.json`.
  - local/dev path: `data/phase3_paper_resume_observation_reports/latest.json`.
- All local quant-tool training paths now carry the paper observation report into the promotion recommendation:
  - `services/local_ai_tools_client.py`.
  - `scripts/train_local_ai_tools_models.py`.
  - `web_dashboard/api/data_collection.py`.
  - `services/trading_service.py`.
- The promotion gate blocks unsafe observation contracts:
  - `starts_trading_service=true`.
  - `submits_orders=true`.
  - `changes_model_routing=true`.
- Current online state `waiting_for_resume` therefore correctly keeps new/rebuilt models in `shadow`, even if offline training metrics later look good.

Verification:

- Local focused regression:
  - `tests/test_model_promotion_policy.py`.
  - `tests/test_local_ai_tools_client.py`.
  - `tests/test_train_local_ai_tools_models.py`.
  - Result: `44 passed`.
- Local py_compile passed for:
  - `services/model_promotion_policy.py`.
  - `services/local_ai_tools_client.py`.
  - `scripts/train_local_ai_tools_models.py`.
  - `web_dashboard/api/data_collection.py`.
  - `services/trading_service.py`.
  - related tests.
- `git diff --check` passed for the touched promotion/training files.
- Expanded related regression:
  - paper observation/preflight/start-entry/data-collection/system-audit/promotion focused suite.
  - Result: `95 passed`.
- Online deployment and verification:
  - uploaded promotion/training code, related tests, and this master-control document.
  - remote py_compile passed for the uploaded Python files.
  - restarted only `bb-dashboard.service`.
  - `bb-dashboard.service=active`, `bb-paper-trading.service=inactive`, `bb-phase3-paper-resume-observation.timer=active`.
  - latest observation: `status=waiting_for_resume`, `can_use_for_promotion=false`.
  - promotion policy probe: `canary_ready=false`, `recommended_stage=shadow`, `canary_blocking_reasons=paper_observation_not_healthy:waiting_for_resume`.

Safety boundary:

- This does not start paper trading.
- This does not train or persist a new artifact.
- This does not promote any model to canary/live.
- It makes promotion stricter: model-server specialists can only advance after paper observation proves OKX/native facts and shadow evidence remain healthy.

### 13.80 Phase 3 Go/No-Go Total Gate Added

Completed hardening:

- Added a read-only Phase 3 total go/no-go aggregation gate:
  - `services/phase3_go_no_go.py`.
  - system audit card: `phase3_go_no_go`.
- The total gate aggregates the critical Phase 3 evidence instead of letting isolated green cards imply overall readiness:
  - server resource-release/migration.
  - quant model-server readiness.
  - paper-resume hard preflight.
  - post-resume paper observation.
  - model training/promotion recommendation.
- The gate has conservative next-step states:
  - `blocked`: stay shadow and fix hard gates.
  - `paper_resume_ready`: paper can only be started through the controlled operator-approved path.
  - `paper_observation_healthy`: canary review may be considered, still requiring operator approval.
- The gate never returns live permission:
  - `can_enter_live=false`.
  - `starts_trading_service=false`.
  - `submits_orders=false`.
  - `changes_model_routing=false`.
  - `live_mutation=false`.
- If any child evidence is missing, critical, unsafe, or missing the paper-observation promotion gate, the total gate blocks advancement.

Verification:

- Local focused regression:
  - `tests/test_phase3_go_no_go.py`.
  - `tests/test_system_audit_api.py -k "go_no_go or phase3_go_no_go or phase3_paper_resume"`.
  - Result: `8 passed`.
- Expanded related regression:
  - `tests/test_phase3_go_no_go.py`.
  - `tests/test_system_audit_api.py`.
  - `tests/test_phase3_paper_resume_observation.py`.
  - `tests/test_phase3_paper_resume_observation_timer.py`.
  - `tests/test_phase3_paper_resume_preflight.py`.
  - `tests/test_model_promotion_policy.py`.
  - `tests/test_local_ai_tools_client.py`.
  - `tests/test_train_local_ai_tools_models.py`.
  - Result: `129 passed`.
- Local py_compile passed for:
  - `services/phase3_go_no_go.py`.
  - `web_dashboard/api/system_audit.py`.
  - related promotion/training files and tests.
- `git diff --check` passed for touched Go/No-Go, system audit, promotion, and training files.

Safety boundary:

- This is read-only aggregation.
- It does not start paper trading.
- It does not place, cancel, or close orders.
- It does not train, persist, promote, or route models.
- It prevents Phase 3 from advancing just because one isolated sub-card looks healthy.

### 13.81 Phase 3 Go/No-Go Report Timer Added

Completed hardening:

- Added a standalone read-only Go/No-Go report script:
  - `scripts/run_phase3_go_no_go_report.py`.
  - reads system audit cards and extracts the `phase3_go_no_go` total gate.
  - writes dated reports plus `latest.json` under `data/phase3_go_no_go_reports`.
- Added a systemd timer installer:
  - `scripts/install_phase3_go_no_go_timer.py`.
  - service: `bb-phase3-go-no-go.service`.
  - timer: `bb-phase3-go-no-go.timer`.
- The timer uses systemd `EnvironmentFile` semantics instead of manual shell sourcing:
  - `EnvironmentFile=-/data/bb/app/.env`.
  - `EnvironmentFile=/etc/bb/bb-runtime.env`.
  - this avoids false DB-user/permission errors from ad-hoc SSH commands.
- The report explicitly carries the no-mutation contract:
  - `starts_trading_service=false`.
  - `submits_orders=false`.
  - `changes_model_routing=false`.
  - `live_mutation=false`.

Verification:

- Local focused regression:
  - `tests/test_phase3_go_no_go.py`.
  - `tests/test_phase3_go_no_go_report.py`.
  - `tests/test_phase3_go_no_go_timer.py`.
  - `tests/test_system_audit_api.py -k "go_no_go or phase3_go_no_go"`.
  - Result: `9 passed`.
- Expanded related regression:
  - Go/No-Go, paper observation/preflight/timer, promotion, Local AI tools, training CLI, and system audit suites.
  - Result: `134 passed`.
- Local py_compile passed for:
  - `services/phase3_go_no_go.py`.
  - `scripts/run_phase3_go_no_go_report.py`.
  - `scripts/install_phase3_go_no_go_timer.py`.
  - related system audit, promotion, training files and tests.
- `git diff --check` passed for touched Go/No-Go, report/timer, system audit, and document files.

Safety boundary:

- This timer is report-only.
- It does not start paper trading.
- It does not place, cancel, or close orders.
- It does not train, persist, promote, or route models.
- It gives Phase 3 a durable `latest.json` total-gate artifact for operator review.

### 13.82 Phase 3 Stage Handoff Report Added

Completed hardening:

- Added a read-only Phase 3 stage handoff evaluator:
  - `services/phase3_stage_handoff.py`.
  - `scripts/run_phase3_stage_handoff_report.py`.
- The handoff report aggregates the existing evidence artifacts:
  - `data/phase3_go_no_go_reports/latest.json`.
  - `data/phase3_paper_resume_observation_reports/latest.json`.
  - `reports/phase3/specialist_shadow_evaluation_latest.json`.
  - `data/phase3_rebuild_preflight_reports/latest.json`.
  - `data/okx_daily_reconciliation_reports/latest.json`.
- It converts scattered evidence into one operator stage:
  - `blocked`: fix hard blockers first.
  - `paper_start_ready`: paper can only be started through the confirmed Phase 3 start entrypoint.
  - `post_resume_observing`: paper has started and observation evidence is warming up.
  - `canary_review_ready`: paper observation is healthy and canary review may be considered.
- Added a repeatable online systemd timer installer:
  - `scripts/install_phase3_stage_handoff_timer.py`.
  - service: `bb-phase3-stage-handoff.service`.
  - timer: `bb-phase3-stage-handoff.timer`.

Verification:

- Local focused regression:
  - `tests/test_phase3_stage_handoff.py`.
  - `tests/test_phase3_stage_handoff_timer.py`.
  - Go/No-Go, paper observation, and promotion related tests.
  - Result: `23 passed`.
- Expanded related regression:
  - stage handoff, market warmup, paper preflight/observation, and promotion suites.
  - Result: `33 passed`.
- Local py_compile passed for:
  - `services/phase3_stage_handoff.py`.
  - `scripts/run_phase3_stage_handoff_report.py`.
  - `scripts/install_phase3_stage_handoff_timer.py`.
- Online validation:
  - uploaded the new service/script/timer/test files.
  - installed `bb-phase3-stage-handoff.timer` with `--run-now`.
  - latest report: `/data/bb/app/data/phase3_stage_handoff_reports/latest.json`.
  - latest report returned `status=paper_start_ready`, `stage=paper_start_pending_operator_approval`, `blockers=[]`.
  - `bb-paper-trading.service=inactive`.

Safety boundary:

- This handoff report is read-only.
- It does not start paper trading.
- It does not place, cancel, or close orders.
- It does not train, persist, promote, or route models.
- It makes the post-Go/No-Go operator sequence explicit so Phase 3 does not depend on manual interpretation of scattered reports.

### 13.83 GPU2 BB-FinQuant Expert Slot Runtime Identity Locked

Completed correction:

- GPU `2` expert-pool runtime identity is now locked to `BB-FinQuant-Expert-14B`.
- The old runtime served-model name `qwen3-14b-expert-pool` is no longer an accepted current service identity.
- `llm_expert_pool` may still use `Qwen/Qwen3-14B-AWQ` as the temporary base carrier, but inventory now marks it as:
  - `served_model_name=BB-FinQuant-Expert-14B`.
  - `specialization_required=true`.
  - `specialization_status=pending`.
  - `base_model_carrier=Qwen/Qwen3-14B-AWQ`.
- `services/phase3_model_server_readiness.py` now blocks the service manifest if GPU `2` / `llm_expert_pool` is missing or if it is exposed under the old `qwen3-14b-expert-pool` name.
- Platform endpoint contracts, tunnel names, shadow probe, service deployment manifest, model-status checker, self-check, and Dashboard public endpoint labels now use `BB-FinQuant-Expert-14B`.
- `phase3_stage_handoff` is also wired into `web_dashboard/api/system_audit.py` as a first-class read-only audit card and topology node.

Current promotion rule:

- A base `Qwen3-14B-AWQ` carrier on GPU `2` is allowed only as shadow bootstrap.
- Canary/live promotion still requires audited BB quant specialization evidence such as `specialization_id`, `specialization_manifest`, adapter path, LoRA artifact, fine-tune id, or training artifact.
- This prevents a generic 14B model from being mistaken for the final BB-FinQuant expert.

Verification:

- Model-server readiness, deployment manifest, inventory sync, maintenance scripts, server monitor, self-check, Dashboard contract, system audit, stage handoff, and Go/No-Go regression: `202 passed`.

Safety boundary:

- No order submission.
- No close execution.
- No trading-service startup.
- No model live-routing enablement.
- No claim that GPU `2` has completed BB-FinQuant specialization yet; the current state is a correctly named, shadow-only expert slot with a hard specialization gate.

### 13.84 Platform Runtime Env Residual Model Name Cleared

Completed correction:

- The model server and code gates were already using `BB-FinQuant-Expert-14B`, but the platform runtime env still had stale `AI_MODELS` entries pointing expert slots at `qwen3-14b-expert-pool`.
- Updated `/etc/bb/bb-runtime.env` on the online platform server so:
  - `decision_maker -> http://127.0.0.1:18000/v1 -> qwen3-32b-trade`.
  - all fixed expert slots -> `http://127.0.0.1:18003/v1 -> BB-FinQuant-Expert-14B`.
  - `HIGH_RISK_REVIEW_MODEL=deepseek-r1-14b-risk` remains on `http://127.0.0.1:18002/v1`.
- Runtime env backup created before the edit:
  - `/etc/bb/bb-runtime.env.bak.20260627133127`.
- Added a reusable safe path in `scripts/sync_to_online_server.py`:
  - `--runtime-env-only`.
  - updates `/etc/bb/bb-runtime.env` from the Phase 3 tunnel contract.
  - creates a backup and emits a JSON summary.
  - does not upload files, restart services, start trading, submit orders, or change model routing.

Online verification:

- Platform runtime env now reports:
  - `old_name_remaining=false`.
  - expert slots use `BB-FinQuant-Expert-14B`.
  - `LOCAL_AI_TOOLS_API_BASE=http://127.0.0.1:18001`.
  - `HIGH_RISK_REVIEW_API_BASE=http://127.0.0.1:18002/v1`.
- Model server status remains healthy:
  - `bb-phase3-llm-decision.service=active`.
  - `bb-phase3-llm-risk-review.service=active`.
  - `bb-phase3-llm-expert.service=active`.
  - `8003 /v1/models` returns `BB-FinQuant-Expert-14B`.
  - deprecated legacy model services remain inactive.
- Re-ran read-only online Go/No-Go:
  - `status=paper_resume_ready`.
  - `go_no_go.status=paper_resume_ready`.
  - `next_step=resume_paper_pending_operator_approval`.
  - `blockers=[]`.
  - `can_start_paper_with_operator_approval=true`.
  - `starts_trading_service=false`.
  - `submits_orders=false`.
- Re-ran read-only online Stage Handoff after the fresh Go/No-Go report:
  - `status=paper_start_ready`.
  - `stage=paper_start_pending_operator_approval`.
  - `blockers=[]`.
  - `can_start_paper_with_operator_approval=true`.
  - next action remains the explicit operator-approved start command only.
- Service state after verification:
  - `bb-paper-trading.service=inactive`.
  - `bb-dashboard.service=active`.
  - `bb-model-tunnels.service=active`.

Local verification:

- `tests/test_phase3_go_no_go.py`.
- `tests/test_phase3_go_no_go_report.py`.
- `tests/test_phase3_stage_handoff.py`.
- `tests/test_phase3_stage_handoff_timer.py`.
- `tests/test_model_server_maintenance_scripts.py`.
- `tests/test_phase3_model_server_readiness.py`.
- Result: `36 passed`.

Safety boundary:

- No order submission.
- No close execution.
- No trading-service startup.
- No paper auto-start.
- No model live-routing enablement.
- Current state is ready for operator-approved paper start only; canary/live remain blocked by observation, specialist evaluation, rebuild, and promotion gates.

### 13.85 Alpha Factory Direction and Same-Source LLM Governance Added

Completed hardening:

- Added the `Alpha Factory` self-growing profit loop as a top-level Phase 3 objective:
  - discover opportunities from OKX-native market/trade facts.
  - verify opportunities with time-series, profit, ML, execution-cost, and risk evidence.
  - collect every paper/canary result into clean Phase 3 training facts.
  - automatically demote or retire models/strategies that cannot prove positive contribution.
  - only promote through `shadow -> paper -> canary -> live`.
- Locked the LLM expert-pool interpretation:
  - 5 `BB-FinQuant-Expert-14B` roles are role views, not 5 independent models.
  - same provider/model roles are source-deduplicated before entry support is evaluated.
  - new entries require either multiple independent LLM/model sources or at least one LLM source plus independent quant evidence.
  - independent quant evidence includes server profit model, time-series model, local ML shadow, and direction competition.
- Updated `ai_brain/ensemble_coordinator.py`:
  - added provider/source grouping for expert opinions.
  - added source-deduplicated entry-support policy into `expert_weight_policy` and `entry_signal_support`.
  - changed new-entry gates and probe-entry gates to use independent source counts instead of raw expert count.
  - kept risk/exit protection paths fast so same-source governance does not delay protective exits.
- Added regression coverage in `tests/test_ensemble_expert_weight_policy.py`:
  - same-provider `BB-FinQuant-Expert-14B` roles no longer count as multiple independent entry sources.
  - same-provider roles may enter only when independent quant support is also present.
- Fixed the Chronos-2 shadow adapter input gate in `scripts/deploy_local_ai_tools_service.py`:
  - valid short `recent_closes` sequences with at least 4 closes are no longer discarded before Chronos/TimesFM shadow inference.
  - this prevents professional time-series models from staying inactive when the platform sends a short but usable close sequence.

Verification:

- Same-source expert governance focused regression:
  - `tests/test_ensemble_expert_weight_policy.py`
  - Result: `9 passed`.
- Broader model-routing / batch-expert / trading-boundary regression:
  - `tests/test_batch_expert_json_stability.py`
  - `tests/test_model_dynamic_routing.py`
  - `tests/test_trading_service_boundaries.py`
  - Result: `177 passed`.
- Local AI tools / ML / Go-No-Go related regression:
  - `tests/test_local_ai_tools_client.py`
  - `tests/test_ml_signal_training_quality.py`
  - `tests/test_phase3_go_no_go.py`
  - Result: `49 passed`.
- Chronos shadow adapter focused regression:
  - `tests/test_local_ai_tools_deploy_service.py::test_local_ai_tools_chronos_shadow_adapter_records_primary_and_challenger`
  - Result: `1 passed`.
- Full local AI tools deploy-service regression:
  - `tests/test_local_ai_tools_deploy_service.py`
  - Result: `30 passed`.
- Python compile passed for:
  - `ai_brain/ensemble_coordinator.py`
  - `tests/test_ensemble_expert_weight_policy.py`
  - `scripts/deploy_local_ai_tools_service.py`
  - `tests/test_local_ai_tools_deploy_service.py`

Current online observation after this local change:

- Platform/model tunnels are healthy:
  - decision endpoint: `qwen3-32b-trade`.
  - quant API endpoint: `phase3_quant_api`.
  - risk endpoint: `deepseek-r1-14b-risk`.
  - expert endpoint: `BB-FinQuant-Expert-14B`.
- `bb-paper-trading.service=inactive`.
- Go/No-Go remains paper-start-ready with operator approval, but this does not mean the profit loop is complete.
- Quant API still reports:
  - `trained_models_available=false`.
  - `shadow_sample_count=0`.
  - `trade_sample_count=0`.
  - specialist model artifacts ready but `actual_inference=false` until the updated adapter is deployed and shadow samples exist.
- `specialist_shadow_evaluation_latest.json` is still missing.
- Rebuild preflight remains blocked until clean Phase 3 training facts and required reports exist.

Safety boundary:

- No order submission.
- No close execution.
- No trading-service startup.
- No paper auto-start.
- No historical deletion.
- No model promotion.
- No claim that the system is profit-ready yet; this step removes a high-risk LLM governance flaw and prepares professional time-series shadow inference for the next deployment.

### 13.86 Profit-First Alpha Factory v2 Locked Into Master Plan

Planning update:

- Added `Profit-First Alpha Factory v2` as the controlling Phase 3 direction.
- Clarified that the expensive 8x RTX 5090 model server must serve a measurable profit loop, not generic AI capability or model stacking.
- Locked the highest-level objective function:
  - maximize realized net profit after fees, slippage, funding, rejects, partial fills, and close failures.
  - constrain drawdown, consecutive losses, tail losses, fast-loss exits, and small-win-big-loss behavior.
  - require every model/strategy change to be replayable, trainable, demotable, and promotable through evidence.
- Locked the model collaboration design:
  - OKX-native facts are the source of truth.
  - Chronos-2 / TimesFM handle time-series direction and path-risk evidence.
  - CatBoost / LightGBM / XGBoost / local ML handle after-cost profit quality and calibration.
  - LLM experts explain, arbitrate, detect event/anomaly context, and propose testable hypotheses; they do not replace quant evidence.
- Locked the same-source expert rule:
  - 5 `BB-FinQuant-Expert-14B` roles are a role-sliced evidence board, not independent majority votes.
  - same-source LLM agreement cannot open new trades without independent quant evidence.
- Locked the resource rule:
  - unused GPU capacity should go to shadow challengers, training, backtests, walk-forward, and hyperparameter search before adding more always-on chat/LLM capacity.
- Completion standard:
  - infrastructure is not enough.
  - paper/canary/live progress must be backed by clean Phase 3 facts, OKX reconciliation, specialist shadow reports, promotion gates, Go/No-Go, and realized contribution attribution.

Safety boundary:

- This planning update does not start paper trading.
- This planning update does not delete old backups.
- This planning update does not promote any model or route to canary/live.
- This planning update is now the guardrail for all remaining Phase 3 implementation.

### 13.87 Phase 3 Clean Vector Memory Reset Gate Added

Completed hardening:

- Vector memory is now treated as a Phase 3 clean-sample index, not as a legacy history index.
- `services/vector_memory/service.py` now writes a Phase 3 reset marker when old indexed documents are cleared:
  - marker: `data/vector_memory/phase3_vector_memory_reset_marker.json`.
  - key field: `reset_at`.
  - policy: `old_vector_index_excluded_from_clean_training`.
- Automatic reindex now respects the reset marker:
  - if the index is empty and `reset_at` exists, status/search no longer auto-rebuilds old pre-reset decisions/news into the index.
  - manual reindex only loads documents whose timestamps are at or after `reset_at`.
- Dashboard vector-memory clear endpoint now records an explicit reason:
  - `phase3_dashboard_clear_old_index`.
- Local ML and data-collection pages use Phase 3 clean-start wording:
  - old data is not shown as training input for new models.
  - vector memory status exposes `clean_rebuild_required` until new Phase 3 samples exist.

Verification:

- Focused local regression:
  - `tests/test_vector_memory_service.py`.
  - `tests/test_vector_memory_api.py`.
  - `tests/test_data_collection_api.py`.
  - `tests/test_dashboard_main_ui_contract.py`.
  - Result: `98 passed`.
- Expanded related local regression:
  - vector memory, data collection, Dashboard contract, server monitor, and local AI tools client suites.
  - Result: `135 passed`.
- Python compile passed for vector memory, Dashboard/data-collection, server-monitor, and local-AI-tools files.
- `git diff --check` passed for the touched Phase 3 UI/API/vector-memory files.

Safety boundary:

- This does not start paper trading.
- This does not submit, cancel, or close orders.
- This does not train or promote a model.
- This does not delete database backups or historical raw archives.
- It prevents cleared legacy vector-memory/index data from silently re-entering Phase 3 clean training context.

### 13.88 Chronos-2 Primary Time-Series Shadow Adapter Fixed

Completed hardening:

- Fixed the Chronos-2 primary time-series shadow adapter in `scripts/deploy_local_ai_tools_service.py`.
- Root cause:
  - Chronos-2 `predict_df` validates timestamp columns by converting them to int64.
  - The previous adapter built a timezone-aware timestamp column, which caused Chronos/Numpy to raise:
    - `Cannot change data-type for array of references.`
- Adapter corrections:
  - timestamp is now passed as timezone-naive `datetime64[ns]`.
  - target values are passed as explicit `float64`.
  - `freq` is provided explicitly.
  - `validate_inputs=False` is used after the adapter has already constructed a regular, sorted, single-series frame.
  - if the DataFrame path still fails, the adapter falls back to Chronos-2's supported direct list-of-array `predict([np.float32])` path.
  - direct Chronos tensor outputs now extract the median forecast path instead of flattening arbitrary quantile values.

Verification:

- Local focused regression:
  - TimesFM challenger shadow adapter.
  - Chronos DataFrame shadow adapter.
  - Chronos direct-predict fallback adapter.
  - Result: `3 passed`.
- Expanded local regression:
  - `tests/test_local_ai_tools_deploy_service.py`.
  - `tests/test_local_ai_tools_client.py`.
  - Result: `57 passed`.
- Python compile passed for:
  - `scripts/deploy_local_ai_tools_service.py`.
  - `tests/test_local_ai_tools_deploy_service.py`.
- `git diff --check` passed for the touched local-AI-tools deployment and test files.
- New model server deployment:
  - `bb-phase3-quant-api.service=active`.
  - smoke test passed.
  - service remains `shadow_only=true`.
- Real model-server inference probe:
  - Chronos-2 primary: `actual_inference=true`.
  - TimesFM 2.5 challenger: `actual_inference=true`.
  - `fallback_reason=specialist_timeseries_shadow_only`.
  - `activation_blocker=walk_forward_required`.
  - `live_mutation=false`.
- Platform-to-model-server probe:
  - `LocalAIToolsClient.status()` reached `/timeseries/deep/predict`.
  - `professional_model_shadow.primary_shadow_result.actual_inference=true`.
  - `professional_model_shadow.challenger_shadow_result.actual_inference=true`.
  - `bb-paper-trading.service=inactive`.

Safety boundary:

- This does not start paper trading.
- This does not submit, cancel, or close orders.
- This does not promote Chronos/TimesFM to canary/live.
- Chronos-2 and TimesFM remain shadow evidence until walk-forward, specialist shadow evaluation, paper observation, promotion, Go/No-Go, and stage-handoff gates pass.

### 13.89 OKX CtVal Audit Fix And Rebuild Gate Reclassified

Completed hardening:

- Fixed OKX authoritative sync contract-size resolution in `services/okx_authoritative_sync.py`.
- Root cause:
  - OKX fill history reports `fillSz` in contract counts.
  - local `orders.quantity` stores base quantity.
  - the previous authoritative sync mainly derived `contract_size` from current exchange positions.
  - when a current position snapshot omitted `ctVal`, NEAR-like contracts could be compared as `3.7` base units instead of `3.7 * 10 = 37`, producing a false `local_order_quantity_differs_from_okx_fill`.
- New contract-size priority:
  - local execution payload evidence: `contract_size`, `contractSize`, nested `info.ctVal`, or `base_quantity / filled_contracts`.
  - OKX native public instruments `ctVal` from `publicGetPublicInstruments`.
  - current exchange position snapshot only as the final source.
- Added `OkxNativeFactsClient.fetch_contract_sizes()` for OKX-native instrument contract values.
- Added NEAR regression:
  - local order quantity `37.0`.
  - OKX fill contracts `3.7`.
  - contract size `10.0`.
  - authoritative sync must not emit `local_order_quantity_differs_from_okx_fill`.

Rebuild gate corrections:

- Fixed `scripts/run_phase3_rebuild_preflight.py` to load runtime env and drop to the online `bb` runtime user, matching other Phase 3 report scripts.
- This removed the false manual-run error `role "root" does not exist`.
- Refined training governance contamination risk in `services/training_data_quality.py`:
  - quarantined samples still never train.
  - high contamination now requires meaningful excluded/severe-reason ratio or low effective weight.
  - small successfully quarantined slices are classified as `medium` warning instead of a hard blocker.
- Fixed `services/model_promotion_policy.py` so an explicit `root` argument takes precedence when loading paper observation reports.

Verification:

- Local OKX/native focused regression:
  - `tests/test_okx_native_facts.py`.
  - `tests/test_okx_authoritative_sync.py`.
  - Result: `15 passed`.
- Local expanded gate regression:
  - OKX native facts, authoritative sync, Phase 3 Go/No-Go, paper observation, stage handoff, and system audit.
  - Result: `106 passed`.
- Local rebuild/training regression:
  - `tests/test_training_data_quality.py`.
  - `tests/test_phase3_rebuild_readiness.py`.
  - `tests/test_phase3_rebuild_preflight.py`.
  - `tests/test_model_promotion_policy.py`.
  - `tests/test_phase3_go_no_go.py`.
  - `tests/test_phase3_stage_handoff.py`.
  - Result: `45 passed`.
- Online deployment used `scripts/sync_to_online_server.py --skip-restart --include-tests`.
- Online verification:
  - `okx_authoritative_sync.status=ok`.
  - `okx_issue_count=0`.
  - Go/No-Go status: `paper_observation_healthy`.
  - Stage handoff: `operator_review_for_canary`.
  - model server readiness: `ready`.
  - paper trading: active.
  - live/canary still disabled.
- Online rebuild preflight after fix:
  - `collection_errors={}`.
  - `contamination_risk=medium`.
  - `high_contamination_risk` cleared.
  - remaining hard blockers are real sample floors:
    - `shadow_sample_floor_not_met`: current `62`, required `200`.
    - `trade_sample_floor_not_met`: current `3`, required `15`.
  - historical trade facts are clean:
    - `trainable_closed_positions=3`.
  - legacy local ML artifacts remain retired/untrusted and cannot influence live.

Current phase state:

- A-line OKX native consistency: current audit clean, continue observation.
- Paper observation: healthy and collecting Phase 3 samples.
- Model rebuild: blocked only by clean sample floors.
- Next development focus:
  - keep paper collection running until clean sample floors are met.
  - investigate `profit_prediction` specialist shadow because `actual_inference_count=0` and direction hit rate is below threshold.
  - do not rebuild/persist artifacts until preflight says ready.

Safety boundary:

- This does not start or stop paper trading.
- This does not submit, cancel, or close orders.
- This does not train, persist, or promote artifacts.
- This does not enable canary or live routing.
- Dirty or quarantined samples remain excluded from training.

### 13.90 Specialist Shadow Runtime And Clean Training Gate Tightened

Completed hardening:

- Fixed `scripts/run_specialist_shadow_evaluation.py` to use the same online runtime bootstrap as other Phase 3 report scripts:
  - loads `/data/bb/app/.env`.
  - loads `/etc/bb/bb-runtime.env`.
  - drops root-run online maintenance commands to the `bb` runtime user before DB access.
  - removes the false manual-run error `role "root" does not exist`.
- Fixed `scripts/run_phase3_go_no_go_report.py` stdout contract:
  - report stdout is now JSON-only.
  - OKX executor/system logs produced during collection are redirected to stderr.
  - automation can parse direct command output without being polluted by `OKX executor initialized` logs.
- Tightened specialist shadow evaluation:
  - `baseline_response=true` outputs are skipped.
  - non-specialist heuristic outputs with `actual_inference=false` are skipped.
  - `eligible_shadow_count` now means rows containing at least one real specialist/professional shadow inference.
  - `local-profit-heuristic-v1` no longer appears as a specialist promotion candidate.
- Tightened clean training quality reports:
  - `specialist_shadow_models` only includes real specialist/professional shadow inference.
  - baseline-only and heuristic-only profit shadows are excluded from specialist model quality statistics.
  - governance now explicitly carries `training_policy=clean_training_view_only`.
  - raw records may remain preserved for audit, but old/dirty records still cannot enter Phase 3 model training.

Verification:

- Local focused regression:
  - `tests/test_specialist_shadow_evaluation.py`.
  - `tests/test_training_data_quality.py`.
  - `tests/test_phase3_rebuild_preflight.py`.
  - `tests/test_model_promotion_policy.py`.
  - Result: `41 passed`.
- Local expanded Phase 3 and Dashboard regression:
  - specialist evaluation.
  - training quality.
  - rebuild preflight.
  - Go/No-Go report.
  - stage handoff.
  - paper resume observation/preflight.
  - system audit API.
  - Result: `132 passed`.
- Python compile passed for:
  - specialist shadow evaluation service and script.
  - training data quality.
  - Go/No-Go report.
  - rebuild preflight.
  - stage handoff report.
- Online deployment used:
  - `scripts/sync_to_online_server.py --skip-restart --include-tests`.
- Online verification:
  - `bb-paper-trading.service=active`.
  - `bb-dashboard.service=active`.
  - `bb-model-tunnels.service=active`.
  - Go/No-Go status: `paper_observation_healthy`.
  - Stage handoff: `operator_review_for_canary`.
  - model server status: `ready`.
  - canary/live remain disabled.
- Online specialist shadow evaluation after tightening:
  - `completed_count=143`.
  - `eligible_shadow_count=143`.
  - `model_count=2`.
  - real specialist candidates:
    - `finbert-shadow-ensemble-v1`.
    - `timesfm_shadow_challenger`.
  - skipped profit shadow reasons:
    - `profit_prediction_baseline_only_shadow=12`.
    - `profit_prediction_non_specialist_shadow=131`.
  - `local-profit-heuristic-v1` is no longer counted as a specialist candidate.
- Online rebuild preflight after tightening:
  - `training_policy=clean_training_view_only`.
  - `contamination_risk=medium`.
  - `collection_errors={}`.
  - remaining hard blockers are real sample floors:
    - `shadow_sample_floor_not_met`: current `143`, required `200`.
    - `trade_sample_floor_not_met`: current `8`, required `15`.
  - `specialist_shadow_models` now contains only:
    - `sentiment_analysis`.
    - `time_series_prediction`.

Current phase state:

- OKX authoritative consistency remains clean.
- Paper observation remains healthy and continues collecting clean Phase 3 facts.
- Model rebuild remains blocked by real sample floors only.
- The system must not rebuild/persist artifacts until preflight clears:
  - at least `200` clean shadow samples.
  - at least `15` clean trade samples.
- Profit specialist remains pending real specialist implementation/training; heuristic profit output may be used as non-promotable feature evidence only, not as a specialist model candidate.

Safety boundary:

- This does not start, stop, or restart paper trading.
- This does not submit, cancel, or close orders.
- This does not train, persist, or promote artifacts.
- This does not enable canary or live routing.
- This does not delete historical raw audit records.

### 13.91 ZIL OKX Native Repair And Phase 3 Report Chain Re-verified

Completed hardening:

- Fixed the specialist shadow report path contract:
  - `scripts/run_specialist_shadow_evaluation.py` now defaults to `data/phase3`.
  - `scripts/install_specialist_shadow_evaluation_timer.py` writes `/data/bb/app/data/phase3/specialist_shadow_evaluation_latest.json`.
  - installer creates the report directory as `bb:bb`, preventing root-owned report directories from breaking the oneshot.
- Fixed Phase 3 report scripts to use one online runtime contract:
  - `scripts/run_phase3_model_server_readiness_audit.py` now loads runtime env and drops to the runtime user before config-heavy imports.
  - `scripts/run_phase3_rebuild_preflight.py` passes the latest paper-observation report into the promotion recommendation.
  - `scripts/repair_missing_position_links_from_okx_fills.py` now loads runtime env, drops to the runtime user, and keeps stdout JSON-only while moving OKX probe logs to stderr.
  - `scripts/start_phase3_paper_with_preflight.py` loads runtime env before running the controlled start flow.
- Repaired the new post-resume ZIL mismatch found by OKX authoritative sync:
  - root cause: OKX had already filled close `ordId=3694289954907328513`, but local `Position.id=59` still showed open and had no local close order.
  - dry-run with `--position-id 59 --window-seconds 7200 --close-missing-exchange-open-position` produced exactly one `open_position_close_plan`.
  - apply closed local `Position.id=59`, created local close `Order.id=80`, and linked `close_exchange_order_id=3694289954907328513`.
  - OKX native contract evidence:
    - `instId=ZIL-USDT-SWAP`.
    - `fill_contracts=43`.
    - `ctVal=100`.
    - `fill_quantity=4300`.
    - `exit_price=0.003349`.
    - `close_fee=0.00720035`.
  - backup written:
    - `data/codex_backups/missing-position-links-from-okx-fills/open_position_closes_before_20260627T231016Z.json`.
  - repair marker added with `source=okx_position_link_repair`, so repaired history remains excluded from clean training until reviewed.

Verification:

- Local focused regressions:
  - specialist shadow script/timer/stage/go-no-go: `21 passed`.
  - model readiness/rebuild/promotion/stage/go-no-go: `40 passed`.
  - missing-position OKX repair/runtime bootstrap: `21 passed`.
  - paper start entrypoint/preflight/observation/stage/go-no-go: `34 passed`.
- Python compile passed for the touched Phase 3 report, repair, and start scripts.
- Online ZIL apply result:
  - `apply_open_position_close_result.applied=1`.
  - `Position.id=59 is_open=false`.
  - `Position.id=59 close_exchange_order_id=3694289954907328513`.
  - close `Order.id=80` created with side `sell`, qty `4300`, price `0.003349`, fee `0.00720035`.
- Online post-repair verification:
  - OKX daily report `status=ok`.
  - `can_open_new_entries=true`.
  - `can_refresh_training=true`.
  - unresolved issue count `0`.
  - paper observation `status=healthy`.
  - `okx_issue_count=0`.
  - Go/No-Go `status=paper_observation_healthy`.
  - Stage handoff `stage=operator_review_for_canary`.
  - `bb-paper-trading.service`, `bb-dashboard.service`, and `bb-model-tunnels.service` all active.
  - recent paper journal check showed no Traceback/Error/Exception/Failed entries.

Current phase state:

- A-line OKX native consistency is clean again after the ZIL repair.
- Model server readiness is `ready`; remaining warning is real and expected: `finquant_expert_specialization_pending`.
- Rebuild preflight is `ready_with_warnings`; sample floors are now met, but confirmed artifact writing still requires explicit rebuild confirmation.
- Promotion remains shadow-only because the time-series specialist direction gate is not strong enough yet:
  - blocker: `time_series_prediction_specialist_direction_hit_rate_low`.
- Canary/live remain disabled and require separate operator approval after stronger specialist evidence and walk-forward review.

Safety boundary:

- ZIL repair was a position-scoped OKX-backed data repair, not a strategy/risk relaxation.
- Paper was stopped only to make the repair atomic, then restarted through the approved Phase 3 preflight entrypoint.
- No canary/live routing was enabled.
- No model artifact was trained, persisted, or promoted.

### 13.92 Dashboard Phase 3 Runtime Display Contract Repaired

Completed hardening:

- Fixed the Dashboard/server-monitor GPU display contract:
  - backend now exposes `phase3_model_server_gpu` from `data/phase3_model_server_readiness_reports/latest.json`.
  - readiness `gpu_rows` are parsed into the same GPU array contract used by live `nvidia-smi`.
  - frontend uses live remote GPU rows when available; if the live probe is empty, it uses the latest Phase 3 readiness GPU audit.
  - this prevents the 8 x RTX 5090 server from being displayed as missing or wrong total VRAM when platform runtime `gpu={}` is empty.
- Fixed local quant tool connection semantics:
  - `available=true` now means the Phase 3 quant API service is reachable.
  - `model_bundle_available=false` separately means the persisted trainable bundle is not ready yet.
  - Dashboard should no longer display "not connected" when `/health` and child endpoints are actually reachable.
- Tightened data-collection training notes:
  - page text now states that Phase 3 starts new training and old data is forbidden from new-model training.
  - old audit/raw-record language is no longer presented as if old samples still train the new model.

Verification:

- Local focused regression:
  - local AI tools client, server monitor probe, data collection API, system audit API, Dashboard UI contract: `191 passed`.
  - Dashboard JavaScript syntax check passed with `node --check web_dashboard/static/js/dashboard.js`.
  - Python compile passed for `services/local_ai_tools_client.py` and `services/server_monitor_status.py`.
- Online targeted deployment:
  - uploaded only:
    - `services/local_ai_tools_client.py`.
    - `services/server_monitor_status.py`.
    - `web_dashboard/static/js/dashboard.js`.
  - restarted only `bb-dashboard.service`.
  - `bb-paper-trading.service` stayed `active` and was not restarted.
- Online read-only verification:
  - `bb-dashboard.service=active`.
  - `bb-paper-trading.service=active`.
  - remote live GPU rows: `8`.
  - Phase 3 readiness GPU rows: `8`.
  - Phase 3 GPU total VRAM: `260856 MB`.
  - GPU source: `phase3_model_server_readiness`.
  - local quant API:
    - `available=true`.
    - `service_available=true`.
    - `model_bundle_available=false`.
    - child endpoints available: `4/4`.
  - OKX daily report `status=ok`.
  - OKX issue ledger: `fixed=4`, `unresolved=0`, `observing=0`.
  - `can_open_new_entries=true`.
  - `can_refresh_training=true`.
  - paper observation `status=healthy`.
  - Go/No-Go `status=paper_observation_healthy`.
  - Stage handoff `stage=operator_review_for_canary`.
  - model-server readiness `status=ready`, `runtime_ready=true`, `gpu_count=8`, `active_endpoint_count=3`.

Current phase state:

- Dashboard display is now aligned with Phase 3 runtime facts:
  - 8-card model server is visible.
  - local quant service is visible as connected.
  - missing persisted model bundle remains a separate shadow/training state, not a connection failure.
- This fixes the user's reported Dashboard symptoms:
  - model-server VRAM mismatch.
  - large-model display abnormal.
  - local quant model shown as disconnected even though the Phase 3 quant API is reachable.
- Feature-coverage gaps remain neutral/read-only and cannot drive live entries by themselves.

Safety boundary:

- No paper restart.
- No order submission, close execution, repair apply, or DB mutation.
- No canary/live routing change.
- No model artifact training, persistence, or promotion.

### 13.93 Phase 3 Local AI Tools Shadow Artifact Rebuilt

Completed hardening:

- Re-ran the Phase 3 local AI tools rebuild through the online runtime bootstrap contract:
  - loaded `/etc/bb/bb-runtime.env` through `scripts.runtime_env_bootstrap`.
  - dropped from root to the configured runtime user before DB/settings-heavy imports.
  - used the confirmed Phase 3 rebuild command with `--persist-artifact --confirm-phase3-rebuild`.
- Confirmed the rebuild used the Phase 3 clean training policy:
  - `artifact_policy_id=phase3_clean_training_artifact_v1`.
  - `training_mode=shadow`.
  - `model_stage=shadow`.
  - old/dirty history remains excluded from new-model training.
- Persisted the new local quant bundle after the preflight gate had no hard blockers:
  - `artifact_persisted=true`.
  - `preflight_only=false`.
  - `persist_artifact_requested=true`.
  - `confirm_phase3_rebuild=true`.
- Confirmed online local quant runtime now reports the bundle separately from service connectivity:
  - `client_available=true`.
  - `client_service_available=true`.
  - `client_model_bundle_available=true`.
  - `client_status=ready`.
  - `client_model_stage=shadow`.
  - `client_training_mode=shadow`.
  - `client_promotion_recommended_stage=canary`.

Observed clean training window:

- `shadow_sample_count=1361`.
- `trade_sample_count=34`.
- `sequence_sample_count=10423`.
- `text_sentiment_sample_count=832`.
- `trained_at=2026-06-28T00:07:33.189331+00:00`.

Current phase state:

- The local quant API is connected and has a persisted Phase 3 shadow bundle.
- The bundle is allowed to support shadow evidence and Dashboard visibility.
- It is not allowed to mutate live trading weights by itself.
- Canary/live promotion remains blocked until the separate promotion gates pass:
  - walk-forward review is still required.
  - `model_stage` remains `shadow`, not `live`.
  - `live_mutation` remains disabled.
  - specialist shadow evaluation still has separate false-signal/direction-quality gates.
- GPU2 `BB-FinQuant-Expert-14B` remains a correctly named Phase 3 expert slot, but the final specialist evidence gate is still pending until audited project-specific specialization exists.

Safety boundary:

- No canary/live routing was enabled.
- No strategy threshold, leverage, sizing, or trade-pair filter was relaxed.
- No order submission, close execution, or OKX repair was triggered by this rebuild.
- This is a shadow artifact rebuild only; promotion requires a later explicit operator-approved gate.

### 13.94 Chronos-2 Primary And TimesFM Challenger Runtime Path Verified

Completed verification:

- Directly requested the online Phase 3 quant API endpoint:
  - `POST http://127.0.0.1:18001/timeseries/deep/predict`.
- Confirmed the specialist time-series chain is now executing real shadow inference:
  - `specialist_inference_active=true`.
  - `fallback_reason=specialist_timeseries_shadow_only`.
  - `live_mutation=false`.
- Chronos-2 primary result:
  - `model=chronos-2-shadow-primary`.
  - `adapter=chronos_2_pipeline_adapter`.
  - `available=true`.
  - `actual_inference=true`.
  - no `Cannot change data-type for array of references` runtime error in the verified call.
- TimesFM challenger result:
  - `model=timesfm-2.5-shadow-challenger`.
  - `adapter=timesfm_transformers_adapter`.
  - `available=true`.
  - `actual_inference=true`.
- Baseline local deep time-series model also returned trained output:
  - `model=local-torch-patch-timeseries-v1`.
  - `status=trained_torch_sequence_model`.
  - `trained=true`.

Current phase state:

- The earlier Chronos adapter array-type failure is no longer blocking the online `/timeseries/deep/predict` path.
- The production trading path can collect primary/challenger shadow evidence from both Chronos-2 and TimesFM.
- These specialist outputs remain comparison evidence until walk-forward and specialist promotion gates pass.
- Specialist shadow evaluation still controls promotion quality; the runtime being callable is necessary but not sufficient for canary/live.

Safety boundary:

- This was a read-only inference check.
- No paper restart.
- No DB mutation, artifact write, order submission, close execution, or strategy/risk change.
- Chronos/TimesFM remain shadow-only and cannot mutate live routing by this verification alone.

### 13.95 Specialist Promotion Gate Tightened To Prevent Premature Canary

Completed hardening:

- Tightened specialist promotion evidence so Phase 3 cannot advance to canary only because the model service is callable or the local quant bundle recommends canary.
- `services/specialist_shadow_evaluation.py` now emits explicit promotion blockers:
  - `promotion_blockers`.
  - `blockers`.
  - `blocked_reasons`.
  - `blocked_reason_counts`.
  - `summary.top_blocked_reasons`.
  - `promotion_gate` thresholds.
- `services/phase3_go_no_go.py` now treats specialist promotion readiness as a canary gate:
  - paper can continue collecting evidence while specialists learn.
  - canary remains disabled when `specialist_promotion_ready_count=0`.
  - next step becomes `stay_shadow_improve_specialists`, not `operator_review_for_canary`.
- `services/phase3_stage_handoff.py` now mirrors the same final operator-facing state:
  - `status=paper_observation_healthy`.
  - `stage=stay_shadow_improve_specialists`.
  - `can_enter_canary_with_operator_approval=false`.
  - `can_enter_live=false`.
- `web_dashboard/api/system_audit.py` now exposes specialist summary/gate details so Dashboard/API consumers can see the same blocker reasons used by Go/No-Go.

Verification:

- Local focused regression:
  - specialist shadow evaluation, Go/No-Go, stage handoff, and system audit API: `97 passed`.
  - extended specialist/stage/go-no-go regression earlier in this step: `101 passed`.
  - Python compile passed for touched services/scripts.
- Online targeted deployment:
  - uploaded:
    - `services/specialist_shadow_evaluation.py`.
    - `services/phase3_go_no_go.py`.
    - `services/phase3_stage_handoff.py`.
    - `scripts/run_specialist_shadow_evaluation.py`.
    - `web_dashboard/api/system_audit.py`.
  - restarted only `bb-dashboard.service` after the Dashboard API output update.
  - `bb-paper-trading.service` stayed `active`.
- Online report refresh result:
  - paper service: `active`.
  - dashboard service: `active`.
  - specialist report:
    - `completed_count=1445`.
    - `eligible_shadow_count=1433`.
    - `model_count=2`.
    - `promotion_ready_count=0`.
    - `blocked_count=2`.
    - `top_blocked_reasons=[false_signal_loss_exceeds_floor x 2]`.
  - Go/No-Go:
    - `status=paper_observation_healthy`.
    - `next_step=stay_shadow_improve_specialists`.
    - `can_enter_canary_with_operator_approval=false`.
    - `specialist_canary_blocked=true`.
  - Stage handoff:
    - `status=paper_observation_healthy`.
    - `stage=stay_shadow_improve_specialists`.
    - `next_action=Keep paper running in shadow and improve specialist false-signal loss before canary review.`
    - `can_enter_canary_with_operator_approval=false`.

Current phase state:

- This is not a hard runtime failure.
- Paper is healthy and can continue collecting clean Phase 3 evidence.
- OKX native consistency remains green in the latest daily/stage inputs.
- The system is intentionally not entering canary yet because both specialist candidates still have unacceptable worst-case false-signal loss:
  - `finbert-shadow-ensemble-v1`.
  - `timesfm_shadow_challenger`.
- Next development focus is to reduce false-signal loss before promotion, not to lower gates.

Safety boundary:

- No canary/live routing was enabled.
- No thresholds, leverage, sizing, strategy filters, or model weights were relaxed.
- No DB repair, artifact write, order submission, close execution, or paper restart was performed by this gate change.

### 13.96 OKX Linked Protection Context Window Root Cause Closed

Root cause:

- AAVE `3694561249469370368` was still able to reappear in the standard Phase 3 paper observation report even after the missing close order row was repaired.
- The remaining issue was not CCXT aliasing and not a missing OKX fact:
  - OKX order history returned the reduce-only close order with `source=7`, `algoId`, and the source entry order in the same native context.
  - The source entry order existed locally, but it was outside the short observation audit window.
  - The authoritative sync only loaded local orders inside the audit window, so the protection-fill classifier could not see the older source entry and misclassified the close fill as `okx_fill_not_linked_to_position`.
- A second transient 2Z/USDT difference appeared during verification immediately after a paper fill, then cleared on the next sync after local persistence caught up. This was classified as a short-lived runtime race, not a persistent repair case.

Completed hardening:

- `services/okx_authoritative_sync.py` now treats OKX order-history rows as native context evidence:
  - explicit local order IDs that are not linked by position entry/close fields are prioritized before generic fill contexts.
  - OKX order-history `ordId` values are scanned for context source orders.
  - source orders outside the main audit window are loaded into a separate read-only context list for protection-fill recognition.
  - context orders are not added to the main audit sample loop, so old source entries do not become new training/audit samples and do not create `local_order_not_found_in_recent_okx_fills` noise.
  - report output includes `context_local_order_count` for visibility.
- `services/okx_native_facts.py` now prioritizes explicit `order_ids` before generic fill-derived context queries while preserving OKX native `instId` when available.
- Added regressions for:
  - explicit OKX order-history IDs staying ahead of generic fill context under a low query cap.
  - unlinked local close orders being queried before 30+ unrelated fills.
  - linked protection close fills whose source entry order is outside the observation window.

Verification:

- Local focused regression:
  - `tests/test_okx_authoritative_sync.py`, `tests/test_okx_native_facts.py`, and `tests/test_repair_missing_position_links_from_okx_fills.py`: `46 passed`.
  - Python compile passed for the touched OKX authoritative/native files and tests.
- Online deployment:
  - uploaded `services/okx_authoritative_sync.py` and `services/okx_native_facts.py`.
  - remote backup:
    - `/data/bb/app/data/codex_backups/code-deploy-okx-context-source-order/20260628T024139Z`.
  - restarted `bb-dashboard.service` and `bb-paper-trading.service`; both returned `active`.
- Online standard Phase 3 report refresh:
  - paper observation:
    - `status=healthy`.
    - `paper_active=true`.
    - `okx_issue_count=0`.
    - `okx_status=ok`.
    - `can_use_for_promotion=true`.
  - rebuild preflight:
    - `status=ready_with_warnings`.
    - runtime probe `status=ok`.
    - unavailable model/runtime count `0`.
  - Go/No-Go:
    - `status=paper_observation_healthy`.
    - next step remains `stay_shadow_improve_specialists`.
    - canary/live remain disabled.
  - stage handoff:
    - `status=paper_observation_healthy`.
    - specialist promotion still blocked by false-signal/direction-quality evidence.
- 24h OKX authoritative sync recheck after the transient 2Z persistence race:
  - `status=ok`.
  - `issue_count=0`.
  - `manual_review_count=0`.
  - `repairable_count=0`.

Current phase state:

- OKX/native paper observation is clean again under the standard gate, not a relaxed diagnostic gate.
- AAVE linked-protection fills are now recognized through native order-history context even when the source entry is outside the short observation window.
- Paper can continue collecting clean Phase 3 samples.
- Canary/live remain intentionally blocked until specialist promotion evidence improves; the OKX cleanup does not relax promotion gates.

Safety boundary:

- No canary/live routing was enabled.
- No strategy threshold, leverage, sizing, symbol filter, or model weight was relaxed.
- No new DB repair apply was run in this step.
- The context source-order load is read-only and is not allowed to feed old source orders into Phase 3 clean training samples.

### 13.97 Specialist Tail-Loss Evidence Added To Training And Promotion Gates

Objective:

- Make specialist promotion depend on real realized PnL evidence, not only sample count and direction hit rate.
- Prevent a model with repeated false-signal tail losses from entering canary even if paper observation and OKX reconciliation are healthy.

Completed hardening:

- `services/training_data_quality.py` now adds realized-return evidence to `quality_report.specialist_shadow_models`:
  - `avg_realized_return_pct`.
  - `worst_realized_return_pct`.
  - `best_realized_return_pct`.
  - `false_signal_count`.
  - `tail_loss_count`.
  - `tail_loss_symbols`.
  - `worst_samples`.
- Training quality promotion blockers now include:
  - `avg_realized_return_below_floor`.
  - `false_signal_loss_exceeds_floor`.
- `services/model_promotion_policy.py` now consumes those same fields in `specialist_shadow_gate` and adds canary/live blockers when:
  - average realized return is below the Phase 3 floor.
  - worst realized return or tail-loss count breaches the false-signal loss floor.
- `services/phase3_stage_handoff.py` now reads the latest rebuild preflight `promotion_recommendation` before displaying `promotion_canary_ready`, so handoff cannot show stale canary readiness after the training/promotion gate has already blocked tail-loss risk.
- `services/phase3_go_no_go.py` now reports `inputs.promotion_canary_ready` as effective promotion readiness after specialist tail-risk gates, while preserving `raw_model_promotion_canary_ready` for diagnostics.
- Added regressions proving:
  - training quality reports zero tail loss for clean specialist wins.
  - training quality reports ACT/USDT-style repeated false-signal tail losses.
  - promotion policy keeps the model in shadow when tail-loss evidence is present.
  - Go/No-Go effective promotion readiness turns false when specialist tail-risk blocks canary.
  - stage handoff uses rebuild promotion readiness over stale Go/No-Go input display.

Verification:

- Local focused regression:
  - `tests/test_training_data_quality.py`
  - `tests/test_model_promotion_policy.py`
  - `tests/test_specialist_shadow_evaluation.py`
  - `tests/test_phase3_go_no_go.py`
  - `tests/test_phase3_stage_handoff.py`
  - Result: `54 passed`.
- Python compile passed for:
  - `services/training_data_quality.py`.
  - `services/model_promotion_policy.py`.
  - `tests/test_training_data_quality.py`.
  - `tests/test_model_promotion_policy.py`.

Current phase state:

- OKX native reconciliation remains a separate hard gate.
- Specialist models now have a second hard gate inside training/promotion itself, so a dashboard or report path cannot accidentally promote a model that still has unacceptable real tail losses.
- Paper/shadow can continue gathering clean Phase 3 evidence.
- Canary/live remain disabled until both reconciliation and specialist PnL evidence pass.

Safety boundary:

- No canary/live routing was enabled.
- No threshold was relaxed.
- No model weight, leverage, sizing, symbol filter, or order-execution behavior was changed.
- No database repair or training artifact mutation was performed by this gate change.

### 13.98 FinBERT Shadow-Only Signal Mutation Removed

Root cause:

- The Phase 3 Quant API sentiment deep endpoint treated FinBERT as shadow-only in metadata, but when FinBERT returned a real score it still overwrote the executable top-level sentiment payload:
  - `score`.
  - `label`.
  - `best_side`.
  - `side`.
  - `model`.
  - `trained`.
- Entry direction competition consumes the top-level payload, so this allowed a shadow-only specialist to influence live/paper direction scoring before passing promotion gates.

Completed hardening:

- `scripts/deploy_local_ai_tools_service.py` now keeps FinBERT output strictly inside `professional_model_shadow`.
- The top-level sentiment payload remains the baseline/trained local sentiment response.
- `professional_model_shadow.baseline_response=true` even when FinBERT actual inference succeeds, because the executable response remains baseline.
- FinBERT still records:
  - `specialist_inference_active=true`.
  - `professional_model_shadow.actual_inference=true`.
  - specialist score/label/disagreement/predictions for evaluation and training evidence.
- Added regression proving FinBERT shadow inference cannot overwrite top-level sentiment side/score/model.

Verification:

- Local focused regression:
  - `tests/test_local_ai_tools_deploy_service.py`
  - `tests/test_phase3_go_no_go.py`
  - `tests/test_phase3_stage_handoff.py`
  - `tests/test_training_data_quality.py`
  - `tests/test_model_promotion_policy.py`
  - `tests/test_specialist_shadow_evaluation.py`
  - Result: `85 passed`.
- Python compile passed for touched services/scripts/tests.
- New model server deploy:
  - updated and restarted `bb-phase3-quant-api.service`.
  - service returned `active`.
  - remote smoke passed.
- Direct model-server verification against `/sentiment/deep/analyze`:
  - `specialist_inference_active=true`.
  - `professional_model_shadow.actual_inference=true`.
  - `professional_model_shadow.baseline_response=true`.
  - top-level `best_side=hold`.
  - top-level `side=hold`.
  - top-level `model=local-sentiment-trained-v2`.
  - `live_mutation=false`.

Current phase state:

- FinBERT remains useful as shadow evidence for evaluation and future training.
- FinBERT no longer mutates executable sentiment direction before canary approval.
- Existing historical FinBERT tail-risk samples remain valid evidence and continue blocking promotion until new clean samples prove improvement.

Safety boundary:

- No canary/live routing was enabled.
- No trade execution, order sizing, leverage, threshold, or symbol filter was changed.
- No training artifact write was performed.

### 13.99 Real Kline Sequence Contract For Chronos/TimesFM

Root cause:

- The platform sent scalar indicator fields to `/timeseries/deep/predict`, but did not consistently include real recent close/volume sequences.
- The Phase 3 Quant API then synthesized a 4-point close path from returns, which made Chronos/TimesFM shadow evidence appear active while using an input that was too short for professional time-series evaluation.

Completed hardening:

- `data_feed/feature_vector.py` now carries:
  - `close_sequence`.
  - `volume_sequence`.
  - `sequence_timeframe`.
  - `sequence_length`.
  - `sequence_quality_warning`.
- `services/data_service.py` now extracts up to 80 recent close/volume points from the selected short-cycle Kline dataframe and adds them to the feature snapshot.
- `services/local_ai_tools_client.py` now preserves and caps sequence payloads before calling the model server.
- `scripts/deploy_local_ai_tools_service.py` now requires at least 30 real close points before Chronos/TimesFM or local deep sequence models count as actual time-series inference.
- Short synthetic 4-point sequences are no longer allowed to masquerade as specialist inference.
- If ticker price and indicator close diverge beyond the existing 20% guard, the sequence is dropped and marked with `indicator_sequence_dropped_due_to_ticker_gap`.

Verification:

- Local focused regression:
  - `tests/test_data_service_security.py`.
  - `tests/test_local_ai_tools_client.py`.
  - `tests/test_local_ai_tools_deploy_service.py`.
  - Result: included in the 138-test focused suite.
- Model server deployed through `scripts/deploy_local_ai_tools_service.py`.
- Platform narrow deploy uploaded:
  - `data_feed/feature_vector.py`.
  - `services/data_service.py`.
  - `services/local_ai_tools_client.py`.
- Online model probe through platform tunnel:
  - 4-point `close_sequence` returns `not_enough_real_close_sequence`.
  - 60-point `close_sequence` makes both Chronos and TimesFM report `actual_inference=true`.
  - `live_mutation=false`.
- Online platform FeatureVector probe as service user `bb`:
  - BTC/USDT generated `sequence_length=80`.
  - `close_sequence_len=80`.
  - `volume_sequence_len=80`.
  - no sequence quality warning.

Safety boundary:

- No canary/live routing was enabled.
- No execution, sizing, leverage, threshold, or symbol filter was changed.
- The change only fixes specialist evidence input quality and diagnostics.

### 13.100 Legacy Short-Sequence Shadow Evidence Quarantined

Root cause:

- After the real sequence contract was fixed, historical shadow rows created before the fix still contained 4-point time-series specialist evidence.
- Reports counted those old rows in promotion metrics, causing stale `sequence_too_short_count` and tail-loss evidence to keep polluting new Phase 3 promotion gates.

Completed hardening:

- `services/specialist_shadow_evaluation.py` now separates:
  - total shadow sample count.
  - clean actual inference count.
  - legacy quarantined count.
  - legacy sequence-too-short count.
- `services/training_data_quality.py` uses the same split for rebuild preflight quality reports.
- `services/model_promotion_policy.py` now blocks on clean sample floor when clean actual inference is insufficient.
- Legacy mixed/short-sequence rows remain visible as quarantine evidence, but no longer enter:
  - direction hit-rate.
  - average realized return.
  - tail-loss count.
  - worst-sample promotion metrics.
- This prevents old 4-point samples from either promoting or permanently poisoning the new model gate.

Verification:

- Local focused regression:
  - `tests/test_data_service_security.py`.
  - `tests/test_local_ai_tools_client.py`.
  - `tests/test_local_ai_tools_deploy_service.py`.
  - `tests/test_specialist_shadow_evaluation.py`.
  - `tests/test_training_data_quality.py`.
  - `tests/test_model_promotion_policy.py`.
  - `tests/test_phase3_go_no_go.py`.
  - `tests/test_phase3_stage_handoff.py`.
  - `tests/test_phase3_rebuild_preflight.py`.
  - Result: `138 passed`.
- Platform narrow deploy uploaded:
  - `services/specialist_shadow_evaluation.py`.
  - `services/training_data_quality.py`.
  - `services/model_promotion_policy.py`.
- Online reports refreshed with real runtime environment:
  - Go/No-Go status: `paper_observation_healthy`.
  - Handoff stage: `stay_shadow_improve_specialists`.
  - Handoff blocker count: `0`.
  - Preflight status: `ready_with_warnings`.
  - Promotion canary ready: `false`.
  - Recommended stage: `shadow`.
  - Time-series `sequence_too_short_count`: `0`.
  - Historical short-sequence evidence moved to `legacy_quarantined_count` and `legacy_sequence_too_short_count`.

Current phase state:

- OKX reconciliation and paper observation have no hard blockers in the latest refreshed reports.
- Canary remains correctly disabled because clean specialist evidence is still insufficient and sentiment specialists still have false-signal tail loss.
- Next work should continue collecting clean paper/shadow samples with the new real-sequence contract, then retrain/evaluate from the clean view before any canary decision.

Safety boundary:

- No old evidence was deleted.
- No training artifact was promoted.
- No canary/live route, sizing, leverage, threshold, or order-execution behavior was changed.

### 13.101 Phase 3 OKX Order Fact Sync And Equity PnL Boundary Completed

Root cause:

- Order fact sync originally mixed the Beijing Phase 3 start day with UTC database timestamps.
- The correct business boundary is `2026-06-28 00:00:00 Asia/Shanghai`, which is `2026-06-27 16:00:00 UTC` in the online database.
- The old DB boundary could treat Phase 3 early-morning orders as missing locally, causing repeated `okx_only_backfilled` rows.
- Dashboard account cards could also confuse local diagnostic trade PnL with OKX account-equity PnL.

Completed hardening:

- `services/okx_order_fact_sync.py` now uses a fixed Phase 3 order boundary:
  - display boundary: `2026-06-28T00:00:00+08:00`.
  - DB comparison boundary: `2026-06-27T16:00:00+00:00`.
- OKX `fills-history` requests include the Phase 3 `begin` timestamp.
- Backfill now queries existing `exchange_order_id` rows from the database before inserting any OKX-only order, instead of trusting only the current in-memory/limited scan window.
- `orders` rows now carry OKX-native fields:
  - `okx_inst_id`.
  - `okx_trade_ids`.
  - `okx_fill_contracts`.
  - `okx_fill_pnl`.
  - `okx_state`.
  - `okx_sync_status`.
  - `okx_synced_at`.
  - `okx_last_error`.
  - `okx_raw_fills`.
- Trade success in the dashboard requires OKX confirmation:
  - `filled + okx_confirmed`.
  - local filled rows without OKX fill confirmation are not shown as successful execution.
- Account dashboard PnL now uses OKX account equity only.
- Local order/position PnL remains diagnostic evidence only and must not become account `today_total_pnl`, cumulative PnL, or risk baseline.
- Daily equity baseline accepts OKX current equity and records `source=okx_snapshot` when available.

Online cleanup and verification:

- Online narrow deploy uploaded only the OKX fact sync/account-equity files, not the whole dirty worktree.
- Online DB migration completed as user `bb`.
- `bb-paper-trading.service` and `bb-dashboard.service` are active after restart.
- Duplicate repeated `okx_only_backfilled` rows created by the previous boundary bug were removed.
- Final online verification:
  - `phase3_order_sync_start=2026-06-27T16:00:00+00:00`.
  - `phase3_order_sync_start_local=2026-06-28T00:00:00+08:00`.
  - sync status: `ok`.
  - `confirmed_count=94`.
  - `unverified_count=0`.
  - `backfilled_count=0`.
  - duplicate OKX-only backfills: `0`.
  - Phase 3 order counts: `okx_confirmed=94`, `null=3`.
- Account PnL sample check:
  - OKX account equity: `4998.15`.
  - local diagnostic trade PnL: `9.22`.
  - displayed account `today_total_pnl`: `null` when no OKX equity baseline exists.
  - local diagnostic PnL is not promoted to account PnL.

Verification:

- Local focused regression:
  - `tests/test_equity_baseline.py`.
  - `tests/test_execution_allocation_service.py`.
  - `tests/test_dashboard_error_safety.py`.
  - `tests/test_dashboard_main_ui_contract.py`.
  - `tests/test_okx_order_fact_sync.py`.
  - `tests/test_okx_native_facts.py`.
  - `tests/test_trade_history_api.py`.
  - `tests/test_okx_authoritative_sync.py`.
  - Result: `134 passed, 1 deselected`.
- The deselected failure is the existing local AI loopback monitor test and is unrelated to OKX order/account fact sync.

Safety boundary:

- No order was submitted during verification.
- No leverage, sizing, strategy threshold, live/canary route, or model promotion was changed.
- The cleanup removed only duplicate OKX-only backfill rows caused by the boundary bug.
- Phase 3 order sync must continue to start at `2026-06-28 00:00 Asia/Shanghai`; older orders must not be re-imported into the Phase 3 account/training truth set.

### 13.102 OKX Account Equity As The Only Account Truth Completed

Root cause:

- Phase 3 account display still had several legacy local-account paths:
  - daily equity baseline could estimate from local positions/allocated balance when OKX equity was unavailable.
  - dashboard account diagnostics could turn unknown OKX equity PnL into `0.0`.
  - PnL history could rebuild an equity curve from local positions or paper executor virtual balances.
  - rejected/no-fill orders could remain `okx_sync_status=null`, making failed attempts look like unresolved sync gaps.
  - local AI closed-position samples only required local order-link fields, not OKX-confirmed order facts.
- These paths could make “orders look profitable” while OKX account equity did not show matching profit.

Completed hardening:

- Account PnL now has a strict rule:
  - current account equity comes from OKX balance snapshot only.
  - daily baseline is `source=okx_snapshot` only.
  - if OKX equity is unavailable, account `today_equity_pnl/today_total_pnl/today_risk_pnl` stays `null` instead of falling back to local PnL or fixed balances.
- `services/equity_baseline.py` no longer reconstructs account equity from local positions, K-lines, allocated balance, or virtual account history.
- Existing non-OKX daily equity baselines are replaced by the current OKX snapshot when available; otherwise the baseline is marked `okx_unavailable`.
- `web_dashboard/api/dashboard.py` no longer builds the dashboard summary from paper executor virtual account summaries before replacing it with OKX data.
- `/dashboard/pnl-history` now reads only `execution_equity_snapshots` rows with `source=okx_snapshot`.
  - It does not read trading-service in-memory local PnL.
  - It does not read paper executor `initial_balance`.
  - It does not rebuild equity from local realized/unrealized PnL.
- `services/okx_order_fact_sync.py` now marks rejected/failed/canceled zero-fill local attempts as `okx_no_fill_rejected`.
  - These rows are not successful trades.
  - They are not OKX sync failures.
  - They are not training facts.
- `scripts/train_local_ai_tools_models.py` now requires closed-position training samples to link to OKX-confirmed order facts:
  - entry order must be `okx_confirmed` or `okx_only_backfilled`.
  - non-flat close order must be `okx_confirmed` or `okx_only_backfilled`.
  - OKX-confirmed order fees are carried into the training sample as `fee_estimate`.

Verification:

- Python compile check passed for:
  - `services/equity_baseline.py`.
  - `services/execution_allocation_service.py`.
  - `services/okx_order_fact_sync.py`.
  - `scripts/train_local_ai_tools_models.py`.
  - `web_dashboard/api/dashboard.py`.
  - related tests.
- Local focused regression:
  - `tests/test_equity_baseline.py`.
  - `tests/test_execution_allocation_service.py`.
  - `tests/test_okx_order_fact_sync.py`.
  - `tests/test_local_ai_trade_fact_training_filter.py`.
  - `tests/test_dashboard_error_safety.py`.
  - `tests/test_dashboard_main_ui_contract.py`.
  - Result: `104 passed` with the known unrelated local-AI loopback monitor test excluded.
- Follow-up regression after unifying Dashboard OKX snapshot paths:
  - Result: `105 passed` with the same known unrelated local-AI loopback monitor test excluded.
- Online narrow deploy completed:
  - uploaded only the OKX/account/training allowlist.
  - remote Python compile passed.
  - DB init now runs as Linux user `bb` to satisfy PostgreSQL peer auth.
  - `bb-dashboard.service` active.
  - `bb-paper-trading.service` active.
  - dashboard health returned `302`.
- Online final read-only verification:
  - `account_pnl_source=okx_authoritative`.
  - `account_equity=5001.360102539451`.
  - `today_equity_baseline=5001.167560139451`.
  - `today_equity_baseline_source=okx_snapshot`.
  - account `today_equity_pnl/today_total_pnl/today_risk_pnl=0.19254240000009304`.
  - `local_trade_today_pnl=7.030117908475312` remains diagnostic and is separate from account PnL.
  - `local_trade_total_pnl=13.592492988475378` remains diagnostic and is separate from account PnL.
  - `initial_balance=None`.
  - PnL history source is `okx_equity_snapshots`.
  - Recent order sync statuses are `okx_confirmed` and `okx_no_fill_rejected`.

Safety boundary:

- No order was submitted.
- No paper/live trading service state was changed.
- No model was trained, promoted, or routed differently.
- This section only removes local/synthetic account-truth paths and tightens training-data eligibility to OKX-confirmed Phase 3 facts.

### 13.103 Local Quant API Tunnel Contract Hard Gate Completed

Root cause:

- The server monitor could already detect `LOCAL_AI_TOOLS_API_BASE=http://127.0.0.1:8001` as a wrong Phase 3 platform loopback port.
- But child probes such as `/profit/predict` could still return HTTP 200, causing `local_ai_tools.available=true`.
- This made the dashboard/server monitor say the local quant API was available even when the platform was not using the required `127.0.0.1:18001` tunnel.

Completed hardening:

- `services/server_monitor_status.py` now treats the Phase 3 tunnel contract as a hard gate.
  - Required platform base remains `http://127.0.0.1:18001`.
  - If the configured loopback base is any other local port, status becomes `wrong_loopback_port`.
  - Child endpoint success remains diagnostic only and cannot make the local quant API available.
- `web_dashboard/static/js/dashboard.js` now applies the same display gate:
  - local/remote child probe success cannot override a failed tunnel contract.
  - the model/local quant status card will not show a wrong-port route as healthy.

Verification:

- `tests/test_dashboard_error_safety.py::test_collect_platform_runtime_status_flags_wrong_local_ai_loopback_port`: passed.
- Related monitor/dashboard/system audit regression passed:
  - `tests/test_dashboard_error_safety.py`.
  - `tests/test_server_monitor_probe.py`.
  - `tests/test_dashboard_main_ui_contract.py`.
  - `tests/test_system_audit_api.py`.
  - Result: `177 passed`.
- Python compile passed for `services/server_monitor_status.py`.
- JavaScript syntax check passed for `web_dashboard/static/js/dashboard.js`.
- `git diff --check` passed for the touched files.
- Online narrow deploy completed:
  - uploaded only `services/server_monitor_status.py`, `web_dashboard/static/js/dashboard.js`, and this master-control document.
  - restarted only `bb-dashboard.service`.
  - `bb-dashboard.service=active`.
  - `bb-paper-trading.service=active` and was not restarted.
  - dashboard HTTP probe returned `302`.
- Online read-only verification from the running dashboard process environment:
  - `api_base=http://127.0.0.1:18001`.
  - `expected_platform_api_base=http://127.0.0.1:18001`.
  - tunnel contract `status=ok`.
  - `available=true`.
  - `service_available=true`.
  - `child_available=true`.
  - `health_service=phase3_quant_api`.
  - child endpoints available: `4/4`.

Safety boundary:

- No order was submitted.
- No paper/live trading service state was changed.
- No model was trained, promoted, or routed differently.
- This change only fixes monitoring truth: the platform must use the Phase 3 `18001` local quant API tunnel before the system reports it as available.

### 13.104 OKX Close Link Convergence And Order Fill Audit Priority Completed

Root cause:

- ACT had an OKX-confirmed reverse filled close order, but the local position row still stayed open.
- AAVE had correct local `orders.price=97.54` and correct OKX `okx_raw_fills.avg_price=97.54`, but the trade-fact integrity audit read the linked decision's stale `execution_result.price=93.57918032786885` first and raised a false `execution_price_mismatch`.

Completed hardening:

- `scripts/repair_missing_position_links_from_okx_fills.py` can now close a stale local open position by reusing an existing OKX-confirmed reverse filled close order.
  - It requires matching `okx_inst_id`, reverse side, filled state, OKX sync status, close order time after position open, and quantity compatibility.
  - It does not create duplicate local close orders when an OKX-confirmed close order already exists.
- Online ACT repair was applied:
  - position `65` is now closed.
  - close order is `3695363208212353024`.
  - `realized_pnl=1.7907`.
- `services/okx_trade_fact_integrity.py` now treats `Order.okx_raw_fills` as the highest-priority OKX execution fact for order price, contracts, contract size, base quantity, and native instrument.
  - Linked decision `execution_result/raw_response` remains a legacy context source only when the order itself has no OKX raw fills.
  - This prevents stale decision payloads from overriding confirmed OKX order facts.

Verification:

- ACT repair dry-run found exactly one plan with source `okx_confirmed_existing_close_order`; apply completed and OKX authoritative current issues became `0`.
- AAVE false warning root cause was reproduced:
  - local order price `97.54`.
  - `okx_raw_fills.avg_price=97.54`.
  - OKX row `fillPx=97.54`.
  - stale linked decision price `93.57918032786885`.
- Added regression `test_order_okx_raw_fills_win_over_stale_decision_execution_price`.
- Local focused regression passed:
  - `tests/test_okx_trade_fact_integrity.py`.
  - `tests/test_repair_missing_position_links_from_okx_fills.py`.
  - Result: `35 passed`.
- Wider reconciliation regression passed:
  - `tests/test_okx_trade_fact_integrity.py`.
  - `tests/test_okx_daily_reconciliation_report.py`.
  - `tests/test_system_audit_api.py`.
  - `tests/test_okx_order_fact_sync.py`.
  - Result: `105 passed`.
- Python compile passed for the touched audit and reconciliation paths.

Safety boundary:

- No order was submitted.
- No paper/live position was opened or closed through the exchange.
- No trading service restart was required for the ACT repair.
- This change only converges local facts to existing OKX-confirmed fills and prevents the audit layer from preferring stale local decision payloads over OKX raw fills.

### 13.105 OKX Native Ledger Unified Remediation Plan Locked

User escalation:

- The repeated ACT/AAVE/FLOKI-style fixes show that continuing to patch one symbol or one local row at a time is not acceptable.
- The user explicitly requires a unified treatment so later implementation does not drift back into local-position patching.
- Backend account, order, position, historical position, fee, fill, and PnL views must match OKX. Local synthetic data must not be presented as exchange truth.

Root problem:

- The platform still has multiple truth layers mixed together:
  - local `orders` as execution records.
  - local `positions` as lifecycle records.
  - OKX native account/order/fill/position facts as the real exchange ledger.
- Dashboard historical positions still depend too much on local `positions` rows, while OKX displays historical positions as grouped lifecycle rows with linked fill/order details.
- Single-symbol repairs can make one row look correct but do not guarantee that the next OKX fill, partial close, protection close, or re-open lifecycle will display and train correctly.

Locked direction:

- Stop treating local `positions` as the primary historical-position truth for Phase 3 display and training.
- Build a Phase 3 OKX-native ledger view from `2026-06-28 00:00 Asia/Shanghai` forward.
- Treat OKX native identifiers and rows as the primary facts:
  - `instId`.
  - `ordId`.
  - `tradeId`.
  - `posSide`.
  - `fillSz`.
  - `fillPx`.
  - `fee`.
  - `fillPnl`.
  - OKX account equity.
- Local rows may provide model attribution, decision context, strategy explanation, and UI annotations, but they must not override OKX native facts.

Implementation plan:

1. OKX native ledger ingestion:
   - Pull account balance/equity from OKX only.
   - Pull current positions from OKX native account positions.
   - Pull order history and fill history from OKX native endpoints.
   - Scope Phase 3 order sync start to `2026-06-28 00:00 Asia/Shanghai`; do not import pre-Phase-3 dirty orders into the clean ledger.
2. Local order cache alignment:
   - Every successful local filled order must be confirmed by OKX `ordId/tradeId/fillPx/fillSz/fee`.
   - Filled local rows not confirmed by OKX are not successful trades, not training facts, and not account-PnL facts.
   - Rejected/no-fill local attempts may be shown as execution attempts but must not be mixed with successful OKX trades.
3. Historical position aggregation:
   - Add an OKX-style grouped historical-position API/view.
   - Group by native instrument, direction, and lifecycle window rather than raw local position row count.
   - The grouped row must expose:
     - symbol / `instId`.
     - leverage where available.
     - position status.
     - average entry price.
     - average close price.
     - realized PnL.
     - realized PnL percentage.
     - max position size.
     - closed quantity.
     - open time.
     - close time.
4. Linked order details:
   - Add a "linked orders" action on each grouped historical-position row.
   - The popup must show OKX-style fill details:
     - buy/sell direction.
     - filled quantity.
     - filled price.
     - PnL / PnL percentage when available.
     - fee.
     - order id.
     - trade id.
     - fill time.
   - The popup must use OKX fills/orders as the display source, not reconstructed local-only rows.
5. Training gate:
   - Only complete OKX-backed grouped historical positions can enter clean training.
   - Any group with missing OKX fill/order evidence stays excluded.
   - Any legacy local-only or manually inferred record stays quarantined unless a later explicit OKX evidence gate marks it trusted.
6. Reconciliation gate:
   - New entries and training refresh remain blocked while OKX/local current-state differences exist.
   - Current open positions must match OKX quantity, side, entry, mark, UPL, and native instrument.
   - Historical grouped positions must match OKX historical-position style totals and linked fill details.

Immediate known open item:

- FLOKI local order `3695537280216961024` remains the current online blocker.
- The next implementation must not only repair the FLOKI row; it must also close the underlying ledger/query gap that allowed a current OKX-backed order to appear as missing from the bounded fill pull.

Acceptance criteria:

- Dashboard account equity and today's account PnL use OKX equity only.
- Dashboard current positions match OKX current positions.
- Dashboard historical positions follow the OKX grouped lifecycle layout.
- Each grouped historical-position row has a linked-order details popup matching OKX fill details.
- OKX daily reconciliation has no unresolved data issue.
- Training refresh is allowed only after the clean OKX grouped ledger has no unresolved evidence gaps.

Non-goals:

- Do not continue one-off symbol patching as the main solution.
- Do not make fixed account balances such as `4000` or `5000` part of account truth.
- Do not use local virtual PnL to explain account equity.
- Do not let old dirty history or local-only synthetic rows enter Phase 3 training.

### 13.106 OKX Native Ledger Implementation And Cleanup Gate

Implementation checkpoint:

- Added `services/okx_position_ledger_view.py` as the read-only Phase 3 historical-position ledger view.
- `GET /api/dashboard/positions?closed_only=true` must return OKX-style grouped lifecycle rows, not raw local `positions` fragments.
- Historical rows now expose `group_id`, `okx_inst_id`, entry/close order ids, linked fill rows, evidence gaps, `evidence_complete`, and `trainable`.
- Dashboard historical position UI must show one grouped row per OKX-style lifecycle and a linked-order popup for OKX fill/order details.
- Linked-order popup must display side, quantity, price, PnL, fee, order id, trade id, fill time, and OKX confirmation status.

Mandatory cleanup and sync gate after code changes:

1. Stop treating any local fixed amount as account truth.
   - Remove old `4000`/`5000` account judgment paths from Phase 3 paper account display and reconciliation.
   - Current equity, available balance, margin, realized/unrealized account PnL, and daily account PnL must be read from OKX account facts.
2. Clean local Phase 3 OKX fact cache from the Phase 3 start boundary.
   - Boundary: `2026-06-28 00:00 Asia/Shanghai`.
   - Do not resync or train from pre-Phase-3 dirty orders unless explicitly quarantined as audit-only.
   - Local orders/positions that cannot be confirmed by OKX `ordId/tradeId/fillPx/fillSz/fee/fillPnl` are not clean trade facts.
3. Resync OKX facts after the cleanup.
   - Balance source: OKX native account balance.
   - Open position source: OKX native account positions.
   - Order source: OKX native order history.
   - Fill source: OKX native fills history.
   - Local DB may only cache/index these facts and attach model/strategy metadata.
4. Run reconciliation before enabling new clean training.
   - Open positions must match OKX exactly by `instId`, side, quantity, entry, mark, and UPL.
   - Historical grouped rows must have linked order/fill evidence or stay non-trainable.
   - Any unresolved discrepancy blocks training refresh and paper resume.

Verification required before this node can be marked complete:

- `rtk pytest tests/test_trade_history_api.py tests/test_dashboard_main_ui_contract.py -q`
- `rtk pytest tests/test_okx_native_facts.py tests/test_okx_order_fact_sync.py tests/test_run_phase3_okx_fact_sync.py tests/test_trade_history_api.py tests/test_okx_trade_fact_integrity.py tests/test_local_ai_trade_fact_training_filter.py tests/test_dashboard_error_safety.py tests/test_dashboard_main_ui_contract.py -q`
- `rtk python -m py_compile services/okx_position_ledger_view.py web_dashboard/api/dashboard.py tests/test_trade_history_api.py tests/test_dashboard_main_ui_contract.py`
- Dry-run first: `rtk python scripts/run_phase3_okx_fact_sync.py --mode paper --json-indent 0`
- Apply order/fill cache sync only after dry-run is understood: `rtk python scripts/run_phase3_okx_fact_sync.py --mode paper --apply-order-sync --json-indent 0`
- Run the OKX daily reconciliation report after sync; unresolved account/order/fill/position differences must remain visible and blocking.

Current status:

- Backend grouped historical-position ledger view: implemented and covered by focused regression tests.
- Dashboard linked-order popup: implemented and covered by UI contract tests.
- Legacy `/api/dashboard/account` endpoint: no longer returns paper virtual account balances as account truth; it returns OKX snapshot fields only.
- Phase 3 OKX fact-sync CLI: added as `scripts/run_phase3_okx_fact_sync.py`; default is read-only, `--apply-order-sync` is required before writing local OKX order/fill cache.
- OKX order fact sync now pulls both native `orders-history` and native `fills-history` from the Phase 3 boundary:
  - `orders-history` confirms order existence/state and caches OKX-only canceled/open/non-filled order rows for dashboard consistency.
  - `fills-history` remains the only source for successful execution price, fee, trade ids, fill size, and realized PnL.
  - OKX order-history-only rows are marked `okx_order_only`; they are not successful trades and cannot enter clean training.
  - Filled local rows that are not confirmed by OKX fills remain `okx_unverified`, even when an OKX order row exists.
- Cleanup/sync apply against online local data: pending; this is required so the fixed code is not reading stale local cache.
- FLOKI order `3695537280216961024`: still tracked as a ledger-query gap until online cleanup/sync and reconciliation prove whether it exists in OKX order/fill history or must be quarantined as a local-only artifact.

### 13.107 OKX Current-Position Confirmation Third State

Problem fixed:

- FLOKI exposed a third OKX evidence state that was not represented cleanly:
  - OKX current positions prove an open position exists (`instId`, `posId`, signed `pos`, `avgPx`, `markPx`, `upl`, `tradeId`, `fee`).
  - OKX `fills-history` / `orders-history` may still return no row for the local `exchange_order_id`.
  - The old binary model treated this as either `okx_confirmed` or `okx_unverified`, which caused repeated false blockers or risked promoting incomplete facts into PnL/training.

Locked rule:

- `okx_confirmed` means OKX fill history confirmed the execution with `ordId/tradeId/fillSz/fillPx/fee/fillPnl`.
- `okx_only_backfilled` means an OKX fill exists and the local order row was created from OKX facts.
- `okx_position_confirmed` means only OKX current position snapshot confirmed the open entry position.
- `okx_position_confirmed` must never be treated as:
  - a completed fill-history fact.
  - realized PnL evidence.
  - a clean closed-trade training sample.
  - proof that order history is complete.

Implementation checkpoint:

- Added `services/okx_position_confirmation.py`.
- `services/okx_authoritative_sync.py` suppresses `local_order_not_found_in_recent_okx_fills` only when the local entry order is strictly matched to an OKX current open position by:
  - `instId`.
  - position side inferred from entry side.
  - local `entry_exchange_order_id`.
  - `posId` when present.
  - entry price tolerance.
  - contract-size-adjusted quantity tolerance.
- `services/okx_order_fact_sync.py` now writes `okx_position_confirmed` for this third state and clears stale `okx_trade_ids`, `okx_fill_contracts`, `okx_fill_pnl`, and fill rows.
- `services/okx_position_ledger_view.py` displays these linked rows as `okx_current_position_snapshot`, not OKX fill-confirmed rows.
- `services/okx_trade_fact_integrity.py` does not use position-snapshot-only payloads as fill execution facts.

Verification:

- Focused OKX native/order/authoritative regression: `38 passed`.
- Trade history, local AI training filter, trade fact integrity, daily reconciliation, system audit regression: `116 passed`.
- Python compile passed for the touched OKX sync, ledger, integrity, and tests.

Remaining mandatory online action:

- Deploy the patch.
- Run `scripts/run_phase3_okx_fact_sync.py --mode paper --apply-order-sync --json-indent 0`.
- Run a fresh reconciliation report.
- Confirm FLOKI is no longer a false current-state blocker while still not counted as a clean closed-trade training fact unless OKX fill/order history later confirms it.

### 13.108 OKX Authoritative Ledger Final Cleanup And Resync Gate

User escalation:

- The backend still showed order/profit records that did not match the actual OKX account equity movement.
- The old `4000` paper budget and the user-mentioned `5000` starting balance must not appear anywhere as account truth or as fallback account math.
- The user explicitly requires removing these fixed-amount judgment paths, not downgrading them to secondary fallbacks.
- After code fixes, local Phase 3 data must be cleaned and resynced from OKX so the dashboard does not keep displaying stale local facts.

Locked rule:

- OKX is the only source of truth for account balance, current positions, orders, fills, fees, realized PnL, unrealized PnL, and historical position lifecycle facts.
- Local database rows are only a cache/index of OKX facts plus model/strategy attribution metadata.
- A local row may explain why the system tried to trade, but it must not prove that money was made or lost unless OKX confirms the execution.
- If an OKX API read is unavailable, the account/position/order fact must be shown as unavailable or blocked; it must not be reconstructed from paper executor balances, `execution_account_balances`, `model_initial_balances`, local open positions, or old virtual-account state.

Official OKX endpoint map:

- Account balance/equity: `GET /api/v5/account/balance`.
- Current positions: `GET /api/v5/account/positions`.
- Historical position lifecycle: `GET /api/v5/account/positions-history`.
- Recent order state: `GET /api/v5/trade/orders-history`.
- Archived order state: `GET /api/v5/trade/orders-history-archive`.
- Execution fills, fees, trade ids, fill PnL: `GET /api/v5/trade/fills-history`.

Field authority:

- Account equity, available balance, frozen/used balance: OKX account balance fields only.
- Current position quantity/side/entry/mark/UPL: OKX positions fields `instId`, `posId`, `posSide`, `pos`, `avgPx`, `markPx`, `upl`, `uTime`.
- Successful execution price/quantity/fee/PnL: OKX fills fields `ordId`, `tradeId`, `instId`, `side`, `posSide`, `fillSz`, `fillPx`, `fee`, `fillPnl`, `ts/fillTime`.
- Order existence/state only: OKX order history fields `ordId`, `instId`, `side`, `state`, `ordType`, `sz`, `accFillSz`, `avgPx`, `cTime`, `uTime`.
- Historical position grouped lifecycle: OKX positions-history fields such as `instId`, `posId`, `mgnMode`, `posSide`, `openAvgPx`, `closeAvgPx`, `realizedPnl`, `uTime`, plus linked fills/orders for detail popup.
- OKX `positions-history.realizedPnl` already includes `pnl + fee + fundingFee + liqPenalty + settledPnl`; dashboard must not add a second local realized PnL on top.

Implementation requirements:

1. Remove fixed-account truth paths:
   - `4000` and `5000` must not be read as account truth.
   - `execution_account_balances`, `initial_virtual_balance`, `model_initial_balances`, paper executor balances, and virtual account rows must not participate in Phase 3 account equity, available balance, total PnL, today PnL, or risk baseline.
   - They may remain only for legacy non-OKX unit tests or disabled historical code, not for the running Phase 3 account path.
2. Remove local open-position fallback from account truth:
   - If OKX current positions are unavailable, dashboard current positions and account UPL must show unavailable/blocked, not local-estimated.
   - Local open positions may be shown only as diagnostic mismatch evidence.
3. Make order display OKX-backed:
   - `okx_confirmed` and `okx_only_backfilled` are successful execution facts.
   - `okx_order_only` is order-state evidence only, not a successful trade.
   - `okx_position_confirmed` is current-position evidence only, not fill/training/realized-PnL evidence.
   - `okx_unverified` local filled rows must display as not OKX-confirmed and must not count as success, account PnL, or clean training.
4. Make historical positions OKX-style:
   - Closed/history page must group by OKX lifecycle rather than raw local fragments.
   - Same lifecycle rows must merge visually like OKX.
   - Each group must expose a linked-orders button/popup with OKX fills/orders details.
   - If OKX positions-history exists, it is the preferred lifecycle source; local grouped cache is only enrichment and attribution.
5. Clean and resync after code fixes:
   - Phase 3 sync boundary is fixed at `2026-06-28 00:00 Asia/Shanghai`.
   - Do not import pre-Phase-3 orders into the clean display/training ledger.
   - After deployment, run local cleanup/resync so stale order status, stale equity baselines, stale linked rows, and old local-only success records do not remain visible as facts.

Acceptance criteria:

- Dashboard account equity equals OKX account equity or shows OKX unavailable; it never falls back to `4000`, `5000`, local virtual balances, or local PnL.
- Dashboard today's account PnL is current OKX equity minus the OKX-backed daily equity baseline.
- Dashboard order success state matches OKX fill confirmation.
- Dashboard historical positions are grouped and include linked OKX order/fill details.
- Local clean-training filters accept only OKX-confirmed Phase 3 facts.
- `scripts/run_phase3_okx_fact_sync.py --mode paper --apply-order-sync --json-indent 0` completes after deployment, then daily reconciliation has no unresolved OKX/local data mismatch that affects trading or training.
- If OKX returns no corresponding order/fill/position fact for a local record, the record remains diagnostic/quarantined instead of being repaired into a fake success.

Implementation checkpoint 2026-06-28:

- `services/okx_order_fact_sync.py` now pulls OKX `positions-history` during the Phase 3 fact sync, in addition to `orders-history` and `fills-history`.
- Synced historical position rows are rebuilt from OKX lifecycle facts:
  - `okx_pos_id` comes from OKX `posId`.
  - `okx_inst_id` comes from OKX `instId`.
  - `side` comes from OKX `posSide`.
  - `entry_price` comes from OKX `openAvgPx`.
  - `current_price` / close price comes from OKX `closeAvgPx`.
  - `realized_pnl` comes directly from OKX `realizedPnl` and must not be locally recomputed or double-counted.
- The sync links nearby OKX fills/orders into `entry_exchange_order_id` and `close_exchange_order_id` for dashboard detail popups, but linked details are evidence enrichment only; lifecycle PnL remains `positions-history.realizedPnl`.
- `scripts/run_phase3_okx_fact_sync.py --apply-order-sync` now cleans the Phase 3 local OKX cache before rebuilding it:
  - deletes Phase 3+ local `orders` for the selected mode.
  - deletes Phase 3+ local `positions` and any current open local position cache for the selected mode.
  - deletes Phase 3+ `execution_equity_snapshots` for `ensemble_trader`.
  - preserves pre-Phase-3 rows outside the clean ledger boundary.
- `/api/positions` now uses the same OKX grouped ledger for closed positions as `/api/dashboard/positions?closed_only=true`, so backend views no longer disagree between raw local fragments and grouped OKX lifecycle rows.
- Open positions are still returned as current-position cache rows because manual close actions need local open-position ids, but they must be refreshed from OKX current positions and are not account truth when OKX reads are unavailable.
- Current open-position cache rebuild is now part of `OkxOrderFactSyncService.sync()`:
  - current positions are pulled account-wide from OKX `GET /api/v5/account/positions`, not filtered by stale local order symbols.
  - local open `Position` rows are backfilled/updated from OKX `instId`, `posId`, signed `pos`/`posSide`, `avgPx`, `markPx`, `upl`, `lever`, `ctVal`, `cTime/uTime`.
  - quantity is rebuilt as OKX contract count times OKX contract size; local paper quantities are not reused.
  - stale local entry/close order ids are not invented or carried forward unless OKX fills/orders safely provide them.
  - when a matching OKX-filled entry order is present in the same clean sync cache, the open-position row links `entry_exchange_order_id` deterministically by `instId`, side, quantity coverage, entry price, and time window.
  - sync reports now expose `current_position_checked_count`, `current_position_backfilled_count`, `current_position_updated_count`, and `current_position_skipped_count`.
- `scripts/install_okx_daily_reconciliation_timer.py` now installs a timer that runs `scripts/run_phase3_okx_fact_sync.py --mode paper --apply-order-sync --json-indent 0`, so the scheduled job performs cleanup/resync plus reconciliation instead of only writing a read-only report.
- `TradingService` OKX authoritative sync diagnostics now include the `positions-history` backfill/update counts.
- Focused local verification passed:
  - `rtk pytest tests/test_okx_order_fact_sync.py tests/test_run_phase3_okx_fact_sync.py tests/test_trade_history_api.py tests/test_okx_daily_reconciliation_timer.py tests/test_dashboard_error_safety.py tests/test_trading_service_boundaries.py -q` -> `213 passed`.
  - `rtk python -m py_compile services/okx_order_fact_sync.py tests/test_okx_order_fact_sync.py scripts/run_phase3_okx_fact_sync.py` -> passed.
  - `rtk pytest tests/test_okx_order_fact_sync.py tests/test_run_phase3_okx_fact_sync.py tests/test_trade_history_api.py::test_trade_positions_api_groups_closed_positions_with_okx_ledger tests/test_trade_history_api.py::test_dashboard_position_history_uses_okx_grouped_ledger_with_linked_fills tests/test_trading_service_boundaries.py::test_okx_order_fact_sync_position_confirmed_does_not_block_runtime_gate tests/test_okx_daily_reconciliation_timer.py -q` -> `14 passed`.
  - `rtk python -m py_compile services/okx_order_fact_sync.py scripts/run_phase3_okx_fact_sync.py web_dashboard/api/trades.py services/trading_service.py scripts/install_okx_daily_reconciliation_timer.py tests/test_okx_order_fact_sync.py tests/test_trade_history_api.py tests/test_okx_daily_reconciliation_timer.py tests/test_trading_service_boundaries.py` -> passed.
  - `rtk git diff --check` -> passed.

Online checkpoint 2026-06-28:

- First online cleanup/resync rebuilt OKX current open positions, but `okx_trade_fact_integrity` still blocked new entries because the newly rebuilt open-position rows had no `entry_exchange_order_id`.
- The report showed 8 repairable deterministic entry links, so the missing link was not a data-truth problem; it was a sync sequencing gap.
- Fix rule: current-position cache rebuild must link matching OKX entry orders during the same sync pass, not rely on a later manual repair step.
- Verification after the fix must show `position_missing_entry_order_link` cleared or reduced only to true manual-review cases.
- Second online cleanup/resync after the link fix reduced trade-fact integrity from `critical_count=8` to `critical_count=0`:
  - OKX current positions: 8.
  - local current positions rebuilt from OKX: 8.
  - OKX fill rows backfilled into local order cache: 104.
  - positions-history rows checked: 36.
  - position link repair candidates: 0.
- Remaining post-sync warning showed `_okx_authoritative_sync_cache.cache.hit=true`, meaning the after-report was still allowed to reuse a stale in-process authoritative-sync cache from before cleanup.
- Fix rule: fresh daily reconciliation (`allow_cache=false`) must clear both `_okx_reconciliation_cache` and `_okx_authoritative_sync_cache` before collecting cards.
- Verification after this cache fix must run a fresh online report and must not treat stale cached pre-cleanup findings as current truth.
- Fresh online preflight then reduced the remaining OKX fact issue to 2 MET fills:
  - MET 2026-06-28 07:00 sell order and 2026-06-28 11:40 buy order were both OKX-confirmed.
  - OKX `positions-history` did not return a lifecycle row for that completed open/close pair.
  - The old local cache therefore had two valid OKX fills but no position lifecycle row linking them.
- Fix rule: when OKX `positions-history` omits a completed lifecycle but OKX `fills-history` provides a deterministic entry/close pair, `OkxOrderFactSyncService` must create a closed `Position` cache row from the OKX fill pair.
- Fill-pair derived closed positions must:
  - use OKX `instId`, `ordId`, `tradeId`, `fillSz`, `fillPx`, `fee`, `fillPnl`, and `ts`.
  - link `entry_exchange_order_id` and `close_exchange_order_id`.
  - compute quantity as OKX contracts times OKX contract size.
  - use only OKX fill PnL/fees for realized PnL; do not recompute account profit from local virtual balances.
  - skip pairs already linked by `positions-history` to avoid duplicates.
- Verification now includes `rtk pytest tests/test_okx_order_fact_sync.py tests/test_okx_authoritative_sync.py tests/test_okx_daily_reconciliation_report.py tests/test_run_phase3_okx_fact_sync.py tests/test_trade_history_api.py tests/test_okx_daily_reconciliation_timer.py tests/test_phase3_paper_resume_preflight.py tests/test_dashboard_error_safety.py tests/test_trading_service_boundaries.py -q` -> `248 passed`.
- Final online OKX fact-sync checkpoint:
  - `scripts/run_phase3_okx_fact_sync.py --mode paper --apply-order-sync --json-indent 0` completed with `order_sync_error=null`.
  - local Phase 3 cache cleanup removed stale local rows, then rebuilt: 104 OKX fills/orders, 36 `positions-history` lifecycle rows, 8 OKX current open positions, and 1 fill-pair-derived MET closed lifecycle.
  - `okx_trade_fact_integrity.issue_count=0`.
  - `okx_authoritative_sync.status=ok`, `issue_count=0`, `severity_counts={}`.
  - `position_price_integrity.status=ok`.
  - `issue_ledger.summary.unresolved=0`.
  - `can_refresh_training=true`.
  - `can_open_new_entries=false` only because `bb-paper-trading.service` was intentionally kept stopped during repair, so runtime heartbeat is stale.
- Interpretation: OKX/backend account facts are now clean enough for training-data refresh and paper-resume preflight; paper entries remain blocked until the runtime is restarted through the approved preflight/start path and publishes a fresh OKX heartbeat.

Mandatory online follow-up:

1. Deploy the allowlisted Phase 3 OKX fact-sync patch.
2. Run `scripts/run_phase3_okx_fact_sync.py --mode paper --apply-order-sync --json-indent 0` online to clear stale Phase 3 local facts and rebuild orders/fills/position history from OKX.
3. Run a fresh reconciliation report and verify:
   - OKX account equity equals dashboard equity.
   - OKX current positions equal dashboard current positions.
   - OKX historical positions match dashboard grouped history.
   - No fixed `4000`/`5000` balance appears as account truth.
4. Only after these checks are clean may Phase 3 clean data collection/training refresh proceed.

---

## Implementation checkpoint 2026-06-29: OKX/account/data-display bug closure

Scope added from operator bug report:

- Daily PnL for `2026-06-28` must not disappear after Phase 3 cleanup.
- Main dashboard cumulative OKX equity PnL must move with OKX account equity; it must never use fixed `4000`, fixed `5000`, local virtual balance, or local trade PnL as account truth.
- `OKX auto sync` label is renamed/treated as OKX authoritative fact sync in the UI.
- Opening funnel and strategy-review panels may show "no Phase 3 sample yet", but must not silently show old dirty samples as current evidence.
- Local ML and local quant-tools sample counts must use only Phase 3 clean/trainable samples. Old values such as `150810`/`144870` are raw legacy diagnostics only and cannot drive training, pending-sample counts, or promotion gates.
- Feature coverage issues `liquidation_risk` and `sector_correlation` remain visible until new Phase 3 market samples provide them; missing/stale features are neutralized and cannot silently drive live entry.
- Server self-check must distinguish real blockers from observing states.
- `BB-FinQuant-Expert-14B / vLLM` must report a model-route mismatch if port `18003` returns another served model such as `qwen3-32b-trade`.
- Vector memory reset must handle existing invalid `/data/bb/app/data/vector_memory/zvec` by quarantining/removing the bad zvec path and recreating a clean Phase 3 index.

Implementation decisions:

- OKX remains the only authority for account balance, current positions, historical positions, orders, fills, fees, and realized PnL.
- Official OKX v5 endpoint contract used by Phase 3:
- Balance: `GET /api/v5/account/balance`.
- Current positions: `GET /api/v5/account/positions`.
- Historical positions: `GET /api/v5/account/positions-history`.
- Order history: `GET /api/v5/trade/orders-history` and archive.
- Fill history: `GET /api/v5/trade/fills-history`.
- Account bills: `GET /api/v5/account/bills` and archive are audit-only balance-change ledgers.
- `scripts/run_phase3_okx_fact_sync.py` must not rebuild Phase 3 account-equity baselines from account bills. OKX bills explain balance/cash changes, but they are not account-equity snapshots and can drift from OKX equity when positions, UPL, fees, or funding are involved.
- If a clean Phase 3 start equity snapshot was missed, the dashboard must either use the first actual OKX equity snapshot captured in the Phase 3 window or mark the Phase 3 cumulative equity baseline unavailable. It must not manufacture an equity baseline from `balChg`, fixed `4000`, fixed `5000`, paper executor balances, or local PnL.
- Existing bill-derived `execution_equity_snapshots` rows at the Phase 3 boundary must be removed during OKX fact sync so values such as `5008.xx` cannot make current OKX equity `5000.xx` display as a false negative Phase 3 equity PnL.
- Dashboard daily and cumulative account PnL read OKX equity snapshots only. Local order/position PnL is shown only as diagnostic trade PnL.
- `services/ml_signal_service.py` and `scripts/train_local_ai_tools_models.py` count completed shadow samples only after `PHASE3_CLEAN_START_UTC`.
- `web_dashboard/api/dashboard.py`, `web_dashboard/api/data_collection.py`, and `web_dashboard/static/js/dashboard.js` expose Phase 3 clean sample counts as the primary UI values; legacy all-time counts cannot be promoted into main training cards.
- `core/server_monitor_probe.py` marks `model_mismatch` when a vLLM endpoint is reachable but does not serve the target model.
- `services/vector_memory/store.py` quarantines invalid existing zvec paths and retries creation, fixing `path validate failed: path[/data/bb/app/data/vector_memory/zvec] exists`.
- `web_dashboard/api/dashboard.py` no longer promotes legacy `auto_train_last_result.new_shadow_sample_count` / `new_sample_count` into Phase 3 `phase3_new_shadow_sample_count`; the Phase 3 new-sample number is now only the explicit Phase 3 value or `phase3_completed - current_training_window`.
- `web_dashboard/static/js/dashboard.js` now labels the status lane as "OKX权威事实同步" and explicitly says stale/warning sync pauses new entries.
- Local ML / local quant-tool cards now use "训练窗口 / 三期完成" and "三期新增未训练样本"; old cumulative samples are not displayed as training sources.
- `services/vector_memory/service.py` adds a service-level one-shot recovery for zvec `path validate failed` on status/reindex/search. The store is discarded, the invalid path is handled by the store layer, and the operation retries once; non-path errors remain visible.

Validation completed locally:

- `rtk pytest tests/test_run_phase3_okx_fact_sync.py tests/test_dashboard_error_safety.py tests/test_data_collection_api.py::test_data_collection_does_not_promote_legacy_training_counts_to_phase3 tests/test_ml_signal_training_quality.py::test_local_ml_training_counts_only_phase3_clean_shadow_rows tests/test_server_monitor_probe.py::test_server_monitor_probe_marks_vllm_port_model_mismatch tests/test_vector_memory_service.py::test_zvec_store_quarantines_invalid_existing_path_before_create -q`
- Result: `41 passed`.
- `python -m pytest tests/test_dashboard_main_ui_contract.py::test_dashboard_runtime_stats_do_not_regress_from_ws_packets tests/test_dashboard_main_ui_contract.py::test_server_monitor_rendering_isolated_from_numeric_format_errors -q`
- Result: `2 passed`.
- `rtk python -m py_compile scripts/run_phase3_okx_fact_sync.py services/ml_signal_service.py scripts/train_local_ai_tools_models.py services/vector_memory/store.py core/server_monitor_probe.py web_dashboard/api/dashboard.py web_dashboard/api/data_collection.py`
- Result: passed.
- `rtk pytest tests/test_okx_order_fact_sync.py tests/test_okx_authoritative_sync.py tests/test_okx_daily_reconciliation_report.py tests/test_run_phase3_okx_fact_sync.py tests/test_trade_history_api.py tests/test_okx_trade_fact_integrity.py tests/test_dashboard_error_safety.py -q`
- Result: `104 passed`.
- `rtk pytest tests/test_model_server_config.py::test_ml_signal_status_does_not_promote_legacy_sample_counts tests/test_vector_memory_service.py::test_vector_memory_recovers_once_from_zvec_path_validation_error tests/test_dashboard_main_ui_contract.py::test_data_collection_ui_explains_phase3_clean_training_view`
- Result: `3 passed`.
- `rtk python -m py_compile web_dashboard/api/dashboard.py services/vector_memory/service.py`
- Result: passed.
- `rtk node --check web_dashboard/static/js/dashboard.js`
- Result: passed.

Deployment verification completed:

- Deployed the OKX/account/data-display checkpoint to the online platform.
- Ran `scripts/run_phase3_okx_fact_sync.py --mode paper --apply-order-sync --json-indent 0`.
- Online cleanup/resync result:
  - deleted stale Phase 3 local cache: 125 local order rows and 53 local position rows.
  - rebuilt from OKX: 119 order/fill facts, 42 OKX historical-position lifecycle rows, and 11 current OKX open positions.
  - `order_sync_error=null`.
- Fresh OKX daily reconciliation:
  - `status=ok`.
  - `requires_attention=false`.
  - `issue_ledger.summary.unresolved=0`.
  - `can_open_new_entries=true`.
  - `can_refresh_training=true`.
- Runtime OKX authoritative sync:
  - `status=ok`.
  - `last_requires_attention_count=0`.
  - `last_error=null`.
  - `task_running=true`.
- Dashboard account verification:
  - Dashboard account equity equals live OKX equity from `GET /api/v5/account/balance`.
  - Account PnL source is `okx_authoritative`.
  - No `4000`, fixed `5000`, paper executor balance, local virtual balance, local trade PnL, or OKX bill-derived value is used as account truth.
  - `execution_equity_snapshots` contains only the real OKX snapshot row for `2026-06-29` captured at `2026-06-28T16:00:00Z`; no synthetic `2026-06-28` boundary snapshot remains.
- Dashboard daily PnL verification:
  - `2026-06-28` daily row is present and shows 38 OKX-confirmed closed trades.
  - `2026-06-28` OKX equity change is intentionally `null` / `okx_snapshot_missing` because no real OKX account-equity snapshot was captured that day.
  - The UI now explicitly says `Missing OKX snapshot` instead of manufacturing a value from fixed balance, local PnL, or OKX account bills.
- ACT verification:
  - ACT historical position is visible in the OKX grouped ledger.
  - The `2026-06-28T05:00:59Z -> 2026-06-28T07:46:45Z` ACT short lifecycle has:
    - `realized_pnl=1.77827305`.
    - `linked_order_count=2`.
    - `evidence_complete=true`.
    - `trainable=true`.
    - linked OKX entry/close fills with `okx_confirmed=true`.
- Additional hardening completed:
  - `services/trade_fact_trust.py` now requires linked entry/close orders to be `okx_confirmed` or `okx_only_backfilled` before closed positions can count as trusted trade facts.
  - `web_dashboard/api/dashboard.py` daily/account local-trade diagnostics now exclude local-only closed positions without OKX-confirmed linked orders.
  - `web_dashboard/static/js/dashboard.js` shows missing OKX equity snapshots explicitly and does not display missing account-equity history as zero profit/loss.
- Validation completed locally after the hardening:
  - `rtk python -m pytest tests/test_dashboard_main_ui_contract.py::test_execution_account_ui_uses_okx_equity_pnl_not_local_trade_fallback tests/test_dashboard_main_ui_contract.py::test_dashboard_runtime_stats_do_not_regress_from_ws_packets tests/test_dashboard_error_safety.py tests/test_trade_fact_trust.py -q` -> `42 passed`.
  - `rtk python -m pytest tests/test_equity_baseline.py tests/test_run_phase3_okx_fact_sync.py tests/test_trade_history_api.py -q` -> `23 passed`.
  - `rtk python -m pytest tests/test_dashboard_error_safety.py tests/test_trade_fact_trust.py tests/test_trade_history_api.py -q` -> `52 passed`.
  - `rtk python -m pytest tests/test_equity_baseline.py tests/test_run_phase3_okx_fact_sync.py tests/test_okx_order_fact_sync.py -q` -> `23 passed`.
  - `rtk node --check web_dashboard/static/js/dashboard.js` -> passed.
  - `rtk python -m py_compile services/trade_fact_trust.py web_dashboard/api/dashboard.py tests/test_dashboard_error_safety.py` -> passed.

