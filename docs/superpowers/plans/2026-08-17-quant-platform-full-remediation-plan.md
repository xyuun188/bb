# 量化平台全量整改方案

日期：2026-08-17  
项目：`E:\code\bb`  
范围：线上市场分析、专家调用、候选池、开仓与持仓退出、资金费、训练闭环、监控和数据展示

> 本方案合并了桌面文件“新建 DOCX 文档.docx”中的问题，以及此前线上排查确认的资金费、不开仓、专家超时、重复分析、训练和持仓数据一致性问题。本文是整改计划，不代表相关问题已经修复或已经上线。

## 1. 整改目标与原则

### 1.1 目标

1. 专家数量、调用状态、分析完成状态和交叉验证结果真实可追溯。
2. 每一轮分析和不开仓结果都能定位到具体阶段、原因和耗时。
3. 已结算资金费进入真实净收益、持仓分析、持仓退出和训练标签；预计资金费进入市场候选、多空方向比较、开仓门禁和继续持仓判断。
4. 消除重复分析、长时间空档、轮次超时、专家串栏和量化工具假死。
5. 在线上真实环境完成影子、灰度和稳定性验收后再全量发布。

### 1.2 不可违反的原则

- 不通过降低门槛或绕过安全门禁制造开仓。
- 不把未调用、超时、空返回或解析失败伪装成完成。
- 不把所有失败统一归类为“不明确”。
- 不直接删除异常资金费；必须先完成 OKX 账单、合约规格和生命周期核验。
- 不把未来资金费、未来价格或未来结算结果泄漏到当前决策特征。
- 不同时放宽多个风控门禁；每次变更都必须有回归和线上验收证据。

## 2. 当前问题全量清单

### 2.1 桌面 DOCX 中确认的问题

1. 市场分析实际应有 4 个专家，但红框页面显示为 5 个。
2. 当前分析没有真正完成，且没有执行交叉验证。
3. 几乎所有分析结果都显示为“不明确”。

### 2.2 专家调用与调度问题

- 部分记录为“未调用”、超时、空返回或 JSON 解析失败。
- 固定 Token 上限可能造成 JSON 截断。
- 同一币种被反复分析，部分记录重复写入。
- 轮次之间出现几十分钟空档，单个任务可能拖住整轮。
- 持仓退出专家内容混入市场分析记录。

### 2.3 开仓与方向问题

- 连续两天没有开仓，尚未完成完整开仓漏斗审计。
- `can_open_new_entries=false` 以及未解决的 OKX 对账问题可能持续阻断新开仓。
- 当前开仓方向明显偏向做空，需要排查模型、样本、收益门禁和多空规则是否不对称。
- 候选池可能长期集中在少数币种，导致市场覆盖不足。

### 2.4 资金费、持仓和收益问题

- 资金费尚未在分析、退出、训练和展示链路统一使用。
- 资金费形成的真实盈利可能没有及时触发收益保护或平仓判断。
- 同一 OKX `posId` 对应多个生命周期时，存在资金费错归属风险。
- 高额资金费样本可能污染 short 的 CVaR、回撤和 LCB。
- 部分仓位出现名义价值、合约张数、合约面值和标记价不一致。

### 2.5 训练和线上基础设施问题

- 训练标签、资金费、手续费和生命周期事实尚未完全统一。
- 可能存在异常数据直接进入训练或未来数据泄漏。
- 线上量化工具存在超时、报错、耗时过长和假死风险。
- 服务器重启、磁盘占用、服务未自动恢复和配置漂移可能影响任务调度。

## 3. 分阶段整改计划

## P0：线上事实审计和基线冻结

只对线上真实环境进行审计，至少覆盖最近 72 小时，不先修改策略门槛。

### 3.1 专家事实审计

- 核对专家注册表、页面显示数量、应调用数量和实际调用数量。
- 当前基线为 4 个市场分析专家；配置变更必须同时更新注册表和展示口径。
- 为每次调用保存角色、请求 ID、币种、轮次、开始时间、结束时间、耗时和返回状态。
- 区分以下状态：

```text
completed
timeout
parse_failed
empty
unavailable
skipped
```

### 3.2 开仓漏斗审计

每轮必须记录：

```text
市场数据
→ 候选池
→ 币种分析
→ 专家调用
→ 交叉验证
→ 多空共识
→ 收益/风险门禁
→ 开仓合同
→ 账户与合约规格检查
→ OKX 下单
```

每一阶段记录输入数量、输出数量、过滤数量、阻断原因、异常和耗时。对两天不开仓的每一轮建立可复核报告。

### 3.3 线上服务基线

- 核对调度器、专家服务、模型服务、量化工具、数据库、缓存和 OKX 接口健康状态。
- 核对服务器重启后的自动启动顺序、定时任务和服务依赖。
- 核对磁盘、日志、临时文件、队列积压和数据库连接。
- 固化当前版本、配置摘要和模型版本，作为后续回滚基线。

## P0：专家调用和分析质量治理

### 4.1 专家数量口径

页面同时展示：

- 配置专家数；
- 本轮应调用数；
- 实际调用数；
- 成功返回数；
- 超时、失败、跳过数。

4 个专家不能显示为 5 个已调用专家；未调用的槽位必须明确显示原因。

### 4.2 完成状态和交叉验证

完整分析必须同时满足：

1. 所有必需专家已经调用，或有明确的不可用记录。
2. 返回内容通过结构化解析和字段校验。
3. 证据摘要完整，不能只有泛化文本。
4. 交叉验证已执行，并记录参与专家、冲突项和最终方向。

若专家不足、结果无效或冲突无法解决，输出 `insufficient_evidence`，不能标记为正常完成，也不能直接用于开仓。

### 4.3 “不明确”治理

允许的结果枚举：

```text
long
short
hold
unclear
```

`unclear` 必须绑定具体原因：数据不足、方向冲突、专家不可用、风险过高、预期净收益不足或交叉验证不足。原始响应、结构化结果、解析错误和证据摘要必须同时保留。

### 4.4 Token、超时和失败隔离

- 使用结构化短字段和自适应 Token 上限，取消可能造成 JSON 截断的固定上限。
- 单专家设置独立超时，整轮设置总预算；一个专家超时不能拖死其他币种。
- 仅允许有限重试，并使用幂等请求 ID，避免重试产生重复分析记录。
- 对连续失败的专家执行熔断，记录 `unavailable`，不伪装成 `unclear` 或 `completed`。

## P0：候选池、重复分析和轮次调度

- 为每次分析生成幂等键：

```text
round_id + symbol + timeframe + analysis_type
```

- 同一幂等键只能保留一条有效分析记录。
- 增加币种冷却和已分析检查，防止同一币种在同一轮或短时间内重复分析。
- 候选池采用去重、分层和轮换策略，记录每个币种入选和淘汰原因。
- 限制单币种连续占用次数，避免页面长期只出现少数币种。
- 采用并行分析、单任务超时和失败隔离；轮次到期后必须生成轮次总结。
- 监控最近一次成功分析时间、队列长度、轮次耗时和未处理任务数。

## P0：不开仓和多空方向治理

所有不开仓必须归类为明确原因：

```text
no_candidate
insufficient_evidence
direction_conflict
risk_blocked
funding_cost_blocked
execution_blocked
account_reconciliation_blocked
service_error
```

重点排查：

- `can_open_new_entries=false` 的对账对象、错误来源和恢复条件；
- 专家未调用、结果截断、交叉验证不足是否导致全部方向失效；
- 余额、杠杆、合约规格、最大持仓、冷却时间和重复订单拦截；
- 资金费、LCB、预期净收益和执行成本是否把机会全部过滤；
- 多头和空头在样本数量、置信度、风险门禁和手续费模型上是否对称。

不以“必须开仓”为验收标准；如果没有合格机会，系统必须给出可审计的无交易原因。

## P1：资金费事实、收益和持仓退出

### 6.1 资金费口径

资金费分为：

1. **已结算资金费**：来自 OKX 账单，结算时已进入账户余额，计入当前仓位生命周期真实收益。
2. **预计资金费**：根据当前费率、结算时间、方向和预计持仓时长估算，只用于未来成本预判。

已结算资金费与预计资金费必须使用不同字段保存和展示。预计值不能写入已实现收益，结算后也不能把同一笔资金费同时作为预计值和已结算值重复计算。

持仓净收益统一为：

```text
净收益
= 浮动盈亏
+ 已结算资金费
- 已发生手续费
- 预计平仓手续费
- 预计滑点
```

### 6.2 市场分析中的资金费维度

市场分析必须在候选池排序、多空方向比较、预期净收益和开仓门禁中显式使用预计资金费，不能只在持仓后处理。

每条市场分析记录至少保存：

```text
funding_rate
funding_rate_source_time
next_funding_time
funding_interval
estimated_holding_period
estimated_settlement_count
long_expected_funding_cashflow
short_expected_funding_cashflow
funding_cost_ratio
funding_risk_level
expected_net_edge_after_funding
```

其中资金费现金流统一采用“正数表示预计收入、负数表示预计支出”的口径，并按 OKX 对应合约的资金费规则、名义价值、方向和预计结算次数分别计算多头与空头结果。

市场分析中的预期净优势统一为：

```text
预期净优势
= 预期价格收益
+ 预计资金费现金流
- 预计开仓手续费
- 预计平仓手续费
- 预计滑点
```

资金费必须参与以下环节：

1. **候选池排序**：比较扣除预计资金费后的净优势，避免只按价格信号排序。
2. **多空方向比较**：同一币种同时计算 long 和 short 的资金费影响，不允许只计算最终选择方向。
3. **开仓门禁**：当预计资金费支出侵蚀预期优势时，输出 `funding_cost_blocked` 及具体金额、比例和结算时间。
4. **风险判断**：预计资金费收入只能增加收益证据，不能绕过流动性、波动、止损、仓位和账户安全门禁。
5. **页面展示**：展示当前费率、下一结算时间、多空预计资金费、资金费后净优势、数据时间和风险等级。

当费率数据缺失、过期或合约规格未通过校验时，资金费状态必须标记为 `funding_evidence_unavailable`；不得默认资金费为零后继续给出完整市场分析。

### 6.3 持仓分析中的资金费维度

每次持仓分析必须同时使用以下三类信息：

1. 当前生命周期已经结算并完成归属的资金费；
2. 从当前时间到下一次或预计退出时间的未来资金费；
3. 浮动盈亏、已发生手续费、预计退出手续费和预计滑点。

持仓分析分别输出当前可确认净收益和继续持仓预计净收益：

```text
当前可确认净收益
= 浮动盈亏
+ 当前生命周期已结算资金费
- 已发生手续费
- 预计退出手续费
- 预计退出滑点

继续持仓预计净收益
= 当前可确认净收益
+ 预计未来价格收益
+ 预计未来资金费现金流
```

持仓分析记录至少保存：

```text
settled_funding_fee
expected_future_funding_cashflow
current_lifecycle_net_pnl
projected_hold_net_pnl
next_funding_time
funding_fee_included
funding_evidence_status
```

资金费证据不完整时，允许继续执行交易所止损等硬保护，但不得把未经核验的资金费用于盈利锁定或模型训练。

### 6.4 生命周期和异常资金费

- 按 `instId`、账单 ID、时间、方向、数量和生命周期归属资金费。
- 同一 `posId` 的多个生命周期必须拆分，不能把多个仓位周期合并成一个样本。
- YB 等高比例资金费先核对 OKX 原始账单、成交数量、合约面值和结算时间；账务一致时不能擅自删除，但在归属和规格核验完成前隔离出训练集。
- 对资金费与名义价值比例异常、生命周期冲突和账单缺失建立隔离状态。

### 6.5 资金费驱动的退出

动态退出除止盈止损外，还应综合：浮动盈亏、已结算资金费、预计资金费、手续费、滑点、趋势、反向压力、盈利回撤和替代机会。

当资金费使仓位净收益达到保护标准时，生成减仓或全平候选，并记录：

```text
funding_fee_included
lifecycle_net_pnl
profit_lock_pressure
close_fraction
reason
```

## P1：合约规格和名义价值一致性

- 使用 OKX 对应 `instId` 的合约类型、`ctVal`、`ctValCcy`、`ctMult` 和标记价计算名义价值。
- 线性、反向和不同计价币种不得套用同一个公式。
- 名义价值、张数、面值和标记价必须使用接近同一时间点的数据。
- 超出容差时标记 `contract_spec_mismatch`，阻断相关证据和开仓，不允许静默继续。
- 每个异常必须保留原始值、计算值、元数据版本、时间戳和处理结论。

## P1：训练闭环整改

- 训练标签使用生命周期级真实净收益，包含平仓收益、资金费、手续费和可确认滑点。
- 训练特征只能使用决策时已可见的数据，禁止未来资金费和未来结算结果泄漏。
- 异常资金费、生命周期冲突、合约规格不一致的数据隔离并保留复核记录。
- 使用时间切分和滚动验证，不使用会混入未来样本的随机切分。
- 分开评估多头和空头的收益、胜率、CVaR、回撤、LCB 和概率校准。
- 新模型先影子运行，再小流量线上验证，禁止直接覆盖生产模型。

## P2：展示、监控和运维

### 8.1 页面展示

- 展示真实专家数量和每个专家状态。
- 展示分析是否完成、是否交叉验证、未完成原因和最终方向。
- 市场分析页不展示持仓退出专家内容。
- 展示候选池、重复分析、轮次耗时和不开仓漏斗。
- 持仓盈亏详情统一展示浮动盈亏、已实现盈亏、资金费和手续费。

### 8.2 线上监控

监控以下指标：

- 专家调用成功率、解析成功率和交叉验证完成率；
- 单专家耗时、整轮耗时、队列积压和最近成功分析时间；
- 重复记录率、长时间无分析和未处理任务数；
- 开仓漏斗各阶段通过率和阻断原因；
- 资金费对账差异和合约规格异常；
- 训练数据质量、模型版本和多空表现；
- 量化工具错误率、超时率、进程存活和磁盘占用。

服务器重启后必须自动执行服务健康检查；磁盘使用率、日志增长和临时文件需要设置告警和清理策略。

## 4. 线上验收标准

- 应调用专家数、实际调用数和页面显示数一致，当前基线差异为 0。
- `completed` 不得包含未调用、超时、空返回或解析失败记录。
- 完整分析必须有交叉验证证据；证据不足只能为 `insufficient_evidence`。
- 每条 `unclear` 都有结构化原因，不能只有一句泛化文本。
- 同一幂等键重复记录数为 0。
- 单轮分析 P95 耗时不超过 120 秒，不能出现无记录的长时间空档。
- 每条不开仓记录都能定位到具体漏斗阶段和阻断原因。
- `can_open_new_entries` 只有在未解决对账项清零并通过线上复核后才能恢复。
- 每条可进入方向决策的市场分析都包含资金费率来源时间、下一结算时间、多空预计资金费和资金费后的预期净优势。
- 市场分析缺少有效资金费证据时必须标记 `funding_evidence_unavailable`，不能默认资金费为零或标记为完整分析。
- 每条持仓分析都同时展示已结算资金费、预计未来资金费、当前生命周期净收益和继续持仓预计净收益。
- 已结算资金费与预计资金费不得重复计入；使用当时已知费率回放时，计算结果必须符合 OKX 合约规则和精度。
- 已结算资金费对账覆盖率为 100%，未归属资金费为 0。
- 名义价值异常必须全部有明确处理状态。
- 训练数据无未来信息泄漏，资金费标签可与 OKX 账单核对。
- 量化工具连续线上运行 24 至 48 小时无未处理超时、重复任务或服务假死。

## 5. 上线顺序

1. 完成线上审计和整改前基线报告。
2. 先修复专家状态、完成判定、交叉验证和不开仓漏斗记录。
3. 再处理资金费归属、持仓退出和训练标签。
4. 再处理候选池轮换、重复分析和性能问题。
5. 影子运行并完成线上验证。
6. 小流量灰度运行 24 至 48 小时。
7. 通过全部验收指标后再全量发布。

## 6. 必须交付的验收材料

- 专家调用和分析完成率报告；
- 四专家数量口径核对报告；
- 交叉验证和“不明确”原因分布报告；
- 两天不开仓的完整漏斗报告；
- 重复分析、轮次超时和长时间空档报告；
- 市场分析多空预计资金费、资金费后净优势和资金费门禁报告；
- 资金费、手续费、浮动盈亏和生命周期净收益对账报告；
- 合约规格和名义价值异常处理报告；
- 训练数据质量、资金费标签和防泄漏报告；
- 线上灰度运行日志、监控截图和回滚验证记录。

只有上述材料齐全、线上验收指标全部通过，才能宣布整改完成。

## 7. 实施与最终验收记录（2026-08-18）

本轮整改已按本方案完成代码、测试、线上同步和运行态验收。历史事实不完整的数据没有被伪造修复或删除，而是保留原始记录并隔离出训练视图；“整改完成”表示系统合同、门禁、调度和证据链已生效，不表示当前模型已经获得生产开仓授权。

### 7.1 代码与测试

- 已落地专家状态/四专家口径、完成判定、交叉验证、`unclear` 原因、候选轮换、幂等去重、轮次预算、开仓漏斗、资金费和合约估值、训练隔离、持仓收益弹窗及监控字段。
- 交叉验证最终运行策略：总预算 18 秒，本地 14B 窗口 6 秒，专用 14B → 最终交易模型 → 风险模型后备，并强制 JSON object 输出；线上版本 `ai_brain/cross_validator.py` SHA256 为 `1c77bd9fc359d93d8e4ccf3b8f214192f2649179b47a6ff592394eecac97ef7d`。
- 本地全量回归：`3365 passed, 4 skipped`；Ruff 全部通过；JavaScript `dashboard.js` 语法检查通过；`git diff --check` 通过。
- Black 全仓库检查仍会报告 343 个既有文件需要重排，未作为本次门禁，也没有引入无关格式重写。

### 7.2 线上连续性与分析覆盖

- 正式窗口：`2026-08-18 04:00:09Z` 至 `04:30:09Z`，覆盖审计 `ready=true`，交易、Dashboard、模型隧道均 `active/running`，重启数均为 0。
- 市场分析 14 条、9 个币种；4/4 专家和 6/6 交叉验证分布稳定；分析完整 14 条，服务错误和不完整记录均为 0；幂等重复 0；最高单币占比 21.43%；最大分析间隔 302.436 秒。
- 持仓分析在同一窗口持续产生 75 条记录；持仓退出内容没有串入市场分析；会诊自然样本均为完成调用，未出现 `invalid_json`、`call_failed` 或队列超时。
- 14 条市场分析最终动作均为 `hold`，原因可追溯为候选没有通过费后收益/LCB 门禁，不是服务停止或隐性丢单。当前没有执行入场，入口漏斗和 `no_candidate` 原因均有记录。

### 7.3 OKX、收益和资金费合同

- 最新只读 OKX 对账：`status=ok`，4 个问题全部归档为 fixed，未解决项、人工复核、可修复项、关键项均为 0；`can_open_new_entries=true`、`can_refresh_training=true`。
- OKX 私有 API 拉取成功，仓位、成交、保护单和合约面值均有事实证据；当前名义价值/张数/`ctVal`/标记价不一致数为 0，保护覆盖缺失和孤儿键均为 0。
- 资金费已纳入市场方向预判、持仓继续/退出收益、动态止盈止损、历史生命周期净收益及训练标签；已结算账单和未来预计资金费分开，避免重复计入。
- 线上文本完整性审计近 24 小时扫描 1507 条记录，疑似乱码记录/字段/可修复字段均为 0。

### 7.4 训练结果与安全边界

- 权威成交审计：826 个生命周期，309 个完整、517 个证据不完整；仅 309 个进入训练，利润符号为盈利 106、亏损 203；合同违规 0，异常资金费/生命周期/滑点缺口保留在隔离清单。
- 当前模型工件可运行且 challenger 对 champion 的质量比较通过，但模型阶段仍为 `canary`、执行范围 `paper_only`、`live_ml_ready=false`。生产门禁继续要求费后预期收益、LCB、执行成本和完整来源证据，故不会因“有候选”而绕过保护开真实仓。
- 当前纸面模型方向资格只包含通过质量门禁的方向；空头在当前训练证据下未达到正收益下界，不人为补齐多空比例。该限制正是此前“不开仓/方向偏差”问题的可审计解决方式。

### 7.5 运维与交付

- 线上磁盘使用率已降至 33%（272G 总量、87G 已用、185G 可用）；三项核心服务连续运行且 `NRestarts=0`。
- 线上页面回归已验证：市场分析专家和交叉验证状态、资金费证据、持仓浮动盈亏点击弹窗（已实现盈亏/浮动盈亏/资金费/手续费）和合约阻断状态均可展示，浏览器控制台无错误。
- 本节记录的是最新线上事实；任一后续部署若改变服务时间线，必须重新执行本方案的 30 分钟连续性审计和全部门禁，不得复用旧窗口结论。

### 7.6 后续闭环修复（2026-08-21）

在上一版上线后的持仓复核观察中发现，快速扫描对没有受治理退出许可的持仓会每轮重复落库 `HOLD` 记录。该问题不会直接提交订单，但会污染持仓分析页面、放大重复分析观感，并增加数据库和审计噪声。本次已补充：

- 对稳定且无动作的快速扫描按持仓组、状态指纹和 5 分钟间隔去重；原因、持仓数量/方向或退出状态变化时立即重新记录。
- 具备动态退出许可、止损/资金费等硬风险或紧急退出标记的扫描不降频，避免影响保护性退出和真实治理事件。
- 完整慢速复评本身已经是有效审计事件，不清空稳定状态指纹；慢速复评后的降级扫描只有在状态变化或间隔达到 5 分钟时才重新记录，避免“慢速复评后立即重复一条 HOLD”。
- 未解决专家重大冲突的模型退出继续受退出门禁阻断；交易所止损、资金费硬风险等保护性退出不受影响。
- 快速风险入口不再执行无专家证据的普通动态退出；只有止损、止盈和资金费硬风险可以走快速保护路径，普通动态减仓必须进入持仓复评门禁。
- 训练循环启动时主动回收已死亡训练进程遗留的 lease；保留 `training_process_interrupted` 历史事实，但新调度器可以自动接管并按重试策略继续训练。

本节代码完成后必须重新执行专项回归、全量测试、线上同步和线上验收；由于重新部署会刷新观察窗口，仍需从新版本启动时间起连续观察至少 24 小时，期间不得复用 2026-08-18 的历史窗口或宣布全部验收完成。

### 7.7 模型隧道和动态退出审计闭环（2026-08-22）

后续线上训练预检定位到一个新的底层问题：模型隧道在共享 SSH transport 上打开并发 channel 超时，可能误关闭仍承载训练长连接的 transport，造成 `RemoteProtocolError`。本次修复包括：

- 活跃 transport 上的单次 channel-open 超时不再关闭共享 transport；只有 transport 已失活、EOF 或明确 session/socket closed 才退役并由池自动恢复。
- CLI 重建隧道时保留每个端点的连接时限，量化训练端点继续使用 1800 秒长连接上限。
- 补充并发长连接、失活 transport 自动恢复和 CLI 超时配置回归测试。

线上策略审计还发现，OKX 外部回补退出记录被错误计入动态退出合同缺口，导致 `position_capacity_release` 和阶段交接误报 critical。本次修复将缺口统计限定为真正的 `dynamic_exit`，外部 OKX 回补仍保留其自身不完整合同状态和原始证据，不会掩盖事实。

本次专项回归 54 项、全量回归 `3507 passed, 4 skipped`，Ruff、compileall 和 diff-check 均通过。线上量化训练预检已完整返回且不再出现远端断连；模型服务 readiness、OKX 对账和执行合同仍按只读安全门禁运行。修复版本重新部署后必须从新的服务启动时间开始连续观察至少 24 小时，覆盖训练长连接、轮次间隔、重复分析、动态退出合同和系统审计；在观察窗口及所有 critical 门禁通过前，不得宣称整改全部完成。

### 7.8 会诊双重排队和 transport 排空收口（2026-08-22）

新版本线上会诊审计发现，深度会诊先等待独立会诊信号量，再等待全局模型容量，两个队列叠加会使请求在真正调用模型前触发 `deep consultation queue wait exceeded`，并连带产生决策模型超时。该问题不是方向冲突或收益门禁造成的，不能通过放宽交易门槛处理。本次收口包括：

- 删除会诊专用的第二个信号量，统一使用与市场、持仓专家相同的全局容量调度，保留会诊优先级和并发上限。
- 将会诊队列等待上限调整为 3 秒，单次模型尝试上限调整为 10 秒；总会诊预算仍受 6 至 18 秒边界和整轮剩余预算约束，失败继续隔离并记录为 `queue_timeout`/`call_failed`，不伪装为完成。
- 首选会诊模型不再预先扣留备用模型的 2 秒预算，首轮调用使用当前会诊剩余预算；只有首选模型真实失败后，备用候选才使用受限的保留预算，避免复杂冲突输入在首轮响应前被总预算取消。
- 回归测试改为直接占满共享模型容量，验证队列超时审计、并发会诊和资源释放，不再只覆盖已经删除的独立信号量。
- 对共享 SSH transport 的 channel-open 超时继续采用排空策略：禁止新 channel，等待现有训练长连接结束后再关闭；已失活、EOF 或明确 session/socket closed 才立即退役并自动恢复。

本次修改后专项会诊/隧道回归 45 项、全量回归 `3507 passed, 4 skipped`，Ruff、compileall 和 diff-check 均通过。该版本部署后必须重新开始至少 24 小时线上观察，重点检查会诊队列超时、模型调用失败、训练长连接断连、轮次空档、重复分析、动态退出合同和系统审计；观察窗口完成前仍不得宣布整改全部验收完成。

### 7.9 Candidate permission boundary and observation scheduling closure (2026-08-22)

This implementation pass closes two code-level gaps found during the post-deployment audit. Market observation candidates are explicitly separated from execution candidates at the single entry-filter boundary. Analysis-only or execution-unverified symbols may be analyzed and persisted, but their entry decisions carry `market_analysis_only_contract` and cannot reach an execution processor.

The observation pool is widened independently from the final expert budget. OKX availability remains an execution permission, preserving rotation and preventing long empty intervals. Shared model-capacity acquisition now honors the per-symbol analysis deadline while waiting for a slot, and local fallbacks retain `timeout`/`call_failed` status instead of being counted as completed expert calls.

Focused regression: 410 passed. Full regression: 3517 passed, 4 skipped. Static lint, compile, and diff checks passed. Online deployment and the new 24-hour observation window remain required before final acceptance.

### 7.10 Consultation deadline propagation and non-blocking OKX capability refresh (2026-08-22)

The post-deployment audit found two remaining latency defects. Deep consultation was using a
separate 6--18 second budget without inheriting the symbol analysis deadline; when the outer
round had only a few seconds left, the consultation was started and then cancelled by the
caller. The validator now receives `_analysis_deadline_monotonic`, records the remaining
budget, and emits an explicit `skipped`/`analysis_deadline_budget_exhausted` observation when
there is not enough time to start a complete consultation. It never converts this state into
`completed` or production permission.

OKX private `fetch_leverage` capability probes are now removed from the market-analysis
critical path. The market round consumes the verified cache immediately; a bounded background
refresh updates execution permissions, revoking cached permissions on definitive unavailable
responses. Symbols without current verification remain analysis-only and are hard-blocked by
the entry boundary. This preserves safety while preventing private API latency from creating
long gaps between market analyses.

Focused regression after this pass: cross-validator runtime policy 14 passed; trading-service
boundary and executor safety suites 403 passed. Full regression and online deployment remain
required before final acceptance.

### 7.11 Shadow sample durability and OKX probe scheduling closure (2026-08-22)

The online audit then confirmed that shadow-label persistence still awaited one
database transaction in the market round and abandoned the sample after the old
2-second timeout. This pass closes that durability gap at the source:

- Market analysis only enqueues a persisted decision payload and returns; a
  bounded background worker performs the database write with isolated retries.
- Worker health, queue depth, enqueue/drop counts and the last write error are
  exposed in maintenance diagnostics. A full queue is recorded as degraded and
  never reported as a successful sample.
- The repository now finds market decisions with fewer than the expected horizon
  rows. Background maintenance reconstructs the decision from durable fields and
  reuses the idempotent writer, so partial writes and process restarts are
  recoverable without duplicate samples.
- OKX private instrument capability refresh remains outside the market critical
  path. Probes are limited to a bounded batch, use a 30-second independent
  deadline, preserve verified cache permissions on temporary errors, and publish
  structured `ok`/`timeout`/`degraded` state with the next retry time instead of
  repeated warning noise. Unverified symbols remain analysis-only and cannot
  enter execution.

Focused regression for the new queue/recovery behavior and the shortlist
boundaries passes. Full regression, online deployment and the fresh 24-hour
observation window are still required; this section is an implementation record,
not a final acceptance declaration.

### 7.12 Coverage backlog, prewarm backoff and independent market budget (2026-08-22)

The final scheduling audit found four remaining root-level risks: shadow recovery
could scan too frequently, indicator prewarm failures could retry without a
symbol-aware backoff, due coverage could remain behind the 30-minute target, and
the market-only loop could inherit a zero or reduced budget from the position
snapshot even though the loops are independent. The implementation now:

- throttles shadow recovery to a bounded five-minute interval with a four-hour
  lookback and a small recovery batch;
- applies exponential per-symbol prewarm backoff and normalizes exchange symbol
  formats before deciding whether a batch failed;
- reserves a second bounded coverage slot only for an accumulated backlog of
  previously observed symbols, while keeping first-discovery behavior stable;
- keeps market-only analysis budget independent from open-position review and
  records `analysis_scope` in both diagnostics and logs;
- tracks repeated blockable defer reasons separately from global scheduling
  deferrals, blocks only after consecutive failures, preserves the concrete
  reason, and exposes blocked-candidate diagnostics.

The focused regression for these boundaries is to be run only after this entire
implementation pass is complete. Online deployment, post-deploy verification,
and a fresh 24-hour observation window remain required before final acceptance.

### 7.13 Unified verification and online rollout (2026-08-22)

The implementation pass is now complete before verification. The unified local
verification completed with:

- focused scheduling/prewarm/defer/coverage regression: `385 passed`;
- full repository regression: `3525 passed, 4 skipped, 1 warning`;
- Ruff, compileall, and `git diff --check`: passed.

The complete source set was synchronized to the online server with the split
service deployment. Immediately after restart, all three platform services
were active with zero restarts, OKX entry/training permissions were allowed
with zero unresolved reconciliation items, and the first 30-minute online
coverage audit showed zero duplicate analyses within the 10-minute cooldown,
33 distinct market symbols, and a maximum decision gap of 158 seconds. The
coverage audit is intentionally not marked complete yet because the new
service window had only been running for about two minutes; the due backlog and
continuity gates must be observed from this deployment forward for at least 24
hours. Model quality gates remain authoritative: the current model is not
production-authorized while fee-after-return, LCB, and profit-factor blockers
remain present, so no entry is forced to make the count look healthy.

### 7.14 Training interruption recovery and final verification pass (2026-08-22)

The post-deployment audit found that a service restart correctly preserved the
historical `training_process_interrupted` event, but left the affected model
rows without `next_check_at`.  That made the scheduler warning durable and
made recovery dependent on an unrelated lease attempt.  The state contract now
assigns a bounded retry time both when a dead `checking`/`running` process is
recovered and when an existing interrupted row is encountered.  Existing
interrupted rows are scheduled idempotently without duplicating the historical
interruption event.  A regression test covers both paths.

The same verification pass removed an invalid no-placeholder f-string from the
remote FinQuant concurrency probe.  Focused regression passed (`50 passed`),
full repository regression passed (`3533 passed, 4 skipped, 1 warning`), and
Ruff, compileall, and `git diff --check` all passed.

The corrected source was synchronized to the online server twice, with the
final deployment reporting model tunnels, trading, and Dashboard active.  The
online state after deployment is `status=ok`; all trainable models are
`skipped`/healthy with a future `next_check_at`, and the OKX reconciliation gate
reports zero unresolved items.  The post-deployment tunnel log contains only
the ready event and no new `Timeout opening channel`, `RemoteProtocolError`, or
transport-draining errors.  A five-minute coverage smoke check observed nine
distinct market symbols, zero duplicate analyses within the ten-minute
cooldown, and seven position-review records.

The plan remains active until the fresh deployment window completes the
required 24-hour online continuity, coverage, expert-call, training-heartbeat,
and audit gates.  Current model quality blockers (fee-after-return, LCB, and
profit-factor evidence) remain authoritative and were not bypassed.

### 7.15 Execution contract audit timeout closure (2026-08-22)

The post-deployment audit found that the dynamic execution-contract report was
queried twice in one audit graph.  A slow database read could exceed the
20-second section deadline, and the resulting exception was then misread by the
Phase 3 gate as missing contract fields.  This pass closes that failure chain:

- The closed-loop and execution-contract cards share one short-lived,
  read-only report and one in-flight task.  A repeated timeout is not allowed to
  start a second identical database scan during the same observation window.
- The contract service now loads only the columns required by validation for
  orders and positions instead of full ORM rows and large unused JSON payloads.
- Successful reports expose `report_available=true`.  Timeout, cancellation or
  database failures expose `report_available=false`, `error_type` and `timeout`
  without fabricating a contract violation.
- Phase 3 go/no-go emits one authoritative
  `dynamic_return_contract_unavailable` blocker for an unavailable report and
  does not add a second false `policy_incomplete` blocker.
- Online daily reconciliation after deployment reported
  `can_open_new_entries=true`, `can_refresh_training=true`, and an execution
  contract card with `status=ok` and zero contract violations.  The focused
  online regression passed 59 tests.

The first narrow projection still left the recent-decision query as the
dominant database cost: PostgreSQL was decompressing the large decision JSON
payload for every eligible row before applying the outer limit.  The query is
now two-stage: it first selects a bounded, ordered set of decision IDs, then
joins that bounded set to load only the fields needed by the contract validator.
The online `EXPLAIN ANALYZE`/service measurement improved the decision read from
about 20.95 seconds to about 4.10 seconds, while preserving the same validation
semantics.  A regression test asserts that the bounded ID query and its limit
remain in place.

The focused regression for the complete change passes `60 tests`.  The isolated
full online regression is rerun after the final deployment with an explicit
pytest root outside the production symlink, so production `.env` and source
discovery cannot contaminate the result.  The new deployment still requires a
fresh 24-hour online continuity, coverage, expert-call, training-heartbeat and
audit observation before final acceptance; no model-quality or OKX safety gate
is bypassed to increase order count.

### 7.16 Clean online regression and current acceptance state (2026-08-22)

本轮按照“先完成整改、再测试”的顺序完成了最终验证：

- 本地全量回归：`3538 passed, 4 skipped`；没有
  `PytestUnhandledThreadExceptionWarning`，仅保留第三方 `StarletteDeprecationWarning`。
- 线上全新隔离目录全量回归：`3542 passed`，退出码为 0；隔离目录位于生产源码和生产 `.env` 之外，排除了环境污染，未发现业务失败、线程回调异常或测试超时。
- 线上最新只读 OKX 对账：`status=ok`，`can_open_new_entries=true`、`can_refresh_training=true`，未解决项、人工复核项和可修复项均为 0。
- 线上执行合同审计：`report_available=true`、`status=ok`，合同违规数为 0；审计耗时约 4 秒，未再出现“超时被误判为字段缺失”。
- 线上服务复核：交易服务、Dashboard 和模型隧道均正常运行；本轮验证没有改动线上交易数据，也没有绕过 OKX、训练或模型质量门禁。

上述结果证明本轮代码和部署已通过即时功能验收。计划中的“连续运行 24--48 小时无超时、重复任务、服务假死，并覆盖训练心跳、轮次连续性、动态退出合同和系统审计”仍是时间性观察项，必须从本次部署时间起完成后，才能将本方案标记为最终全部验收通过。当前模型质量门禁仍保持原状态，不因测试通过而强行开仓。

### 7.17 后台异步任务生命周期闭环与正式同步验收（2026-08-22）

本轮继续追查线上隔离回归中残留的 `aiosqlite` 线程回调异常，确认不是业务断言失败，而是两个后台刷新链路在测试事件循环关闭后仍持有数据库工作：行情快照持久化任务，以及策略上下文的绩效/学习刷新任务。整改已落到生产生命周期边界：

- `DataService` 只有在 `start()` 后才接受行情持久化回调，并登记所有持久化任务；`stop()` 会取消、等待并清空任务集合。
- `TradingService.stop()` 在关闭数据库和模型客户端前，统一取消并等待策略上下文绩效刷新、学习上下文刷新任务。
- 测试夹具在每个异步测试结束时先取消当前循环中残留的测试任务，再释放共享数据库引擎；新增回归测试锁定策略上下文任务必须被回收。

本地全量回归通过 `3539 passed, 4 skipped, 1 warning`；线上全新隔离目录使用严格门禁 `-W error::pytest.PytestUnhandledThreadExceptionWarning` 回归通过 `3543 passed, 1 warning`。两端仅保留第三方 `StarletteDeprecationWarning`，没有 `PytestUnhandledThreadExceptionWarning`、`Event loop is closed`、业务断言失败或测试超时。诊断探针确认 SQLite worker 只在真实数据库连接期间短暂存在，并在事件循环关闭前确定性退出，不再残留线程回调异常。

Phase 3 正式维护同步为订单事实同步配置独立的 60 秒上限，实时交易默认超时没有被放宽。线上正式同步全部阶段完成，`deferred_stages=[]`、`error=null`、`okx_pull_available=true`；最新对账未解决项为 0，`can_open_new_entries=true`、`can_refresh_training=true`。对账总状态仍为 `warning`，仅因为两项信息级历史残留和方向集中度处于观察态；它们不要求人工处理、不阻断开仓或训练，也没有被伪装成 `ok`。

最终线上即时健康检查确认交易服务、Dashboard、模型隧道和每日对账 timer 均为 active/enabled，最近 10 分钟没有 hard deadline、traceback、unhandled、`Timeout opening channel`、`RemoteProtocolError` 或 transport-draining 日志。执行合同最近窗口违规数为 0，训练调度状态为 `ok`、心跳未过期、模型运行时和工件均可用。当前没有新增开仓的直接原因仍是模型费后平均收益、收益下界和 profit factor 未通过质量门禁，不是 OKX、后台任务、轮次调度或模型隧道故障；这些质量门禁没有被人为放宽。

这仍不替代计划要求的连续 24--48 小时观察。观察期间模型质量门禁、OKX 对账门禁和训练调度门禁继续按合同执行，不通过人为放宽阈值来制造开仓数量。

### 7.18 会诊生命周期复用与硬截止闭环（2026-08-22）

本轮在最终统一验证前完成了剩余运行时边界整改：

- 深度会诊在普通共享模型容量已满时最多使用一个受控溢出租约；取消时仍会释放，且不会降低市场分析或持仓分析的普通容量。
- 会诊外层、模型调用和候选尝试均使用硬截止；超时或取消会取消并消费迟到任务，队列超时、模型超时、预算跳过和调用失败保持不同状态，不会把未调用记录伪装成 completed。
- 持仓会诊复用指纹覆盖规范化 symbol、全部匹配的 OKX 生命周期、方向、数量/张数、合约面值、入场/标记价、浮动盈亏、已结算及预期资金费、手续费、名义价值和资金费证据。缓存只复用短期 completed 结果；没有权威匹配持仓时禁用复用。
- 非字典 OKX `info`、空 symbol 和不可直接 JSON 序列化的上下文均采用 fail-safe 处理；底层 OKX 快照解析器遇到异常 `info` 结构不会使整条持仓上下文失败。

最终代码状态验证结果：

- 本地专项回归：`116 passed`。
- 本地全量回归：`3553 passed, 4 skipped, 1 warning`；唯一 warning 为第三方 Starlette/httpx 弃用提示。
- 线上隔离专项回归：`116 passed`。
- 线上隔离全量回归：`3557 passed, 1 warning`；唯一 warning 为同一第三方弃用提示。
- `compileall`、Ruff、`git diff --check` 均通过。
- 最终线上部署约发生于 `2026-08-22 23:29 UTC`；交易服务、Dashboard 和模型隧道均为 `active/running`，`NRestarts=0`，批准的三个模型服务均为 active。
- 部署后只读巡检确认 `can_open_new_entries=true`、`can_refresh_training=true`，OKX unresolved 为 0，保护订单覆盖缺口为 0，执行合同违规为 0，最近服务 warning 日志为空；训练心跳未过期，模型无超时记录。
- 部署后 30 分钟策略巡检记录 `123` 条分析，执行合同违规 0、废弃策略字段 0、重复执行 0；没有新增开仓是模型质量门禁（fee-after-return、收益 LCB、profit factor）真实阻止的结果，未放宽门禁。

本节是代码整改和即时线上验证记录，不是最终验收声明。新的观察窗口从最终部署时间重新开始，至少需要连续 24 小时（计划目标为 24--48 小时）同时满足分析连续性、专家调用、队列/模型超时、重复分析、训练心跳、OKX 对账、执行合同和系统审计门禁；观察窗口完成前，整改计划必须保持 active。
