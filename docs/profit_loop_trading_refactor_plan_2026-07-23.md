# 盈利闭环交易系统重构计划

日期：2026-07-23

项目根目录：`E:\code\bb`

## 1. 重构目标

把系统从“模型晋升前交易停滞、模型训练缺少真实盈利闭环、状态判断分散”的结构，重构为“规则小仓先真实交易、模型旁路学习、真实净收益驱动晋升、无效旧逻辑及时删除”的闭环系统。

最终目标只有一个：持续产生可追溯的真实净收益证据，并用 `net_return_after_all_cost_pct` 驱动训练、晋升、降级和交易权限。

## 2. 重构后的交易逻辑

交易只保留三种模式：

- `observe`：只观察，不下单。
- `live_rules_canary`：规则和风控主导的小仓真实交易；模型只能旁路预测和学习，不能直接决定交易。
- `live_ml`：模型通过真实净收益晋升后，才允许参与方向、筛选、仓位或退出决策。

统一交易顺序：

1. 市场信号进入。
2. `production_trade_gate` 判断当前模式和权限。
3. 规则或模型生成交易候选。
4. 风控服务裁决仓位、最大亏损、最大持仓数和冷却时间。
5. 执行服务下单。
6. 平仓后回写 OKX 真实订单、成交、手续费、资金费、滑点和 PnL。
7. 真实结果进入训练数据、策略复盘和模型晋升评估。

模型未晋升前，系统仍然可以通过 `live_rules_canary` 小仓真实交易。这样不会等模型晋升才有数据，也不会让未验证模型直接造成大额亏损。

## 3. 数据清理原则

准确结论：清掉旧的衍生训练数据、旧 artifact、旧 cursor 和旧 dashboard cache，会让后续训练更干净；但不能删除 OKX 订单、成交、持仓、平仓、手续费、资金费、realized PnL 和审计日志这些事实数据。

执行原则：

- 事实数据保留。
- 衍生数据重建。
- 没有真实盈亏闭环的数据隔离，不参与晋升。
- 训练主目标统一为 `net_return_after_all_cost_pct`。
- `net_return_after_cost_pct`、`realized_net_return_pct`、`fee_after_return` 已从运行代码和新衍生数据中删除；历史数据重建时直接丢弃这些旧字段，不做双读兼容。

## 4. 训练数据闭环

每条可训练交易样本必须包含：

- `symbol`
- `side`
- `entry_order_id`
- `close_order_id`
- `entry_price`
- `close_price`
- `quantity`
- `notional`
- `entry_fee`
- `close_fee`
- `funding_fee`
- `slippage`
- `realized_pnl`
- `net_return_after_all_cost_pct`
- `holding_minutes`
- `decision_authority`
- `model_shadow_prediction`
- `evidence_fingerprint`

训练必须同时吸收盈利样本和亏损样本。亏损不是直接“惩罚模型”，而是先做归因：

- 规则开仓导致的亏损，归因给规则采样和市场条件。
- 模型旁路预测曾提前避开亏损，记为模型加分证据。
- 模型旁路预测支持亏损方向，记为模型扣分证据。
- 模型接管后产生的亏损，才记为模型实盘失败。

## 5. 模型晋升标准

晋升不再以单纯准确率为核心。准确率只做诊断，不能单独决定上线。

晋升必须看：

- 真实净收益均值：`net_return_after_all_cost_pct`
- Profit Factor
- 最大回撤
- 尾部亏损
- long / short 分方向表现
- 收益下界 LCB
- 样本量和时间覆盖
- 是否有真实 OKX 证据链

模型晋升前的小仓亏损不会直接阻断模型晋升；只有当证据显示模型本身的旁路判断或实盘判断持续降低真实净收益时，才影响晋升。

## 6. 统一交易闸门

新增唯一权威入口：

`services/production_trade_gate.py`

统一输出：

```json
{
  "can_trade": true,
  "mode": "observe | live_rules_canary | live_ml | blocked",
  "decision_authority": "none | rules | model",
  "model_can_influence": false,
  "reason": "ok | data_insufficient | model_not_profitable | risk_blocked | okx_unhealthy",
  "risk": {
    "max_notional_usdt": 10,
    "max_open_positions": 1
  },
  "evidence": {}
}
```

执行、策略、dashboard、审计、训练调度都只读取这个闸门结果，不再各自拼交易资格判断。

## 7. 需要闭环处理的功能

专家记忆：

- 输入：真实复盘结论和成交事实。
- 输出：结构化经验。
- 回写：下次决策辅助。

影子复盘：

- 输入：模型旁路预测和当时决策。
- 输出：未来 10 / 30 / 60 分钟真实行情结果。
- 回写：训练样本和模型晋升证据。

开仓演练：

- 输入：候选信号。
- 输出：是否允许开仓、允许原因或拒绝原因。
- 回写：风控统计和漏斗统计。

策略复盘：

- 输入：真实订单、真实平仓、真实成本。
- 输出：真实净盈亏归因。
- 回写：策略学习和训练数据。

策略调度：

- 输入：调度条件、交易模式、服务健康。
- 输出：执行结果、失败、超时、降级。
- 回写：调度健康状态和超时治理。

## 8. 保留、重构和删除

保留：

- OKX 订单、成交、持仓和平仓事实。
- 真实手续费、资金费、滑点、realized PnL。
- 审计日志和证据 fingerprint。
- 已经能证明真实闭环的数据生成逻辑。

必须重构：

- 分散的 production eligibility 判断。
- 多处 shadow / canary / live 分支。
- 模型晋升前无法正常交易的流程。
- 训练数据字段混乱和旧收益字段优先级。
- dashboard 把内部 blocker 当主状态展示的逻辑。
- 响应超时后缺少降级和恢复的调度链路。

重构完成后删除：

- 不进入新闭环的旧 blocker 拼接逻辑。
- 不再使用的旧 training cursor 兼容分支。
- 只展示、不回写、不参与决策的 helper。
- 旧 dashboard 状态词和旧收益字段主读逻辑。
- 无人调用的 service 和 script。
- 与 `production_trade_gate` 重复的交易资格判断。

重点清理对象：

- `services/entry_opportunity_scoring.py`
- `services/live_ml_profit_contract.py`
- `services/trading_policies.py`
- `services/execution_service.py`
- `services/ml_signal_service.py`
- `web_dashboard/static/js/dashboard.js`
- `web_dashboard/static/js/training.js`

## 9. 实施顺序

1. 冻结旧衍生训练层：旧样本、旧 artifact、旧 cursor、旧 dashboard cache 不再参与晋升。
2. 建立干净训练视图：只允许真实闭环样本进入 `net_return_after_all_cost_pct` 训练目标。
3. 建立 `production_trade_gate`：统一交易模式、权限、风控和 blocker。
4. 接通 `live_rules_canary`：模型未晋升前也能规则小仓真实交易。
5. 回写真实交易样本：打通订单、成交、手续费、资金费、滑点、平仓 PnL。
6. 训练吸收亏损和盈利：按归因更新模型证据，不把规则亏损误记到模型头上。
7. 以真实净收益晋升：用盈利率、Profit Factor、回撤、尾损和 LCB 判断。
8. 删除旧逻辑：每完成一个新闭环，就删除对应旧分支，避免项目继续堆积。

## 10. 验收标准

- 模型未晋升前，系统能通过规则小仓真实交易。
- 每笔交易都能追溯到 OKX 真实事实。
- 亏损样本进入训练，并能判断模型当时是否应该避开。
- 模型晋升只看真实净盈利能力，不看单纯准确率。
- 交易权限只有一个权威入口。
- Dashboard 展示交易可执行状态，而不是内部 blocker 噪音。
- 旧逻辑不堆积；未进入新闭环的代码必须删除。

## 11. 首批落地任务

- 修正训练收益显示和 registry 输出，主读 `net_return_after_all_cost_pct`。
- 修正本地 AI tools 服务的实际收益分布读取，避免继续依赖旧字段。
- 建立 `production_trade_gate.py`。
- 接入规则小仓 `live_rules_canary`。
- 清理重复交易资格判断。
- 简化 dashboard 主状态。
- 建立旧衍生训练数据清理脚本和重建流程。

## 12. `live_rules_canary` 已落地合同

规则小仓实盘已拆分为四个唯一权威阶段：

1. `live_rules_canary_signal`：只读当前技术特征生成方向；原模型结果写入 `model_shadow_decision`，不能改变真实 action。
2. `profit_risk_sizing`：只读统一交易闸门、OKX 当前账户/档位/订单簿、合约规格和市场压力损失；模型仓位、杠杆和收益声明不参与定仓。若风险上限低于 OKX 最小合约金额，当前候选直接归零并继续尝试其他可成交币种。
3. `live_rules_canary_contract`：执行前和成交后共用同一个合同构建器，同时验证方向来源、1 倍杠杆、风险代数、执行成本、名义金额上限和 OKX 最小合约金额；规格不完整或计算不一致时禁止提交。
4. OKX 平仓事实回写：交易责任标记为 `decision_authority=rules`；模型旁路预测单独生成 `model_shadow_alignment`，用真实 `net_return_after_all_cost_pct` 学习“支持了亏损方向”或“避开了亏损方向”。

上述四阶段不读取 `live_ml_profit_contract` 才能运行，也不允许旧 `net_return_after_cost_pct` 进入机会评分、定仓或收益分布组合。规则小仓的单笔风险预算独立取自当日剩余亏损预算和最大并发仓位，再由压力损失反推名义金额，避免风险预算与仓位互相定义。

## 13. 本轮晋升与旧逻辑清理结果

- 模型晋升唯一权限字段为 `live_ml_ready`；`allow_live_position_influence` 和 `production_influence_authorized` 已从代码、接口和测试中删除。
- 晋升收益只读取两类模型证据：影子样本中明确记录的 `model_shadow_action` 费后收益，以及 `decision_authority=model` 的 OKX 权威成交收益。
- `decision_authority=rules` 的真实成交收益不进入模型收益分布，只保留真实手续费、资金费、滑点和证据指纹，用于规则归因、执行成本校准和 `model_shadow_alignment` 诊断。
- 模型影子方向缺失时直接隔离，不能从规则成交方向、旧字段或默认值推导。
- 晋升报告的均值、下四分位和 Profit Factor 使用样本相关性权重；规则亏损不会通过另一条 readiness 或远端 artifact 校验链间接惩罚模型。
- `model_stage` 不再是训练请求的授权输入；`live_mutation` 已从运行代码、接口和预检 blocker 中删除，artifact 只按晋升报告生成 candidate -> shadow -> canary -> active。
- registry 和远端量化服务已升级为新版本，旧 artifact 不迁移、不双读，必须通过显式衍生数据清理后重新生成。

## 14. 派生训练层彻底重置合同

- 唯一清理入口为 `python scripts/reset_training_derived_state.py --apply --confirm RESET_TRAINING_DERIVED_STATE`；默认只输出计划，不执行删除。
- 清理对象只有影子样本、复盘、专家记忆、策略快照、模型收益快照、权益缓存、模型 artifact、训练 cursor、scheduler state、向量索引和 Dashboard 最新缓存。
- `orders`、`positions`、`okx_position_history`、`okx_account_bills`、`ai_decisions`、`strategy_learning_events`、`risk_events`、费用、资金费、滑点、realized PnL、用户和密钥审计永不删除。
- 清理完成后原子写入 `data/training_epoch.json`；影子、OKX 真实成交、Kline 序列、新闻和社交训练加载器都只读取该 epoch 之后的数据。
- 删除 `scripts/phase3_cold_start_reset.py`，不再提供删除 paper 订单、持仓、决策或重置虚拟账户的旧入口。
- 删除 OKX authoritative sync 和 order fact sync 对 `phase3_cold_start_reset_marker.json` 的读取；旧 marker 不能再截断或隐藏原始成交事实。
- 删除只服务于废弃 `strategy_learning_state.json` 的乱码修复脚本；不保留弃用壳、迁移器或双读路径。
- 执行清理前必须停止 `bb-paper-trading.service`、`bb-model-tunnels.service` 和 `bb-dashboard.service`；清理后先生成新样本，再进行 preflight、正式训练和收益晋升评估。

## 15. 当前训练纪元唯一口径

- `data/training_epoch.json` 是训练、晋升和当前交易契约审计的唯一时间边界；marker 缺失或损坏时必须阻断，不再回退到固定 Phase 3 日期。
- Dashboard、本地 ML、本地 AI tools 和系统审计只发布 `training_shadow_sample_count`、`training_trade_sample_count`、`completed_*`、`last_trained_*` 和 `artifact_training_*`；删除 `phase3_*`、`legacy_*`、`raw_*` 训练计数双读。
- `ExpertMemoryService`、FinQuant 专家 LoRA、专家影子评估、影子质量隔离和策略历史回放只读取当前 epoch 数据；重置前交易不得重新生成当前复盘、专家记忆或晋升证据。
- `TradeExecutionContractService` 使用 `max(请求窗口起点, 当前 epoch)`；epoch 前合同违规继续保留为历史审计事实，但不能永久阻断当前规则小仓开仓。
- 向量记忆缺少 epoch 时返回明确错误状态并禁止检索、重建和清理操作；禁用状态接口仍可读，不得因 marker 缺失返回 HTTP 500。

## 16. 当前 OKX 审计唯一时间边界

- `OkxTradeFactIntegrityService`、`OkxAuthoritativeSyncService` 和 Dashboard 当前对账统一使用 `max(配置回看起点, 当前 training epoch)`；epoch 前问题只留在历史审计，不得阻断当前交易或训练。
- 删除合约面值缺失时使用 `ctVal=1` 的默认回退。数量一致性只接受 OKX `base_quantity`，或 `filled_contracts * ctVal` 的可验证换算。
- 缺少 `base_quantity/ctVal` 时明确输出 `contract_specification_evidence_missing`，禁止生成虚假的数量差异和名义金额差异；该事实必须补齐证据后才能进入训练。
- 当前 epoch marker 缺失或损坏时三个当前态审计全部失败关闭，不读取固定日期、不读取旧 marker、不放行历史窗口。

## 17. 训练策略与模型授权唯一字段

- 训练策略只保留 `current_training_epoch_only`，唯一常量为 `services.training_epoch.CURRENT_TRAINING_EPOCH_POLICY`；删除 `clean_training_view_only` 和对应 Phase 3 常量。
- 删除训练接口和预检报告顶层的 `raw_*`、`trainable_*`、`quarantined_*` 重复计数；当前训练样本只发布 `training_shadow_sample_count` 和 `training_trade_sample_count`，隔离统计留在质量报告。
- 模型生产授权只保留 `live_ml_ready`；删除 `live_mutation`、`live_trading_mutation` 和 `live_influence`，Dashboard、registry、远端量化客户端和回放统一读取新字段。
- 单次预测能否参与当前决策使用 `prediction_eligible` / `ml_prediction_eligible` 表示，但它不能授予生产权限；生产权限仍只来自 `production_trade_gate`。
- 盈亏归因不再接受默认 ML 影响开关。只有当 `production_trade_gate` 同时满足 `can_trade=true`、`mode=live_ml`、`decision_authority=model`、`model_can_influence=true` 时，交易才标记为模型生产责任；无闸门和规则小仓交易一律不归责给模型。
- 本地量化 API 没有当前 epoch artifact 时必须显式返回 `trained=false`、`live_ml_ready=false`、`production_permission=false`、`production_eligible=false`，不得伪造 `loss_probability` 或收益预测；部署 smoke 同时验证无 artifact 的观察态和有 artifact 的晋升态。

## 18. 生产交易授权单一路径

- `production_trade_gate` 已升级为 `2026-07-24.profit-loop-trade-gate.v3`；生产开仓只接受当前版本，不兼容旧版本、不接受缺字段字典。
- 唯一合法生产模式只有 `live_rules_canary + rules + model_can_influence=false` 和 `live_ml + model + model_can_influence=true`；模式、权责或版本不一致时统一失败关闭。
- `ExecutionService` 对非模拟开仓强制要求权威门禁提供器；缺少提供器、返回空值、返回关闭门禁或返回旧门禁时，必须在获取 OKX 执行器前拒绝。
- `EntryPolicy` 和最终执行合同只复核同一个门禁结果；`live_ml_profit_contract` 只提供收益安全证据，不能单独授予生产权限。
- 仓位计算、价格保护、候选排序、规则信号、成交合同和训练责任归因全部调用同一个门禁校验器；删除各模块手写的规则/模型授权判断和旧阻断码。
- 模拟交易不读取生产门禁；进入执行服务时会清除裁决载荷中夹带的生产门禁，避免模拟样本被错误标记为模型实盘责任。
- 策略学习只提供历史先验上下文，不提供生产授权。`production_influence_enabled` / `production_influence_eligible` 已删除并替换为 `historical_prior_context_enabled` / `historical_prior_context_eligible`，旧键不迁移、不双读。
- Dashboard、持续策略路由、纸面冠军和线上健康检查统一使用“历史先验上下文”命名，并继续显式声明 `can_authorize_entry=false`、`production_permission=false`。

## 19. 删除无执行效果的动态模型路由

- 原 `services/model_dynamic_routing.py` 在全部专家已经调用完成后才生成报告，始终选择全部专家、跳过 0 个专家、理论调用减少量为 0，不会改变任何真实执行路径。
- 该功能只向决策载荷写入 `dynamic_model_routing`，再由系统巡检重复统计 shadow/readiness 和永远为 false 的 mutation 字段；它不是路由器，也不产生训练或盈利闭环价值。
- 已删除动态路由服务、ensemble 决策载荷写入、`/model-dynamic-routing/status` 接口、系统巡检卡、依赖图节点、Dashboard 映射和原专用测试，不保留弃用接口或兼容返回。
- `live_route_mutation`、`applied_to_live_calls`、`can_apply_live_route` 和 `unsafe_live_mutation_attempts` 已从运行代码删除；模型生产授权继续只由 `production_trade_gate v3` 和 `live_ml_ready` 决定。
- 模型专家健康与竞赛仍保留为只读质量证据，但它们直接服务于训练诊断和策略决策，不再指向一个不存在执行效果的中间路由节点。
- 新增删除契约测试，持续验证服务文件、ensemble 源码、系统巡检卡/API 和 Dashboard 都不能重新出现该功能。

## 20. 完整保留 OKX 成交 ID 事实

- 线上 `order_fact_sync` 的 SQLAlchemy autoflush 降级根因不是事务查询顺序，而是 `orders.okx_trade_ids VARCHAR(500)` 无法容纳一个订单的 57 个真实 OKX trade IDs。
- 禁止使用 `session.no_autoflush` 隐藏长度错误，也禁止截断 ID 串；全部成交 ID 都属于训练、费用、成交数量和责任归因的权威证据。
- `Order.okx_trade_ids` 已改为无长度上限 `TEXT`；SQLite 新库直接创建 `TEXT`，PostgreSQL 启动迁移将现有非 text 列原位转换为 `TEXT`。
- `okx_raw_fills.trade_ids` 与 `okx_trade_ids` 继续保存同一完整 ID 集，训练和审计仍可按逗号拆分，不引入双字段或兼容读取。
- 回归测试使用线上同规模的 57 个 trade IDs，验证字符串长度超过 500 且顺序、数量和原始事实完全保留。

## 21. 人工暂停与自动市场扫描单一路径

- 本轮确认生产交易门禁已经放行，但进程共享的 `data/trading-control-state.json` 仍保存此前人工设置的 `paused=true`；交易循环因此在权威生产门禁之外停止新市场分析，长期输出 `scan_symbol_count=0`。这不是模型晋升或余额风控导致，而是未恢复的人工控制状态。
- 保留唯一有实际作用的人工暂停开关：暂停只停止新市场分析和新开仓，已有仓位继续行情刷新、复盘、止盈止损与平仓；恢复操作立即允许下一轮自动市场扫描。人工紧急暂停不得被服务重启或健康门禁自动解除。
- 自动全市场扫描改为固定执行路径。删除 `scan_mode` 持久化字段、`/control/scan-mode` 接口、Dashboard 假切换状态、`is_auto_scan` 条件分支、永远回写自动模式的 `switch_to_manual`，以及仅由已删除手动分支调用的 `MarketDirectEntryProcessor` 和专用测试。
- 模型身份只保留 `active_model_name` / `active_model`。删除 `live_model_name` 的持久化兼容读取、模型注册表兼容方法、`live_model` API 重复返回、`/control/select-model` 固定值接口和无调用前端函数；纸面与实盘执行账户共享同一活动模型，不再暴露第二套模型指针。
- 交易控制状态文件后续只写入 `mode`、`paused`、`active_model_name` 和 `mode_changed_at`。旧 `scan_mode`、`live_model_name` 即使仍存在于历史文件中也不再读取；首次暂停、恢复、账户切换或模型选择后会以唯一字段集合覆盖旧文件。
- 新增删除契约测试，持续禁止旧扫描入口、旧模型别名和旧直连执行器回流。部署验收必须同时满足 `paused=false`、`run_market_analysis=true`、`scan_symbol_count>0`，不能只凭生产门禁状态推断交易循环已恢复。
- 线上继续验收发现实盘候选全部被 `okx_private_entry_instrument_probe_failed` 拒绝，精确错误为 `OKX API credentials are not configured`；加密配置服务本身正常，但线上仅配置 paper/demo 三项凭据，live 的 API Key、Secret、Passphrase 均缺失。
- 生产门禁升级为失败关闭：必须明确满足 `execution_mode=live`、三项 live 凭据齐全、OKX 当前同步为 `status=ok`，或 `status=degraded` 但明确存在新鲜成功快照且该快照覆盖随后失败，同时 `can_open_new_entries=true`；否则返回 `okx_execution_mode_not_live`、`okx_live_credentials_missing`、`okx_unhealthy`、`okx_new_entries_blocked` 或 `okx_status_not_ok`，不再把空字典和未明确失败误判为健康。
- live 凭据缺失时，交易循环在新市场分析前直接给出缺失字段并停止扫描，禁止继续消耗行情、排序和私有接口探测资源。没有真实 live 凭据时只能运行 OKX demo 交易，系统不得伪装成实盘可交易。
- 2026-07-24 线上部署验收：控制状态已明确恢复为 `mode=paper`、`active_model_name=ensemble_trader`、`paused=false`；交易心跳新鲜且无 market error，当前自动轮次 `run_market_analysis=true`、`scan_symbol_count=240`、`feature_fetch_requested_count=48`、`feature_valid_count=8`、`rank_selected_count=2`，已进入 `market_ai:ETH/USDT`。这证明模型晋升前的 OKX demo 规则交易路径已恢复实际分析候选，不再被旧暂停或已删除手动模式阻断。
- 同一线上验收确认 v3 对缺少 live 凭据返回 `can_trade=false`、`mode=blocked`、`reason=okx_live_credentials_missing`，且服务器旧 `services/market_direct_entry_processor.py` 已不存在。真实小仓实盘必须先由用户在 Dashboard 安全配置三项 OKX live 凭据；在此之前系统只运行 demo，禁止自动复制 paper 密钥或绕过门禁。

## 22. 删除无效单模型竞争与错误绩效快照

- `CompetitionService` 已不具备竞争前提：执行模型固定为唯一 `ensemble_trader`，paper/live 启动器也只把该模型加入活动集合；每小时给唯一模型排第一既不能晋升模型，也不能改变权威生产授权。
- 该服务的收益口径不符合训练合同：它从持仓 `realized_pnl` 再次减入场/平仓手续费并加资金费，随后除以账户初始余额而非该笔交易权威 notional；最大回撤还用买卖订单现金流近似权益曲线。它写出的排名、Sharpe、回撤和 `model_performance_snapshots` 不能作为 `net_return_after_all_cost_pct` 晋升证据。
- 已彻底删除 `services/competition_service.py`、对应测试、paper/live 启动时评估、每小时评估任务、Dashboard service 注入、模型设置保存后的重复评估，以及无人使用的 `/api/models`、模型 performance/decisions 路由；不保留废弃接口或兼容响应。
- 同时删除只有构造/自测、没有任何运行调用的 `NotificationService`、通知配置字段和测试；删除没有入口或调用者的整个旧 `workers` 编排包，禁止正式启动器之外再形成独立采集、交易或模型评估循环。
- 删除 `ModelPerformanceSnapshot` ORM、RiskRepository 读写方法和训练重置引用；数据库启动迁移直接执行 `DROP TABLE IF EXISTS model_performance_snapshots`，旧错误派生数据不再留作隐式回退。
- 模型晋升唯一保留 `model_promotion_policy` 的真实收益分布评估、artifact 生命周期和 `production_trade_gate` 授权；专家竞争服务只保留为专家质量诊断，不得切换活动模型或授予生产权限。
- 本地验证 `2786 passed, 4 skipped`，Ruff 和前端 Node 语法检查通过。线上部署后 7 个 stale 源文件被删除，PostgreSQL `to_regclass('public.model_performance_snapshots')` 返回空，FastAPI 路由中不存在 `/api/models`；交易进程仍为 `paper + paused=false`、心跳新鲜、market error 为空并继续运行策略上下文阶段。

## 23. 专家记忆与影子训练标签彻底解耦

- 专家记忆只接受完整 OKX 权威结果合同：`source=authoritative_trade_outcome`、`authority_level=okx_settlement_and_execution`、当前 `outcome_version`、`cost_complete=true`、`production_evidence_eligible=true`，并且 `outcome_id` 与 `outcome_fingerprint` 均非空。该判定集中在 `core.training_contracts.is_authoritative_expert_memory_extra`，写入仓储、提示词检索、FinQuant 训练加载和反馈策略统一调用，不再各自维护宽松条件。
- 本地平仓只创建等待 OKX 回写的 `TradeReflection`，不再生成 `local_provisional_reflection` 专家记忆，也不再把本地 PnL 比例伪装成 `net_return_after_all_cost_pct`。权威回填只处理 `settlement_fact_trusted=true` 且 `outcome_complete=true` 的结果。
- `ShadowBacktestService` 只生成带费用合同和证据指纹的影子训练标签；删除影子结果复制到 `ExpertMemory` 的事务、配置开关、相关特征分桶/文本模板/关联 memory key 和兼容反馈计数。影子样本继续服务训练与晋升评估，但不能进入下一次交易提示词。
- 数据库启动迁移对 PostgreSQL 和 SQLite 都执行结构化 JSON 合同检查，所有非权威、不完整或伪造布尔值的 `expert_memories` 直接物理删除，不保留 inactive 兼容记录。`MemoryRepository.upsert_memory` 同时拒绝不完整合同，旧来源无法再次写入。
- 删除无人调用的一次性 `scripts/cleanup_expert_memory_text.py`、其专用测试和只验证影子专家记忆可用的旧测试文件；新增删除契约，禁止 `local_provisional_reflection`、`shadow_memory_enabled`、`_record_memory_in_session` 和 `build_expert_lessons` 回流运行代码。
- 部署前线上 `expert_memories` 共 56 条：权威 0、`shadow_backtest` 56、合同不完整 56；部署后总数、影子数、临时数和合同不完整数全部为 0，FinQuant 专家记忆训练加载结果为 0。后续只有新的完整 OKX 权威结果能重新生成专家记忆。
- 本地 Ruff 全通过，完整测试为 `2784 passed, 4 skipped`。线上服务和模型隧道均 active，控制态为 `paper + paused=false + ensemble_trader`，交易心跳和训练调度心跳新鲜，最近十分钟无 warning/error 日志。
- 当前自动交易仍存在独立阻断，不能记为已恢复：部署后市场轮次 `run_market_analysis=true`、扫描 20 个标的、14 个特征有效，但 `rank_selected_count=0`、决策 0；四个二级候选均因 `okx_private_entry_instrument_probe_failed` 被拒绝。下一批必须定位并彻底修复 OKX 私有标的探测/超时链路，再以非零候选和实际规则决策验收。
