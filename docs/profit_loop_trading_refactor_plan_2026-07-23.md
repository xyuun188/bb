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
- `model_stage`、`evaluation_policy.live_mutation` 不再是训练请求的授权输入；artifact 只按晋升报告生成 candidate -> shadow -> canary -> active。
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
