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
- `net_return_after_cost_pct`、`realized_net_return_pct`、`fee_after_return` 只作为历史兼容读取，不再作为主目标。

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
- `services/return_execution_policy.py`
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
