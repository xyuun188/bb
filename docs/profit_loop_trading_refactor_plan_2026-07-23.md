# 盈利闭环交易重构计划

日期：2026-07-23

项目根目录：`E:\code\bb`

## 1. 目标

把系统从“模型晋升后才允许交易”的闭环，重构为“规则先做真实小仓交易，模型旁路学习亏损与盈利，证据达标后再接管”的闭环。

最终目标不是单纯提升准确率，而是让系统持续产生可追溯的真实盈利证据，并以此驱动模型晋升、交易权限和策略优化。

## 2. 交易侧最终逻辑

交易只保留三种模式：

- `observe`：只观察，不交易。
- `live_rules_canary`：规则与风控主导的小仓真实交易，模型只旁路学习。
- `live_ml`：模型晋升后，模型可以参与方向、筛选和仓位。

统一交易顺序：

1. 市场信号出现。
2. 统一交易闸门判断模式。
3. 规则或模型生成交易建议。
4. 风控裁决仓位和风险。
5. 执行服务下单。
6. 平仓后回写真实结果。
7. 结果进入训练与晋升评估。

## 3. 训练侧最终逻辑

训练必须同时吸收盈利样本和亏损样本。

每条样本必须能追溯到真实闭环：

- symbol
- side
- entry_order_id
- close_order_id
- entry_price
- close_price
- quantity
- notional
- entry_fee
- close_fee
- funding_fee
- slippage
- realized_pnl
- net_return_after_all_cost_pct
- holding_minutes
- decision_authority
- model_shadow_prediction
- evidence_fingerprint

训练主目标只认一个：

- `net_return_after_all_cost_pct`

辅助指标只做诊断：

- Profit Factor
- 回撤
- 尾部亏损
- long / short 分方向收益
- 收益下界 LCB

## 4. 亏损学习规则

亏损必须进入训练，但不能乱归因。

- 规则开的亏损，先归因给规则采样。
- 模型旁路预测若能提前避开亏损，记为加分证据。
- 模型旁路预测若支持亏损方向，记为扣分证据。
- 模型真正接管后产生的亏损，才算模型实盘失败。

结论很明确：

- 亏损要学。
- 归因要准。
- 不能把所有亏损粗暴算到模型头上。

## 5. 统一交易闸门

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

所有执行、策略、Dashboard 和审计逻辑都只读这个闸门结果，不再各自重复裁决。

## 6. 五个功能的闭环

### 6.1 专家记忆

输入：真实复盘结论和成交事实。  
输出：结构化经验。  
回写：下次决策辅助。

### 6.2 影子复盘

输入：决策。  
输出：未来 10 / 30 / 60 分钟后的真实行情结果。  
回写：训练样本。

### 6.3 开仓漏斗

输入：候选信号。  
输出：是否允许开仓。  
回写：风控统计和漏斗统计。

### 6.4 策略复盘

输入：真实订单和真实平仓。  
输出：真实净盈亏归因。  
回写：策略学习和训练。

### 6.5 策略调度

输入：调度条件。  
输出：执行结果、失败、超时、降级。  
回写：调度健康状态。

## 7. 数据层处理

### 保留

- OKX 订单
- 成交
- 持仓
- 平仓
- 手续费
- funding
- realized PnL
- 审计日志

### 清理或隔离

- 旧 shadow 样本
- 旧 artifact
- 旧 training cursor
- 旧 dashboard cache
- 旧 contract 训练视图
- 没有真实盈亏闭环的数据

原则是：

- 事实保留。
- 衍生层重建。

## 8. 要删除的旧逻辑

重构完成后，下面这些如果不进入新闭环，就直接删除：

- 重复的 production eligibility 判断
- shadow / canary / live 多处散落分支
- 旧 blocker 拼接逻辑
- 旧 training cursor 兼容逻辑
- 只展示、不回写的 helper
- 旧 Dashboard 状态词
- 旧 contract / version 分叉
- 无人调用的 service 和 script

重点清理对象：

- `services/entry_opportunity_scoring.py`
- `services/return_execution_policy.py`
- `services/trading_policies.py`
- `services/execution_service.py`
- `services/ml_signal_service.py`
- `web_dashboard/static/js/dashboard.js`

## 9. 实施顺序

1. 冻结旧训练层。
2. 建立干净训练视图。
3. 上统一交易闸门。
4. 接通 `live_rules_canary`。
5. 回写真实交易样本。
6. 用亏损和盈利一起重训。
7. 以真实净收益做晋升标准。
8. 删除旧逻辑和废状态。

## 10. 验收标准

1. 模型没晋升前，系统也能通过规则小仓真实交易。
2. 每笔交易都能追溯到 OKX 真实事实。
3. 亏损样本会进入训练，并能判断模型当时是否应该避开。
4. 模型晋升只看真实净盈利能力，不看单纯准确率。
5. 交易权限只有一个权威入口。
6. Dashboard 不再把内部 blocker 当主状态。
7. 旧逻辑不堆积，没进新闭环的代码删除。

## 11. 首批落地任务

- 新建 `services/production_trade_gate.py`
- 冻结旧训练层，建立新训练视图
- 接通规则小仓真实交易模式
- 清理重复的交易资格判断
- 简化 Dashboard 主状态

