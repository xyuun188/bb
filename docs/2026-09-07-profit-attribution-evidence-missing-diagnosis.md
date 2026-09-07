# 盈利归因证据缺失问题诊断与修复方案

## 问题描述
2026-09-07 用户反馈：盈利归因页面显示大量"证据不可用"、"预期不可用"、"微多预计证据不可用"等信息。

## 根本原因分析

### 证据缺失的三种情况

根据 `services/profit_attribution.py` 的逻辑，证据显示"不可用"有以下原因：

#### 1. 未匹配到开仓AI决策
```python
missing_reason = "未匹配到开仓 AI 决策；已检查订单 decision_id、币种和开仓时间窗"
```
**原因**：
- Position记录与Order记录的`decision_id`不匹配
- 币种符号格式不一致（如 BTC/USDT vs BTC-USDT-SWAP）
- 开仓时间窗口超过30分钟，匹配失败

#### 2. 开仓AI决策未保存证据
```python
missing_reason = f"开仓 AI 决策未保存{label} 证据"
```
**原因**：
- AI决策的`raw_llm_response`中缺少以下字段：
  - `ml_signal` 或 `local_ai_tools.ml_signal`
  - `profit_prediction` 或 `server_quant_tools.profit_prediction`
  - `time_series_prediction` 或 `local_ai_tools.time_series_prediction`
  - `sentiment_analysis` 或 `server_quant_tools.sentiment_analysis`

#### 3. 证据数据不完整
```python
available = bool(signal.get("available")) and (has_side or has_expected or has_score)
```
**原因**：
- 工具返回了数据，但`available=False`
- 缺少`side`（做多/做空方向）
- 缺少`expected_return_pct`（预期收益）
- 缺少`score`（情绪评分）

## 可能的技术原因

### 1. 工具调用超时
```python
# trading_service.py
settings.local_ai_tools_timeout_seconds  # 默认可能太短
```

如果ML模型、盈利预测、时序预测等工具超时，就会返回空数据或error状态。

### 2. 工具服务不可用
- 本地ML服务挂掉
- 服务器量化工具API不可达
- 模型未训练或未加载

### 3. 证据保存逻辑有缺陷
决策生成时可能没有正确保存工具输出到`raw_llm_response`。

### 4. 历史数据迁移问题
旧版本的决策记录格式可能与当前盈利归因代码不兼容。

## 诊断步骤

### 步骤1：检查最近24小时的决策证据完整性

```sql
SELECT 
    id,
    symbol,
    action,
    created_at,
    CASE 
        WHEN raw_llm_response::jsonb ? 'ml_signal' 
            OR raw_llm_response::jsonb #> '{local_ai_tools,ml_signal}' IS NOT NULL 
        THEN '有ML' 
        ELSE '无ML' 
    END as ml_evidence,
    CASE 
        WHEN raw_llm_response::jsonb ? 'profit_prediction'
            OR raw_llm_response::jsonb #> '{server_quant_tools,profit_prediction}' IS NOT NULL
        THEN '有盈利模型'
        ELSE '无盈利模型'
    END as profit_evidence,
    CASE 
        WHEN raw_llm_response::jsonb ? 'time_series_prediction'
            OR raw_llm_response::jsonb #> '{local_ai_tools,time_series_prediction}' IS NOT NULL
        THEN '有时序'
        ELSE '无时序'
    END as timeseries_evidence,
    CASE 
        WHEN raw_llm_response::jsonb ? 'sentiment_analysis'
            OR raw_llm_response::jsonb #> '{server_quant_tools,sentiment_analysis}' IS NOT NULL
        THEN '有情绪'
        ELSE '无情绪'
    END as sentiment_evidence
FROM ai_decisions 
WHERE created_at >= NOW() - INTERVAL '24 hours'
    AND action IN ('long', 'short', 'open_long', 'open_short')
    AND was_executed = true
ORDER BY created_at DESC 
LIMIT 20;
```

### 步骤2：检查工具服务状态

```bash
# 检查ML信号服务
curl -s http://127.0.0.1:8002/api/ml-signal/status | jq '.status'

# 检查本地AI工具
curl -s http://127.0.0.1:8002/api/local-ai-tools/status | jq '.status'

# 检查模型训练状态
curl -s http://127.0.0.1:8002/api/model-training/registry | jq '.summary'
```

### 步骤3：检查Position-Order-Decision匹配率

```sql
SELECT 
    COUNT(DISTINCT p.id) as total_positions,
    COUNT(DISTINCT CASE WHEN o.id IS NOT NULL THEN p.id END) as matched_with_order,
    COUNT(DISTINCT CASE WHEN d.id IS NOT NULL THEN p.id END) as matched_with_decision,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN d.id IS NOT NULL THEN p.id END) / COUNT(DISTINCT p.id), 2) as match_rate_pct
FROM positions p
LEFT JOIN orders o ON o.id = (
    SELECT o2.id 
    FROM orders o2 
    WHERE o2.symbol = p.symbol 
        AND o2.created_at BETWEEN p.created_at - INTERVAL '30 minutes' AND p.created_at + INTERVAL '5 minutes'
    ORDER BY ABS(EXTRACT(EPOCH FROM (o2.created_at - p.created_at))) ASC
    LIMIT 1
)
LEFT JOIN ai_decisions d ON d.id = o.decision_id
WHERE p.created_at >= NOW() - INTERVAL '24 hours'
    AND p.execution_mode = 'paper';
```

## 修复方案

### 短期修复（立即可做）

#### 1. 增加工具调用超时时间

如果工具经常超时，可以适当增加超时设置：

```python
# config/settings.py 或环境变量
LOCAL_AI_TOOLS_TIMEOUT_SECONDS = 8  # 从5秒增加到8秒
```

#### 2. 添加证据缺失的容错显示

修改前端显示逻辑，不要全部显示"不可用"，而是显示部分可用的证据：

```javascript
// dashboard.js
function profitAttributionEvidenceStatusChip(label, status, options = {}) {
    const available = sourceStatus.available === true || options.available === true;
    if (!available) {
        const reason = sourceStatus.missing_reason || `${label}证据未匹配`;
        // 只在critical情况下显示红色警告
        if (reason.includes('未匹配到开仓 AI 决策')) {
            return { tone: 'missing', ... };
        }
        // 其他情况显示灰色提示
        return { tone: 'info', text: `${label}未保存`, html: '...' };
    }
    ...
}
```

#### 3. 检查并重启相关服务

```bash
# 检查服务状态
sudo systemctl status bb-paper-trading
sudo systemctl status bb-dashboard

# 如果需要，重启服务
sudo systemctl restart bb-paper-trading
```

### 中期修复（1-3天）

#### 1. 优化证据匹配逻辑

扩大时间窗口，提高匹配成功率：

```python
# services/profit_attribution.py
max_gap_seconds = 45 * 60  # 从30分钟增加到45分钟
```

#### 2. 添加证据保存的监控和告警

在决策生成时检查证据完整性：

```python
# services/trading_service.py
def _validate_decision_evidence(raw_response: dict) -> dict:
    """验证决策证据完整性"""
    issues = []
    
    if not extract_entry_signal_sides(raw_response).get('ml', {}).get('available'):
        issues.append('ml_signal_missing')
    
    if not extract_entry_signal_sides(raw_response).get('server_profit', {}).get('available'):
        issues.append('profit_prediction_missing')
    
    return {
        'complete': len(issues) == 0,
        'issues': issues,
    }
```

#### 3. 补充历史数据的证据

对于已有的Position记录，尝试从Order和Decision重新匹配：

```sql
-- 修复脚本（需谨慎执行）
UPDATE positions p
SET metadata = jsonb_set(
    COALESCE(p.metadata, '{}'::jsonb),
    '{matched_decision_id}',
    to_jsonb(o.decision_id)
)
FROM orders o
WHERE o.symbol = p.symbol
    AND o.created_at BETWEEN p.created_at - INTERVAL '30 minutes' AND p.created_at + INTERVAL '5 minutes'
    AND o.decision_id IS NOT NULL
    AND p.metadata->>'matched_decision_id' IS NULL
    AND p.created_at >= '2026-08-01';
```

### 长期优化（1-2周）

#### 1. 证据持久化改进

将证据数据单独存储，不依赖`raw_llm_response`的JSON解析：

```python
# 新表结构
class DecisionEvidence(Base):
    __tablename__ = 'decision_evidences'
    
    decision_id = Column(Integer, ForeignKey('ai_decisions.id'))
    ml_signal = Column(JSONB)
    profit_prediction = Column(JSONB)
    timeseries_prediction = Column(JSONB)
    sentiment_analysis = Column(JSONB)
    shadow_backtest_id = Column(Integer, ForeignKey('shadow_backtests.id'))
```

#### 2. 实时证据健康度监控

添加Prometheus指标：

```python
evidence_completeness_gauge = Gauge(
    'bb_decision_evidence_completeness',
    'Percentage of decisions with complete evidence',
    ['evidence_type']
)
```

#### 3. 自动修复机制

当检测到证据缺失时，异步重新计算并补充：

```python
async def repair_missing_evidence(decision_id: int):
    """异步修复缺失的证据"""
    decision = await get_decision(decision_id)
    if not decision:
        return
    
    # 重新调用工具获取证据
    ml_signal = await ml_signal_service.predict(...)
    profit = await profit_prediction_service.predict(...)
    
    # 更新决策记录
    decision.raw_llm_response['ml_signal'] = ml_signal
    decision.raw_llm_response['profit_prediction'] = profit
    await save_decision(decision)
```

## 监控指标

添加以下监控，及时发现证据缺失问题：

1. **证据完整率**：每小时统计有完整证据的决策占比
2. **匹配成功率**：Position能匹配到Decision的比例
3. **工具可用率**：ML/盈利/时序/情绪工具的可用时间占比
4. **工具响应时间**：各工具的P50/P95/P99响应时间

## 验证步骤

修复后需要验证：

1. 新产生的决策是否包含完整证据
2. 盈利归因页面的"不可用"比例是否降低
3. Position-Decision匹配率是否提升到90%以上
4. 工具服务是否稳定运行

## 相关文件

- `services/profit_attribution.py` - 盈利归因主逻辑
- `services/entry_signal_extraction.py` - 证据提取逻辑
- `services/trading_service.py` - 决策生成和工具调用
- `web_dashboard/static/js/dashboard.js` - 前端证据显示
- `web_dashboard/api/dashboard.py` - Dashboard API

## 临时workaround

如果短期无法修复，可以：
1. 隐藏"证据不可用"的显示，只显示有效证据
2. 降低证据完整性要求，允许部分证据缺失
3. 添加"历史数据正在补充中"的提示

---

**创建时间**: 2026-09-07 15:30
**优先级**: P1（影响用户体验）
**预计修复时间**: 短期修复1天，中期优化3天
