# Quant Closed-Loop Eradication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 根治“观望错过机会未充分反馈、ML 长期 learning_only、乱码反复出现、Ruff/质量债反复漏检、策略修一处坏一处”的闭环问题。

**Architecture:** 采用“入口治理 + 决策闭环 + 训练治理 + 巡检可观测 + 分批质量债”的方式处理，不再按页面或单个现象打补丁。所有影响交易行为的改动必须先写失败测试，再实现，再用线上巡检和策略健康脚本验证。

**Tech Stack:** Python 3.12、FastAPI/Streamlit-style dashboard static UI、SQLAlchemy、pytest、ruff、black、OKX API、本地量化工具服务、Hindsight 项目记忆、线上 systemd 服务。

---

## 0. 根治边界与完成定义

### 必须一起处理的问题

1. **影子错过机会没有充分形成策略能力**
   - 现象：系统巡检显示大量 `missed_opportunity_sample`，但真实开仓仍偏少。
   - 根因假设：错过机会目前主要作为 `shadow_memory` 收益提示和 `memory_habit_adjustment` 小探针，不足以形成“同币种/同方向/同市场结构”的可验证放大规则。
   - 约束：不能绕过净收益、亏损概率、尾部风险、模型一致性、OKX 规则。

2. **本地 ML 长期 `learning_only`**
   - 现象：模型服务可用，但系统不能证明高分组更赚钱。
   - 根因假设：训练质量、样本隔离、达标指标、收益分组验证和 UI 解释没有形成明确状态机。
   - 约束：不能把状态硬改成 `ready`，必须由指标达标触发。

3. **乱码反复出现**
   - 现象：中文文案、历史记录、模型输出、控制台显示混杂真假乱码。
   - 根因假设：缺少统一文本治理模块；只扫源码/页面，不扫新增 DB 记录、缓存 JSON、模型响应写入口。
   - 约束：必须区分“终端显示假乱码”和“源码/数据库真污染”。

4. **Ruff/格式/静态质量债反复漏出**
   - 现象：bug/security 规则已清理，但全量 Ruff 仍存在历史导入排序/E402 等债务。
   - 根因假设：历史脚本手工 `sys.path` 注入较多，未分批纳入质量门。
   - 约束：不能一次性全仓格式化造成巨大无关 diff；必须分批提交。

5. **系统巡检还不够能指导修复**
   - 现象：巡检能显示问题，但还需要更强地分清已修复、观察中、未修复、责任链路。
   - 根因假设：巡检未对所有新增治理点建立单独节点和趋势指标。
   - 约束：巡检必须直接服务后续修改判断，不做展示型假指标。

### 不算完成的情况

- 只改页面文案或按钮样式。
- 只把 `learning_only` 改成 `ready`。
- 只放宽开仓阈值导致小仓、快平、亏损复开增加。
- 只清源码乱码，不清数据库/缓存/模型写入口。
- 只跑本地测试，不部署线上验证。
- 没有 Git 提交、没有 Hindsight 同步、没有线上巡检记录。

### 完成验收总标准

- `pytest` 全量通过。
- `security_secret_scan.py --fail-on high` 通过。
- `ruff check . --select F,B,E9,S608,E722` 通过。
- 新增的文本治理、ML 状态、影子反馈、巡检测试全部通过。
- 线上 `system_audit` 无 `critical`，问题台账可区分 `fixed/unresolved/observing`。
- 线上 2 小时策略健康：失败订单为 0；弱证据执行为 0；15 分钟内快亏平不新增；开仓少时必须能解释是收益/风险/模型冲突导致，而不是链路卡死。

---

## 1. 文件结构规划

### 新增文件

- `services/text_integrity.py`
  - 统一文本治理模块：检测 mojibake、修复常见 UTF-8/GBK 错解、返回修复报告。
- `tests/test_text_integrity_runtime.py`
  - 覆盖源码之外的运行时文本治理：模型响应、执行原因、数据库写入前文本。
- `scripts/audit_runtime_text_integrity.py`
  - 线上运行时文本巡检：扫描最近分析记录、执行详情、策略事件、专家记忆、JSON 缓存。
- `tests/test_shadow_missed_opportunity_closed_loop.py`
  - 覆盖“反复错过机会如何形成受控策略放大”的契约。
- `tests/test_ml_readiness_state.py`
  - 覆盖 ML `learning_only -> ready` 的状态机达标条件和原因展示。

### 修改文件

- `services/entry_evidence.py`
  - 清理源码内乱码文案；让 `memory_missed_opportunity_relief` 给出明确中文原因；增加“同币种/同方向/同市场结构”证据要求。
- `services/entry_opportunity_scoring.py`
  - 调整影子记忆从单一 cap 提示，升级为分层贡献：提示、排序、受控 probe、禁止放大。
- `services/memory_feedback.py`
  - 输出更细的 missed opportunity 聚合：symbol/side/timeframe/regime/avg_return/risk_count。
- `services/ml_signal_service.py`
  - 输出 ML readiness 诊断：训练样本、隔离样本、AUC/收益分组、最近训练时间、下一次训练触发。
- `web_dashboard/api/system_audit.py`
  - 增加运行时文本完整性节点、ML readiness 节点、影子错过机会闭环节点。
- `web_dashboard/api/data_collection.py`
  - 数据采集页展示文本/训练数据质量，不再只显示粗略样本数。
- `web_dashboard/static/js/dashboard.js`
  - 系统巡检和本地 ML 页面展示新增诊断字段；不做交易逻辑。
- `docs/repair_retrospective_and_monitoring_2026-06-21.md`
  - 增补本次根治计划执行记录和每批结论。

---

## 2. 批次执行总表

- [ ] **Batch A：文本乱码根治**
- [ ] **Batch B：影子错过机会闭环**
- [ ] **Batch C：ML readiness 状态机**
- [ ] **Batch D：系统巡检深层节点**
- [ ] **Batch E：Ruff/格式历史债分批清理**
- [ ] **Batch F：线上部署、观测、Git/Hindsight 闭环**

---

## Batch A：文本乱码根治

**目标：** 让乱码治理从“页面替换”升级为“写入前拦截 + 历史扫描 + 巡检统计”。

**Files:**
- Create: `services/text_integrity.py`
- Create: `tests/test_text_integrity_runtime.py`
- Create: `scripts/audit_runtime_text_integrity.py`
- Modify: `web_dashboard/api/system_audit.py`
- Modify: `services/entry_evidence.py`

### Task A1：写文本治理失败测试

- [ ] 新增 `tests/test_text_integrity_runtime.py`，测试以下行为：
  - `looks_like_mojibake("鏈轰細璇勫垎")` 返回 `True`。
  - `repair_mojibake("鏈轰細璇勫垎")` 返回包含正常中文的文本或标记为不可安全修复。
  - 正常中文 `机会评分为正` 不被误判。
  - 英文、数字、交易对 `BTC/USDT` 不被误判。

Run:

```powershell
rtk python -m pytest tests/test_text_integrity_runtime.py -q
```

Expected RED:

```text
FAILED ... ModuleNotFoundError: No module named 'services.text_integrity'
```

### Task A2：实现 `services/text_integrity.py`

- [ ] 新增函数：
  - `looks_like_mojibake(text: str) -> bool`
  - `repair_mojibake(text: str) -> TextIntegrityResult`
  - `sanitize_runtime_text(value: Any) -> Any`
- [ ] `TextIntegrityResult` 必须包含：`original`、`text`、`changed`、`suspected`、`method`、`reason`。
- [ ] 修复只允许确定性编码逆转；不确定时返回 `suspected=True`，但不乱改。

Run:

```powershell
rtk python -m pytest tests/test_text_integrity_runtime.py -q
rtk ruff check services/text_integrity.py tests/test_text_integrity_runtime.py
rtk black --check services/text_integrity.py tests/test_text_integrity_runtime.py
```

Expected GREEN:

```text
passed
Ruff: No issues found
```

### Task A3：清理源码内乱码文案

- [ ] 修改 `services/entry_evidence.py` 中 positive/strong/memory relief 的中文 reason，全部改为正常 UTF-8 中文。
- [ ] 不修改策略数值，只修文本来源。
- [ ] 增加测试断言这些 reason 不包含 `鏈|璇|锛|銆|�`。

Run:

```powershell
rtk python -m pytest tests/test_entry_evidence_policy.py tests/test_no_mojibake_source.py -q
```

### Task A4：运行时文本巡检脚本

- [ ] 新增 `scripts/audit_runtime_text_integrity.py`。
- [ ] 线上扫描范围：最近 500 条 `AIDecision.raw_llm_response`、`execution_reason`、`StrategyLearningEvent.payload`、`ExpertMemory.content`、关键 JSON 缓存。
- [ ] 输出：`scanned_records`、`suspected_records`、`by_table`、`examples`、`repairable_count`。
- [ ] 默认 dry-run，不直接改数据库。

Run:

```powershell
rtk python scripts/audit_runtime_text_integrity.py --help
rtk python -m pytest tests/test_text_integrity_runtime.py -q
```

### Batch A 验收

- [ ] 源码乱码测试通过。
- [ ] 运行时脚本能在本地或线上 dry-run 输出统计。
- [ ] 系统巡检新增“运行时文本完整性”节点。
- [ ] 不能再只用 PowerShell 控制台显示判断乱码真假。

---

## Batch B：影子错过机会闭环

**目标：** 让 missed opportunity 成为可解释、可验证、受风控约束的策略信号，而不是简单放宽开仓。

**Files:**
- Create: `tests/test_shadow_missed_opportunity_closed_loop.py`
- Modify: `services/memory_feedback.py`
- Modify: `services/entry_opportunity_scoring.py`
- Modify: `services/entry_evidence.py`
- Modify: `scripts/inspect_online_strategy_health.py`

### Task B1：写失败测试：重复错过机会不能只停留在 shadow-only

- [ ] 构造同一 symbol/side 最近多次 missed opportunity，平均收益为正，风险证据低，模型方向一致。
- [ ] 断言输出：
  - `shadow_memory_component.available is True`
  - `memory_missed_opportunity_relief.applied is True`
  - `tradeable_probe is True`
  - `expected_net_breakdown.components.shadow_memory.contribution_pct > 0`
- [ ] 同时构造风险证据高或模型冲突场景，断言仍不能执行。

Run:

```powershell
rtk python -m pytest tests/test_shadow_missed_opportunity_closed_loop.py -q
```

Expected RED:

```text
FAILED ... expected tradeable_probe True
```

### Task B2：增强 `memory_feedback` 聚合粒度

- [ ] `services/memory_feedback.py` 输出字段增加：
  - `symbol_side_missed_count`
  - `symbol_side_avg_return_pct`
  - `recent_missed_count_2h`
  - `regime_consistency_score`
  - `risk_evidence_ratio`
- [ ] 不允许单纯全局 missed count 推动任意币种开仓。

Run:

```powershell
rtk python -m pytest tests/test_memory_feedback.py tests/test_shadow_missed_opportunity_feedback.py -q
```

### Task B3：把 missed opportunity 分层进入策略

- [ ] 修改 `services/entry_opportunity_scoring.py`：
  - 全局 missed opportunity 只进入解释和轻微排序。
  - symbol/side missed opportunity 连续、收益正、风险低，才进入受控 probe。
  - 如果模型冲突或亏损概率高，只记录为“错过机会观察”，不执行。
- [ ] 修改 `services/entry_evidence.py`：
  - `memory_relief_allowed` 不再只看 `memory_missed_count >= 6`。
  - 改为综合 `symbol_side_missed_count`、`avg_return`、`risk_evidence_ratio`、`expected_net`、`profit_quality`、模型冲突。

Run:

```powershell
rtk python -m pytest tests/test_shadow_missed_opportunity_closed_loop.py tests/test_entry_evidence_policy.py tests/test_entry_opportunity_scoring.py -q
```

### Task B4：策略健康脚本增加错过机会闭环统计

- [ ] 修改 `scripts/inspect_online_strategy_health.py` 输出：
  - `missed_feedback_applied_count`
  - `missed_feedback_tradeable_probe_count`
  - `missed_feedback_blocked_reasons`
  - `missed_feedback_symbol_side_examples`

Run:

```powershell
rtk python -m pytest tests/test_inspect_online_strategy_health.py -q
```

### Batch B 验收

- [ ] 不是简单增加开仓；弱证据执行仍为 0。
- [ ] 同币种/同方向反复错过且质量达标时，能转为受控策略信号。
- [ ] 策略健康报告能解释错过机会有没有被利用、为什么没利用。

---

## Batch C：ML readiness 状态机

**目标：** 让 `learning_only` 变成可解释、可达标、可退回的状态，而不是模糊标签。

**Files:**
- Create: `tests/test_ml_readiness_state.py`
- Modify: `services/ml_signal_service.py`
- Modify: `scripts/deploy_local_ai_tools_service.py`
- Modify: `web_dashboard/api/system_audit.py`
- Modify: `web_dashboard/static/js/dashboard.js`

### Task C1：写 ML readiness 失败测试

- [ ] 测试样本不足时状态为 `learning_only`，原因包含 `sample_count_below_target`。
- [ ] 测试高分组收益不优于低分组时状态为 `learning_only`，原因包含 `high_score_group_not_profitable`。
- [ ] 测试 AUC、样本数、高分组收益、最近训练时间全部达标时状态为 `ready`。
- [ ] 测试新脏数据比例升高时状态从 `ready` 退回 `learning_only`。

Run:

```powershell
rtk python -m pytest tests/test_ml_readiness_state.py -q
```

Expected RED:

```text
FAILED ... readiness diagnostics missing
```

### Task C2：实现 readiness 诊断结构

- [ ] 在 `services/ml_signal_service.py` 或本地 AI tools 状态返回中统一输出：
  - `status`
  - `ready`
  - `blocking_reasons`
  - `metrics.sample_count`
  - `metrics.quarantined_sample_count`
  - `metrics.auc`
  - `metrics.high_score_group_return_pct`
  - `metrics.low_score_group_return_pct`
  - `metrics.last_trained_at`
  - `next_training_due_at`
- [ ] 不满足达标时，不允许真实仓位放大依赖 ML。

Run:

```powershell
rtk python -m pytest tests/test_ml_readiness_state.py tests/test_local_ai_tools_client.py tests/test_train_local_ai_tools_models.py -q
```

### Task C3：系统巡检和 UI 展示 ML 阻塞原因

- [ ] `web_dashboard/api/system_audit.py` 的模型训练卡片展示 `blocking_reasons`。
- [ ] 本地 ML 页面展示：达标项、未达标项、样本结构、下一次训练时间。
- [ ] 页面不能只显示“未就绪”。

Run:

```powershell
rtk python -m pytest tests/test_system_audit_api.py tests/test_dashboard_main_ui_contract.py -q
node --check web_dashboard/static/js/dashboard.js
```

### Batch C 验收

- [ ] 不能硬改 `ready`。
- [ ] `learning_only` 必须能说清楚差什么。
- [ ] 线上系统巡检能看到 ML 是否影响仓位放大。

---

## Batch D：系统巡检深层节点

**目标：** 系统巡检能定位责任链路，而不是只给一个 warning。

**Files:**
- Modify: `web_dashboard/api/system_audit.py`
- Modify: `web_dashboard/static/js/dashboard.js`
- Modify: `tests/test_system_audit_api.py`
- Modify: `tests/test_dashboard_main_ui_contract.py`

### Task D1：新增治理节点

- [ ] 增加节点：
  - `runtime_text_integrity`
  - `missed_opportunity_feedback`
  - `ml_readiness`
  - `ruff_quality_gate`
- [ ] 每个节点必须输出：`status`、`state`、`evidence`、`next_actions`、`owner_path`。

Run:

```powershell
rtk python -m pytest tests/test_system_audit_api.py -q
```

### Task D2：巡检页面展示“已修复/未修复/观察中”

- [ ] UI 按三列展示 issue ledger。
- [ ] 每个问题显示最近一次发现时间、最近一次修复提交、当前状态。
- [ ] 不展示原始 JSON 作为主视图。

Run:

```powershell
rtk python -m pytest tests/test_dashboard_main_ui_contract.py -q
node --check web_dashboard/static/js/dashboard.js
```

### Batch D 验收

- [ ] 用户不用看日志，也能知道问题在哪条链路。
- [ ] 巡检能告诉后续代码修改优先级。

---

## Batch E：Ruff/格式历史债分批清理

**目标：** 把质量债纳入门禁，但不一次性制造巨大无关 diff。

**Files:**
- Modify: `pyproject.toml` if needed
- Modify: affected scripts/tests by batch

### Task E1：固定 bug/security 门禁

- [ ] 保持以下命令为提交前必跑：

```powershell
rtk ruff check . --select F,B,E9,S608,E722
```

- [ ] 如果失败，必须当批修复。

### Task E2：分批清 I001 导入排序

- [ ] 每批只处理一个目录：`scripts/`、`services/`、`web_dashboard/api/`、`tests/`。
- [ ] 每批跑：

```powershell
rtk ruff check <dir> --select I --fix
rtk black <dir>
rtk python -m pytest <related-tests>
```

### Task E3：E402 历史脚本治理

- [ ] 对 CLI 脚本保留必要 `sys.path` 时，使用局部 `# noqa: E402` 并说明原因。
- [ ] 对可模块化脚本，改为包内导入入口。

### Batch E 验收

- [ ] bug/security 规则全仓通过。
- [ ] 被处理目录 Ruff/Black 通过。
- [ ] 不做无测试的大面积格式化。

---

## Batch F：线上部署、观测、Git/Hindsight 闭环

**目标：** 每批改完都有可验证线上结果，避免“本地好了线上又坏”。

### Task F1：每批固定验证

Run:

```powershell
rtk python -m pytest
rtk python scripts/security_secret_scan.py --fail-on high
rtk ruff check . --select F,B,E9,S608,E722
```

### Task F2：每批固定部署

Run:

```powershell
rtk python scripts/sync_to_online_server.py --split-services
```

Expected:

```text
model-tunnels-ok
active
active
active
dashboard-ok:302
```

### Task F3：每批固定线上复查

Run:

```powershell
rtk python scripts/inspect_online_strategy_health.py --minutes 120
```

检查：

- `failed_orders == 0`
- `fast_loss_close_under_15m == 0`，或若历史窗口有遗留，当前 runtime window 为 0
- `weak_shadow_executed_count == 0`
- `missed_feedback_*` 指标存在
- `local_ai_tools.status` 和 blocking reasons 可解释

### Task F4：每批固定 Git/Hindsight

Run:

```powershell
rtk git status --short
rtk git add <changed-files>
rtk git commit -m "<batch-specific message>"
rtk git push
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Administrator\.codex\hindsight-memory\sync-project-memory.ps1" -ProjectRoot "F:\BB" -Event "manual"
```

### Batch F 验收

- [ ] Git 干净。
- [ ] 线上服务 active。
- [ ] 系统巡检记录本批结果。
- [ ] Hindsight 已同步。

---

## 3. 执行顺序与停止规则

### 执行顺序

1. Batch A：先堵乱码入口，避免后续所有记录继续污染。
2. Batch B：处理观望错过机会，但不直接放宽开仓。
3. Batch C：处理 ML readiness，明确模型是否能参与仓位放大。
4. Batch D：把新治理点纳入系统巡检。
5. Batch E：分批清质量债。
6. Batch F：每批都执行部署和验证。

### 停止规则

- 如果某批测试失败，不能进入下一批。
- 如果线上出现 `critical`，先修线上硬故障，不继续策略优化。
- 如果策略健康出现失败订单、弱证据执行、快亏平新增，停止放大逻辑，回到对应批次定位。
- 如果连续 3 次局部修改仍不能解决同一问题，停止打补丁，重审架构边界。

---

## 4. 第一批开始前检查命令

Run:

```powershell
rtk git status --branch --short
rtk python -m pytest tests/test_no_mojibake_source.py tests/test_text_sanitize.py -q
rtk python scripts/security_secret_scan.py --fail-on high
```

Expected:

```text
clean or only plan doc changed
passed
source safety scan ok
```

---

## 5. 计划执行记录

| Batch | 状态 | Git Commit | 线上验证 | 备注 |
| --- | --- | --- | --- | --- |
| A 文本乱码根治 | 未开始 | - | - | 先堵写入口 |
| B 影子错过机会闭环 | 未开始 | - | - | 不放宽硬风控 |
| C ML readiness | 未开始 | - | - | 不硬改 ready |
| D 系统巡检深层节点 | 未开始 | - | - | 加责任链路 |
| E Ruff/格式债 | 未开始 | - | - | 分目录清理 |
| F 部署观测闭环 | 未开始 | - | - | 每批都跑 |
