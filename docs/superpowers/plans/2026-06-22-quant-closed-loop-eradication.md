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
- `services/training_data_governance.py`
  - 训练数据生命周期治理：识别旧数据/脏数据、隔离不可训练样本、生成训练视图和训练集版本。
- `tests/test_training_data_governance.py`
  - 覆盖乱码、OKX 对账异常、极小探针、快亏平、重复记录、未来函数泄露等污染类型。
- `scripts/audit_training_data_governance.py`
  - dry-run 扫描线上训练数据污染，输出隔离、修复、归档建议，不默认直接删除。
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
- `services/shadow_training_quarantine.py`
  - 接入训练数据治理结果，避免脏影子样本进入本地 ML。
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
- [ ] **Batch B0：旧数据/脏数据生命周期治理**
- [ ] **Batch B：影子错过机会闭环**
- [ ] **Batch C：ML readiness 状态机**
- [ ] **Batch C2：模型/专家全面体检**
- [ ] **Batch C3：模型/专家竞赛框架**
- [ ] **Batch C4：数字货币特征补强**
- [ ] **Batch C5：模型组合动态路由**
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

## Batch B0：旧数据/脏数据生命周期治理

**目标：** 在任何训练、模型评估、策略放大之前，先建立数据生命周期治理，防止历史旧数据、脏数据、错误对账、极小探针、乱码和未来函数泄露继续污染训练结果。

**Files:**
- Create: `services/training_data_governance.py`
- Create: `tests/test_training_data_governance.py`
- Create: `scripts/audit_training_data_governance.py`
- Modify: `services/ml_signal_service.py`
- Modify: `services/shadow_training_quarantine.py`
- Modify: `web_dashboard/api/data_collection.py`
- Modify: `web_dashboard/api/system_audit.py`
- Modify: `web_dashboard/static/js/dashboard.js`

### Task B0.1：定义训练数据治理状态

- [ ] 新增样本状态：`raw`、`clean`、`quarantined`、`repairable`、`repaired`、`archive_only`、`training_excluded`。
- [ ] 新增污染原因：`mojibake`、`okx_mismatch`、`bad_close_state`、`fee_missing`、`price_outlier`、`micro_probe_pollution`、`fast_loss_unexplained`、`duplicate_sample`、`json_parse_failed`、`future_leakage`、`stale_market_data`、`mode_mixed`。
- [ ] 原始数据必须保留，训练读取 `clean_training_view`，不能直接删除历史记录。

Run:

```powershell
rtk python -m pytest tests/test_training_data_governance.py -q
```

Expected RED:

```text
FAILED ... ModuleNotFoundError: No module named 'services.training_data_governance'
```

### Task B0.2：实现脏数据识别和隔离规则

- [ ] 识别乱码样本：分析记录、专家意见、策略原因、模型返回包含疑似 mojibake。
- [ ] 识别 OKX 不一致数据：平台订单、仓位、收益与 OKX 对账缺失或冲突。
- [ ] 识别错误平仓状态：全部平仓显示部分平仓、手续费缺失、收益错算。
- [ ] 识别极小探针污染：弱证据小仓、测试单、低质量 probe 不进入正常盈利模型训练。
- [ ] 识别快亏平异常：几分钟亏损平仓且无强风控原因时标记为 `fast_loss_unexplained`。
- [ ] 识别重复样本：同一分钟同交易对重复分析、重复影子复盘、重复专家记忆。
- [ ] 识别无效模型输出：JSON 解析失败、未返回、超时、兜底结果。
- [ ] 识别未来函数泄露：训练特征不能包含决策后才知道的收益、平仓结果。
- [ ] 识别过期行情/K线：ticker、K线、盘口时间不一致的数据不能训练。
- [ ] 识别模式混淆：模拟盘、实盘、影子盘必须带来源标签，不能混成同一执行结果。

Run:

```powershell
rtk python -m pytest tests/test_training_data_governance.py tests/test_training_data_quality.py tests/test_shadow_training_quarantine.py -q
```

### Task B0.3：训练集版本化和回滚能力

- [ ] 每次训练生成 `training_dataset_version`，记录样本范围、样本数量、排除数量、排除原因、数据时间跨度。
- [ ] 模型 artifact 记录使用的训练集版本。
- [ ] 如果新模型上线后表现变差，能回滚到上一版训练集和模型。
- [ ] 数据修复不覆盖原始记录，只生成修复版本或修复视图。

### Task B0.4：旧数据归档和安全清理

- [ ] 原始旧数据默认归档，不直接删除。
- [ ] 可修复数据生成 `repaired` 版本。
- [ ] 不可确认真实结果的数据标记为 `archive_only`，保留审计但不训练。
- [ ] 清理脚本默认 dry-run，只有显式 `--apply` 且输出待处理清单后才允许修改状态。

### Task B0.5：系统巡检和数据采集页展示治理指标

- [ ] 系统巡检新增“训练数据治理”节点。
- [ ] 数据采集管理页展示：新增脏数据数、脏数据来源分布、自动隔离数量、自动修复数量、不可修复数量、当前可训练样本数、本次训练排除样本数、极小探针样本占比、快亏平样本占比、OKX 对账异常样本数、训练集版本号和最近训练时间。
- [ ] 本地 ML 页面展示训练使用的数据版本，不能只显示样本总数。

### Batch B0 验收

- [ ] 本地 ML 训练不能直接读取脏数据。
- [ ] 乱码数据不能进入专家记忆和模型训练。
- [ ] OKX 对不上的订单/仓位不能进入收益训练。
- [ ] 极小探针和快亏平异常样本必须有标签，不能污染正常盈利模型。
- [ ] 每次训练都能解释用了多少样本、排除了多少、为什么排除。
- [ ] 系统巡检能提前发现新增污染来源。

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

## 3. 模型训练与模型/专家竞赛专项方案

### 3.1 专项目标

本专项不默认现有大模型、专家、本地模型是最优方案。所有模型和专家都必须进入统一评估、竞赛、淘汰和替换机制，最终只保留对手续费后真实净收益、回撤控制、错过机会减少、快平仓减少有正贡献的组合。

### 3.2 大模型评估与调整范围

- `qwen3-14b-trade`：评估交易方向判断、JSON 稳定性、过度保守率、开仓后收益贡献。
- `deepseek-r1-14b-risk`：评估风险识别、极端行情防守、反转判断、误杀好机会率。
- 其他候选大模型：只能先进入影子评估池，不能直接接管真实执行权重。
- 高耗时低贡献模型降级为后台复盘模型；JSON 错误、乱码、未返回率高的模型自动禁用。
- 只有在影子盘或模拟盘证明净收益/风险控制优于基线后，模型才能提高真实决策权重。

### 3.3 专家体系重构范围

现有专家不再固定全部调用，改成动态专家路由。每个专家必须记录参与次数、建议方向、采纳率、采纳后收益、错误建议率、平均耗时、重复输出率。

专家池按能力重组：

- 市场结构专家：识别趋势、震荡、突破、假突破。
- 风险专家：识别尾部风险、插针、连续亏损、极端波动。
- 盈利专家：判断手续费后净收益、盈亏比、机会质量。
- 平仓专家：判断止盈、止损、反转、继续持有。
- 事件专家：处理新闻、公告、社媒、宏观事件。
- 执行专家：处理 OKX 规则、滑点、最小下单、成交质量。
- 复盘专家：总结错过机会、亏损原因、策略退化。

处理规则：长期无贡献专家降权；重复专家合并；错误率高专家停用；缺失能力新增；高耗时专家只在关键候选上调用。

### 3.4 本地模型评估与调整范围

本地模型统一进入 `learning_only / shadow_ready / ready / degraded / disabled` 状态机。任何本地模型如果不能证明高分组更赚钱，不能参与真实仓位放大。

必须评估的本地模型：

- 盈利预测模型：预测手续费后净收益。
- 方向模型：预测 long/short/hold 哪个更优。
- 风险模型：预测亏损概率、尾部风险、快平仓风险。
- 平仓模型：预测继续持有、止盈、止损、反转平仓。
- 仓位模型：根据证据质量、风险、余额、OKX 规则给出合理仓位。
- 新增候选模型：资金费率模型、盘口滑点模型、清算风险模型、同板块联动模型、BTC/ETH 牵引模型。

每个模型必须输出训练样本数、隔离样本数、AUC/PR-AUC、高分组收益、低分组收益、最近训练时间、下一次训练条件、是否允许影响真实仓位。

### 3.5 模型/专家竞赛机制

每个模型和专家都必须和基线策略比较，不能只看单次建议是否合理。竞赛分三层：

1. **离线回放竞赛**：用历史 K线、订单、影子复盘、外部事件数据回放，比较净收益、回撤、错过机会、快亏平。
2. **影子盘竞赛**：模型/专家只给建议，不真实执行，统计如果采纳会怎样。
3. **模拟盘 A/B 竞赛**：通过小权重真实模拟执行，比较手续费后结果。

竞赛指标：

- 手续费后净收益。
- Profit factor。
- 平均盈利、平均亏损、盈亏比。
- 最大回撤。
- 快亏平比例。
- 小仓占比及小仓原因。
- 错过机会收益。
- 错误 JSON/未返回/超时率。
- 单次耗时和资源占用。

竞赛处理：

- 连续贡献为正：提高权重或进入关键决策链路。
- 只降低风险但不提升净收益：保留为风控复核，不参与放大。
- 贡献持续为负：降权。
- 连续未返回或 JSON 错误高：暂停。
- 高耗时低收益：降级为后台复盘。
- 某行情状态表现好：只在对应市场状态启用。

### 3.6 数字货币投资能力训练方案

训练目标不是让模型“会解释”，而是让系统更懂数字货币交易中的赚钱结构。训练数据必须覆盖：

- OKX 真实订单、历史仓位、成交价、手续费、滑点、资金占用。
- `1m/5m/15m/1h` K线、ticker、盘口、资金费率、未平仓量、爆仓/清算风险。
- 新闻、公告、社媒情绪、事件日历、来源可信度、事件影响衰减。
- 专家建议、大模型输出、本地模型输出、最终执行结果。
- 影子复盘：没开仓后 10/30/60/240 分钟收益，判断是否错过机会。
- 平仓复盘：最大浮盈、最大浮亏、持仓时长、手续费后净收益、是否过早平仓。

训练标签必须以结果为准：手续费后净收益、最大回撤、持仓时长、是否快亏平、是否错过机会、是否低质量仓位。不能只用模型当时的主观信心当标签。

### 3.7 大模型微调策略

不在脏数据未清理前做微调。先做结构化 prompt、工具调用和影子评估；只有当高质量训练样本足够、标签稳定、模型输出 JSON 稳定后，再考虑 LoRA/轻量微调。

微调目标：

- 稳定输出交易结构化 JSON。
- 更准确识别数字货币事件影响。
- 更好解释数值模型与专家冲突。
- 更少过度保守、错过强机会。
- 更少无证据开仓、快平仓和亏损复开。

### 3.8 新增 Batch：模型/专家体检与竞赛

- [ ] **Batch C2：模型/专家全面体检**
  - 输出现有大模型、专家、本地模型清单。
  - 统计最近 24/72 小时贡献、耗时、失败率、收益表现。
  - 标记 `keep / reduce / shadow_only / disable / replace / add_candidate`。
- [ ] **Batch C3：模型/专家竞赛框架**
  - 建立离线回放、影子盘、模拟盘 A/B 的统一统计表。
  - 每个模型/专家必须有对比基线。
  - 系统巡检展示模型/专家排行榜和淘汰原因。
- [ ] **Batch C4：数字货币特征补强**
  - 接入资金费率、盘口深度、未平仓量、清算风险、板块联动等特征。
  - 缺失的数据源必须在数据采集管理页显示状态和用途。
- [ ] **Batch C5：模型组合动态路由**
  - 根据市场状态、模型 readiness、专家历史贡献、实时风险决定调用哪些模型/专家。
  - 不再固定所有专家每轮都跑，也不让低贡献模型拖慢主链路。

### 3.9 竞赛机制验收标准

- 每个模型/专家都有独立贡献统计。
- 系统能回答：哪个模型赚钱、哪个模型亏钱、哪个专家无效、哪个专家耗时高但没贡献。
- 模型/专家权重不是人工固定，而是由历史表现、当前市场状态、readiness 和风险共同决定。
- 新模型不能直接上线，必须先走影子评估。
- 任何模型/专家如果拖累净收益或增加快亏平，必须自动降权或暂停。

---

## 4. 执行顺序与停止规则

### 执行顺序

1. Batch A：先堵乱码入口，避免后续所有记录继续污染。
2. Batch B0：先治理旧数据/脏数据生命周期，防止训练和模型评估继续被污染。
3. Batch B：处理观望错过机会，但不直接放宽开仓。
4. Batch C：处理 ML readiness，明确模型是否能参与仓位放大。
5. Batch C2-C5：评估、淘汰、替换和新增模型/专家，建立竞赛机制。
6. Batch D：把新治理点纳入系统巡检。
7. Batch E：分批清质量债。
8. Batch F：每批都执行部署和验证。

### 停止规则

- 如果某批测试失败，不能进入下一批。
- 如果线上出现 `critical`，先修线上硬故障，不继续策略优化。
- 如果策略健康出现失败订单、弱证据执行、快亏平新增，停止放大逻辑，回到对应批次定位。
- 如果连续 3 次局部修改仍不能解决同一问题，停止打补丁，重审架构边界。

---

## 5. 第一批开始前检查命令

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

## 6. 计划执行记录

| Batch | 状态 | Git Commit | 线上验证 | 备注 |
| --- | --- | --- | --- | --- |
| A 文本乱码根治 | 未开始 | - | - | 先堵写入口 |
| B0 旧数据/脏数据生命周期治理 | 未开始 | - | - | 训练前置门禁 |
| B 影子错过机会闭环 | 未开始 | - | - | 不放宽硬风控 |
| C ML readiness | 未开始 | - | - | 不硬改 ready |
| C2 模型/专家全面体检 | 未开始 | - | - | 不默认现有组合最优 |
| C3 模型/专家竞赛框架 | 未开始 | - | - | 和基线策略对比 |
| C4 数字货币特征补强 | 未开始 | - | - | 资金费率/盘口/清算等 |
| C5 模型组合动态路由 | 未开始 | - | - | 动态选择有效组合 |
| D 系统巡检深层节点 | 未开始 | - | - | 加责任链路 |
| E Ruff/格式债 | 未开始 | - | - | 分目录清理 |
| F 部署观测闭环 | 未开始 | - | - | 每批都跑 |
