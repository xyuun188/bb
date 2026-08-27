# 训练有效性与模型晋级闭环整改计划

日期：2026-08-25  
项目：`E:\code\bb`  
状态：计划文件，仅供后续执行；创建本文件时不启动训练、不执行线上测试、不修改交易逻辑。

## 1. 目标与非目标

### 1.1 目标

建立一条可以证明“训练确实改善了模型和交易结果”的闭环：

```text
新数据
-> 正确标签
-> 有效训练
-> 样本外指标改善
-> 候选模型验收
-> 线上受控观察
-> 真实交易反馈
-> 下一轮有效训练
```

本计划要解决的不是“训练时间够不够长”，而是：

- 训练任务成功但模型没有改善；
- 最新 challenger 已生成但没有进入线上；
- 线上仍使用旧 active 模型而界面不够清楚；
- 影子样本、真实成交、手续费、滑点、资金费和最终收益口径混杂；
- 无法知道是模型判断错误，还是风控、下单、成交、平仓链路导致亏损；
- 六个专家被一个整体门禁一起拦截，无法定位具体拖累者；
- 训练或验证任务可能因重复运行而长期占用资源。

### 1.2 非目标

- 不因训练时间长而强行晋级；
- 不为增加开仓次数而降低晋级门槛；
- 不把单笔资金费收益当成模型训练成果；
- 不在审计期间启动新的训练任务；
- 不在候选模型未完成验收前切换线上 active；
- 不把一次成功的任务执行等同于模型已经有效。

## 2. 严格执行规则：防止重复运行和无限运行

以下规则适用于本计划的所有审计、训练和线上验证步骤。

### 2.1 一次运行原则

- 每个阶段只创建一个唯一 `run_id`；
- 同一阶段存在未完成 `run_id` 时，禁止再次启动；
- 同一输入指纹已经产生完整结果时，禁止重复计算；
- 重复请求只能读取已有结果，不能重新启动任务；
- 本计划不允许使用“失败后无限自动重试”。

### 2.2 输入指纹原则

每次运行必须记录：

- 数据时间范围；
- 样本数量和样本指纹；
- 模型版本和代码版本；
- 标签版本；
- 成本合同版本；
- 运行阶段和启动时间。

如果上述输入没有变化，后续只读取已有结果，不重新训练或重新验证。

### 2.3 超时和停止原则

- 审计任务必须有固定超时时间；
- 训练、评估、线上验证分别设置独立超时；
- 超时后立即记录 `timeout` 和已完成进度，然后停止该次运行；
- 超时不得在同一任务中自动重试；
- 只有查明超时原因、改变输入或修复问题后，才允许创建新的 `run_id`；
- 任何阶段完成后立即停止，不自动串联下一阶段。

### 2.4 线上安全原则

- 数据审计和离线评估期间不调用下单、平仓、训练启动接口；
- 线上验证必须由单独的、明确授权的阶段启动；
- 一次线上验证结束后必须回收运行锁；
- 没有验收结论时，不允许自动切换 active 模型；
- 发现内存、CPU、磁盘或接口异常时，立即停止验证，不继续追加任务。

## 3. 阶段一：建立版本真相和效果基线

### 3.1 需要记录的版本状态

必须同时列出：

- 最新训练版本；
- 最新评估版本；
- 当前线上 active 版本；
- 被拒绝的 challenger 版本；
- 每个版本的训练时间、评估时间和生命周期；
- 晋级或拒绝原因；
- 当前 active 距离最新训练版本的时间差。

### 3.2 固定三个不可变基线

- 不使用模型的原有规则策略；
- 当前线上 active 模型；
- 资金费计入和不计入的对照结果。

### 3.3 统一收益口径

```text
费后净收益 = 毛盈亏 - 手续费 - 滑点 + 资金费
```

结果必须按多头、空头、币种、持仓时间、预测周期和市场状态拆分。

阶段产物：一份基线报告，必须包含新旧模型的增量，而不是只有“训练成功”状态。

阶段停止条件：基线报告完整生成后立即停止，不自动进入训练。

## 4. 阶段二：训练数据和标签审计

### 4.1 样本权威层级

必须分别统计：

- 影子市场机会样本；
- 反事实执行成本样本；
- OKX 实际成交样本；
- OKX 实际手续费；
- OKX 实际资金费；
- OKX 实际平仓收益。

影子样本只能表示市场机会，不能直接当作模型已经实现的收益。

### 4.2 完整持仓生命周期校验

每条真实训练样本必须能够关联：

```text
开仓 -> 成交 -> 持仓 -> 资金费 -> 平仓 -> 手续费 -> 滑点 -> 最终收益
```

重点检查：

- 一个 `posId` 是否关联多个生命周期；
- 一笔资金费是否重复或漏记；
- 平仓缺失的记录是否被错误纳入完整样本；
- 手续费和滑点是否来自同一成交；
- 标签时间是否与实际平仓时间一致；
- 资金费是否只计入一次。

阶段产物：样本权威层级报告、异常样本清单、标签版本和数据指纹。

阶段停止条件：异常清单和处理策略生成后停止，不自动重新训练。

## 5. 阶段三：交易执行链路审计

对每个候选信号统计完整状态：

```text
产生信号
-> 通过证据门禁
-> 通过风险检查
-> 提交订单
-> 成交
-> 建立持仓
-> 平仓
-> 结算
```

分别统计：

- 模型方向判断错误；
- 盈利质量判断错误；
- 证据门禁阻断；
- 风控阻断；
- 下单失败；
- 成交失败；
- 开仓后过早部分平仓；
- 平仓记录缺失；
- 资金费、手续费或滑点导致的收益变化。

阶段目标是把“模型失败”和“执行失败”分开，避免用被执行链路污染的标签继续训练。

阶段停止条件：链路统计和最大影响因素明确后停止，不自动触发修复或训练。

## 6. 阶段四：专家独立贡献分析

必须做单模型和消融对照：

- 方向专家单独评估；
- 盈利质量专家单独评估；
- 损失过滤专家单独评估；
- 时间序列专家单独评估；
- 情绪专家单独评估；
- 退出专家单独评估；
- 移除单个专家后的整体评估；
- 不同专家组合的对照评估。

每个专家都要回答：

- 是否改善费后收益；
- 是否降低错误开仓；
- 是否改善多空平衡；
- 是否降低回撤或尾部损失；
- 是否只在特定币种或市场状态有效。

阶段停止条件：每个专家的贡献、拖累或无效结论明确后停止，不自动晋级任何模型。

## 7. 阶段五：确定问题后才允许重新训练

只有满足以下任一条件，才允许启动新训练：

- 有足够的新真实成交样本；
- 有新的市场状态样本；
- 发现并修正数据、资金费、手续费、滑点或平仓标签问题；
- 发现明确的特征、目标函数或退出策略问题；
- 发现模型输入分布发生明显漂移。

每次训练必须绑定一个明确假设，例如：

- 修正资金费标签后，费后收益是否改善；
- 修正退出标签后，过早部分平仓是否减少；
- 纳入真实滑点后，模型是否仍保持正收益；
- 删除无效特征后，样本外稳定性是否提高。

禁止对输入不变的数据重复训练来“等待效果出现”。

## 8. 阶段六：候选模型对照验收

新模型必须在同一份冻结的样本外数据上，同时比较：

- 当前 active 模型；
- 无模型基线；
- 新 challenger 模型。

验收条件：

- 费后样本外平均收益为正；
- 收益下置信界限为正；
- Profit Factor 高于盈亏平衡；
- 时间滚动验证通过；
- 市场状态验证通过；
- 留一币种验证不失效；
- 真实成交样本达到合同要求；
- 相比 active 和无模型基线存在明确增益。

如果只有某个方向、币种或专家有效，只能限制在有效范围内，不得整包晋级。

阶段停止条件：产生明确的“晋级、保留 challenger 或退回整改”结论后停止。

## 9. 阶段七：有限期线上观察

线上观察模式必须有固定有效样本数和固定结束条件：

- 达到有效样本数后只做一次晋级判定；
- 通过则进入下一阶段；
- 不通过则退回整改；
- 超过观察期限仍无改善则停止放大资源；
- 观察模式不得无限期维持 1 倍小单。

观察期间必须核对：

- 预测与实际方向是否一致；
- 费后收益是否改善；
- 资金费是否正确归因；
- 是否发生开仓后立即部分平仓；
- 是否存在信号通过但订单未成交；
- active 版本是否与观察版本一致。

## 10. 页面可视化训练效果

训练闭环整改完成后，不能只生成后台报告或接口数据，页面必须能够展示可审计的训练事实。页面展示必须来自已经生成并缓存的审计报告，打开页面、刷新页面或切换筛选条件不得启动训练、评估、回测或线上验证。

### 10.1 版本状态面板

必须同时展示：

- 当前线上 active 版本；
- 最新 challenger 版本；
- 无模型基线版本或基线标识；
- 各版本训练时间、评估时间和状态；
- challenger 是否已经进入线上；
- 未晋级的具体原因；
- 当前 active 距离最新训练版本的时间差。

### 10.2 训练前后效果对比

页面必须能够在同一时间范围、同一冻结样本外数据上对比：

- 当前 active；
- 最新 challenger；
- 无模型基线。

至少展示以下指标的数值和变化量：

- 费后净收益；
- 毛盈亏；
- 手续费；
- 滑点；
- 资金费；
- Profit Factor；
- 收益下置信界限；
- 最大回撤；
- 胜率；
- 亏损概率；
- 有效样本数。

页面必须支持多头、空头、币种、持仓时间、预测周期和市场状态筛选，并明确显示筛选后的样本数量和数据时间范围。

### 10.3 收益组成和资金费归因

页面必须把以下金额分开显示，禁止只展示一个总收益：

```text
毛盈亏
- 手续费
- 滑点
+ 资金费
= 费后净收益
```

如果账户盈利主要来自少数资金费订单，页面必须显示资金费占费后净收益的比例，并标注“资金费贡献”不能直接等同于模型预测能力。

### 10.4 专家贡献面板

页面必须展示各专家单独使用、组合使用和移除后的效果：

- 方向专家；
- 盈利质量专家；
- 损失过滤专家；
- 时间序列专家；
- 情绪专家；
- 退出专家。

每个专家至少显示对费后收益、回撤、错误开仓和多空平衡的影响，避免只显示一个整体模型分数而无法定位拖累来源。

### 10.5 交易链路漏斗

页面必须展示每个阶段的数量和损失率：

```text
产生信号
-> 通过证据门禁
-> 通过风险检查
-> 提交订单
-> 成交
-> 建立持仓
-> 平仓
-> 结算
```

必须单独标识：

- 模型判断错误；
- 证据阻断；
- 风控阻断；
- 下单或成交失败；
- 开仓后过早部分平仓；
- 平仓记录缺失；
- 资金费、手续费或滑点造成的损失。

### 10.6 样本可信度面板

页面必须区分并展示：

- 影子市场机会样本；
- 反事实成本样本；
- OKX 实际成交样本；
- OKX 实际资金费样本；
- OKX 实际平仓收益样本；
- 被排除的异常样本；
- 有效样本数和有效样本指纹。

影子样本不得被页面标记为“真实盈利样本”。

### 10.7 结论和数据新鲜度

页面必须明确显示：

- 相比 active，challenger 是改善、无变化还是变差；
- 是否满足晋级条件；
- 未晋级的阻断原因；
- 报告生成时间；
- 数据截止时间；
- 数据是否过期；
- 报告是否完整或仅为部分结果。

如果报告过期、样本不足或链路数据不完整，页面必须显示“结论不可用”，不得显示为正常训练成功。

### 10.8 页面运行安全

- 页面接口只读取缓存的审计报告和版本元数据；
- 页面刷新不得触发训练、评估、回测或线上验证；
- 手动刷新必须有冷却时间和请求去重；
- 同一报告指纹只允许读取，不允许重复生成；
- 页面加载超时只显示已有缓存和数据过期状态，不自动重试长任务。

阶段产物：页面可视化训练效果面板、版本对照、收益归因、专家贡献、交易漏斗和样本可信度视图。

阶段停止条件：页面能够展示一份完整、可追溯的报告后停止，不自动启动下一阶段。

## 11. 最终验收标准

本计划完成的标准不是“训练任务成功”，而是以下资料全部齐全：

- 版本状态和 active/challenger 对照报告；
- 训练前后效果差异报告；
- 基线对照报告；
- 样本权威层级和标签完整性报告；
- 执行链路失败归因报告；
- 各专家独立贡献报告；
- challenger 样本外验收报告；
- 有限期线上观察报告；
- 页面可视化训练效果面板；
- 页面显示的报告指纹与后台报告一致；
- 页面刷新不会启动任何训练或验证任务；
- 明确的晋级或退回整改结论。

在这些结果完成前，不得宣称“训练有效”“模型已改善”或“模型已晋级”。

## 12. 本计划的执行顺序

```text
阶段一：版本与基线
-> 阶段二：数据与标签
-> 阶段三：执行链路
-> 阶段四：专家贡献
-> 阶段五：有明确假设后训练
-> 阶段六：冻结样本外验收
-> 阶段七：页面可视化训练效果
-> 阶段八：有限期线上观察
-> 最终晋级或退回整改
```

每个箭头只允许在前一阶段产生完整产物后推进一次。页面可视化阶段只能读取前面阶段的缓存产物，不能反向触发训练或验证。任何阶段失败、超时或输入指纹未变化，都必须停止并报告，不得自动重复执行。

## 13. 逐步实施说明：代码、文件和验证方式

本节把前面的原则转换成可以逐项执行的工程任务。执行时严格按顺序推进；每一步都要先完成代码和验证，再开始下一步。下面的代码片段是接口形状和伪代码，实际实现必须复用项目已有的 repository、配置、日志和鉴权工具，不要另造一套数据库连接或训练调度器。

### 13.1 第 0 步：建立工作分支和变更边界

要做的事：

1. 从当前 `main` 创建 `codex/training-effectiveness-closed-loop` 分支。
2. 先确认工作区没有把正在进行的训练、评估或线上观察作为本次变更的一部分。
3. 只允许修改以下范围：
   - `services/training_effectiveness_report.py`（新增）；
   - `scripts/build_training_effectiveness_report.py`（新增）；
   - `web_dashboard/api/dashboard.py`；
   - `web_dashboard/static/index.html`；
   - `web_dashboard/static/js/dashboard.js`；
   - `web_dashboard/static/css/dashboard.css`；
   - 对应 `tests/` 测试文件；
   - 本计划文件。

验证命令：

```powershell
rtk git status --short
rtk git branch --show-current
```

完成标准：当前分支、基线 commit、变更文件清单已经记录；没有在本步启动训练、回测、下单或平仓。

### 13.2 第 1 步：先定义不可变报告合同

新增 `services/training_effectiveness_report.py`，先写类型和纯函数，再接数据库。报告必须能独立保存、读取和复现，不允许把 ORM 对象直接塞进 JSON。

建议的顶层结构：

```python
TRAINING_EFFECTIVENESS_REPORT_VERSION = "2026-08-25.v1"

{
    "report_version": TRAINING_EFFECTIVENESS_REPORT_VERSION,
    "report_id": "te-<run_id>",
    "generated_at": "...Z",
    "data_cutoff_at": "...Z",
    "status": "complete|partial|invalid",
    "input_fingerprint": "sha256:...",
    "run": {
        "run_id": "...",
        "stage": "baseline|data_audit|execution_audit|expert_ablation|evaluation",
        "source_code_version": "...",
        "label_version": "...",
        "cost_contract_version": "...",
    },
    "versions": {
        "active": {...},
        "challenger": {...},
        "baseline": {...},
    },
    "filters": {
        "mode": "paper|live|all",
        "side": "long|short|all",
        "symbol": "...|all",
        "market_state": "...|all",
        "hold_minutes": {"min": 0, "max": 0},
    },
    "metrics": {...},
    "cost_attribution": {...},
    "expert_contributions": [...],
    "execution_funnel": {...},
    "sample_quality": {...},
    "conclusion": {...},
    "freshness": {...},
}
```

必须实现的纯函数：

- `build_input_fingerprint(inputs) -> str`：对排序后的版本、样本 ID、时间窗口、标签和成本合同做 SHA-256；不把随机时间写入指纹。
- `calculate_fee_after_return(gross_pnl, fee, slippage, funding_fee) -> float`：严格使用“毛盈亏 - 手续费 - 滑点 + 资金费”。
- `calculate_metric_delta(active, challenger, baseline) -> dict`：同时返回绝对变化和百分比变化，遇到分母为 0 返回 `None`，不能返回无穷大。
- `classify_sample_authority(sample) -> str`：只能返回 `shadow_opportunity`、`counterfactual_cost`、`okx_realized`、`excluded` 之一。
- `validate_report(report) -> list[str]`：检查必填字段、时间顺序、样本指纹、收益组成是否相等；有错误时将报告标为 `invalid`。

先为这些纯函数写单元测试，使用固定字典输入，不连接数据库。这样收益公式和样本分层可以在不启动训练的情况下先锁定。

### 13.3 第 2 步：实现只读报告服务

在同一个文件新增 `TrainingEffectivenessReportService`。它只负责读取已有结果并聚合，不负责训练、评估、回测、模型切换或交易。

服务内部应按以下顺序实现：

1. 读取 `services/model_training_registry.py` 提供的模型生命周期信息，取得 active、challenger、训练时间、评估时间和晋级原因。
2. 读取现有权威成交/结算数据，优先使用 `services/authoritative_trade_outcome.py` 的结果；不能把 `Position.realized_pnl` 当作唯一权威来源。
3. 读取影子复盘和反事实成本样本，调用 `classify_sample_authority` 分层；影子样本只能进入机会指标，不能进入真实成交收益指标。
4. 读取执行阶段记录，按 `DecisionStage` 聚合信号、门禁、风控、提交、成交、持仓、平仓、结算数量。
5. 分别计算 active、challenger、baseline 的指标和 delta。
6. 运行 `validate_report`；有缺失或冲突时返回 `status="partial"` 或 `"invalid"`，并列出 `blocking_reasons`。

伪代码：

```python
class TrainingEffectivenessReportService:
    async def build(self, *, filters: TrainingEffectivenessFilters) -> dict[str, Any]:
        versions = load_model_registry_snapshot()
        samples = await self._load_samples(filters)
        authoritative = [s for s in samples if classify_sample_authority(s) == "okx_realized"]
        shadow = [s for s in samples if classify_sample_authority(s) == "shadow_opportunity"]
        report = assemble_report(
            versions=versions,
            metrics=compare_versions(versions, authoritative, shadow),
            cost_attribution=aggregate_costs(authoritative),
            expert_contributions=aggregate_expert_ablation(samples),
            execution_funnel=aggregate_execution_funnel(samples),
            sample_quality=aggregate_sample_quality(samples),
        )
        report["blocking_reasons"] = validate_report(report)
        report["status"] = "complete" if not report["blocking_reasons"] else "partial"
        return sanitize_payload(report)
```

每个数据库查询都要有明确时间范围、最大条数和超时；使用 `get_read_session_ctx()`，禁止在该服务中调用 `get_session_ctx()`、`commit()` 或任何写操作。数据库不可用时返回结构化错误报告，不抛出到页面。

### 13.4 第 3 步：生成并缓存报告，保证幂等

新增 `scripts/build_training_effectiveness_report.py`，作为唯一的报告生成入口。它不是训练入口，也不能隐式调用训练脚本。

脚本执行顺序：

1. 解析 `--stage`、`--from`、`--to`、`--mode` 和可选 `--run-id`。
2. 读取当前训练 epoch、模型 registry、标签版本和成本合同版本。
3. 生成输入指纹；如果同一指纹已有 `complete` 报告，直接输出已有路径并退出 0。
4. 使用文件锁 `data/training_effectiveness_report.lock`；锁已存在时输出 `already_running` 并退出，不创建第二个任务。
5. 调用 `TrainingEffectivenessReportService.build()`，设置固定总超时。
6. 先写临时文件，再使用原子替换写入：
   - `data/training_effectiveness_reports/<report_id>.json`；
   - `data/training_effectiveness_reports/latest.json`。
7. 将 `run_id`、输入指纹、状态、耗时和错误写入审计日志。
8. 报告生成完成后立即释放文件锁并退出；不得自动启动训练或评估。

建议命令：

```powershell
rtk proxy .venv\Scripts\python.exe scripts/build_training_effectiveness_report.py `
  --stage baseline --from 2026-08-01T00:00:00Z --to 2026-08-25T00:00:00Z
```

验证：同一命令连续运行两次，第二次必须复用相同 `report_id` 和 `input_fingerprint`，不能新增训练、回测或数据库写入。

### 13.5 第 4 步：增加只读 API

在 `web_dashboard/api/dashboard.py` 增加接口，不新建页面专用服务：

```python
@router.get("/training-effectiveness/report")
async def get_training_effectiveness_report(
    mode: str = "all",
    side: str = "all",
    symbol: str | None = None,
    market_state: str | None = None,
    report_id: str | None = None,
) -> dict[str, Any]:
    """Read cached training-effectiveness evidence; never start a job."""
    report = load_cached_training_effectiveness_report(report_id=report_id)
    filtered = apply_report_filters(report, mode=mode, side=side, symbol=symbol, market_state=market_state)
    return sanitize_payload(filtered)
```

接口要求：

- 只读取 `latest.json` 或指定 `report_id` 文件和模型 registry 快照；
- 不调用 `scripts/build_training_effectiveness_report.py`；
- 不调用训练、评估、回测、线上观察或交易服务；
- 缓存命中时直接返回；缓存不存在时返回 `status="missing"`；
- 过滤条件只能改变展示结果，不能改变原始报告指纹；
- 响应必须包含 `report_id`、`input_fingerprint`、`generated_at`、`data_cutoff_at` 和 `freshness`。

为接口添加三类测试：缓存完整、缓存缺失、报告过期。测试中 monkeypatch 所有训练启动函数，断言它们没有被调用。

### 13.6 第 5 步：把可视化放进“专家记忆”页面

训练效果可视化**必须放在现有专家记忆页面中**，不增加独立导航项，不放到“服务器量化模型”页面，也不把训练按钮混入专家记忆页。

修改 `web_dashboard/static/index.html`：

1. 在 `#expert-memory-tabs` 现有“专家长期记忆”“交易复盘”之后增加：

```html
<button class="trade-tab"
        data-expert-memory-view="training-effectiveness"
        onclick="setExpertMemoryView('training-effectiveness')">
    训练效果
</button>
```

2. 在 `#page-expert-memory` 内增加同级 `tab-panel`，不要再包一层 card：

```html
<div class="tab-panel" id="expert-memory-panel-training-effectiveness">
    <div class="training-effectiveness-toolbar">
        <select id="training-effectiveness-mode"></select>
        <select id="training-effectiveness-side"></select>
        <select id="training-effectiveness-symbol"></select>
        <button class="btn btn-sm" onclick="fetchTrainingEffectivenessReport()">刷新</button>
    </div>
    <div id="training-effectiveness-freshness"></div>
    <div id="training-effectiveness-version-comparison"></div>
    <div id="training-effectiveness-metrics"></div>
    <div id="training-effectiveness-cost-attribution"></div>
    <div id="training-effectiveness-expert-contributions"></div>
    <div id="training-effectiveness-execution-funnel"></div>
    <div id="training-effectiveness-sample-quality"></div>
    <div id="training-effectiveness-conclusion"></div>
</div>
```

3. 页面首次打开专家记忆时只调用已有的 `fetchExpertMemories()` 和新的只读 `fetchTrainingEffectivenessReport()`；切换 Tab 只能切换 DOM，不得触发报告生成。

### 13.7 第 6 步：实现专家记忆页内的前端渲染

修改 `web_dashboard/static/js/dashboard.js`：

1. 在 state 中新增 `trainingEffectivenessReport`、`trainingEffectivenessRequest` 和筛选状态。
2. 实现 `fetchTrainingEffectivenessReport()`，使用现有 `fetchLatestPageJSON` 或同等请求去重工具：

```javascript
async function fetchTrainingEffectivenessReport() {
    const query = new URLSearchParams(trainingEffectivenessFilters());
    const report = await fetchLatestPageJSON(
        'training-effectiveness',
        `/api/training-effectiveness/report?${query.toString()}`,
    );
    if (!report) return;
    state.trainingEffectivenessReport = report;
    renderTrainingEffectiveness(report);
}
```

3. 实现以下渲染函数，每个函数只接收报告 JSON，不做数据库计算：
   - `renderTrainingEffectivenessFreshness(report)`：显示报告时间、数据截止时间、过期状态和指纹；
   - `renderTrainingEffectivenessVersions(report)`：显示 active、challenger、baseline 及晋级状态；
   - `renderTrainingEffectivenessMetrics(report)`：显示费后净收益、Profit Factor、置信下界、回撤、胜率和样本数及 delta；
   - `renderTrainingEffectivenessCostAttribution(report)`：按“毛盈亏 - 手续费 - 滑点 + 资金费 = 费后净收益”显示，并计算资金费占比；
   - `renderTrainingEffectivenessExperts(report)`：显示六个专家单独、组合、移除后的影响；
   - `renderTrainingEffectivenessFunnel(report)`：显示八段交易漏斗和每段损失率；
   - `renderTrainingEffectivenessSamples(report)`：显示影子、反事实、OKX 实际、排除样本的数量和权威等级；
   - `renderTrainingEffectivenessConclusion(report)`：显示改善/不变/变差、是否满足晋级、阻断原因。

4. 报告为 `missing`、`partial`、`invalid`、过期或样本不足时，所有指标显示“结论不可用”，禁止显示绿色“训练成功”。
5. 在 `setExpertMemoryView()` 中增加 `training-effectiveness` 分支，只切换 `#expert-memory-panel-training-effectiveness` 的 active 状态。
6. 过滤条件变化时只重新读取同一缓存报告并在前端应用过滤，不触发后台生成任务；若后端过滤，必须保持原始 `input_fingerprint`。

### 13.8 第 7 步：样式和图表实现

修改 `web_dashboard/static/css/dashboard.css`，复用专家记忆页已有的 Tab、表格、状态色和分页样式。只增加训练效果面板所需的类：

- `.training-effectiveness-toolbar`：筛选控件和刷新按钮；
- `.training-effectiveness-summary`：版本与总体结论；
- `.training-effectiveness-metric-grid`：固定列宽的指标网格；
- `.training-effectiveness-cost-equation`：收益组成等式；
- `.training-effectiveness-funnel`：漏斗阶段和损失率；
- `.training-effectiveness-status`：complete、partial、stale、invalid 状态。

图表优先使用已有 `web_dashboard/static/js/charts.js` 的封装；如果没有合适封装，先用语义化表格和 CSS 条形图，不引入新的图表依赖。图表只能使用报告中的数值，不能在浏览器中重新计算标签或收益。

页面验收必须覆盖：桌面宽度、窄桌面、移动宽度；长模型名、空数据、超长错误原因、负收益和 0 样本都不能溢出或互相遮挡。

### 13.9 第 8 步：每个原阶段对应的具体代码任务

| 计划阶段 | 代码任务 | 主要产物 | 完成后检查 |
| --- | --- | --- | --- |
| 阶段一 版本与基线 | 读取 `model_training_registry`，实现 active/challenger/baseline 快照 | `versions`、基线指标 | 三个版本都可追溯，缺失版本显示阻断 |
| 阶段二 数据与标签 | 聚合权威成交、资金费、手续费、滑点、影子和排除样本 | `sample_quality`、标签指纹 | 影子样本不会进入真实收益 |
| 阶段三 执行链路 | 按 `DecisionStage` 聚合漏斗和失败原因 | `execution_funnel` | 模型失败与执行失败分开 |
| 阶段四 专家贡献 | 读取已有专家输出和消融结果；没有结果就标记缺失 | `expert_contributions` | 不用整体分数冒充专家贡献 |
| 阶段五 重新训练 | 只记录明确假设和新 `run_id`；训练入口保持独立 | `run`、假设说明 | 输入指纹不变时拒绝重复运行 |
| 阶段六 候选验收 | 在冻结样本外比较 active/challenger/baseline | `conclusion` | 晋级条件逐条可见 |
| 阶段七 页面可视化 | 在专家记忆页增加训练效果 Tab | 页面面板 | 页面只读缓存，不启动任务 |
| 阶段八 线上观察 | 读取固定期限观察报告，不在页面启动观察 | 观察结论 | 达到结束条件后只判定一次 |

### 13.10 第 9 步：测试清单

新增或修改测试时至少覆盖：

1. `tests/test_training_effectiveness_report.py`
   - 收益公式和四舍五入；
   - 资金费正负值；
   - 样本权威分层；
   - 指纹稳定性；
   - 报告完整性校验；
   - active/challenger/baseline delta。
2. `tests/test_training_effectiveness_api.py`
   - API 只读缓存；
   - 缓存过期和缺失；
   - 筛选不改变原始指纹；
   - 训练/评估函数未被调用；
   - 超时返回结构化 `partial`。
3. `tests/test_dashboard_main_ui_contract.py`
   - 专家记忆页包含 `training-effectiveness` Tab；
   - 三个专家记忆面板都能切换；
   - 页面不存在独立训练效果导航；
   - 前端请求路径为 `/api/training-effectiveness/report`。
4. `tests/test_training_effectiveness_ui_contract.py`
   - `partial`、`invalid`、过期和 0 样本不会渲染为成功；
   - 成本等式字段全部显示；
   - 影子样本标签明确为非真实成交。

最小验证命令：

```powershell
rtk proxy .venv\Scripts\python.exe -m pytest -q `
  tests/test_training_effectiveness_report.py `
  tests/test_training_effectiveness_api.py `
  tests/test_dashboard_main_ui_contract.py `
  tests/test_training_effectiveness_ui_contract.py
```

### 13.11 第 10 步：部署、线上验证和停止条件

部署前：

1. 运行格式、类型和相关测试；
2. 用固定 fixture 生成一份 `complete` 报告和一份 `partial` 报告；
3. 检查页面专家记忆 Tab，不启动任何训练或交易服务；
4. 检查报告 JSON 中没有 API key、私钥或账户敏感信息。

部署后：

1. 执行仓库规定的同步命令：

```powershell
python scripts/sync_to_online_server.py --split-services
```

2. 只读取线上 `/api/training-effectiveness/report`，确认 HTTP 成功、`report_id` 和 `input_fingerprint` 存在。
3. 打开专家记忆页，切换“训练效果” Tab，确认页面请求只读 API；检查服务日志没有训练启动、评估启动或下单日志。
4. 使用同一筛选条件刷新两次，确认报告指纹不变、后台没有新增 run。
5. 运行线上只读审计，确认报告状态、样本数量和后台报告一致。

停止条件：

- 报告缺失、指纹变化、权威成交样本不足、成本等式不平衡、接口超时或页面触发任务时，立即停止本阶段；
- 不得通过降低晋级门槛、增加影子样本权重或隐藏失败阶段来让页面显示“有效”；
- 只有报告完整、页面可追溯、线上观察达到固定结束条件后，才进入晋级判定；
- 晋级判定完成后立即停止，不自动切换 active，除非另有明确授权的发布步骤。
