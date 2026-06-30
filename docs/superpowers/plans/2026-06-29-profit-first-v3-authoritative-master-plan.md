# Profit-First v3 Authoritative Master Plan

Date: 2026-06-29
Project: `F:\BB`
Status: authoritative execution plan for the next optimization phase

## 0. Document Authority

This document is the single execution entrypoint for the next phase.

It consolidates and supersedes the execution guidance in:

- `docs/superpowers/plans/2026-06-29-system-profitability-optimization-plan.md`
- `docs/superpowers/plans/2026-06-29-profit-first-master-control-v3-plan.md`

It inherits historical facts, safety boundaries, and Phase 3 infrastructure constraints from:

- `docs/superpowers/plans/2026-06-26-phase-3-quant-master-control-plan.md`
- `docs/superpowers/plans/2026-06-22-quant-closed-loop-eradication.md`

Use the older documents as evidence logs and background references only. Do not execute a new change from an older plan if it conflicts with this document.

## 1. Premise

The new large-model server is already the target model foundation. The next problem is not "deploy more models" or "add more dashboard pages". The next problem is making the trading master control consume model evidence correctly and optimize every real trading action for realized net profit after fees, slippage, funding, and execution constraints.

Current symptom cluster:

- Entries still often become tiny/probe orders.
- Some windows still have long no-entry periods.
- Closed-position results can be all-loss windows.
- Exit/release logic can turn low-quality probes into realized losses.
- Model and strategy outputs are visible, but not yet governed by one realized-net-profit control contract.

Therefore the next phase is named:

```text
Profit-First Master Control v3
```

## 2. Highest-Level Objective

Upgrade the system from a safety-rule-heavy trading robot into a profit-first master control system.

Every real entry, add, hold, reduce, full close, model weight, and strategy promotion must be justified by expected or realized net profit evidence.

The system is successful only when it can answer:

- Why did it open?
- Why did it size this large or this small?
- Why did it not open?
- Why did it close?
- Was the close profitable or loss-making after fees?
- Which model, strategy, symbol, side, and market regime helped or hurt realized net profit?
- What will be promoted, demoted, paused, or kept shadow-only next time?

## 3. Non-Goals And Safety Boundaries

Do not solve the current problems by:

- Globally increasing leverage.
- Globally increasing position size.
- Loosening entry gates without realized or shadow evidence.
- Letting a single LLM force real entries.
- Counting multiple role views of the same model as independent sources.
- Promoting old or untrusted artifacts into live influence.
- Training from dirty or legacy samples.
- Continuing tiny probes after recent tiny/probe closes are all losing.
- Closing stale probes at repeated small losses without hard risk or replacement opportunity evidence.
- Building more pages while the core trading contract remains fragmented.

The new big-model server can provide trade reasoning, estimates, and risk review, but it cannot bypass:

- complete trade plan requirement,
- fee/slippage-adjusted positive expectancy,
- max loss budget,
- tail-risk cap,
- OKX fact/account truth,
- clean training gates,
- shadow/canary/live promotion discipline.

## 4. Canonical Trade Contract

Implement one canonical plan object:

```text
ProfitFirstTradePlan
```

Every candidate that may become a real order must have this plan.

Required groups:

1. Identity
   - `symbol`
   - `side`
   - `analysis_type`
   - `strategy_profile_id`
   - `model_sources`
   - `independent_source_count`
   - `plan_version`

2. Expected Profit
   - `expected_gross_return_pct`
   - `expected_fee_pct`
   - `expected_slippage_pct`
   - `expected_net_return_pct`
   - `expected_profit_usdt`
   - `profit_quality_ratio`
   - `reward_risk_ratio`

3. Risk
   - `loss_probability`
   - `expected_loss_usdt`
   - `tail_loss_probability`
   - `tail_loss_usdt`
   - `max_stop_loss_usdt`
   - `portfolio_side_pressure`
   - `same_symbol_side_edge`
   - `recent_realized_edge`

4. Timing
   - `expected_hold_minutes`
   - `max_hold_minutes`
   - `entry_price_reference`
   - `invalidation_price`

5. Position
   - `decision_lane`
   - `position_size_pct`
   - `leverage`
   - `promotion_reasons`
   - `block_or_downgrade_reasons`

6. Exit
   - `exit_plan_id`
   - `stop_loss_pct`
   - `take_profit_pct`
   - `trailing_profit_trigger_pct`
   - `profit_drawdown_exit_pct`
   - `partial_exit_plan`
   - `full_exit_plan`
   - `do_not_close_conditions`

Hard rules:

- Missing required profit, risk, timing, position, or exit fields means `shadow_only`.
- `expected_net_return_pct <= 0` means `shadow_only`.
- No exit plan means no real entry.
- Low payoff quality cannot receive meaningful size.
- Any live action must persist the plan fields for later attribution.

## 5. Canonical Decision Lanes

All candidates must be assigned exactly one lane.

### `shadow_only`

No real order. Used for missing plan fields, weak evidence, negative expectancy, untrusted model/data state, excessive tail risk, or negative recent realized edge.

### `tiny_probe`

Controlled learning sample only.

Initial size:

- 1%-2%
- normally 2x-3x

Requirements:

- positive expected net return,
- acceptable loss probability,
- acceptable tail loss,
- at least 2 independent aligned source groups,
- no active recent probe all-loss brake.

### `validated_probe`

Real but still cautious trade.

Initial size:

- 3%-5%
- normally 3x-5x

Requirements:

- positive expected net return after fees/slippage,
- profit quality above threshold,
- loss probability below validated threshold,
- tail loss within budget,
- at least 3 independent aligned source groups,
- recent realized edge not materially negative.

### `meaningful_entry`

High-quality opportunity that may solve the "forever small order" problem.

Initial size:

- 5%-8%

Requirements:

- strong expected net return,
- strong profit quality,
- controlled loss probability,
- controlled tail loss,
- at least 4 independent aligned source groups,
- clean recent realized/shadow support,
- complete exit plan.

### `high_conviction`

Rare strongest opportunity.

Initial size:

- 8%-12%

Requirements:

- all `meaningful_entry` requirements,
- independent high-risk review approval,
- positive recent realized edge for same lane/regime,
- portfolio drawdown and side concentration permit it,
- Phase 3/go-no-go/model gates healthy.

Default:

- disabled until paper observation proves positive realized net PnL.

## 6. Unified Profit-First Score

Add a master score:

```text
profit_first_score
```

It should combine:

- fee/slippage-adjusted expected net return,
- expected profit in USDT,
- profit quality ratio,
- reward/risk ratio,
- loss probability,
- tail loss,
- independent source alignment,
- recent realized edge,
- same symbol/side edge,
- portfolio side pressure,
- execution cost,
- model reliability,
- exit plan quality.

This score maps to lanes and is persisted. Lower-level gates can still protect hard risk, but they should not create unexplained behavior outside the plan.

## 7. Entry And Exit Binding

Every real entry must create and persist an exit plan at the same time.

Exit plan must define:

- stop loss,
- invalidation thesis,
- max hold time,
- take profit,
- trailing/profit drawdown rule,
- partial close conditions,
- full close conditions,
- loss repair probability,
- do-not-close conditions,
- replacement-opportunity requirement for capital rotation.

Rules:

- Opening without an exit plan is forbidden.
- Exit decisions must reference `exit_plan_id`.
- If an exit happens outside the original plan, the system must record why the original plan failed.
- Profit exits, risk exits, and capital-rotation exits must be attributed separately.

## 8. No-Entry Governance

Every no-entry must be classified.

Canonical categories:

- `profit_insufficient`
- `evidence_insufficient`
- `risk_gate_blocked`
- `model_disagreement`
- `budget_insufficient`
- `position_capacity_occupied`
- `same_side_crowded`
- `okx_unavailable_or_rejected`
- `market_data_incomplete`
- `phase3_model_unavailable`
- `shadow_only_missing_plan_fields`
- `recent_realized_edge_negative`

Daily report must answer:

- Is the market genuinely unattractive?
- Or is the system over-conservative?
- Which blocker caused the most missed profitable shadow outcomes?
- Which missing fields most often forced shadow?
- Which symbols/sides should be reviewed for threshold adjustment?

## 9. Losing-Exit Governance

Every losing close must be attributed.

Canonical categories:

- `entry_wrong_direction`
- `entry_late`
- `stop_too_tight`
- `position_too_small_fee_drag`
- `hold_too_short`
- `trend_reversal`
- `model_false_positive`
- `server_profit_overestimated`
- `timeseries_false_signal`
- `sentiment_false_signal`
- `okx_slippage_or_execution`
- `exit_too_early`
- `exit_too_late`
- `capital_release_forced_loss`
- `unknown_requires_review`

Attribution must feed the next cycle:

- repeated `model_false_positive` demotes model source;
- repeated `position_too_small_fee_drag` stops tiny probes in that regime;
- repeated `exit_too_early` tightens capital-release exits;
- repeated `stop_too_tight` adjusts exit plan generation;
- tail loss forces shadow/review.

## 10. Model And Strategy Ranking

Create realized-PnL leaderboards by:

- model route,
- strategy profile,
- symbol,
- side,
- market regime,
- decision lane,
- exit intent,
- holding-time bucket.

Metrics:

- realized net PnL,
- win rate,
- profit factor,
- average win,
- average loss,
- max adverse excursion,
- max favorable excursion,
- fast-loss rate,
- tail-loss rate,
- fee drag ratio,
- opportunity miss rate.

Promotion:

- only after clean realized/shadow evidence;
- only through shadow -> canary -> live;
- only when OKX/account/training/model gates are healthy.

Demotion:

- consecutive losses,
- profit factor below threshold,
- tail loss,
- false signal loss,
- repeated fee-drag churn.

## 11. New Model Server Integration Policy

Expected new-server routes:

- `qwen3-32b-trade`
- `phase3_quant_api`
- `deepseek-r1-14b-risk`
- `BB-FinQuant-Expert-14B`

The master control must verify:

- model output is normalized into `ProfitFirstTradePlan`,
- required fields are present,
- route identity is correct,
- latency and health are acceptable,
- model source is calibrated against realized outcomes,
- same-provider role views are not treated as independent votes.

Model output can recommend but not force:

- side,
- expected return,
- loss probability,
- exit plan,
- risk review,
- size tier.

The final action is decided by the Profit-First master control.

## 12. Model Training, Coordination, And Brain Training

Model training, model cooperation, and the intelligent trading brain are part of this plan. They are not separate side projects.

The rule is:

```text
Models may learn and recommend, but Profit-First master control decides when their evidence is trusted, sized, promoted, demoted, or kept shadow-only.
```

### 12.1 Per-Model Training

Each model family must train or calibrate against clean Phase 3 facts only.

Included model groups:

- decision LLM route, such as `qwen3-32b-trade`;
- high-risk review route, such as `deepseek-r1-14b-risk`;
- quant API models for profit prediction, time-series prediction, sentiment, and exit advice;
- local ML signal model;
- specialist/expert route, such as `BB-FinQuant-Expert-14B`;
- strategy-learning profiles and symbol/side performance memories.

Training objectives:

- predict fee/slippage-adjusted net return;
- predict loss probability;
- predict tail loss;
- predict expected holding time;
- predict exit timing and profit drawdown risk;
- classify losing-exit root cause;
- classify no-entry root cause;
- estimate model/source reliability by regime.

Training gates:

- clean OKX-confirmed Phase 3 facts only;
- no legacy dirty samples in live influence;
- shadow-first artifacts;
- canary only after sample floors and walk-forward checks;
- live influence only after realized net-PnL evidence passes promotion gates.

### 12.2 Model Cooperation

Models must cooperate through the `ProfitFirstTradePlan`, not by directly competing to place orders.

Each source should fill or validate specific fields:

- decision LLM: thesis, direction, scenario reasoning, candidate exit thesis;
- quant profit model: expected net return, loss probability, profit quality;
- time-series model: expected path, trend continuation, adverse-move risk;
- sentiment model: event/news shock direction and confidence;
- high-risk review: veto, downgrade, or approve high-conviction risk;
- local ML: calibrated historical edge and regime fit;
- strategy memory: recent realized edge by symbol, side, lane, and regime.

The coordinator must record:

- which fields each model provided;
- whether the field was complete and valid;
- whether sources were independent;
- conflicts and conflict strength;
- which model was overridden and why;
- later realized outcome by model contribution.

### 12.3 Intelligent Brain Training

The intelligent brain is the meta-controller that learns how to use models and strategies, not a single model that blindly overrides the system.

It must learn:

- which model to trust by market regime;
- when to keep a candidate shadow-only;
- when to promote from tiny probe to validated or meaningful size;
- when to demote a model/strategy after losses;
- when no-entry is correct versus over-conservative;
- when an exit was too early, too late, or correct;
- how to allocate capital across symbols, sides, and strategies.

Brain training inputs:

- `ProfitFirstTradePlan` fields;
- realized net PnL;
- max adverse/favorable excursion;
- entry lane;
- exit attribution;
- no-entry attribution;
- model contribution records;
- OKX slippage/execution facts;
- strategy profile history.

Brain training outputs:

- source weights;
- strategy profile weights;
- lane thresholds;
- size promotion/demotion recommendations;
- no-entry threshold recommendations;
- exit-policy adjustment recommendations;
- shadow/canary/live promotion decisions.

Hard boundary:

- The intelligent brain can recommend changes, but any live behavior change must pass the same Profit-First gates, tests, shadow/canary observation, and rollback discipline.

### 12.4 Completion Criteria For Model And Brain Layer

This layer is complete only when:

- model outputs are normalized into `ProfitFirstTradePlan`;
- each model contribution can be tied to realized net PnL;
- unreliable models are automatically demoted or kept shadow-only;
- profitable model/strategy combinations can earn more budget after clean evidence;
- the brain can explain why it trusted, ignored, promoted, or demoted each model source.

## 13. Implementation Sequence

### Stage 0 - Plan Consolidation And Guard Rails

Goal:

- Prevent plan drift.

Actions:

- Treat this document as the only execution entrypoint.
- Mark the two 2026-06-29 sub-plans as superseded.
- Keep the 2026-06-26 Phase 3 plan as historical infrastructure and implementation log.

Acceptance:

- Future work references this file in task descriptions and PR notes.

### Stage 1 - Read-Only Profit Diagnosis

Goal:

- Build the unified diagnosis layer before changing trade behavior.

Deliverables:

- `ProfitFirstTradePlan` schema/service.
- Lane classifier.
- No-entry reason normalizer.
- Losing-exit attribution normalizer.
- Probe-loop health report.
- Model/strategy realized-PnL report skeleton.

Rules:

- No order behavior changes in this stage.
- Unknown categories must be visible and reduced.

Acceptance:

- Recent entries/non-entries can be rendered into the canonical plan shape.
- Recent closes have attribution or `unknown_requires_review`.
- Required field gaps are measurable.

### Stage 2 - Probe-Loss Brake And Net-Benefit Release

Goal:

- Stop the current losing probe loop.

Deliverables:

- Recent probe PnL brake.
- Stale-probe release net-benefit policy.
- Exit-intent profit factor tracking.

Behavior changes:

- If recent tiny/probe closes are all losing, new tiny probes go shadow unless upgraded to validated or better.
- Stale probes cannot be closed at repeated small losses without hard risk, true capacity pressure, or stronger replacement opportunity.

Acceptance:

- No repeated 8h all-loss probe close window.
- Fast-loss closes require strong evidence.
- Release decisions include net-benefit evidence.

### Stage 3 - Dynamic Position Ladder

Goal:

- Solve "always small orders" safely.

Deliverables:

- Lane-based sizing.
- Promotion/de-promotion state.
- Max stop-loss and tail-loss integrated with lane sizing.

Behavior changes:

- `validated_probe` can use 3%-5%.
- `meaningful_entry` can use 5%-8%.
- `high_conviction` remains disabled until observation gates pass.

Acceptance:

- No low-payoff trade gets meaningful size.
- Meaningful entries show promotion reasons.
- Average notional rises only for validated/high-quality lanes.

### Stage 4 - Entry-Exit Binding

Goal:

- Every entry carries its exit thesis.

Deliverables:

- `ProfitFirstExitPlan`.
- Exit-plan persistence.
- Exit decisions reference original plan.
- Plan-failure attribution.

Acceptance:

- Every new position has `exit_plan_id`.
- Every close references the plan or a plan-failure reason.
- Losing closes become attributable.

### Stage 5 - Model Training, Coordination, And Brain Training

Goal:

- Make model training, model cooperation, and intelligent brain training operate inside the Profit-First loop instead of as a separate side project.

Deliverables:

- Per-model training objectives mapped to `ProfitFirstTradePlan` fields.
- Model contribution records for each plan field.
- Model cooperation contract showing which source provided or validated each field.
- Clean Phase 3 training gates for each model family.
- Brain-training dataset built from plan fields, realized PnL, no-entry attribution, losing-exit attribution, and model contribution outcomes.
- Brain outputs for source weights, strategy weights, lane thresholds, size promotion/demotion recommendations, no-entry threshold recommendations, and exit-policy adjustment recommendations.

Behavior changes:

- Models can recommend but cannot bypass Profit-First gates.
- Missing model fields force shadow for fields that are required for real trading.
- Unreliable models are demoted or kept shadow-only.
- Profitable model/strategy combinations can earn more budget only after clean evidence.
- The intelligent brain can recommend live behavior changes only through shadow/canary/live promotion and rollback discipline.

Acceptance:

- Model outputs are normalized into `ProfitFirstTradePlan`.
- Each model contribution can be tied to realized net PnL.
- The coordinator records which model supplied which field, whether it was valid, and whether sources were independent.
- The brain can explain why it trusted, ignored, promoted, or demoted each model source.
- No trained model or brain recommendation can influence live size unless clean Phase 3 facts, sample floors, walk-forward checks, and promotion gates pass.

### Stage 6 - Strategy And Model Ranking

Goal:

- Make strategy learning act on realized net profit.

Deliverables:

- Realized-PnL leaderboard.
- Model/source contribution ranking.
- Strategy profile promotion/demotion.
- Regime-specific performance memory.

Acceptance:

- Losing profiles cannot keep live size.
- Profitable profiles earn more budget only after clean samples.
- Ranking changes are auditable.

### Stage 7 - Controlled Paper Resume / Canary Readiness

Goal:

- Resume or continue auto trading only after the new control loop is active and foundations are healthy.

Required gates:

- OKX authoritative facts healthy.
- Account equity truth from OKX.
- Clean training view available.
- New model server endpoints healthy.
- Quant API healthy.
- Go/no-go non-critical or explained non-blocking warnings.
- `ProfitFirstTradePlan` active.
- Exit binding active.
- Probe-loss brake active.

Observation:

- 2h: no contract violations, no missing plan fields for real entries.
- 8h: no repeated all-loss probe loop.
- 24h: improving or positive profit factor.
- 72h: stable lane distribution and no uncontrolled size escalation.

## 14. Concrete First Coding Slice

The first slice must be read-only.

Implement:

- `services/profit_first_trade_plan.py`
- `ProfitFirstTradePlan` dataclass or typed dict.
- builder from existing `DecisionOutput` and raw response.
- lane classifier.
- missing-field detector.
- no-entry reason normalizer.
- losing-exit attribution normalizer.
- probe-loop health summary.

Tests:

- `tests/test_profit_first_trade_plan.py`
- `tests/test_profit_first_lane_classifier.py`
- `tests/test_no_entry_reason_taxonomy.py`
- `tests/test_losing_exit_attribution.py`

Do not change:

- leverage,
- max position size,
- live order execution,
- model routing,
- high-risk review behavior.

## 15. Definition Of Done

Profit-First v3 is not done when the system simply opens more trades.

It is done when:

- every real order has a complete profit-first plan;
- every no-entry has a canonical reason;
- every losing close has attribution;
- small probes stop after recent all-loss evidence;
- strong opportunities can graduate beyond tiny size through audited lanes;
- exits follow or explain deviation from the original plan;
- model and strategy weights move according to realized net PnL;
- OKX account/fact/training/model gates protect promotion;
- rolling paper/canary windows show positive or improving realized net PnL after fees.

Online validation rule:

- A stage can be called locally implemented only after the local tests listed for that stage pass.
- A stage can be called completed only after the same stage is synced or verified on the online server through a read-only validation pass.
- The online validation pass must confirm `read_only=true`, `audit_only=true`, `starts_trading_service=false`, `submits_orders=false`, `changes_model_routing=false`, `changes_live_sizing=false`, and `live_mutation=false`.
- If online new market analysis or new entries are intentionally paused, the validation must record the control-state pause explicitly and must not treat the pause as proof that entry logic is healthy.
- No paper/canary/live resume is allowed until the read-only online validation says resume is ready and the operator explicitly approves recovery.
- Use selective sync for staged validation when the local worktree is dirty:

```powershell
rtk python scripts/sync_to_online_server.py --skip-restart --include-tests --only services/profit_first_trade_plan.py --only scripts/verify_profit_first_online_readiness.py
```

Then run:

```powershell
rtk python scripts/verify_profit_first_online_readiness.py --json-indent 2
```

## 16. Final Operating Rule

When future work is ambiguous, choose the change that improves this loop:

```text
complete plan -> expected net edge -> controlled size -> bound exit plan ->
realized outcome -> attribution -> model/strategy promotion or demotion
```

If a task does not improve that loop, it is not part of the next optimization phase.

## 17. 2026-06-29 Implementation Status

Current implementation status:

- Stage 1 locally implemented: canonical `ProfitFirstTradePlan`, no-entry taxonomy, losing-exit attribution, probe-loop diagnosis, and realized-PnL leaderboard skeleton are implemented and covered by tests.
- Stage 2 locally implemented: recent probe-loss brake and net-benefit release guard are implemented. Tiny/probe all-loss windows force new tiny probes back to shadow unless the opportunity upgrades.
- Stage 3 locally implemented: dynamic position ladder is implemented with `tiny_probe`, `validated_probe`, `meaningful_entry`, and disabled-by-default `high_conviction`. Low-payoff candidates cannot receive meaningful size.
- Stage 4 locally implemented: entry decisions now attach exit plans and entry-exit binding metadata. Exit decisions must reference the original exit plan or record why the plan failed.
- Stage 5 locally implemented at the audit/control-contract layer: model contributions from the new model-server/quant/local sources can be tied back to realized net PnL; brain-training datasets and recommendations are read-only and remain shadow/canary/live governed. The brain output is now explicit and test-covered: `source_weights`, `strategy_weights`, `lane_threshold_recommendations`, `size_promotion_demotion`, `no_entry_threshold_recommendations`, `exit_policy_adjustments`, and `shadow_canary_live_decisions` must all be present before recovery gates can pass.
- Stage 6 locally implemented at the read-only master-control layer: `services/profit_first_ranking.py` ranks model, strategy, symbol, side, and lane combinations by realized net PnL, profit factor, consecutive losses, tail loss, fast-loss rate, and fee drag. The ranking input is now restricted to OKX-confirmed closed-position facts; untrusted positions are quarantined with reason counts and cannot drive ranking, training, promotion, or budget increases.
- Stage 6.5 locally implemented as the 24h governance report layer: `services/profit_first_governance_report.py` and `scripts/run_profit_first_governance_report.py` produce a read-only `profit_first_governance` report for the current no-entry and losing-exit loop. The report summarizes no-entry diagnosis, losing-exit attribution, missing brain outputs, next-cycle actions, and the strict no-live-mutation safety boundary. `scripts/install_profit_first_governance_timer.py` installs the online systemd timer for the report without starting paper trading or submitting orders. Current local execution proves the script stays read-only and reports `unavailable` when the DB/remote source is unreachable instead of silently allowing recovery.
- Stage 7 locally implemented at the gate layer, not as an automatic resume: system audit and Phase 3 go/no-go now require `ProfitFirstTradePlan`, position ladder, exit binding, active probe-loss brake contract, OKX account-equity truth, Profit-First ranking readiness, complete Stage 5 brain-output coverage, and `profit_first_governance` readiness before paper/canary resume can be considered. Any executed entry that bypasses `profit_first_probe_loss_brake` is a hard trade-contract violation.
- Stage 7 failure semantics locally implemented: if the trade-contract or ranking audit itself fails, the card now exposes `report_available=false` plus conservative read-only policy, and Phase 3 go/no-go blocks on `profit_first_trade_contract_unavailable` or `profit_first_ranking_unavailable` instead of misclassifying the issue as an ordinary policy or sample-floor state.
- Stage 7 governance failure semantics locally implemented: if `profit_first_governance` is missing, unavailable, not read-only, or missing required brain outputs, Phase 3 go/no-go blocks on a Profit-First governance code before any resume. Empty recent no-entry/loss-exit samples are warnings, not permission to bypass the governance layer.

Online validation status as of 2026-06-29T11:06:21Z:

- Full rolling paper/canary performance observation has not yet been completed. The current status is "technical hard blockers cleared, operator pause still active", not automatic resume approval.
- Read-only online validation command: `rtk python scripts/verify_profit_first_online_readiness.py --json-indent 2`.
- Current runtime control state: `bb-paper-trading.service=active`, `bb-dashboard.service=active`, `bb-model-tunnels.service=active`, `data/trading-control-state.json` has `paused=true`, mode is `paper`, and `live_model_name=null`.
- This means the service process is still up for existing-position review and dashboard/API service, but new market analysis and new entries remain intentionally paused.
- `scripts/verify_profit_first_online_readiness.py` is the canonical online read-only validation command for this plan. It runs governance and go/no-go checks without starting trading, submitting orders, changing model routing, or changing sizing.
- First online selective sync used `scripts/sync_to_online_server.py --skip-restart --include-tests --only ...` and uploaded only validation/governance/Profit-First files. It did not restart `bb-paper-trading.service`.
- Latest readiness result is read-only/audit-only: `read_only=true`, `audit_only=true`, `starts_trading_service=false`, `submits_orders=false`, `changes_model_routing=false`, `changes_live_sizing=false`, and `live_mutation=false`.
- Online governance is available and read-only: `profit_first_governance.status=ready`, `read_only=true`, `can_start_trading_service=false`, `can_submit_orders=false`, `missing_brain_outputs=[]`.
- Current online go/no-go hard blockers are clear: `go_no_go.status=post_resume_observing`, `go_no_go.blocker_codes=[]`, and `go_no_go.critical_blocker_count=0`.
- Current recovery repair plan is clear: `recovery_repair_plan.status=clear`, `blocking_actions=[]`, `operator_approval_required_count=0`.
- Current historical recovery package is empty: `historical_recovery_package.status=empty`, `items=[]`, `proposed_raw_patch_count=0`, and `operator_approval_required_count=0`.
- `resume_allowed_by_this_check=false` remains expected because the operator pause switch is still on. This is a resume boundary, not a remaining technical blocker.
- The latest persisted go/no-go cache may still show an older `blocked` snapshot; the current read-only validation above is the active readiness snapshot for this implementation handoff.
- Historical trade-contract blockers were resolved by an operator-approved allowlisted recovery apply on 2026-06-29. The apply only patched `ai_decisions.raw_llm_response` for the exact approved legacy decisions, marked the recovered samples as excluded from trusted training until manual trust, did not repair OKX facts, did not touch orders/positions/ranking/sizing/model routing, and did not start or restart trading services.
- Historical recovered/quarantined decision ids include `11462`, `11484`, `11486`, `11456`, `11444`, `11441`, `11435`, `11430`, and `11428`. They are `exclude_until_manual_trust`, not clean training or live-promotion samples.
- The remaining non-technical hold is the operator pause switch: `data/trading-control-state.json` still has `paused=true`, so new market analysis and new entries remain intentionally stopped until an explicit resume action flips the pause state.
- The former `LAB/USDT` order `2678` / exchange order `3697115296869093376` quantity issue was cleared online after `okx_authoritative_sync` was changed to prefer current OKX instrument contract size or confirmed `orders.okx_raw_fills` cache over stale legacy decision payload fields.
- The former ranking disable blocker was cleared online after single-sample tail-loss rows were changed from hard `disable` to `demote` / no-budget-increase. Consecutive-loss profiles and multi-sample tail-loss profiles still hard-disable. Demotions remain visible as warnings and cannot receive budget increases.
- The known add-entry/replenishment duplicate-position bug has been addressed at all current display/persistence boundaries:
  - open-position execution application merges same model/symbol/side add-entry fragments;
  - DB position persistence merges same model/symbol/side open rows instead of inserting duplicate current rows;
  - dashboard open-position snapshots group duplicate local same symbol/side fragments;
  - closed-position history now groups add-entry fragments closed by the same OKX close order into one OKX-style lifecycle row.
- The BNB duplicate current-position case was repaired and later confirmed truly closed by OKX close order `3698624752522076160` at `2026-06-29T10:46:47Z`; no current BNB open row should remain. Related linked entry ids were `3693875684642099200`, `3698357883789615104`, `3698366268773736448`, and `3698374742710657024`.
- OKX authoritative sync after the BNB repair reported clean state (`status=ok`, `issue_count=0`) in the online repair/validation sequence.
- `services/okx_order_fact_sync.py` now preserves/merges existing `entry_exchange_order_id` links when OKX current-position snapshots update local rows, and chooses the canonical current row among duplicates by entry-link presence and quantity.
- On 2026-06-29, `bb-paper-trading.service` was restarted once only to load the OKX sync fix while `paused=true`; the pause remained true afterward. A later dashboard-only restart loaded the historical grouped-ledger display fix and did not restart/resume the paper trading service.
- `services/profit_first_recovery_blockers.py` is the recovery-blocker diagnosis layer. It does not repair history or mutate strategy state; it turns trade-contract, ranking, and OKX blockers into an operator cleanup checklist before resume.
- `services/profit_first_recovery_repair_plan.py` and `scripts/plan_profit_first_recovery_repairs.py` add the recovery dry-run planner. It converts the recovery-blocker checklist into explicit operator-reviewed actions: historical ProfitFirstTradePlan backfill-or-quarantine, position-ladder backfill-or-quarantine, exit-reference repair or legacy failure marker, ranking shadow/disable review, and OKX exact-order quantity repair review. The planner is read-only, dry-run, does not write database history, does not mutate routing/sizing, does not start services, and never treats its own output as resume approval.
- Latest online validation for this recovery-planner step completed at `2026-06-29T11:06:21Z` through `scripts/verify_profit_first_online_readiness.py`. The remote readiness report included `recovery_repair_plan.status=clear`, `dry_run=true`, `read_only=true`, `mutates_database=false`, `starts_trading_service=false`, `submits_orders=false`, `changes_model_routing=false`, `changes_live_sizing=false`, `live_mutation=false`, and `blocking_actions=[]`.
- The remote recovery dry-run currently reports 41 ranking demotion observation actions and 0 resume-blocking approval-required actions. This is a verified online blocker-clear map, not permission to bypass the operator pause switch.
- `services/profit_first_historical_recovery_package.py` and `scripts/plan_profit_first_historical_recovery_package.py` add the next dry-run layer: an operator approval package for the exact recovery targets. It reads current blockers, loads the affected decisions/orders, proposes raw JSON patches for historical entry/exit decision records, marks all proposed historical fixes as `exclude_until_manual_trust`, and leaves OKX quantity differences as exact-order review items. It does not apply the patches.
- Latest online validation for the historical recovery package after the approved apply, OKX cleanup, ranking cleanup, and grouped-ledger display fix completed at `2026-06-29T11:06:21Z`. The package now returns `status=empty`, `dry_run=true`, `read_only=true`, `mutates_database=false`, `starts_trading_service=false`, `submits_orders=false`, `changes_model_routing=false`, `changes_live_sizing=false`, `live_mutation=false`, `items=[]`, and `proposed_raw_patch_count=0`.
- Each remaining stage update must record both local validation and online read-only validation before the stage can be renamed from locally implemented to completed.

Strict resume boundary:

- Online new market analysis and new entries remain paused.
- This implementation does not start `bb-paper-trading.service`.
- This implementation does not submit orders.
- This implementation does not change live model routing, live model weights, live strategy weights, leverage, or live sizing.
- Paper/canary resume is allowed only after OKX facts, equity, clean training, model-server health, quant API health, trade contract, ranking gate, and go/no-go all pass, then the operator explicitly approves.

Added/updated implementation artifacts:

- `services/profit_first_trade_plan.py`
- `services/profit_first_stage2.py`
- `services/profit_first_position_ladder.py`
- `services/profit_first_exit_binding.py`
- `services/profit_first_brain_training.py`
- `services/profit_first_governance_report.py`
- `services/profit_first_recovery_blockers.py`
- `services/profit_first_recovery_repair_plan.py`
- `services/profit_first_ranking.py`
- `services/model_contribution_performance.py`
- `services/entry_profit_risk_sizing.py`
- `services/trading_policies.py`
- `services/position_release_decision.py`
- `services/open_positions_execution_applier.py`
- `services/okx_order_fact_sync.py`
- `services/okx_position_ledger_view.py`
- `services/position_execution_persistence.py`
- `services/trade_execution_contract.py`
- `services/phase3_paper_resume_preflight.py`
- `services/phase3_go_no_go.py`
- `scripts/run_profit_first_governance_report.py`
- `scripts/plan_profit_first_recovery_repairs.py`
- `scripts/plan_profit_first_historical_recovery_package.py`
- `scripts/verify_profit_first_online_readiness.py`
- `scripts/install_profit_first_governance_timer.py`
- `web_dashboard/api/system_audit.py`
- `web_dashboard/api/dashboard.py`
- `web_dashboard/api/trades.py`

Validation completed locally:

- `rtk pytest tests/test_profit_first_ranking.py tests/test_phase3_go_no_go.py tests/test_system_audit_api.py -q`
- `rtk pytest tests/test_profit_first_trade_plan.py tests/test_profit_first_stage2.py tests/test_profit_first_position_ladder.py tests/test_profit_first_exit_binding.py tests/test_profit_first_brain_training.py tests/test_model_contribution_performance.py tests/test_trade_execution_contract.py tests/test_trading_service_boundaries.py tests/test_position_release_decision.py tests/test_open_positions_execution_applier.py -q`
- `rtk python -m py_compile services/profit_first_ranking.py services/phase3_go_no_go.py web_dashboard/api/system_audit.py tests/test_phase3_go_no_go.py tests/test_system_audit_api.py`
- `rtk pytest tests/test_profit_first_ranking.py tests/test_trade_execution_contract.py tests/test_phase3_go_no_go.py tests/test_phase3_paper_resume_preflight.py tests/test_system_audit_api.py -q`
- `rtk pytest tests/test_profit_first_ranking.py tests/test_phase3_go_no_go.py tests/test_system_audit_api.py tests/test_profit_first_trade_plan.py tests/test_profit_first_stage2.py tests/test_profit_first_position_ladder.py tests/test_profit_first_exit_binding.py tests/test_profit_first_brain_training.py tests/test_model_contribution_performance.py tests/test_trade_execution_contract.py tests/test_trading_service_boundaries.py tests/test_position_release_decision.py tests/test_open_positions_execution_applier.py tests/test_phase3_paper_resume_preflight.py tests/test_equity_baseline.py tests/test_run_phase3_okx_fact_sync.py tests/test_trade_fact_trust.py -q`
- `rtk python -m py_compile services/profit_first_ranking.py services/trade_execution_contract.py services/phase3_go_no_go.py services/phase3_paper_resume_preflight.py web_dashboard/api/system_audit.py`
- `rtk pytest tests/test_phase3_go_no_go.py tests/test_system_audit_api.py -q`
- `rtk pytest tests/test_profit_first_recovery_repair_plan.py tests/test_profit_first_recovery_repair_plan_script.py -q`
- `rtk pytest tests/test_profit_first_online_readiness.py tests/test_profit_first_recovery_repair_plan.py tests/test_profit_first_recovery_repair_plan_script.py -q`
- `rtk pytest tests/test_profit_first_recovery_blockers.py tests/test_phase3_go_no_go.py tests/test_profit_first_online_readiness.py tests/test_system_audit_api.py tests/test_model_server_maintenance_scripts.py tests/test_profit_first_recovery_repair_plan.py tests/test_profit_first_recovery_repair_plan_script.py -q`
- `rtk python -m py_compile services/profit_first_recovery_repair_plan.py scripts/plan_profit_first_recovery_repairs.py scripts/verify_profit_first_online_readiness.py`
- `rtk pytest tests/test_profit_first_historical_recovery_package.py tests/test_profit_first_historical_recovery_package_script.py -q`
- `rtk pytest tests/test_profit_first_online_readiness.py tests/test_profit_first_historical_recovery_package.py tests/test_profit_first_historical_recovery_package_script.py tests/test_profit_first_recovery_repair_plan.py -q`
- `rtk python -m py_compile services/profit_first_historical_recovery_package.py scripts/plan_profit_first_historical_recovery_package.py scripts/verify_profit_first_online_readiness.py`
- `rtk pytest tests/test_trade_history_api.py -q` -> `13 passed`.
- `rtk python -m py_compile services/okx_position_ledger_view.py web_dashboard/api/trades.py tests/test_trade_history_api.py`
- `rtk pytest tests/test_okx_order_fact_sync.py tests/test_okx_authoritative_sync.py tests/test_trade_execution_contract.py tests/test_profit_first_recovery_blockers.py tests/test_phase3_go_no_go.py tests/test_system_audit_api.py::test_trade_execution_contract_audit_and_endpoint_force_read_only tests/test_system_audit_api.py::test_trade_execution_contract_audit_treats_historical_only_as_observing tests/test_system_audit_api.py::test_trade_contract_violation_counts_use_unresolved_profit_first_counts tests/test_open_positions_execution_applier.py tests/test_position_execution_persistence.py tests/test_dashboard_error_safety.py::test_display_open_positions_snapshot_groups_same_symbol_side_fragments tests/test_trade_history_api.py -q` -> `108 passed`.

Validation completed online:

- `rtk python scripts/sync_to_online_server.py --skip-restart --include-tests --only services/profit_first_recovery_repair_plan.py --only scripts/plan_profit_first_recovery_repairs.py --only tests/test_profit_first_recovery_repair_plan.py --only tests/test_profit_first_recovery_repair_plan_script.py --only docs/superpowers/plans/2026-06-29-profit-first-v3-authoritative-master-plan.md`
- `rtk python scripts/sync_to_online_server.py --skip-restart --include-tests --only scripts/verify_profit_first_online_readiness.py --only tests/test_profit_first_online_readiness.py`
- `rtk python scripts/sync_to_online_server.py --skip-restart --include-tests --only services/profit_first_historical_recovery_package.py --only scripts/plan_profit_first_historical_recovery_package.py --only scripts/verify_profit_first_online_readiness.py --only tests/test_profit_first_historical_recovery_package.py --only tests/test_profit_first_historical_recovery_package_script.py --only tests/test_profit_first_online_readiness.py`
- `rtk python scripts/verify_profit_first_online_readiness.py --json-indent 2`
- Online result: `control_state.paused=true`, `governance.status=ready`, `go_no_go.status=blocked`, `recovery_repair_plan.status=blocked`, `recovery_repair_plan.summary.action_count=46`, `blocking_action_count=5`, `operator_approval_required_count=5`, `historical_recovery_package.status=ready`, `historical_recovery_package.summary.item_count=9`, `proposed_raw_patch_count=8`, and `resume_allowed_by_this_check=false`.
- `rtk pytest tests/test_profit_first_ranking.py tests/test_trade_execution_contract.py tests/test_phase3_go_no_go.py tests/test_phase3_paper_resume_preflight.py tests/test_system_audit_api.py tests/test_profit_first_trade_plan.py tests/test_profit_first_stage2.py tests/test_profit_first_position_ladder.py tests/test_profit_first_exit_binding.py tests/test_profit_first_brain_training.py tests/test_model_contribution_performance.py tests/test_trading_service_boundaries.py tests/test_position_release_decision.py tests/test_open_positions_execution_applier.py tests/test_equity_baseline.py tests/test_run_phase3_okx_fact_sync.py tests/test_trade_fact_trust.py -q`
- `rtk pytest tests/test_phase3_go_no_go.py tests/test_system_audit_api.py tests/test_profit_first_brain_training.py tests/test_profit_first_ranking.py tests/test_profit_first_trade_plan.py -q`
- `rtk python -m py_compile services/phase3_go_no_go.py services/profit_first_brain_training.py services/profit_first_trade_plan.py services/profit_first_ranking.py web_dashboard/api/system_audit.py tests/test_phase3_go_no_go.py tests/test_system_audit_api.py tests/test_profit_first_brain_training.py`
- `rtk pytest tests/test_profit_first_governance_report.py tests/test_profit_first_governance_report_script.py tests/test_phase3_go_no_go.py tests/test_system_audit_api.py -q`
- `rtk pytest tests/test_profit_first_governance_report.py tests/test_profit_first_governance_report_script.py tests/test_profit_first_ranking.py tests/test_trade_execution_contract.py tests/test_phase3_go_no_go.py tests/test_phase3_paper_resume_preflight.py tests/test_system_audit_api.py tests/test_profit_first_trade_plan.py tests/test_profit_first_stage2.py tests/test_profit_first_position_ladder.py tests/test_profit_first_exit_binding.py tests/test_profit_first_brain_training.py tests/test_model_contribution_performance.py tests/test_trading_service_boundaries.py tests/test_position_release_decision.py tests/test_open_positions_execution_applier.py tests/test_equity_baseline.py tests/test_run_phase3_okx_fact_sync.py tests/test_trade_fact_trust.py -q`
- `rtk python scripts/run_profit_first_governance_report.py --stdout-only --json-indent 0`
- `rtk pytest tests/test_profit_first_governance_timer.py tests/test_profit_first_governance_report_script.py tests/test_profit_first_governance_report.py -q`
- `rtk python scripts/sync_to_online_server.py --skip-restart --include-tests --only services/okx_position_ledger_view.py --only tests/test_trade_history_api.py --only docs/superpowers/plans/2026-06-29-profit-first-v3-authoritative-master-plan.md`
- Online dashboard-only load validation: remote `py_compile` passed for `services/okx_position_ledger_view.py`, `web_dashboard/api/trades.py`, and `tests/test_trade_history_api.py`; `bb-dashboard.service` was restarted and returned `active`; `data/trading-control-state.json` remained `paused=true`; `bb-paper-trading.service` remained `active` but was not restarted by this display fix.
- Online positions display spot-check used the dashboard service environment in read-only mode. Result: `open_count=2`, open rows were `ETHW/USDT short` and `TRX/USDT short`, and `open_symbol_side_duplicates={}`. This validates that the current open-position display no longer shows duplicate same-symbol/side rows after the add-entry merge fixes.
- `rtk python scripts/verify_profit_first_online_readiness.py --json-indent 2`
- Latest online result: `checked_at=2026-06-29T11:06:21Z`, `control_state.paused=true`, `governance.status=ready`, `go_no_go.status=post_resume_observing`, `go_no_go.blocker_codes=[]`, `critical_blocker_count=0`, `recovery_repair_plan.status=clear`, `recovery_repair_plan.blocking_actions=[]`, `historical_recovery_package.status=empty`, `read_only=true`, `audit_only=true`, `starts_trading_service=false`, `submits_orders=false`, `changes_model_routing=false`, `changes_live_sizing=false`, `live_mutation=false`, and `resume_allowed_by_this_check=false` because the operator pause remains enabled.
