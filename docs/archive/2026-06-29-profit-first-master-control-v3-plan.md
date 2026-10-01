# Profit-First Master Control v3 Optimization Plan

> Status: superseded as an execution plan by `2026-06-29-profit-first-v3-authoritative-master-plan.md`.
> Keep this file as the expanded Profit-First v3 design draft. Do not use it as the primary implementation roadmap when it differs from the authoritative plan.

Date: 2026-06-29
Project: `F:\BB`
Premise: the new large-model server is already the target model foundation. This plan does not treat the problem as "deploy more models". It treats the problem as "make the trading master control consume model evidence correctly and optimize for realized net profit".

## 1. One-Sentence Direction

Upgrade the system from a safety-rule-heavy trading bot into a profit-first master control system where every entry, size increase, hold, exit, model weight, and strategy promotion is justified by expected and realized net profit after fees and slippage.

## 2. Why This Plan Replaces The Previous Fragmented Approach

The current symptoms are connected:

- Small entries exist because opportunity evidence often reaches only `exploration/small`, and sizing promotion rarely triggers.
- Losing exits exist because small probes are being released or stopped at realized loss, often before they prove useful.
- Long no-entry windows exist because the system has many independent gates, but no single profit-first decision contract explaining whether the market lacks opportunity or the system is over-conservative.
- No profitability exists because entry, sizing, exit, model ranking, and strategy learning are not governed by one realized-net-PnL control loop.

Therefore the next phase should not keep adding pages, warnings, or fallback branches. The system needs a single trading plan contract and a master scoring layer.

## 3. New Core Contract: `ProfitFirstTradePlan`

Every candidate must produce a complete `ProfitFirstTradePlan` before it can open or add to a real position.

Required fields:

- `symbol`
- `side`
- `decision_lane`: `shadow_only`, `tiny_probe`, `validated_probe`, `meaningful_entry`, `high_conviction`
- `expected_gross_return_pct`
- `expected_fee_pct`
- `expected_slippage_pct`
- `expected_net_return_pct`
- `expected_profit_usdt`
- `loss_probability`
- `expected_loss_usdt`
- `tail_loss_probability`
- `tail_loss_usdt`
- `profit_quality_ratio`
- `reward_risk_ratio`
- `expected_hold_minutes`
- `max_hold_minutes`
- `entry_price_reference`
- `invalidation_price`
- `stop_loss_pct`
- `take_profit_pct`
- `trailing_profit_trigger_pct`
- `profit_drawdown_exit_pct`
- `partial_exit_plan`
- `full_exit_plan`
- `position_size_pct`
- `leverage`
- `max_stop_loss_usdt`
- `model_sources`
- `independent_source_count`
- `strategy_profile_id`
- `recent_realized_edge`
- `same_symbol_side_edge`
- `portfolio_side_pressure`
- `block_or_downgrade_reasons`
- `promotion_reasons`
- `exit_plan_id`
- `plan_version`

Hard rule:

- If any required profit/risk/exit field is missing, the candidate is `shadow_only`.
- If `expected_net_return_pct <= 0`, the candidate is `shadow_only`.
- If `expected_profit_usdt` is too small to beat fees/slippage/noise, the candidate is `shadow_only` or `tiny_probe`, never meaningful size.
- If there is no exit plan, there is no entry.

Primary implementation targets:

- `services/profit_first_trade_plan.py`
- `services/entry_opportunity_scoring.py`
- `services/entry_profit_risk_sizing.py`
- `services/entry_opportunity_gate.py`
- `services/position_release_decision.py`
- `services/trading_service.py`

## 4. Decision Lanes

### 4.1 `shadow_only`

Use when:

- Required plan fields are missing.
- Expected net return is not positive.
- Model sources disagree strongly.
- Recent similar realized edge is negative.
- Phase 3 model/quant API evidence is unavailable or untrusted.
- Tail risk is too high.

Allowed actions:

- Log decision.
- Run shadow outcome tracking.
- Feed clean training view after outcome is known.

Not allowed:

- Real order.
- Position-size promotion.
- Strategy/profile promotion.

### 4.2 `tiny_probe`

Purpose:

- Controlled learning sample only.

Suggested size:

- 1%-2% position size.
- Low leverage, normally 2x-3x.

Required:

- Positive fee/slippage-adjusted expected net return.
- Loss probability below probe ceiling.
- Tail loss within small loss budget.
- No recent all-loss probe brake.
- At least 2 independent source groups aligned.

Hard limits:

- If recent probe closes are all losing, `tiny_probe` becomes `shadow_only` unless upgraded to `validated_probe`.

### 4.3 `validated_probe`

Purpose:

- A real but still cautious trade that can produce measurable PnL.

Suggested size:

- 3%-5% position size.
- Leverage normally 3x-5x.

Required:

- Positive expected net return after fee/slippage.
- Profit quality ratio above threshold.
- Loss probability below validated threshold.
- Tail loss within budget.
- At least 3 independent source groups aligned.
- No active same-side realized-PnL penalty, or strong enough quality override.

### 4.4 `meaningful_entry`

Purpose:

- Solve the "always tiny orders" problem without blind risk expansion.

Suggested size:

- 5%-8% position size.
- Leverage only if invalidation is clear and stop budget supports it.

Required:

- Strong expected net return.
- Strong profit quality.
- Loss probability clearly favorable.
- Tail loss controlled.
- At least 4 independent source groups aligned.
- Recent strategy/model/symbol-side realized edge not negative.
- Complete exit plan.

### 4.5 `high_conviction`

Purpose:

- Rare strongest opportunities after independent review.

Suggested size:

- 8%-12% position size.

Required:

- All `meaningful_entry` requirements.
- High-risk review approves.
- Recent realized edge for same lane/regime is positive.
- Portfolio drawdown and side concentration allow it.
- Phase 3 go/no-go and model promotion gates are healthy.

Default:

- Keep disabled until paper observation proves positive realized net PnL.

## 5. Unified Profit Score

Add `profit_first_score` as the master scalar score.

Suggested components:

- `expected_net_return_score`
- `expected_profit_usdt_score`
- `profit_quality_score`
- `loss_probability_score`
- `tail_loss_penalty`
- `source_alignment_score`
- `recent_realized_edge_score`
- `same_symbol_side_edge_score`
- `portfolio_concentration_penalty`
- `model_reliability_score`
- `execution_cost_penalty`
- `exit_plan_quality_score`

Example formula:

```text
profit_first_score =
  expected_net_return_score
  + expected_profit_usdt_score
  + profit_quality_score
  + source_alignment_score
  + recent_realized_edge_score
  + model_reliability_score
  + exit_plan_quality_score
  - loss_probability_penalty
  - tail_loss_penalty
  - execution_cost_penalty
  - portfolio_concentration_penalty
```

Decision mapping:

- `< 0`: `shadow_only`
- `0.0 - 0.35`: `shadow_only`
- `0.35 - 0.55`: `tiny_probe`
- `0.55 - 0.75`: `validated_probe`
- `0.75 - 0.90`: `meaningful_entry`
- `> 0.90`: `high_conviction`, subject to high-risk review

The exact thresholds should be configurable and validated by shadow replay.

## 6. New Position Sizing System

The old behavior makes too many trades economically meaningless. The new system should use a tiered size ladder.

Initial ladder:

- `shadow_only`: 0%
- `tiny_probe`: 1%-2%
- `validated_probe`: 3%-5%
- `meaningful_entry`: 5%-8%
- `high_conviction`: 8%-12%

Risk constraints:

- Max stop-loss USDT always caps size.
- Tail loss always caps size.
- Drawdown mode caps size.
- Same-side crowding caps size.
- Recent all-loss lane demotes size.
- Low payoff quality blocks notional floor.

Promotion rules:

- Promote only when plan quality and recent realized/shadow evidence both support it.
- A single model opinion cannot promote size alone.
- Same provider role views, such as multiple `BB-FinQuant-Expert-14B` roles, must not count as independent sources.

De-promotion rules:

- 2 consecutive losses in a lane/regime: demote one level.
- Profit factor below 1.0 over recent sample: demote one level.
- Fast-loss event: demote related symbol/side/model route.
- Tail loss event: force shadow until reviewed.

Primary implementation targets:

- `services/entry_profit_risk_sizing.py`
- `services/entry_sizing.py`
- `services/trading_params.py`
- `services/strategy_learning.py`
- `services/model_contribution_performance.py`

## 7. Entry And Exit Must Be Bound

Every real entry must create an exit plan at the same time.

Exit plan must include:

- Initial stop loss.
- Invalidation thesis.
- Max hold time.
- Profit target.
- Trailing/profit drawdown rule.
- Partial close conditions.
- Full close conditions.
- Loss repair probability.
- Do-not-close conditions.
- Replacement-opportunity requirement for capital rotation.

Rules:

- Opening without exit plan is forbidden.
- Exit decisions must reference the original `exit_plan_id`.
- If the system exits outside the plan, it must record why the original plan failed.
- Profit exits and loss exits must be evaluated separately.

This avoids the current pattern where entry is made by one logic and exit is later improvised by another logic.

Primary implementation targets:

- `services/profit_first_exit_plan.py`
- `services/exit_fast_risk.py`
- `services/exit_fee_churn_guard.py`
- `services/position_quality.py`
- `services/position_release_decision.py`
- `services/exit_arbitrator.py`
- `services/trading_service.py`

## 8. Governance For "No Entry"

Every non-entry must be categorized.

Required no-entry reason categories:

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

Every 24 hours the system must answer:

- Was the market genuinely unattractive?
- Or was the system over-conservative?
- Which gate blocked the most profitable later-missed opportunities?
- Which missing field most often forced shadow?
- Which symbols had positive shadow outcomes but no real entries?

Required output:

- A daily no-entry report.
- A top blocker table.
- A missed-opportunity replay table.
- A recommendation: keep thresholds, relax specific gate, tighten specific gate, or improve data/model source.

Primary implementation targets:

- `services/strategy_signal_root_cause_audit.py`
- `services/shadow_missed_opportunity_closed_loop.py`
- `services/entry_feature_ranker.py`
- `web_dashboard/api/system_audit.py`

## 9. Governance For Losing Exits

Every losing close must be attributed.

Required loss attribution categories:

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

Required fields:

- entry plan id
- exit plan id
- model route
- strategy profile
- symbol/side
- expected net return at entry
- actual realized net PnL
- expected hold time
- actual hold time
- entry-to-exit adverse move
- fee/slippage estimate
- attribution confidence
- next-cycle penalty or fix

Rules:

- Any attribution with low confidence goes to review/shadow learning, not immediate strategy promotion.
- If `exit_too_early` repeats, increase minimum hold or require stronger replacement opportunity.
- If `stop_too_tight` repeats, adjust stop model before changing entry model.
- If `model_false_positive` repeats, demote model contribution for that regime.
- If `position_too_small_fee_drag` repeats, stop tiny probes in that regime instead of making more small trades.

Primary implementation targets:

- `services/profit_attribution.py`
- `services/trade_execution_contract.py`
- `services/order_position_reconciliation.py`
- `services/strategy_learning.py`
- `services/model_contribution_performance.py`

## 10. Model And Strategy Ranking

Build real PnL leaderboards by:

- model route
- strategy profile
- symbol
- side
- market regime
- decision lane
- exit intent
- holding-time bucket

Metrics:

- realized net PnL
- win rate
- profit factor
- average win
- average loss
- max adverse excursion
- max favorable excursion
- fast-loss rate
- tail-loss rate
- fee drag ratio
- opportunity miss rate

Promotion:

- Increase weight only when realized net PnL is positive and sample quality is clean.
- Shadow evidence can nominate a model/strategy, but cannot directly promote to live influence.
- Canary requires healthy paper observation and clean training gates.

Demotion:

- Consecutive losses demote.
- Tail loss forces shadow.
- False signal loss beyond threshold blocks promotion.
- Model disagreement with better-performing sources reduces weight.

Primary implementation targets:

- `services/model_contribution_performance.py`
- `services/model_dynamic_routing.py`
- `services/strategy_learning.py`
- `services/symbol_side_performance.py`
- `services/daily_side_performance.py`

## 11. New Big Model Server Integration Policy

The new model server is part of the foundation, but it should not bypass profit-first control.

Expected model endpoints:

- Decision maker: `qwen3-32b-trade`
- Quant API: `phase3_quant_api`
- High-risk review: `deepseek-r1-14b-risk`
- Expert pool: `BB-FinQuant-Expert-14B`

Rules:

- The model server may propose direction, expected return, risk, and exit plan.
- The master control must normalize the model output into `ProfitFirstTradePlan`.
- Missing model output fields force shadow.
- Same model family role views do not count as multiple independent sources.
- High-risk review can approve/downgrade/veto, but cannot force size beyond risk budget.
- Quant API predictions must be calibrated against realized outcomes before live weight increases.

Main question to verify:

- Are new-server model outputs entering the master trade plan, or are they only visible in diagnostics/shadow?

## 12. Implementation Roadmap

### Stage 1 - Profit Diagnosis Unification

Goal:

- Create one profitability diagnosis table/report that combines entry funnel, small-order reason, no-entry reason, losing-exit attribution, missed opportunity, model contribution, and strategy profile.

Deliverables:

- `ProfitFirstTradePlan` schema.
- `ProfitFirstDiagnosticService`.
- Daily/rolling profitability report.
- Dashboard card can display it, but the core is the service/report, not the page.

Do first:

- Add read-only reports.
- Backfill recent 24h/72h decisions into the new diagnostic shape.
- Do not change trading behavior yet.

Acceptance:

- Every recent entry and non-entry has a lane and reason.
- Every recent close has an exit attribution or `unknown_requires_review`.
- Unknown categories are below 10% before behavior changes.

### Stage 2 - Dynamic Position Ladder

Goal:

- Replace fixed tiny/probe behavior with quality-based size tiers.

Deliverables:

- Lane-based sizing policy.
- Recent realized-edge brake.
- Promotion/de-promotion state.
- Max loss budget integration.

Behavior changes:

- Strong opportunities can reach 5%-8%.
- Extreme opportunities can reach 8%-12% only after high-risk review and health gates.
- Bad recent lane performance forces demotion to shadow/tiny.

Acceptance:

- No `low_payoff_quality=true` trade gets meaningful size.
- `meaningful_entry` trades always show promotion reasons.
- Average notional rises only for validated/high-quality lanes.

### Stage 3 - Entry-Exit Binding

Goal:

- Make every entry carry its own exit plan.

Deliverables:

- `ProfitFirstExitPlan`.
- Exit-plan persistence.
- Exit decisions reference original plan.
- Exit attribution compares actual close against planned close conditions.

Behavior changes:

- No plan, no entry.
- Capital-release exits must prove net benefit or hard risk.
- Stale probe release cannot repeatedly lock small losses without replacement evidence.

Acceptance:

- Every new position has an `exit_plan_id`.
- Every close references either the original plan or a plan-failure reason.
- Losing closes decrease in frequency or become clearly attributed.

### Stage 4 - Strategy Learning Becomes Profit Ranking

Goal:

- Strategy profiles and model routes should be promoted/demoted by realized net PnL, not just descriptive profile labels.

Deliverables:

- Strategy/model leaderboard.
- Regime-specific ranking.
- Auto-demotion for continuous losses.
- Shadow nomination for promising but unproven strategies.

Acceptance:

- A losing profile cannot keep pushing live entries at the same size.
- A profitable profile earns more budget only after clean samples.
- Ranking changes are auditable.

### Stage 5 - Controlled Paper Resume And Online Observation

Goal:

- Resume automatic opening only when the foundation and profit-first controls are ready.

Required gates:

- OKX reconciliation healthy.
- Realized PnL accounting trusted.
- Clean training view available.
- Model server and quant API healthy.
- Go/no-go non-critical.
- `ProfitFirstTradePlan` and exit binding active.
- Probe-loop brake active.

Observation windows:

- 2h: no contract violations, no missing plan fields.
- 8h: no repeated all-loss probe loop.
- 24h: positive or improving profit factor, clear attribution for losses.
- 72h: stable lane distribution, no uncontrolled size escalation.

## 13. Hard Safety Boundaries

Do not:

- Increase global leverage first.
- Increase all position sizes globally.
- Let high-risk review bypass max loss budget.
- Count one model's multiple roles as independent agreement.
- Promote ML/local tools while clean training view is unavailable.
- Keep executing tiny probes after all recent tiny probes lose.
- Close stale probes at small losses repeatedly without replacement evidence.

Do:

- Optimize for realized net PnL.
- Keep all behavior auditable.
- Use shadow before live influence.
- Make every entry and exit explainable by the same plan.

## 14. Success Metrics

Short-term success:

- New entries have complete `ProfitFirstTradePlan`.
- Unknown no-entry and unknown exit reasons drop below 10%.
- Tiny probes stop during all-loss windows.
- Losing stale-release loop stops.

Medium-term success:

- `validated_probe` and `meaningful_entry` become the main real-entry lanes.
- `profit_factor > 1.0` over rolling 24h paper windows.
- Fast-loss rate trends down.
- Average realized loss does not increase while meaningful entries are introduced.

Long-term success:

- Realized net PnL is positive after fees and slippage.
- Model/strategy ranking automatically moves capital toward profitable regimes.
- The system can explain whether no-entry periods are market-driven or system-driven.
- Strong opportunities are not trapped forever as tiny orders.

## 15. Recommended First Coding Slice

Implement the smallest slice that changes the control architecture without risky size expansion:

1. Add `services/profit_first_trade_plan.py`.
2. Build read-only `ProfitFirstTradePlan` from existing decisions.
3. Add lane classification.
4. Add no-entry reason normalization.
5. Add losing-exit attribution normalization.
6. Add probe-loop health report.
7. Add tests for required fields and shadow fallback.

Only after this read-only layer is stable should behavior changes begin:

1. Add recent probe PnL brake.
2. Add lane-based sizing.
3. Add entry-exit binding.
4. Add strategy/model PnL ranking.

Suggested tests:

- `tests/test_profit_first_trade_plan.py`
- `tests/test_profit_first_lane_classifier.py`
- `tests/test_no_entry_reason_taxonomy.py`
- `tests/test_losing_exit_attribution.py`
- `tests/test_recent_probe_pnl_brake.py`
- `tests/test_entry_exit_plan_binding.py`

## 16. Final Position

This is the right next direction:

- not more UI,
- not more fallback logic,
- not blind bigger orders,
- not model worship,
- but one profit-first master control loop.

The main system objective becomes:

```text
Only take real risk when the complete plan says expected realized net profit justifies it,
scale only when quality and realized evidence justify it,
exit only when the original plan or new hard evidence says the expected net outcome improves,
and demote every model/strategy that fails this test.
```
