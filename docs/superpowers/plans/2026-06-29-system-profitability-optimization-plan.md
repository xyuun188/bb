# 2026-06-29 System Profitability Optimization Plan

> Status: superseded as an execution plan by `2026-06-29-profit-first-v3-authoritative-master-plan.md`.
> Keep this file as the 2026-06-29 read-only diagnosis snapshot and supporting evidence. Do not use it as the primary implementation roadmap when it differs from the authoritative plan.

## 0. Current Check Summary

This plan is based on a read-only system check run from `F:\BB` on 2026-06-29 CST.

Recent 480-minute strategy health:

- `decisions=1600`
- `market_decisions=281`
- `market_entry_decisions=77`
- `executed_entries=10`
- `orders=34`
- `filled_orders=33`
- `rejected_orders=1`
- `positions_created=12`
- `positions_closed=10`
- `open_positions=8`
- `fast_loss_close_under_15m=2`

Closed-position PnL diagnostics:

- `closed_count=9`
- `win_count=0`
- `loss_count=9`
- `win_rate=0.0`
- `total_realized_pnl=-2.703428 USDT`
- `profit_factor=0.0`
- worst losses: `MAGIC/USDT -0.996403`, `CELO/USDT -0.635491`, `GRAM/USDT -0.600153`
- short side drove most loss: `short=-2.549418`, `long=-0.15401`

Entry evidence and sizing diagnostics:

- Market entry evidence tiers: `blocked=38`, `exploration=22`, `small=12`, `weak_conflict_probe=5`.
- There were no recent trade execution contract violations in the checked window.
- No weak-evidence or negative-expected entries were executed by the contract audit.
- The executed entries that did occur were still only `exploration/small` and sizing was mostly capped by low payoff quality, strategy probe caps, and notional-floor blocks.
- Local ML readiness is still `degraded`, with live position influence disabled.
- Phase 3 go/no-go is `blocked`: model-server readiness, paper resume preflight, post-resume observation, model training/promotion, quant API tunnel, and OKX authoritative sync checks are not healthy from this environment.

Conclusion:

The system is not simply "unable to open positions". It can open, but most accepted entries are low-quality probe-style entries. The system then releases or closes those small positions at a realized loss, while the model/data layer is not healthy enough to justify scaling. The result is the exact symptom pattern: small entries, loss-making closes, long wait periods, and no net profitability.

## 1. Final Goal Gap

The target system should:

1. Scan enough symbols and find tradable opportunities regularly.
2. Separate weak/noisy entries from genuinely high-quality opportunities.
3. Automatically promote high-quality opportunities from tiny/probe size into meaningful size.
4. Avoid paying fees and slippage repeatedly on low-edge micro positions.
5. Let winners run or protect profit before it turns into a loss.
6. Stop losers only when the thesis is invalidated or risk is genuinely worsening, not just because a small probe looks inefficient.
7. Use Phase 3 quant API, specialist models, clean training views, and ML only after readiness and promotion gates pass.
8. Measure everything by realized net PnL after fees and slippage.

Current gaps:

- The entry layer reaches `exploration/small` too often and `medium/normal/high_profit` too rarely.
- The sizing layer has promotion paths, but live data rarely satisfies their thresholds.
- The exit/release layer is successfully cleaning up small inefficient probes, but the realized outcome is consistently negative.
- The model layer is not ready for live influence, so evidence quality stays weak.
- The system lacks a hard daily "do not keep sampling if all closes are losing" brake tied to recent realized PnL.

## 2. Root Causes

### A. Entry Quality Is Not Strong Enough

The entry evidence layer is doing its safety job: it blocks weak conflicts and does not execute negative expected-return entries. However, recent executed entries mostly come from `exploration` and `small` tiers, not high-conviction tiers.

Observed evidence:

- `exploration=22`, `small=12`, `blocked=38`, `weak_conflict_probe=5`.
- Effective score median was far below normal entry quality.
- Server profit, sentiment, and timeseries signals often conflict or contribute weakly.
- ML component is ignored/degraded, so it cannot strengthen entry conviction.

Primary files:

- `services/entry_evidence.py`
- `services/entry_candidate_evidence.py`
- `services/entry_opportunity_scoring.py`
- `services/entry_opportunity_gate.py`
- `services/entry_feature_ranker.py`

### B. Meaningful-Size Promotion Exists But Does Not Trigger Often Enough

The sizing code has quality tiers such as `good_probe`, `strong_probe`, `quality_override`, `high_profit`, `elite`, and `winner_add`. But recent trades stay at base/probe sizing because expected net return, profit quality, score, aligned source count, and ML readiness do not jointly pass.

Observed evidence:

- Recent executed entries were sized around small notionals, often near 20-30 USDT fills.
- Reasons included `low_payoff_quality`, `strategy_probe_cap_applied`, `notional_floor_blocked`, `expected_net_below_min`, and `profit_quality_below_min`.
- `notional_floor_blocked` explicitly says profit quality or small-win-large-loss risk is too high.

Primary files:

- `services/entry_profit_risk_sizing.py`
- `services/entry_sizing.py`
- `services/trading_params.py`
- `ai_brain/ensemble_coordinator.py`

### C. Exit/Release Is Converting Probe Rotation Into Realized Loss

The release logic is designed to free capacity from stale or low-quality probes. That part works, but the current result is bad: all checked closed positions were losses.

Observed evidence:

- `closed_count=9`, `loss_count=9`, `profit_factor=0.0`.
- Common close triggers: `stale_probe_capital_inefficient`, `loss_pressure`, `fee_efficiency_weak`, `signal_reversal_watch`, `time_cost_flat_12h`.
- Two fast losses closed under 15 minutes.

Primary files:

- `services/position_quality.py`
- `services/position_release_decision.py`
- `services/exit_fast_risk.py`
- `services/exit_fee_churn_guard.py`
- `services/exit_arbitrator.py`
- `services/trading_service.py`

### D. Portfolio Is Over-Concentrated In One Side

Recent diagnostics show many more shorts than longs and the crowded-side cap repeatedly blocks new same-side shorts.

Observed evidence:

- Market entry actions: `short=58`, `long=19`.
- Closed losses: `short=7`, `long=2`.
- Side PnL: `short=-2.549418`, `long=-0.15401`.
- Some entry skips were due to crowded short-side exposure.

Primary files:

- `services/entry_crowded_side_cap.py`
- `services/entry_position_exposure.py`
- `services/daily_side_performance.py`
- `services/symbol_side_performance.py`

### E. Phase 3 Foundation Is Still Blocked

The system cannot responsibly scale entries while the Phase 3 model and clean training stack are blocked.

Observed evidence:

- Go/no-go status: `blocked`.
- Model readiness: `unverified`; `BB_SECURE_SETTINGS_KEY` missing in ad-hoc readiness run.
- Phase 3 quant API on `127.0.0.1:18001` unavailable or returning 502/read errors.
- Decision/risk/expert model tunnels on `18000/18002/18003` unavailable or returning 502/read errors.
- Clean training view unavailable.
- Legacy ML artifacts are classified as `retired_legacy`.
- Promotion recommendation stays `shadow`.

Primary files:

- `services/phase3_go_no_go.py`
- `services/phase3_model_server_readiness.py`
- `services/phase3_paper_resume_preflight.py`
- `services/phase3_paper_resume_observation.py`
- `services/phase3_rebuild_readiness.py`
- `services/local_ai_tools_client.py`
- `scripts/start_online_model_tunnels.py`
- `scripts/deploy_local_ai_tools_service.py`
- `scripts/train_ml_signal_model.py`
- `scripts/train_local_ai_tools_models.py`

## 3. Optimization Strategy

Do not solve this by blindly increasing leverage or position size. The recent sample is all losing closes. Scaling that behavior would only make losses larger.

The correct path is:

1. Restore the Phase 3 model/data foundation.
2. Make the entry funnel produce fewer low-payoff probes and more verifiable high-quality candidates.
3. Add a strict promotion ladder from probe to meaningful size after evidence improves.
4. Change exit/release so stale small probes are not repeatedly closed at small guaranteed losses unless capacity pressure or hard risk truly requires it.
5. Add realized-PnL brakes and experiment scorecards so the system stops sampling bad regimes automatically.

## 4. Phase Plan

### Phase 1 - Restore Observability And Hard Gates

Goal: make the system truthfully know whether it is allowed to learn, trade, and scale.

Actions:

1. Fix model-server readiness execution context.
   - Ensure readiness audits run with the real platform runtime env, including `BB_SECURE_SETTINGS_KEY`.
   - Re-run `scripts/run_phase3_model_server_readiness_audit.py`.
   - Required result: `runtime_ready=true`, `artifact_ready=true`, active endpoints for `qwen3-32b-trade`, `phase3_quant_api`, `deepseek-r1-14b-risk`, and `BB-FinQuant-Expert-14B`.

2. Restore tunnels and quant API.
   - Bring back `18000`, `18001`, `18002`, `18003`.
   - Verify `18001 /health` reports `service=phase3_quant_api`.
   - Verify child endpoints: `/profit/predict`, `/exit/advise`, `/timeseries/deep/predict`, `/sentiment/deep/analyze`.

3. Restore OKX authoritative sync and clean fact audits.
   - Fix `[WinError 1225]` connection failures for OKX/native sync and DB-dependent reports.
   - Re-run trade-fact integrity and reconciliation reports.
   - Do not train or promote models while this is unavailable.

4. Produce a fresh go/no-go baseline.
   - Run `scripts/run_phase3_go_no_go_report.py --stdout-only`.
   - Required result for strategy work: no `critical` blockers, or explicit known false positives caused by local ad-hoc environment mismatch.

Acceptance:

- Phase 3 go/no-go no longer reports model server, quant API, OKX sync, and training source as hard blockers.
- Dashboard/system audit can distinguish true online service state from local Windows dev limitations.

### Phase 2 - Stop The Losing Probe Loop

Goal: prevent repeated small trades from turning fees, slippage, and minor reversals into realized loss.

Actions:

1. Add a recent realized-PnL brake for probe entries.
   - If the last N closed probe/small positions have `win_rate=0` or `profit_factor<0.8`, block new probe entries unless the candidate reaches `strong_probe` or better.
   - Suggested initial gate: last 8 closed positions, or last 6 hours, whichever has enough samples.
   - Scope this to `exploration`, `small`, `good_probe`, and `strategy_learning` probe entries, not forced exits.

2. Tighten probe admission after all-loss windows.
   - Require `expected_net_return_pct >= 0.55`, `profit_quality_ratio >= 0.65`, `loss_probability <= 0.42`, and at least 3 independent aligned sources before a new small/probe can execute during a recent all-loss window.
   - Keep weak/conflict candidates as shadow-only.

3. Make stale probe release PnL-aware.
   - Current `stale_probe_capital_inefficient` can close tiny flat/slightly losing positions after 1 hour.
   - Change it so a small losing stale probe is not released solely for capital efficiency unless:
     - capacity pressure is real,
     - reversal is strong,
     - expected hold recovery is negative,
     - or loss exceeds a defined risk budget.
   - Otherwise classify it as `watch`, not `release_candidate`.

4. Add fee/slippage break-even guard before discretionary release.
   - Do not close a low-risk small position if projected exit locks a loss smaller than normal noise and no stronger replacement opportunity exists.
   - The release decision should compare:
     - realized loss if closed now,
     - expected recovery/decay,
     - stronger candidate available now,
     - current capacity pressure.

Acceptance:

- New 8h window should not show `closed_count>5` with `win_count=0`.
- `fast_loss_close_under_15m` should be 0 unless stop breach or hard adverse evidence is logged.
- Release reasons should show replacement-opportunity or hard-risk context, not only stale/fee inefficiency.

### Phase 3 - Rebuild Entry Quality

Goal: create fewer entries, but entries with enough edge to either avoid trading or trade meaningfully.

Actions:

1. Split entry candidates into four lanes.
   - `shadow_only`: positive but insufficient evidence.
   - `tiny_probe`: allowed only for learning when recent realized probe loop is healthy.
   - `validated_probe`: expected net, profit quality, loss probability, and aligned sources pass.
   - `meaningful_entry`: strong evidence plus clean recent outcomes.

2. Add "entry cannot be executed unless it has a target lane".
   - Every executed entry must record lane, reason, thresholds, and promotion path.
   - `unknown` skip kinds should be eliminated from entry diagnostics.

3. Reweight evidence after Phase 3 API recovers.
   - Do not count multiple `BB-FinQuant-Expert-14B` role views as independent sources.
   - Count independent source groups only: decision LLM, risk reviewer, quant API profit, timeseries, sentiment, local ML, shadow memory, symbol-side history.
   - Require at least 3 groups for `validated_probe`, 4 groups for `meaningful_entry`.

4. Add negative realized side-memory penalties.
   - Since recent short losses dominate, new shorts should require stronger expected net and lower loss probability until short-side realized PnL recovers.
   - This is not a permanent short ban; it is an adaptive side-quality gate.

5. Adjust market ranking to avoid repeatedly selecting noisy low-liquidity names.
   - Recent filtered-out reasons are dominated by `analysis_volume_ratio_below_floor`, `analysis_notional_below_floor`, and `missing_indicator_snapshot`.
   - Add a "post-filter missed winner review" before relaxing filters.
   - Only relax analysis filters for symbols that later show positive clean shadow outcomes, not globally.

Acceptance:

- In the next 24h paper window, `validated_probe + meaningful_entry` should become the majority of executed entries.
- `exploration` should mostly stay shadow unless recent probe-loop health is good.
- `unknown` entry skip kind should approach 0.

### Phase 4 - Add A Real Promotion Ladder

Goal: solve "opens only small orders" without blindly scaling.

Actions:

1. Define strict promotion stages.
   - `tiny_probe`: 0.5%-1.5% size, only for learning.
   - `validated_probe`: 2%-4% size, only if expected net and profit quality pass.
   - `meaningful_entry`: 6%-12% size, only if recent realized loop is positive or clean high-quality evidence is present.
   - `high_conviction`: only after Phase 3 canary, walk-forward, and realized PnL gates pass.

2. Promotion should require realized or high-confidence shadow proof.
   - Do not promote because one model says positive.
   - Promote when the same lane/symbol/side/regime has clean realized or shadow outcomes.

3. Make notional floor conditional on expected profit in USDT.
   - If expected profit after fees is less than a minimum useful amount, either do not trade or keep it shadow.
   - A larger notional floor should only apply when the expected profit/loss structure is favorable.

4. Add automatic de-promotion.
   - If a lane causes 2 consecutive realized losses or a profit factor below 1.0, demote to shadow/tiny until recovered.

Acceptance:

- Meaningful entries must show `meaningful_size_reason`.
- No entry should be enlarged when `low_payoff_quality=true`.
- Average realized loss per closed trade should not grow while testing promotion.

### Phase 5 - Fix Exit Logic For Profitability

Goal: exits should protect net profit and stop thesis failure, not generate fee churn.

Actions:

1. Add exit outcome attribution by intent.
   - Track realized PnL grouped by:
     - `stale_probe_capital_inefficient`
     - `loss_pressure`
     - `signal_reversal`
     - `profit_drawdown`
     - `fast_adverse`
     - `fee_efficiency_weak`
   - If an intent has negative profit factor across recent samples, reduce or suspend that discretionary exit path.

2. Require stronger replacement opportunity for capital-rotation exits.
   - Do not close a small losing stale probe unless a higher-quality candidate is ready and capacity is actually constrained.

3. Improve winner handling.
   - If a position reaches protectable profit, prefer profit drawdown protection before it turns into loss.
   - Ensure peak PnL tracking is reliable and not lost across restarts.

4. Separate hard risk exits from low-quality release.
   - Hard risk remains immediate.
   - Low-quality release must pass net-benefit checks.

Acceptance:

- Closed-position diagnostics should show at least some profitable closes before any further sizing increase.
- No exit-intent bucket should keep firing with `profit_factor=0`.
- Fast loss must include stop breach, hard adverse evidence, or predictive reversal confirmation.

### Phase 6 - Rebuild And Promote Models Safely

Goal: make ML and specialist evidence actually improve entries/exits.

Actions:

1. Restore clean training view.
   - Clear `clean_training_view_unavailable`.
   - Clear `historical_trade_fact_audit_unavailable`.
   - Keep legacy artifacts retired.

2. Rebuild local ML and local tools from clean samples only.
   - Run preflight commands first:
     - `python scripts/train_ml_signal_model.py`
     - `python scripts/train_local_ai_tools_models.py`
   - Only persist after readiness passes:
     - `python scripts/train_ml_signal_model.py --persist-artifact --confirm-phase3-rebuild`
     - `python scripts/train_local_ai_tools_models.py --persist-artifact --confirm-phase3-rebuild`

3. Keep rebuilt models shadow-first.
   - Require sample floors and walk-forward checks.
   - Require top-decile expected returns to be positive for both long and short before live influence.

4. Add specialist shadow evaluation.
   - Run `scripts/run_specialist_shadow_evaluation.py`.
   - Do not let specialist roles count as independent votes until evaluation passes.

Acceptance:

- ML readiness no longer `degraded`.
- Top long and top short average returns are positive.
- Specialist shadow report exists and does not show tail-loss blockers.
- Promotion recommendation can advance from `shadow` only after paper observation is healthy.

## 5. Concrete Priority Order

Priority 0:

- Do not increase leverage, global max position size, or bypass risk reviews now.
- Recent realized data is too poor for blind scaling.

Priority 1:

- Restore Phase 3 endpoints, runtime env, quant API, OKX sync, and clean training view.
- Fix audit false positives caused by local ad-hoc environment mismatch.

Priority 2:

- Implement the realized-PnL probe brake and PnL-aware stale-probe release change.
- This directly attacks the current all-loss close loop.

Priority 3:

- Add entry lanes and promotion/de-promotion reporting.
- This attacks "opens tiny forever".

Priority 4:

- Rebuild ML/local tools from clean data and keep shadow until quality gates pass.
- This attacks "long time no entries" and weak evidence.

Priority 5:

- Tune filters and ranking only after shadow replay shows missed profitable opportunities.
- Do not relax liquidity/volume filters globally.

## 6. New Monitoring KPIs

Track these every 2h, 8h, 24h, and 72h:

- `executed_entry_count`
- `tiny_probe_count`
- `validated_probe_count`
- `meaningful_entry_count`
- `tiny_probe_count / executed_entry_count`
- `closed_count`
- `win_rate`
- `profit_factor`
- `total_realized_pnl`
- `fast_loss_close_under_15m`
- `avg_notional_usdt`
- `notional_floor_blocked_count`
- `low_payoff_quality_count`
- `stale_probe_release_loss_count`
- `exit_intent_profit_factor`
- `short_side_realized_pnl`
- `long_side_realized_pnl`
- `model_readiness_status`
- `phase3_go_no_go_status`

Hard stop conditions:

- Last 6-8 closed positions all losing.
- `profit_factor=0` over 8h with more than 5 closes.
- Any exit intent fires 3+ times with 0 profitable outcomes.
- Fast-loss closes occur without strong exit evidence.
- ML or Phase 3 model gates are critical while sizing attempts to promote.

## 7. Definition Of Done

This round is not done when the system merely opens more trades.

It is done when:

1. Phase 3 foundation reports healthy or clearly explained non-blocking warnings.
2. New entries are classified into lanes with no unknown execution path.
3. Recent closed positions show positive profit factor after fees.
4. Tiny probes are a controlled minority, not the default execution mode.
5. Meaningful size appears only on audited high-quality opportunities.
6. Exit/release decisions no longer produce repeated all-loss windows.
7. Model promotion remains shadow/canary/live gated by clean sample outcomes.

## 8. Immediate Next Engineering Work

Recommended first implementation slice:

1. Add `RecentProbePnLBrakePolicy`.
2. Wire it into entry gate/sizing so all-loss probe windows force shadow-only unless the candidate reaches strong quality.
3. Add `PositionReleaseNetBenefitPolicy`.
4. Require stale-probe release to prove either hard risk, real capacity pressure plus better replacement, or negative recovery expectancy.
5. Add dashboard/audit cards for:
   - probe-loop health,
   - stale-release realized PnL,
   - entry lane distribution,
   - promotion/de-promotion decisions.

Suggested focused tests:

- `tests/test_recent_probe_pnl_brake.py`
- `tests/test_position_release_net_benefit.py`
- `tests/test_entry_lane_promotion.py`
- `tests/test_strategy_signal_root_cause_audit.py`
- `tests/test_system_audit_api.py`

This slice directly targets the current symptoms without unsafe leverage or size expansion.
