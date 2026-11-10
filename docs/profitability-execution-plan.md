# Profitability Execution Plan

Updated: 2026-11-11, Asia/Shanghai.

Status: ACTIVE. Profitability acceptance has not passed.

Repository: `G:\数字量化\bb`; remote application: `/data/bb/app`.

History through 2026-11-08 is archived at
`docs/archive/profitability-execution-plan-through-2026-11-08.md`. That file is
audit history only; current code, tests, and fresh runtime evidence override
anything written there.

## Current Decision

Profitability acceptance remains incomplete. There were also concrete
engineering defects: shadow funding labels used continuous accrual across
non-settlement periods, passive entry fallback could race an unconfirmed
cancellation, and the training dataset lookup timed out. These have separate
fixes and must not be conflated with an unprofitable strategy.

The current funding supervision policy is
`2026-11-10.scheduled-shadow-funding.v5`; v4 artifacts are incompatible.
Rebuild both model branches before interpreting their readiness. Do not force
trades, shorten the primary horizon, mix horizons, or relax the positive-LCB,
profit-factor, disjoint-exam, or live-readiness gates to manufacture entries.

## Primary-Horizon Training Contract

`services/ml_signal_training_contract.py` defines the 240-minute primary
prediction horizon and the `fit_primary_horizon_only` policy.

`services/ml_signal_service.py` now filters the training frame to the primary
horizon before decision-group partitioning, tail-policy derivation, model
fitting, walk-forward evaluation, and holdout evaluation. The standalone
`scripts/phase3_quant_api_service.py` applies the same rule and rejects a
training request that has no 240-minute rows.

Artifacts record `fitted_horizons`, `source_training_horizons`,
`source_horizon_sample_counts`, `unfitted_horizon_minutes`,
`training_horizon_policy`, and `mixed_horizon_fit_allowed=false`. Non-primary
horizons remain visible as diagnostics only and cannot contribute promotion
evidence.

This removes a real contamination path; it does not create profitability.

## Acceptance Gates

A model can influence paper or live routing only after all existing gates pass:

- Chronologically disjoint decision groups and labels.
- Walk-forward folds fitted independently on the primary horizon.
- Positive fee-after lower confidence bound.
- Fee-after profit factor above `1.0`.
- Stable leave-one-symbol-out and market-regime evidence.
- Authoritative OKX execution-cost and realized-return calibration.
- The independent promotion exam, with no reuse of development or holdout data.

Elapsed time alone does not clear these conditions. A valid result requires
fresh, mature, authoritative samples and a candidate that passes the gates.

## Current Workflow

1. Deploy the verified code and preserve the runtime environment.
2. Verify source integrity, runtime imports, model tunnels, OKX connectivity,
   paper trading, and the Dashboard.
3. Run a non-persisting training preflight with fresh source data.
4. Persist and rebuild the primary-horizon model only if the preflight is valid.
5. Recheck Go/No-Go, readiness, profit factor, lower bound, fold stability, and
   whether a paper canary is permitted.
6. Keep paper-only execution and live routing disabled until every gate passes.

## Evidence Policy

Do not delete databases, raw authoritative audit records, metric samples, or
`data/profitability_remediation/`. Generated cache directories are disposable.
Retired model names may remain in leak-check denylists; that is not evidence
that a retired service is active.

## Current Limitation

There is no defensible time-to-profit estimate. The system is ready to collect
more evidence, not proven profitable.

The five remediation items are not all accepted: development separation and
recipe comparison exist, but there is no qualifying replacement, completed
fresh independent exam, complete maker-fill/exit-cost portfolio replay, or
qualified strategy's paper entry-to-settlement acceptance. Shadow costs still
project the round trip from entry-time market facts. The separately saved
settlement observation is not an actual exit fill and must not be presented as
one or leaked into decision-time features.

## November 11 Remediation

- Funding projections count settlements in `(decision_time, exit_time]`.
  Original decision time takes precedence; missing/invalid timing and future
  observations cannot supply complete production cost evidence. Truly legacy
  snapshots without clocks keep explicitly degraded projection metadata.
- Replay outcome emphasis uses the recomputed fee-after optimal side; stored
  historical audit labels remain unchanged.
- Market fallback requires the passive order's confirmed terminal state and
  final cumulative fill. An unavailable order ID/detail or active original
  order blocks fallback rather than risking duplicate exposure.
- Training group counts no longer compute a window sort just to count groups.
  A partial covering index excludes history-only observations before sampling.
  The online count of 52,428 eligible groups took 0.333 seconds, compared with
  the previous 30-second timeout.
- Diagnostic and manual training use the scheduler's bounded 30-second
  statement / 15-second idle-transaction budgets. API read budgets stay at
  their existing defaults.
- The diagnostic CLI uses the production primary-horizon loader, so other
  horizons no longer consume its fitting-window budget.
- Specialist shadow evaluation uses the same settlement-aware funding
  calculation as training and entry. Its former duplicate continuous formula
  could otherwise distort the quality evidence used for entry.
- The explicit standardized linear comparison now pins its estimator family
  at all sample counts. All four registered candidates failed; the best mean
  was approximately break-even and its confidence bound remained negative.

Use the existing online index utility for deployment or rollback:

```bash
python scripts/ensure_online_reconciliation_indexes.py --scope training --apply
python scripts/ensure_online_reconciliation_indexes.py --scope training --rollback --apply
```

These operations use concurrent index creation/drop, short lock waits, bounded
maintenance resources, and recovery of an invalid index left by an interrupted
concurrent build. They do not delete samples.

## Horizon Economics Finding (2026-11-09)

A read-only study of all four source horizons measured fee-after net return by
score quantile, using the same walk-forward split the gate uses.

| Horizon | Side | Top 5% mean | Top 5% LCB | Top 5% PF | Sample |
| --- | --- | ---: | ---: | ---: | ---: |
| 5 min | long | -0.0496% | -0.2014% | 0.798 | 80 |
| 5 min | short | -0.1349% | -0.2046% | 0.193 | 80 |
| 15 min | long | -0.0501% | -0.2872% | 0.866 | 80 |
| 15 min | short | -0.1688% | -0.4478% | 0.691 | 80 |
| 60 min | long | -0.2772% | -0.4482% | 0.372 | 80 |
| 60 min | short | **+0.7092%** | -0.1777% | 2.366 | 80 |
| 240 min | long | -0.6678% | -1.4046% | 0.547 | 79 |
| 240 min | short | -0.6882% | -1.4621% | 0.617 | 79 |

The 60-minute short top-quantile bucket is the only combination with positive
mean fee-after return and profit factor above one. It still fails the gate:
the return lower confidence bound is negative, fold 4 turns negative, and
leave-one-symbol-out is negative for every excluded symbol. The 240-minute
primary horizon shows negative fee-after returns on both sides at every
quantile tested. No tested horizon, side, or quantile simultaneously passes
positive LCB, profit factor above one, and split stability.

The practical reading: execution cost (about 0.14% round trip) is comparable to
the smallest horizon's gross edge, and the primary 240-minute horizon has no
generalizing directional edge in the current sample. This is a strategy and
feature-evidence problem, not a gate-calibration problem; relaxing the LCB, PF,
or stability gates would only manufacture entries without proven edge.

### Go/No-Go Evidence Binding

`services/phase3_go_no_go.py` previously read profitability from the legacy
`local_ai_tools` bundle, so the audit could display a stale profit factor from a
different model. It now prefers `local_ml_profit_quality` (the current local ML
artifact), reads side-specific profit factor, return LCB, mean return, and
fallback/ walk-forward state, and blocks when those fields are missing instead
of silently falling back. `web_dashboard/api/system_audit.py` exposes the
artifact evidence plus `top-level live_ml_ready`, and
`services/model_training_registry.py` surfaces per-side metrics for the same
identity.

### Counterfactual Cost Sign

Negative shadow execution costs were investigated. They come from funding
payments that credit a position (for example, negative funding paid to longs),
and each observed negative-cost sample carries a real `funding_interval_minutes`
and `next_funding_time`. The sign is semantically correct and is not clamped;
the cost model remains `shadow_counterfactual_live_microstructure`.
