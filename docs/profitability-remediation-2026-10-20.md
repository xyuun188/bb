# Profitability Remediation And Acceptance

Updated 2026-10-21. The account used for this investigation is paper trading.
No live trading is enabled, and no result below is a promise of profit.

## Current Conclusion

The system-health problem and the verified data-pipeline defects have been repaired, but
the profitability problem is **not solved**. A completed training run, a
positive short paper window, or a healthy replay worker is not sufficient
evidence to promote a model.

The same-window diagnostic generated at 2026-10-20 17:33:43 UTC (October 21
01:33:43 Beijing time) used the same two clean, mature, cost-complete five-minute
decisions after both registered artifacts were created. The decisions were
made at 17:26:31-17:26:45 UTC. Current and challenger both selected long and
averaged `-0.2318803%`, with two losses, profit factor zero, and zero samples
passing the fee-after lower-bound gate. This is a counterfactual shadow
comparison, not exact production execution or portfolio PnL. Two samples are
insufficient for a statistical conclusion and do not justify promotion.

The six-hour paper snapshot ending at 2026-10-20 17:26:19 UTC contained 1,419
hold decisions, 13 executed exits (seven close-long and six close-short), three
unexecuted exits, and seven officially settled lifecycles. The exit pipeline
was active. This window is separate from the earlier 15:41 UTC snapshot and
must not inherit its PnL.

A separate bounded audit of 300 analyses from 15:18:07 to 17:33:33 UTC found
300 complete, decision-eligible analyses and no service errors. All finished
with `no_candidate`. None of 600 side evaluations were production eligible;
42 long and 23 short mean-return estimates were positive, but their lower
bounds were negative. Directional support, fee-after quality and economics
remain the entry problem. Ordinary paper execution does not universally
require model promotion or a positive per-trade lower bound, so those facts
must not be described as a blanket paper-order prohibition.

## Repaired Causes

1. Replay now loads the complete immutable artifact holdout: 1,062 holdout
   rows plus 200 bounded forward observations. Eight development groups are
   purged at the time boundary; the resulting development and exam partitions
   have zero source overlap.
2. Shadow training selection is now outcome-independent. Clean hold decisions
   and clean losing/no-opportunity observations are retained instead of being
   admitted only when their future action label happens to be directional.
   The selection-policy version is recorded in metadata and causes governed
   retraining when it changes.
3. Delayed-label maintenance now services both the oldest due rows and a
   bounded slice of newer matured rows. This prevents a long backlog from
   starving recent evidence needed by forward comparison.
4. Same-window comparison is artifact-bound and read-only. It verifies
   registry identity, source IDs, hashes, model versions, prediction counts,
   maturity, quality and fee/funding completeness. It cannot train, promote,
   change pointers or submit orders.
5. Training preflight is resource-bounded and now writes an explicit
   diagnostic report on completion, scheduler skip, or exception. It uses at
   most two CPU cores, 2 GiB memory and a 60-minute wall-clock limit; it never
   persists an artifact.
6. Positive but stale cached order-book depth no longer suppresses a blocking
   REST refresh. WebSocket book/mark facts must be current and newer before
   replacing usable snapshot facts. Mark/index merging preserves a newer
   usable native fact when a derivative supplies zero price, missing timestamp
   or an older value. Nonblocking reads retain their existing behavior.
   After deployment, DOGE row 598509 and LIT row 598529 had clean REST facts
   with sub-two-second source skew. Other ETH/OP/SOL observations still showed
   quote inconsistency or missing one-minute paths; data quality is not fully
   resolved.
7. A bounded, development-only baseline diagnostic now compares no-trade,
   constant sides, standardized Ridge and the existing tree defaults on three
   expanding chronological folds. It purges all label horizons at boundaries,
   fixes the primary horizon at five minutes, reports ranking/calibration and
   cannot persist a model, promote it or submit orders.
8. A separate bounded nested diagnostic fixes the same five-minute horizon,
   compares Ridge strengths 1/10/100 and predicted-net entry thresholds
   0/0.05/0.1 percent, and selects exclusively inside two purged inner folds.
   Three outer chronological folds evaluate the selected policies. Failed
   inner evidence selects no-trade. This research changes no runtime policy
   and cannot authorize promotion.

## Verified Historical Findings

- The current artifact is `20260915T123923735471Z-a10583f2`.
- Challenger `20261020T114306322775Z-9557978e` was rejected by existing
  governance because its fee-after return, tail loss and drawdown evidence were
  worse.
- The automatic trainer subsequently generated challenger
  `20261020T171459921362Z-f7df1af6`, trained at 17:14:57 UTC. The current
  pointer did not change. This was separate from the diagnostic preflight.
- The corrected preflight completed at 17:01:25 UTC with exit code zero and
  no artifact persisted. It loaded 19,266 rows, built 19,252 training-frame
  rows and included 499 authoritative trade samples.
- The preflight split contained 9,252 training rows/2,442 decision groups and
  9,786 holdout rows/2,498 groups. It purged 56 groups/214 rows, with zero
  group overlap and disjoint label periods. Training composition included
  8,596 hold, 425 short and 231 long labels.
- Corrected five-minute holdout top buckets (49 per side out of 2,469 rows)
  remained negative: long mean `-0.1387573045%`, lower bound
  `-0.1689556985%`, profit factor `0.05150658`; short mean `-0.0992743616%`,
  lower bound `-0.1241377415%`, profit factor `0.07973909`. Mean execution
  cost was `0.14115%` long and `0.14240%` short. Other horizons also failed;
  those already-read results cannot be used to select a winning horizon.
- Exact replay generated 125 candidates and zero governed profitable
  strategies. A successful replay worker therefore means only that the
  calculation completed.

## Development Baseline Results

The diagnostic completed at 2026-10-20 17:48:02 UTC (October 21 01:48:02
Beijing time). Its current-snapshot development partition contained 9,246
rows/2,443 groups after purging 55 boundary groups. The three expanding folds
evaluated 1,212 unique five-minute decision groups; no exam labels were used
for model fitting, selection or metrics.

| Policy | Selected | Mean Net Return | Profit Factor |
| --- | ---: | ---: | ---: |
| No-trade | 0 | N/A | N/A |
| Always long | 1,212 | -0.109308% | 0.456306 |
| Always short | 1,212 | -0.150868% | 0.313542 |
| Standardized Ridge, positive prediction | 73 | -0.192714% | 0.335832 |
| Existing tree defaults, positive prediction | 26 | -0.293295% | 0.347871 |
| Ridge, diagnostic top 20% | 243 | -0.133676% | 0.391534 |
| Tree, diagnostic top 20% | 243 | -0.171795% | 0.381492 |

The fixed top-20% ranking slices were negative in all three folds for both
models. Ridge long/short rank correlations were 0.09973/0.07668; tree
correlations were 0.06857/0.04126. This supports weak ranking as an additional
research problem, rather than treating conservative entry thresholds as the
only defect. The tree here predicts net return directly; it is not the full
production multitask/risk-adjusted model.

These are correlated shadow opportunity returns, not executed portfolio PnL.
Their naive lower bounds and accumulated drawdowns must not be presented as
independent confidence or portfolio risk estimates. Later report revisions
also expose selected gross return and execution cost separately, and include
sample weights and horizons in the reproducibility fingerprint.

## Nested Development Results

The bounded nested experiment completed at 2026-10-20 17:58:49 UTC (October 21
01:58:49 Beijing time), with data fingerprint
`e559d4dd04f5e8e4bd79998d829d11d7a7bec16fb41220ef7f54db0ccab9a03c`.
Its snapshot was newer than the baseline snapshot and contained 9,272
development rows/2,444 groups; do not combine their sample accounting.

Each of three outer folds selected exclusively from two earlier purged
inner folds. All 27 candidate evaluations failed the fixed inner selection
requirements. A third-fold candidate (alpha 10, threshold 0.05 percent) had
23 selected shadow opportunities, mean net return `+0.187327%` and profit
factor `1.915438`, but lower bound `-0.161590%`; this is not a qualified
profit result. None of the three outer folds selected a trading policy.
The resulting 1,215 outer opportunities therefore had zero selected trades
and zero return per opportunity, not a profitable strategy.

The gross/cost decomposition confirms both weak signal and cost pressure.
For alpha 1/threshold zero, first-fold inner selections averaged gross
`+0.013701%`, cost `0.136822%` and net `-0.123122%`; second-fold inner
selections already had negative gross `-0.056627%` before cost `0.131176%`.
Reducing an entry gate does not repair that economics.

Both development diagnostics leave artifact pointers and production risk/
execution rules unchanged. No fresh sealed exam has been consumed for this
search, and no controlled paper round trip has been authorized by its results.

## Required Research Order

1. Use the completed corrected preflight as a negative benchmark. Do not
   manually persist or promote that diagnostic model.
2. Treat the completed development ranking/calibration comparison as a negative
   baseline. Do not reverse scores or choose a horizon using the already-read
   holdout.
3. Treat the completed nested Ridge diagnostic as another negative experiment;
   do not deploy any of its candidate parameters. Further model families,
   feature subsets and horizons require
   separate nested chronological development experiments with purged label
   boundaries. Select
   model family, feature subset, horizon and entry threshold inside development
   folds only.
4. Freeze that policy, its code/data fingerprint and its cost/threshold rules.
   Evaluate it once on a fresh sealed exam containing
   independent cost-complete observations. Keep current and challenger on the
   exact same rows. The holdout already inspected above is not a fresh exam.
5. Require positive fee-after return with an independent lower bound above zero,
   profit factor above break-even, sufficient decision-group coverage and
   acceptable drawdown/tail risk. Report insufficient coverage as insufficient
   evidence.
6. Only after those gates pass, run a controlled paper round trip and reconcile
   entry, exit, protection orders and official settlement over subsequent
   independent windows.

Risk controls, leverage limits, quality gates and promotion rules must not be
weakened to manufacture more trades or apparent profit.

## Verification

The cache/native-fact regression group passed 146 tests. The final label-queue,
training-quality, comparison, replay, governance, preflight, baseline and nested
development group passed 156 tests. Ruff and whitespace checks passed.
Normal split-service synchronization passed source-integrity/import checks,
model-tunnel checks, OKX network checks and the Dashboard health check.
