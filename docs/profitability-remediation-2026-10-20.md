# Profitability Remediation And Acceptance

Updated 2026-10-22. The account used for this investigation is paper trading.
No live trading is enabled, and no result below is a promise of profit.

## Latest Remediation: October 22

The ordinary-paper authorization defect is repaired and deployed, but a
profitable replacement strategy is **not yet qualified**. Do not interpret
fewer unauthorized entries, passing software tests or positive unrealized PnL
as proof of profitability.

### Verified Authorization Defect

The deployed v19 policy allowed a positive current modeled net return to
override a denied historical fee-after quality permission. Executed DOGE
decision 680868 at October 21 16:02:56 UTC (October 22 00:02:56 Beijing time)
had modeled net return `+0.05340004%`, but its local-ML historical permission
was false. Its `authoritative_trade:ml_profit_model` evidence contained 230
samples, mean `-0.294319 USDT`, lower bound `-0.440138 USDT`, and profit factor
`0.592868`. These historical metrics use per-trade USDT, not percentages.
CFX decision 680847 showed the same permission bypass.

Contract `2026-10-21.normal-paper-strategy-trade.v20` now checks historical
permissions at side selection, contract construction, current-cost
reconciliation and final entry validation. Every supplied quality source
must grant execution and contain positive historical fee-after mean and
lower bound, a positive sample count, and profit-factor-above-break-even
evidence. Malformed sources cannot be filtered away to manufacture permission.
A missing profit-factor ratio is allowed for an all-winning history only
with the producer's explicit above-break-even evidence.

Failed historical quality remains eligible for shadow training where its
current signal meets the existing training rules; it cannot submit an order.
A per-trade downside quantile remains sizing stress, not a confidence bound
on the historical mean. Recognized v19 envelopes can still support settlement
and protection of existing positions, but cannot authorize new submissions.

### Account And Model Evidence

The read-only OKX account-bill audit at October 21 16:15:48 UTC (October 22
00:15:48 Beijing time) covered the preceding 24 hours, reached the time
boundary in two pages and deduplicated 170 bills. Trading bills contained
realized price PnL `-14.3243872861 USDT` and fees `-6.5054990700 USDT`.
Funding bills contributed `+0.1042119040 USDT`, giving trading/funding cash
change `-20.7256744521 USDT`. This includes entry fees and is not a statistic
of completed round trips or total marked-to-market account return.

At that snapshot there were 18 positions, six long and twelve short. All
had live exchange-confirmed OCO protection with matching closing direction,
adequate quantity, and both stop-loss and take-profit triggers. Unrealized
PnL was `+4.3620589898 USDT`; it was not treated as realized profit.
Position-history rows include partial lifecycle updates and must not be
summed as exact 24-hour trading PnL.

The bounded same-window model comparison at October 21 16:19:29 UTC
(October 22 00:19:29 Beijing time) evaluated 225 clean, mature, cost-complete
five-minute decision groups shared by current and challenger. Three rows
were excluded for missing executable order-book levels. The shared decision
window was 11:12:52-16:11:14 UTC, strictly after the common artifact cutoff
of 11:09:18 UTC.

| Immutable Artifact | Mean Net Return | Profit Factor | Diagnostic Gate Passes |
| --- | ---: | ---: | ---: |
| Current `20260915T123923735471Z-a10583f2` | -0.19259738% | 0.33917223 | 0/225 |
| Challenger `20261021T110918124657Z-9f094cf0` | -0.08713328% | 0.64132905 | 0/225 |

The input fingerprint is
`05e36086d8cc899fc113609457b3c2f70a536fa46a52fb00ea663a56e3db4ee0`.
These are correlated shadow opportunities, not executed portfolio PnL or a
fresh sealed strategy-selection exam. The diagnostic's distribution gate is
not the ordinary-paper historical-quality gate. Neither artifact was promoted,
trained, or used to submit orders by this comparison. A less-negative
challenger is not a profitable replacement.

### Deployment And Acceptance

The three changed service files were backed up and uploaded individually.
Remote source hashes match local files; imports and v20 policy flags passed.
Only paper trading and Dashboard were restarted at October 21 16:21:27 UTC
(October 22 00:21:27 Beijing time). Dashboard subsequently returned HTTP 200
with `{"status":"ok"}`; both services were active with zero automatic restarts.
Live trading remained inactive. No model tunnel, runtime environment, exchange
account setting or manual order was changed.

The final local worktree passed all 4,770 tests, Ruff on the nine changed
Python files, and whitespace checks. A read-only remote replay of all 26
executed entry decisions found in the preceding 24-hour bounded query changed
each v19-authorized signal into an unauthorized `paper_training_entry`.
A constructed positive-history/positive-current smoke case still authorized
an entry without submitting one. This verifies entry gating, not profitability.

At October 21 16:25:04 UTC (October 22 00:25:04 Beijing time), the first 40
post-restart decisions contained five market decisions: XRP 681014 and ZRO
681028 were v20 unauthorized training entries, while three had no direction.
The position loop also executed ADA close-long decision 681005. A DOGE partial
exit was separately blocked as below the exchange execution minimum. The
quality fix therefore did not disable existing-position exits; it did not
establish profitable exits either.

The remaining acceptance criteria in Required Research Order still apply.
Do not reopen historical-quality-denied execution to increase trade count,
select a favorable side from this already-inspected comparison, or promote
the challenger merely because it loses less in this window.

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

A bounded audit of 300 analyses from 15:18:07 to 17:33:33 UTC found 300
complete, decision-eligible analyses and no service errors. Its then-current
records all finished with `no_candidate`. A later read-only audit found that
the funnel's hold classifier also mislabeled a selected `paper_training_entry`
shadow as `no_candidate`; therefore the earlier result describes that earlier
window only and is not a complete classification of later decisions.

The later online example at 18:51:39 UTC on October 20 selected XRP short for
shadow training, with modeled net return `+0.026606%`, gross return
`+0.134172%`, execution cost `0.107566%`, objective net return `-0.862138%`
and loss probability `0.524228`. Its historical local-ML quality evidence
contained 265 samples with fee-after mean `-0.044642%`, lower bound
`-0.174464%`, and profit factor `0.922553`. The contract correctly remained
unauthorized because historical fee-after quality had not reached break-even.
This is a selected shadow signal, not an executable entry, and its small
positive current mean does not override the negative quality evidence.

The latest sampled online window (100 rows, 17:59:05-18:54:35 UTC on October
20) still contained only final `hold` actions. The audit exposed selected
shadow examples in that window as well. No result here indicates that a paper
order was opened. Ordinary paper execution does not universally require model
promotion or a positive per-trade lower bound, but the existing historical
quality contract can deliberately keep training candidates unauthorized.
Do not describe all holds as either "no candidate" or a universal ban on
ordinary paper orders.

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
9. A frozen, nested multi-horizon development diagnostic compared 5/15/60/240
   minutes, existing/market-only features, Ridge strengths 1/10/100 and net
   thresholds 0/0.05/0.1 percent. Two folds selected 240-minute research
   policies; the third selected no policy. Aggregate all-opportunity evidence
   was positive, but the single-global-position view selected only 15
   observations and had a negative naive lower bound. This remains an
   unqualified development result, not a deployable strategy.
10. The entry-funnel diagnostics now distinguish selected
    `paper_training_entry` and `paper_quality_observation` shadows from
    `no_candidate`. This is display/classification only and does not change
    authorization.

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
The multi-horizon result also fails its own promotion guard: one outer fold
selected no policy, and its nonoverlapping aggregate had only 15 selected
observations with a nonpositive naive lower bound.

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
