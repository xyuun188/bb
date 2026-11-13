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

The current supervision policy is
`2026-11-11.observed-shadow-exit-cost.v6`; v5 artifacts are incompatible.
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
qualified strategy's paper entry-to-settlement acceptance. Legacy shadow rows
still project costs from entry-time market facts. New native settlements use
separately recorded exit microstructure and current exit fees, scaled to exit
notional. These are counterfactual estimates, not actual fills or complete
portfolio replay. They cannot become decision-time features.

## November 11 Exit-Cost Repair

- New native settlements preserve exit bid/ask, depth, account fees, scheduled
  time, source/receipt clocks and fact identity as label-only evidence.
- Costs are computed separately for both sides. Sized estimates close the same
  base quantity on the exit book; they do not require closing liquidity on the
  earlier entry book. Counterfactual labels cannot assume a passive maker fill.
- Missing, late, unverified or mismatched exit evidence cannot silently fall
  back to an entry projection once the observed-exit contract is declared.
- Training supervision, settlement labels and specialist quality evaluations
  share the same cost implementation. Historical replay uses the entry-time
  projection for its entry gate and observed exit costs for its outcome.
- Training snapshot v8 makes settlement cost observations required metadata,
  ahead of optional diagnostics, in both Python and PostgreSQL. A bounded
  repair recovers observations dropped from older compact snapshots.
- Required contracts cannot be overwritten by the optional-metadata pass.
  Specialist evidence is admitted one group at a time, in the same order in
  Python and PostgreSQL, rather than losing both groups at the byte limit.
- Mixed nested/flat market contracts preserve label fingerprints, identity and
  availability clocks. An explicit nested null, false or zero takes precedence
  over a flat value. PostgreSQL also preserves the shape of flat compact
  contracts instead of adding null fields that Python never stored.
- Prediction replay strips settlement, label and specialist outcome metadata
  before single or batch inference.
- The label cost basis is explicit. Passive fill probability, unfilled orders,
  actual exits and concurrent portfolio PnL remain unaccepted; the code records
  `execution_replay_complete=false` instead of claiming those tests passed.

Before this deployment, the November 11 13:26 Beijing bounded audit still
showed zero entries, a current paper canary and no live permission. The latest
candidate had not replaced the champion. These are pre-deployment findings,
not validation of v6.

The deployed v6 settlement code was checked on twelve fresh samples covering
5, 15, 60 and 240 minutes: stored labels, training recalculation and specialist
returns agreed, exit observations survived compaction, and input seals were
unchanged. That audit exposed the compaction discrepancies above. The full
PostgreSQL/Python parity test passed using temporary functions, including full,
flat and mixed contracts, explicit null/false/zero and the single-group byte
budget; its transaction was rolled back without changing permanent functions.
Final deployment and model acceptance must still be checked after the repair.

The final storage-boundary repair also removes synthetic empty specialist
objects from PostgreSQL projections and applies Python's recursive 512-character
string bound. Missing results stay missing, observed `false` stays `false`, and
an explicitly observed empty result stays empty. Temporary-function PostgreSQL
tests cover these cases and verify that a second compaction changes nothing.
The pre-deployment 24-row source audit found no Python/updated-SQL differences,
but four recent stored projections still reflected the old function and three
older rows had lost an existing cost observation. Historical repair and fresh
post-deployment settlements must be counted separately.

A runtime audit also reproduced an unhandled exception in the finalizer of a
timed-out OKX SDK request when its detached worker later failed. The finalizer
now consumes that failure, closes the client once and releases its lock;
cancelling the finalizer still waits for worker completion before propagating
cancellation. Both failure paths have regression coverage.

Both v6 model branches were rebuilt on November 11. Local ML activated
`20261111T062814771915Z-d60502ab` as a paper canary; the independent quant API
loaded `20261111T061626933821Z-50e17452` in shadow. Live permission remained
false. The local-ML supervision report explicitly contains 6 observed-exit
samples and 4,927 legacy projections; rebuilding does not turn the historical
projections into observed fills.

At 14:31 Beijing, the canonical Go/No-Go audit still failed profitability:
short-side OOS mean was `+0.244700%`, profit factor `1.775850`, and return lower
bound `-0.043182%`; the long side remained negative. Eight fresh decisions had
complete analysis and the new contract, but none qualified for entry. This is
not an accepted profitable strategy or completed paper execution workflow.

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

## November 11 Post-Deployment Acceptance

The `ad74bee3` deployment completed at 15:52 Beijing. The priority bounded
repair then scanned 7,231 recent completed paper shadows, recompacted 5,070,
restored 4,735 existing settlement-cost observations, and preserved the
availability seals and raw source for all 5,070 updated rows. No settlement
outcome or order record was rewritten.

Twelve post-repair settlements across 5, 15, 60 and 240 minutes passed the
storage and label checks: Python and PostgreSQL projections matched, a second
compaction was idempotent, seals and observed exit costs were retained, and
training labels agreed with both persisted labels and specialist evaluation.
All were within the 128 KiB storage budget. This verifies data handling, not
profitability or actual fills.

Both training branches ran again using the repaired sample pool. The local-ML
challenger `20261111T081327725052Z-bca1c670` and quant challenger
`20261111T081704499285Z-f0138dc1` were rejected by their existing
non-degradation gates; the prior champion artifacts remain active. The current
canonical Go/No-Go check remains `no_go`: fee-after mean `-0.069554%`, profit
factor `0.898108`, and return lower confidence bound `-0.489961%`. The short
side alone has positive mean (`+0.244700%`) and PF (`1.775850`), but its lower
bound remains negative (`-0.043182%`), so it does not qualify.

At 16:25 Beijing, all three online services were active with zero restarts and
no post-deployment unhandled SDK-finalizer or service errors. In the latest
30-minute window, 144 decisions completed analysis, selected the authorized
240-minute horizon, and had complete funding projections; all remained HOLD.
The current blockers are negative fee/risk-adjusted direction evidence and
historical quality gates, not a stopped decision loop. Server-profit and
timeseries specialists expose a tail-loss event, not ordinary loss
probability; they remain observation-only rather than substituting one event
for the other.

The daily briefing now labels `paper_model_eligible` as model paper-call
eligibility, not an order-opening permission. Every decision still needs to
pass the separate current-return, cost, and risk gates.

Profitability acceptance is still open. There is no qualifying independent
exam, complete passive-fill and concurrent-portfolio replay, or successful
paper entry-to-settlement lifecycle. Do not respond to the absence of entries
by forcing orders or weakening the return gates.

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

## November 12 Entry And Exit Compatibility

The chronological partition repair (`808e9c61`) restored training without
relaxing label-availability isolation. The online automatic training at 14:42
Beijing activated paper canary `20261112T064153047017Z-f4b4e072`, using 398
training and 397 holdout groups after purging 1,074 groups. Live permission
remains closed. Model holdout performance is not realized trading profit.

The first complete post-repair PEPE lifecycle reconciled against OKX history
with fee-after realized PnL of `+0.065515 USDT`. This proves the entry, exit and
settlement path works; it does not establish aggregate profitability.

A subsequent exit review found a separate circular authorization defect.
`portfolio_exposure_pressure` is concentration multiplied by pressure derived
from the same position's loss and adverse move. Testing it for `> 0` as
independent loss-reduction evidence bypassed the existing stop-budget and
continuation requirements, even for an arbitrarily small concentration.
At 15:25 Beijing, 17 of 19 executed post-training exits matched this path:
negative current fee-after PnL, no hard stop, insufficient market loss evidence,
no model exit request and no qualifying replacement opportunity. This is a
bounded diagnosis of exit authorization, not a counterfactual profit estimate.

The exit policy now uses concentration only to scale an independently
authorized exit. Existing market-loss, model, replacement, stop and funding
conditions retain their authority; no entry condition or risk budget changes.
Policy provenance is `2026-11-12.independent-loss-exit-evidence.v20`.
Regression coverage includes both directions, tiny and large concentration,
repeat evaluation, market/model/hard-stop exits, and the combined paper-entry,
temporary-loss hold, fee-after profit exit and stop lifecycle. Entry tests
also verify that a current negative edge still rejects a new order.

Remaining profitability acceptance requires complete post-fix lifecycle
settlements and independent fee-after evidence for the actual execution
policy. Partial reductions are not finalized lifecycles; positive holdout or
a profitable isolated trade cannot authorize live trading.

## November 13 Strategy Replay Identity And Entry Cohort

Strategy replay previously checked only the model artifact identity. A prediction
from an obsolete score policy could still produce a complete replay. In addition,
observations that failed the current normal-entry return gate could enter strategy
candidate development and validation. That made strategy evidence describe a
different entry population from the execution policy.

The blueprint, replay cache and prediction fingerprint, development and exam
reports, persisted paper champion, and entry permission now bind both the score
policy and entry-cohort versions. Rejected entry observations remain available
only as diagnostics. A model blueprint with missing or invalid replay evidence
cannot fall back to legacy selector-matched returns. Background evaluation
rechecks the current artifact, strategy identity and evidence contract before
activating a champion. A legacy champion cannot regain authority through rollback.

The running trading service is `bb-paper-trading.service`; `bb-trader.service`
does not exist on this deployment. An inactive result for the latter is not
evidence that the trading loop stopped.

A bounded chronological diagnostic found an actual regime reversal. Training
contained 4,134 groups and the later test contained 397, with training labels
available before test decisions. Fee-after short returns changed from a training
mean of `+0.32661%` and PF `1.5705` to a test mean of `-1.20707%` and PF `0.1872`.
Frozen-model predictions correlated `-0.31581` with realized returns; all eleven
positive-objective selections were short and only one won. This is development
diagnosis, not an untouched profitability exam. Neither reversing the prediction
nor refitting the unchanged recipe establishes an executable edge.

The evidence-chain fixes do not establish profitability. Independent fee-after
validation and completed execution-policy lifecycles remain required; failed
evidence cannot be relabeled as a reason to authorize an order.

Online acceptance of the new replay completed at `2026-11-13T14:54:35Z`.
Of 1,641 replayed direction observations, 1,591 failed the current entry-return
gate and are now diagnostic only. The executable cohort contains 50 entries:
one development observation and 49 exam observations, including six forward
observations. Three strategy candidates remain; none passes validation. These
counts explain why the old 136-candidate report overstated strategy coverage.

A recent ZRO rejection was traced to the size-aware cost pass: estimated cost
rose from `0.109064%` to `0.630496%` and the final return lower bound became
`-0.66591305%`. Its rejection is a current profitability failure. A separate
final-risk check still used the permissive sizing validator, which omitted the
strict entry lower-bound check. It now uses the same entry validator as order
submission. Regression tests reproduce the former acceptance of a fingerprinted
zero or negative bound after otherwise valid sizing and require rejection.
Reconciliation also retires a persisted champion from an obsolete score or entry
contract before considering current validated candidates. This prevents stale
metrics or a rollback from deferring an otherwise eligible current candidate to
the next evaluator cycle; it grants no permission when current validation fails.

## November 13 Indicator Period Identity

A frozen 4,988-group development window contains 3,056 short-return snapshots
from one-minute candles and 1,932 from five-minute candles. Both used the same
`returns_1`, `returns_5`, `returns_20`, and volatility columns, while the model
discarded their observed candle period. A 20-minute return and a 100-minute
return could therefore produce identical model inputs. The trend, volume and
entry-activity periods were also omitted.

The feature encoder now preserves all four observed periods as explicit minute
features in training, single-row inference and batch replay. Missing or unknown
periods remain zero (unknown); they are never guessed from current settings or
the prediction horizon. The feature contract and immutable development snapshot
schemas change together, so old artifacts and frozen frames cannot silently use
the new input schema. Existing sealed market snapshots are not rewritten.

The regression first reproduced the collision, then verified distinct inputs,
known and unknown intervals, frozen training rows and live encoder parity. This
fix addresses an input ambiguity, not proven profitability. On the same purged
development folds, adding period context alone still selected 11 shorts with
negative fee-after mean; it is not a profitable challenger.

A separate preregistered four-arm diagnostic compared the unchanged recipe,
price-unit normalization, ATR-normalized targets, and both changes. The combined
arm improved aggregate fee-after opportunity mean to `+0.12011%` (PF `1.16771`),
but its return LCB was `-0.03188%`. Under the existing three-position capacity
limit it produced 12 observations with mean `+1.00614%`, PF `3.11860`, and LCB
`-0.12786%`; the first chronological fold still lost money. All four arms failed
qualification. These are inspected development counterfactuals, not executed
PnL or an untouched exam. Neither normalization change is enabled in production,
and no failed result authorizes a trade or an increase in exposure.

The same missing period identity also affected the standalone model server.
Its training projection discarded the four period strings before transport,
and its separate feature encoder omitted them. Both are repaired together with
the local ML encoder. An end-to-end regression verifies that compaction,
transport, standalone training and live inputs preserve the same period values;
legacy remote probability artifacts require rebuilding under the new contract.

A second registered diagnostic tested direction-reflected fitting to remove a
historical short intercept. Reflected rows had half weight and were used only
for fitting; all validation outcomes were original observations. Costs, tail
classifiers, risk rules and capacity limits remained in force. The symmetric
candidate selected four longs and one short, but only two capacity observations
and a negative return LCB remained. This demonstrates that removing directional
bias alone does not establish a profitable executable strategy. It is not enabled.

## November 14 Rebuild And Acceptance Follow-Through

The period-contract rebuild completed on both deployed training paths. Local ML
artifact `20261113T155824628907Z-76761f83` entered paper canary with no live
permission. Remote quant artifact `20261113T155529804163Z-e39f9e55` is available
in shadow. Both report the explicit indicator-period feature contract. The two
automatic training timers were restored after the controlled rebuilds.

The remote status endpoint previously preferred the challenger pointer over the
current artifact regardless of training time. After a contract migration this
could report no latest training even while the verified current model was new.
It now chooses the newest verified current, candidate or challenger by parsed
training time. Invalid pointers remain invalid, and reporting does not change
model activation or execution permissions.

The runtime text audit also failed on large strategy snapshots. Twenty recent
strategy events contained about 64 million JSON characters; a single 200-row
projection could exceed the shared database statement timeout. The audit now
reads at most five payload rows per query, releases the read transaction before
inspection, scans off the async event loop, and merges only bounded counts and
examples. ASCII-only leaves bypass the non-ASCII mojibake detector. It still
inspects all selected records and fields; it never rewrites historical text.
An isolated twenty-row scan passed, but a fresh integrated run reproduced the
shared 12-second statement timeout. Five-row batches alone then hit the section
deadline during JSON decoding and traversal. The final implementation reads JSON
as serialized text and applies a conservative marker prefilter in the worker
thread. Any possible marker, including an escaped Unicode marker, still causes
full decoding and per-leaf inspection. Clean serialized documents need no object
graph. Regression checks cover every detector marker, Unicode escape casing,
false positives in keys, and equality with the original per-leaf report. The
scan coverage and all database and section timeouts remain unchanged.

The same integrated acceptance exposed full-payload reads in the independent
high-risk and shadow missed-opportunity audits. High-risk review now fixes its
original scalar window before projecting only entry review keys. Missed-opportunity
review retains its original windows and only fetches feature payloads for completed
misses and original decision contracts for executed entries, five rows at a time.
SQLite integration regressions verify review violations, unchanged fee-after
observations, excluded old rows and the narrow payload queries.

Final integrated report `final-acceptance-v4-20261114/latest.json`, generated at
`2026-11-13T16:43:54Z`, completed in 94.77 seconds within the 180-second total
budget. All three corrected audit cards passed, with no deferred audit sections.
Text inspection covered 711 records with zero suspected defects; high-risk review
and missed-opportunity review completed in 0.36 and 0.69 seconds respectively.
The full suite passed 5,907 tests with one skip before the final two audit
projections; all final edits passed 395 focused regressions, Ruff and compilation.

The exact rebuilt local model replay completed at `2026-11-13T16:13:23Z`.
It retained 17 qualifying entry observations, all in the strategy exam and none
in strategy development. Another 1,553 observations failed the current entry
return gate, and 91 were outside governed sides. Replay is complete, but there
are zero validated strategy candidates. This is a failed profitability
qualification, not an unfinished replay or a reason to force an entry.

Profitability acceptance still fails. The final report records 99 settled
positions with net PnL `-17.79576402 USDT` in the current system execution window,
not solely the rebuilt model's trades. The current-model holdout entry cohort
has 23 short observations, mean `+0.577111%`, PF `1.60354`, but return LCB
`-0.400948%` and no long observations. Recent decisions use the rebuilt artifact
with complete required market inputs, yet fail their current fee-after return
and loss-risk conditions. Neither the positive holdout mean nor corrected audit
availability establishes a profitable strategy or grants live permission.

## November 14 Entry Veto And Fresh Audit Corrections

A fresh decision timeline showed no executed market entries after the November
13 09:00 UTC hour, before the period-contract rebuild and latest audit deployment.
The automatic training scheduler recovered to `skipped/not_due` with no error;
the prior interrupted state did not remain stuck. Required current market inputs
were complete. The latest model still frequently produces nonpositive fee-after
risk objectives; availability and missing historical permission alone do not
explain all holds.

An additional reproducible veto existed in direction-family aggregation. The
current-direction check retained positive raw-return families even when their
risk-adjusted fee-after objective failed the final entry gate. A negative
objective family could therefore cancel another family's complete positive
entry, despite being unable to authorize an entry by itself. Direction support
v18 applies the existing positive mean/objective requirement to the participating
entry families before combining them. All independently qualifying families
remain in the combination; rejected families remain in the full diagnostics.
When none qualify, the original observation summary and rejection remain.
Historical permission, loss probability, opposing evidence, costs, horizon and
the final order authority continue to apply.

The frozen decision `801409` at `2026-11-13T13:59:58Z` reproduced the defect with
its original costs, model outputs and expert opinions: the old combined objective
was `-0.132542%`; the valid local-ML family alone had `+0.289312%`, expected net
`+1.368336%` and loss probability `0.089501`. The corrected calculation authorized
only the existing bounded paper-training contract. Three other frozen decisions
tested alongside it remained holds. These are counterfactual contract checks,
not submitted orders, realized returns or new profitability qualification.

Profiling the fresh artifact audit found 1,051 file entries, 1,056 JSON decodes
and 350 binary hash checks taking 63.54 seconds on the async event loop. It now
runs filesystem and hashing work in a worker thread, shares a per-scan metadata
reader, and caches only classification/identity fields. It still scans every
artifact, checks every required hash and rereads all metadata on the next audit.
Regression tests verify that tampering between scans remains blocked and that
runtime probes can proceed while a file scan is in progress.

The follow-up fresh scan exposed a separate training-count timeout. PostgreSQL
was scanning completed shadow rows and decoding their feature JSON to find the
native crypto category. A partial index now covers the existing completed,
two-return, crypto-only predicate and includes the decision ID. The online
maintenance command creates it concurrently and retains invalid-build recovery;
startup schema setup also registers it for new databases. No sample eligibility,
training epoch or return gate changed. The online index was verified ready and
valid; EXPLAIN used it and completed in 60.452 ms, and the unchanged training
counter returned 41,591 samples in 0.0326 seconds.

The complete regression suite passed 5,915 tests with one skip. After the index
registration change, 88 focused database, maintenance and collection-status
tests passed. Two forced scans under an artificial 70% CPU quota exhausted
their total budget and are not clean acceptance results. The final scan used
the normal dashboard resource conditions, with Nice=10 and the unchanged
180-second budget. Report
`native-audit-acceptance-20261113-180232/latest.json`, generated at
`2026-11-13T18:04:52Z`, completed all required fresh sections in 135.6231 seconds
with no deferred or missing sections. Training counts were known, with no
warnings: 41,723 shadow rows and 238 trade rows. Of 27 cards, 21 were OK, four
were warnings and two were critical. Both critical cards concern failed
profitability/promotion qualification; collection success does not pass them.

The v18 entry change also produced a real bounded paper fill. AVAX/USDT short
decision `802082` at `2026-11-13T17:46:41Z` used local model
`20261113T155824628907Z-76761f83`. Order `18713` filled 1.8 AVAX at 8.366,
with OKX order `4096649205102915585` confirmed at `17:47:40Z`. Its final
fee-after objective was `+0.0219984%`; the ineligible remote family remained
diagnostic. Production permission stayed false. Position `14461` subsequently
had a 0.2 AVAX partial exit, represented by fragment `14462`; the initial
settling fragment is not authoritative settled profit. The system execution
window still had 96 settled positions and `-25.13324202 USDT`, which must not
be attributed entirely to the current model. Profitability remains unqualified.

## November 14 Observation-Only Audit Classification

The final fresh scan reported two rejected entries as 24 execution-contract
defect categories. Frozen decisions `801669` and `801683` establish the actual
stage: both had `market_analysis_only=true`, a final observation contract with
entry permission false, denied top-level entry permission, no sizing contract,
and no execution. Their final reason was the shared market-observation veto;
their candidate reason was `okx_private_entry_instrument_probe_deferred`. Both
also failed the current return objective. These were observation records, not
orders submitted with missing leverage, cost or account-risk contracts.

Root-cause auditing now counts this narrow, fully evidenced final observation
veto separately, preserving its candidate-reason counts. It continues complete
execution-contract validation for executed or promoted records, conflicting
permissions, existing sizing, missing evidence and different final reasons.
Neither entry authorization nor profitability qualification changes.

Commit `2e5472be` passed 211 focused regressions and repository-wide Ruff, was
pushed to main and deployed with all 531 source hashes verified. Online replay
of the same three frozen records now reports two observation vetoes and one
fully validated executed entry, with zero execution-contract gaps. The fresh
500-decision root-cause report also passed. Complete fresh report
`observation-audit-acceptance-20261113-1831/latest.json` at
`2026-11-13T18:32:39Z` finished all sections in 134.0083 seconds within the
unchanged 180-second budget: 23 OK cards, two observation warnings and two
profitability/promotion critical cards. There were no execution-contract
violations across 85 executed entries and 286 exits in its historical window.
The refreshed system window contains 94 settled positions, all shorts, with
net `-37.54282267 USDT`; this is not exact attribution to the rebuilt model.
The model's independent positive-objective cohort remains 23 shorts, PF
`1.60354`, mean `+0.577111%`, and LCB `-0.400948%`. Live qualification still
fails. Both auto-train timers and paper/dashboard/model-tunnel services were
healthy after deployment; completed temporary acceptance units were recycled.
