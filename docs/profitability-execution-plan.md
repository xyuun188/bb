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

## November 14 用户授权的纸盘准入调整

用户明确选择“放宽盈利/风险门槛”和“直接修复并部署”。本次限定在已有的
paper-training 合约中调整准入，不开启实盘，也不修改训练标签、模型晋级条件
或独立检验结果。

线上决策 `804049` 复现了阻断：费后均值 `+0.2665481857%`，风险调整目标
`-0.8251118651%`，亏损概率 `0.549535456`。原实现仍要求受限纸盘训练单的
风险调整目标大于零。因此即使方向和费后均值合格，也无法收集实际成交样本。
这条要求是既有策略限制，本次根据用户授权调整，不将其描述为已经证明的
收益预测计算错误。

调整后，费后均值为正、下行估计完整且有限、亏损概率不超过 `0.60`、方向
证据完整、无强反对证据的候选，可以使用受限纸盘训练合约。负的风险调整值
继续进入压力损失计算，不能改写为正数。普通纸盘交易仍要求正的风险调整
目标与历史盈利证据。最终执行成本复核、下单前合约校验和最终风险检查采用
一致的训练准入规则。

单笔训练压力风险上限仍为权益的 `0.0001`，组合上限仍为 `0.0003`；保护单、
交易所最小订单、流动性、账本和权限检查继续适用。缺少成本、非正费后均值、
非有限的下行估计、超限亏损概率仍拒绝。日报和漏斗将明确的盈利门槛拒绝
分类为 `profit_gate_blocked`，真实执行失败与风险拒绝优先，避免继续显示为
笼统的“没有方向”。

新增只读脚本 `scripts/audit_online_paper_entry_policy.py`，对线上有界历史快照
使用本地规则重算候选与合约。脚本仅删除旧选择器追加的风险目标拒绝原因，
保留上游阻断；不会提交订单、修改历史记录或授予实盘权限。最近 128 条记录
中原选中 0 条，新规则下 95 条通过候选与合约校验，33 条仍被拒绝。95 条全为
short，这不是 95 笔成交，也不是盈利验收；还需经过当前订单成本、组合预算
及实际成交检查。该窗口已用于政策核对，不能再称为独立盈利检验集。

执行策略公开投影与 Phase 3 验收同样按新规则区分受限训练单和普通交易，
不再要求训练单的下行目标为正。验收仍校验有限下行压力、正的费后均值、
固定训练风险预算及禁止实盘权限；历史观察记录也不会因此取得下单权限。

此前启动的无持久化训练预检已在 `2026-11-14T02:36:00Z` 正常完成，无遗留
临时训练单元。预检成功不等同于模型盈利或晋级通过。

## November 14 成交验收与超时回写修复

准入调整 `4830ff19` 已推送并部署，532 个源码哈希校验通过，运行导入通过，
paper、dashboard、model tunnels 与 nginx 均正常运行。该版本全量回归为
5,967 项通过、1 项跳过，仓库 Ruff 通过。

截至 `2026-11-14T03:09:53Z`，新策略窗口实际新增 SHIB、SOL、ETHFI、JUP
四笔纸盘空单，订单均为 `filled`、`okx_confirmed`。窗口执行合约验收覆盖完整：
4 笔开仓合约齐全、成交同步待定 0、违规 0。四个当前仓位的交易所原生 OCO
均处于 live，止盈止损覆盖完整、无超额覆盖。ETHFI 的 OCO 在成交时已存在，
此前本地管理合约短暂未同步，不能据此认定交易所没有保护单。

继续验收发现 ETHFI 决策 `804195` 的真实回写缺口：订单 `18772` 已获交易所
确认，但决策缺失 execution_result，并保留“150 秒交接超时、结果未知”的
陈旧说明。日志证明交易所成交先于外层超时；原执行路径将完整成交回报放在
本地持仓同步之后保存，外层取消可能跳过最后一步。原生对账对已存在且数值
匹配的订单直接跳过，也没有恢复原决策。

本次在交易所确认后、本地慢同步前先保存成交回报；原生对账对严格匹配的
纸盘订单恢复决策成交回报、成交时间、状态和终态说明。恢复要求完整的原生
成交、已验证的合约规格、精确客户端订单身份、币种和方向一致；不根据本地
布尔状态伪造成交，不重复开仓或改动持仓，也不把开仓 fillPnl 当作已实现盈利。
回归覆盖同步中断、已有匹配订单、幂等恢复及身份/方向/币种/完整性不符拒绝。

四笔新仓尚未结算。已有独立正目标模型样本仍为 23 笔空单，PF `1.60353955`、
均值 `+0.57711117%`、下界 `-0.40094780%`；方向集中与盈利置信下界不足仍然
存在。当前允许受限纸盘收集执行样本，不代表稳定盈利或实盘晋级已经通过。

成交回写修复已以 `6fec329f` 推送并部署，532 个源码哈希和运行导入通过；
全量回归 5,975 项通过、1 项跳过。`2026-11-14T04:22:44Z` 复查确认，
SOL `804192`、ETHFI `804195` 已由新版后台原生对账自动恢复成交回报；
两笔决策的原始记录与学习快照均为交易所确认，原超时原因已清除，成交时间
取自原生成交。四笔已核实订单的保护回报仍完整，四项线上服务正常运行。

后续验收又定位到影子错失机会复盘的独立误报：PEPE 决策 `804387` 成交
名义金额 `4.24 USDT`，低于下单前价格折算的最小名义金额 `4.2444 USDT`，
但实际成交 `0.1` 张符合规格，已有完整原生成交和有效下单合约。主执行合约
验收已能使用这些证据，影子复盘却仅验证原始决策，没有传入实际成交证据。
本次将该复盘接入同一执行合约验收，并按精确决策关联读取订单；未验证的
合约规格仍报缺口。追加回归覆盖完整和未验证成交，96 项定向测试通过。

`2026-11-14T04:22Z` 的最近 30 分钟窗口已有 5 笔已执行开仓、9 笔已执行
平仓决策，持续交易已恢复。`04:21Z` 权威历史窗口累计 57 个已结算仓，净
`-20.1205365 USDT`，这是当前系统窗口累计结果，不能归为新策略独立收益。
模型独立检验样本和盈利下界仍未通过，不能把回写修复或交易恢复等同于
稳定盈利验收通过。

复盘修复 `d2443224` 已推送并部署，532 个源码哈希和运行导入通过，仓库
Ruff 通过。部署后实际复验：影子复盘执行缺口从 1 变为 0；最近两小时主
执行合约覆盖完整，22 笔已执行开仓、28 笔已执行平仓，成交同步待定 0、
合约违规 0。该窗口 14 个权威已结算仓净 `+0.56284546 USDT`，其中 2 个
亏损；窗口包含多个修复版本，不能作为当前版本独立检验或稳定盈利证明。

`2026-11-14T04:28:53Z` 直接读取 OKX 纸盘原生仓位、挂单、保护单和规格：
13 个当前持仓对应 13 个 live OCO，待成交订单 0，缺失保护、孤立保护单、
数量不匹配和无效保护单均为 0。最近六小时日报确认实际开仓 22 次、完成
影子结算 3,139 条，运行模型仍是 canary，实盘权限关闭。当前方向仍集中
做空，独立模型盈利置信下界不足仍需后续训练检验，不能以放宽纸盘准入
代替盈利证明。

## November 14 小单预算与漏记成交恢复

小单的直接原因是纸盘训练政策：单笔压力损失仅允许权益的 `0.0001`，
训练组合仅允许 `0.0003`。权益约 4,439 USDT 时，单笔预算约 0.44 USDT，
按实际下行压力折算后，订单名义金额通常只有 5 至 17 USDT。这是原有
风险政策的实际结果，不是合约张数或保证金换算错误。

按用户已授权的纸盘门槛调整，新合同升为 v23，动态仓位升为 v12。
纸盘训练单与普通纸盘共用 `0.005` 单笔压力风险上限，删除训练专用
组合微仓限额。相同账户、盘口、预测和下行估计现在产生相同预算；
最终金额仍取决于实时成本、盘口、原生价格区间、动态杠杆、可用保证金
及模型显式上限。负的下行估计继续增加压力损失，不能改写为正数。
该调整提高了纸盘单笔可承担损失，不证明模型已有盈利优势。旧合同仍按
原单笔和组合上限结算验收，不能用新政策追认历史超限。

同时核实了另一项独立开仓阻断：BTC 决策 `805511` 已于
`2026-11-14T08:11:32.838Z` 成交，订单 `4098388415837089793`，客户端
身份 `BBNP805511`；本地未落库，决策还被标为未执行。保存的成交回执
存在，但优先同步只从本地订单找目标，完全缺少本地订单时会漏过。
已用既有原生成交恢复流程补回订单和决策，后台随后自动创建精确关联
持仓 `14583`，没有复用昨天已关闭的 BTC 生命周期。

自动同步现增加有界的未落库成交回执发现，按回执中的订单 ID 重新拉取
交易所原生成交，沿用客户端身份、币种、方向、成交完整性和公开合约
规格核验后才恢复事实。回执只用于发现查询目标，不能直接充当成交证据。
回归覆盖无本地订单的自动恢复、幂等、未确认和拒绝回执不得伪造成交。

`2026-11-14T09:56:30Z` 重新生成的日对账：严重项 0、未解决项 0，
`can_open_new_entries=true`、`can_refresh_training=true`。窗口仍有方向
集中观察项及费后亏损，不能把对账恢复或预算提高称为盈利闭环完成。

预算与漏记成交修复 `8836a70e` 已推送并部署，532 个运行源码哈希和
运行导入校验通过，全量测试 5,983 项通过、1 项跳过。新版已实际成交：
NEAR 决策 `805950`，225 NEAR，成交名义金额 `750.383 USDT`，入场保证金
`24.20590323 USDT`；HMSTR 决策 `805952`，1,000,000 HMSTR，实际成交名义
金额 `142.48778 USDT`，入场保证金 `23.74796333 USDT`。后者经过执行上限
缩量，不能把提交前估算的约 526 USDT 当作实际成交。两笔均已核实原生
仓位和全量 live/reduce-only OCO 覆盖；旧有小仓不会自动补大。

继续验收发现后台原生成交同步反复超时：优先拉取只涉及最多四笔订单，
落库前却读取全部可写历史订单的完整决策，连保护单历史也包括在内。
线上最近 500 笔订单中有 85 个这样的决策；单独抽取的 18 个完整决策
JSON 已合计约 46.7 MB。优先落库现先按目标订单筛选，再读取对应决策；
保留无交易所订单 ID 的拒单提交恢复，继续严格核实完整原生成交与身份。
新增回归验证只读取目标决策、纠正真实成交且不改动无关记录，对账相关
98 项测试及全仓 Ruff 通过。

`2026-11-14T10:21:14Z` 日对账确认开仓和训练门禁均开放，严重项及人工
处理项为零。Phase 3 仍未通过盈利验收：模型独立正目标样本 23 笔，PF
`1.60354`，平均费后收益 `+0.577111%`，置信下界 `-0.400948%`；运行模型
仍为 canary，实盘未放行。扩大纸盘预算和修复成交回写均不能替代该检验。

优先查询修复 `6fe6caa0` 已推送部署，532 个源码哈希和运行导入通过。
`2026-11-14T10:39:12Z` 原生事实恢复运行成功：无未验证成交、无待重试
目标、无延迟阶段。NEAR 已由后台补成已执行；HMSTR 成交和执行说明也已
恢复。继续核对状态机发现旧提交恢复逻辑把所有成交统一标成平仓进度，
并未关闭超时阶段。现按真实决策动作区分开仓和平仓，保留保护单回执，
补齐成交确认终态；已有确认回执但仍为未知阶段或误标平仓的开仓也允许
通过严格的原生成交核验纠正。回归覆盖开仓、平仓、保护回执保留、陈旧
状态恢复和幂等。

状态修复 `618e398c` 已推送部署，532 个源码哈希及运行导入通过，模型
通道、纸盘、Dashboard 和 Nginx 均运行正常。最新对账回归 101 项及相邻
状态机、成交仓库和离场关联回归 14 项通过，全仓 Ruff、编译及差异检查
通过。`2026-11-14T10:48:23Z` 精确原生成交恢复返回 OK，未验证成交、
延迟阶段和错误均为零；NEAR `805950` 和 HMSTR `805952` 均为已执行、
原生成交已确认、开仓标识正确、成交确认阶段完成，订单均为
`okx_confirmed`。`10:48:31Z` 新日对账严重项 0、人工处理项 0，开仓和
训练门禁均开放。以上为交易链路验收，不改变独立模型盈利尚未通过的结论。

## November 14 巡检超时与页面误报

用户确认当前纸盘开仓和当日收益改善后，定位截图中两项执行审计不可用：
SQL statement timeout，另有持仓经济性审计因整批读取原始决策 JSON 超时。
只读执行审计现按 50 条读取，只有入场决策提取完整的执行调和历史，保留
提交指纹验证；平仓不再读取无关的原始入场字段。持仓审计优先读取已有
学习快照，旧记录没有快照时继续读取原始证据，不修改交易规则或订单。

在线以未部署候选代码对比：386 条执行决策审计，78 个已执行开仓、201 个
已执行平仓，违规为 0；新旧逐笔合约结果相同。默认读取耗时从 5.251 秒
降至 2.857 秒。持仓审计默认查询原来 12.908 秒超时，候选 0.745 秒完成，
覆盖完整，持仓经济性缺口和动态退出缺口均为 0；另用 25 条分页读取原始
证据完成独立对照，305 条平仓决策的结果没有差异。

中文原因映射补齐 average_net_return_after_all_cost_not_positive，未知
诊断码不再一律伪装成未确认结算。巡检读取失败的回退文案改为中文，并
明确不等同于已确认违规。收益曲线保留原有 OKX 权益百分比基准和全部
历史亏损，仅将横轴从时分改为北京时间日期加时分，避免每日午夜快照
看起来重复；更新静态资源版本。模型晋升仍按真实费后证据判断，历史
权益下降与今日交易盈利是不同时间和统计口径，不能互相替代。

修复 `3395e9f6` 已推送并部署，532 个运行源码哈希及运行导入检查通过，
纸盘、Dashboard、Nginx、模型通道与 OKX 网络均正常。相关 361 项回归、
全仓 Ruff、JavaScript 语法及差异检查通过。Chromium 实际运行页面脚本
确认盈利诊断正确显示为平均费后收益不为正，未知码不再伪报结算，跨日
标签显示 `11/13 00:10`、`11/14 00:10`，曲线数值保持不变。

`2026-11-14T15:31:13Z` 全量新巡检未修复从 4 降至 0，正常 21、观察 6。
执行审计覆盖完整，78 个已执行开仓、203 个已执行平仓，执行违规与待定
成交均为 0；闭环审计和持仓审计正常，持仓经济性和动态退出缺口为 0。
这些是最近 24 小时窗口，不表示部署后新增 78 个开仓。当前窗口权威已
结算净收益 `+65.86474105 USDT`，同样不能代替当前版本的独立盈利检验。
`15:31:22Z` 新日对账严重项 0、人工处理项 0，允许开仓及训练。剩余
Phase 3 阻断仅为 `local_ml_not_live_ready`、`model_return_lcb_not_positive`；
做空集中和历史单证据族授权仍明确保留为观察项，没有为巡检变绿而改动
交易条件、仓位预算、模型或止盈止损，也未放行实盘模型晋升。
## November 15 磁盘耗尽导致分析和 OKX 同步异常

北京时间 2026-11-15 02:26:29，PostgreSQL 因根盘无剩余空间，无法写入 WAL，
停止运行；分析、OKX 事实落库和页面查询随后发生连带错误。数据库文件共约
198 GiB，其中影子回测、决策和策略事件的大 JSON 占据主要空间。旧维护每天
每表最多处理 5,000 条，11 月 13 日报告仍有约 55.5 万条待处理，不能追上积压。

已先回收旧系统日志约 3.4 GiB，恢复 PostgreSQL；确认没有进程使用后删除
8 月遗留验收环境的独立虚拟环境，并删除 7 月已被较新完整备份替代的迁移前
快照，保留 8 月备份、当前数据库、订单和模型产物。空间由 0 恢复至约 14.6 GiB。
恢复后 OKX 行情请求成功，自动同步状态为 ok，分析继续运行。

维护定时器改为每小时分批处理，保留每表 5,000 条和全部既有证据保护条件；
不修改交易条件、仓位预算、模型和止盈止损。安装器同时限制持久系统日志
512 MiB、运行日志 128 MiB，并尽量给其他写入保留 4 GiB。页面自检增加
不依赖数据库的磁盘检查，低于 10% 或 10 GiB 时提示关注，低于 5% 或
5 GiB 时提示异常。磁盘恢复后仍需清理数据维护积压，不能把进程恢复等同于
全部历史数据已经压缩完成。

首轮新的有界维护已成功处理 15,000 条，未删除交易行。完整日对账恢复后允许
开仓、允许更新训练，需人工处理项为 0。另在验收中发现独立的 BCH 保护单
缺口：2026-11-14T18:49Z 同时读取公开 last=206.45、模拟盘 last=212.5，
旧保护单参考价错误复用了公开行情，因此短仓止损反复被 OKX 以 51278 拒绝。
保护单校验现单独使用执行环境行情；普通分析和开仓取价保持原路径。缺失保护
只有新增单时直接使用重新校验后的补单计划，不再进入依赖旧 OCO 的替换分支。
没有改变再锚定距离策略、止盈止损百分比、模型或仓位预算。定向回归 188 项及
全仓 Ruff 通过；其中验证两个报价可以不同、执行报价失败不会静默使用公开价，
以及没有旧保护单时补单不会执行无关撤单或改变持仓数量。

修复 `6d17b292` 和 `4580c0a2` 已推送并部署，532 个运行源码哈希、导入、
模型通道、OKX 网络和 Dashboard 启动检查通过。2026-11-14T18:54:00Z 后台
自动补回 BCH OCO，18:55:25Z 独立重读确认保护单 `4099681598042779649`
为 live/reduce-only，24.61 张恰好覆盖当前 24.61 张短仓，无缺失、超量、孤单或
非法保护单；持仓指纹与补单前一致。保护价 215.1/210.7 来自既有再锚定规则，
此次仅纠正该规则使用的报价环境。市场循环无错误、OKX 同步为 ok，失败服务为 0。
剩余约 14.4 GiB，磁盘仍显示空间偏紧；首轮压缩后的历史积压未全部消失，
由新的每小时任务继续处理，不能宣称整个系统已达到稳定盈利或实盘晋升条件。

## November 15 XPL 秒级止损与超时开仓清理

用户反馈的单笔百余 USDT 亏损对应 OKX 模拟盘 XPL 空仓，权威历史记录 2903，
入场决策 808102。北京时间 2026-11-15 02:16:40.897 创建开仓订单，直到
02:19:46.648 才成交 805 张（8,050 XPL），均价 0.0453。止损 0.0466，
止盈 0.0443；OCO 4099612944903155718 的 actualSide=sl，02:19:47.181
触发，02:19:47.184 完成买回，均价 0.0608529192546584，共 81 笔成交，
价格范围 0.0545 至 0.0628。费后亏损 125.6282655 USDT，持仓仅 0.536 秒。
已确认是止损成交和巨大滑点，不能归因于模型持续忽略已经亏损很久的持仓；
现有记录不足以重建触发瞬间完整盘口，不能把滑点的交易所成因编造成定论。

同单日志证明：02:16:26 外层交接超时，交易所执行继续运行，02:17:03 本地
执行取消，但交易所订单仍在，约 186 秒后才迟到成交。修复取消保护分支里的
二次等待超时未清理子任务问题；执行器在开仓提交被取消时，按准确客户端订单
身份读取本单，校验合约和订单身份后撤销未成交部分，并对已确认部分成交补验
原生保护。确认市价开仓后也终结零成交的挂起余单，不再仅清理部分成交余单。
不改变既定仓位、杠杆、止损百分比、模型或盈利门槛，不主动平掉已成交仓位。
取消无法确认时明确保留不确定证据，不能宣称远端一定撤单成功。

同时纠正当前持仓保护契约用历史入场价判断保护方向的问题：移入盈利区的
止损可按当前原生保护单执行报价和各自触发类型校验；持仓契约验收使用同一
验证逻辑，真实错误方向仍拒绝。BCH 模拟盘 last 和公开估值不同的回归已覆盖。
相关执行、持仓保护、挂单、执行契约与流水线 290 项回归通过，全仓 Ruff 通过。

## November 17 近期训练窗口与方向比较修复

`e70d0cb9` 已部署：开仓限价同时受信号收益预算、既有止盈止损和交易所
价格带约束，价格带拒绝与信号收益预算耗尽使用不同原因。影子候选在普通
入场比较样本为零时，只有共同未拟合样本的六项预测误差全部严格改善，才
允许更新影子预测；该路径不授予普通入场或实盘权限。

`26418b8f` 已推送并部署，533 个运行源码哈希和导入校验通过。训练采样
改为先取最近 4,096 条原始合格范围 ID，再按稳定优先级选择最多 512 条；
质量排除后不向更早历史补样。保留 240 分钟预测、50% 时序检验集和完整
标签隔离。原来先从全部历史按 ID 取样，导致训练仍使用 11 月 14 日标签。

正式训练于 `2026-11-16T20:05:17Z` 完成，新影子模型为
`20261116T200517068863Z-a59fb25a`，训练 188 条、检验 256 条，拟合标签
截止推进至 `2026-11-16T05:44:45.909337Z`。共同未拟合比较 471 条，六项
误差均改善；普通入场比较样本仍为零，`live_ml_ready=False`。排序子集的
正收益只作开发诊断，不能替代独立交易盈利证据。

手动训练旧入口遗漏新版训练传输报告且在 Dashboard 内无界读取历史。
现与自动训练共用隔离子进程，统一时序训练、近期采样、资源限制、产物
治理和结构化失败结果；超时或取消会终止并回收本次子进程。

实际决策 `830384` 暴露另一处错误阻断：server_profit 的 long 收益分布
完整，但其普通亏损概率未通过独立 Brier 检验。旧逻辑连反方向收益基线
也一起丢弃，令证据完整且收益更优的 short 被误报方向冲突。现仅保留
通过分布校验的反向比较资格，缺失概率侧仍无决策、纸盘或实盘开仓权限；
蓝图拒绝不会覆盖已有比较资格，不同收益契约仍拒绝混用。方向支持版本
升级为 v20。线上原始特征加真实预测 API 的只读候选回放确认冲突消失，
未伪造缺失概率。相关 570 项测试和全仓 Ruff 通过。

`2026-11-16T20:23:31Z` 原生检查为 18 个仓位、17 个保护单和 1 个待成交
退出单，无孤立保护单、覆盖数量错误或非法保护单。唯一缺口为 IOTA
short 残仓，原 reduce-only 买回单 `4104963763034894337` 仍 live，成交
为零；模拟盘最低卖价 0.03307，高于交易所允许买价 0.03252。保留原订单
和真实未完成状态，不能用重复下单、删除仓位或伪报成交消除该外部阻断。

方向比较与共享训练入口修复 `6a35ae24` 已推送并完整部署，533 个运行源码
哈希及导入通过，纸盘、Dashboard、Nginx、模型通道和 OKX 网络正常。全量
测试 6,315 项通过、1 项跳过，全仓 Ruff 通过。已部署 v20 对决策 `830384`
的真实预测重放确认：short 不再误报冲突，long 仍因费后预测收益为负而被
拒绝，server_profit 缺失概率侧只有比较资格。

部署后做多决策 `830698`（ETHFI）已通过方向比较和纸盘训练授权，因最近
真实平仓后的同币种冷却期被拦；`830694`（JCT）已选中 long，但属于仅观察
币种。不能把这些记录说成系统写死做空，也不能把选中信号说成实际成交。
部署后至 `2026-11-16T20:43:15Z` 检查的 33 条开仓方向决策尚无新增成交。

最终 `2026-11-16T20:42:40Z` 巡检仅剩真实结算收益不为正、local_ml 未达到
实盘晋升条件两项阻断；当前交易合约违规、持仓经济性缺口和已执行动态
离场合约缺口均为零。`20:40:25Z` 原生检查为 16 个仓位、16 个保护单、
待成交单零，缺失、孤立、覆盖数量错误和非法保护单均零。`20:43:15Z`
精确读取确认 IOTA 原退出单已 filled 1 张，均价 0.03303，原生 IOTA 仓位
为空；不再把此前价格带阻断保留为当前未解决项。

`2026-11-16T20:37:13Z` 六小时简报为已结算 14 笔、净亏损
`33.7615726 USDT`、PF `0.28049553`。该窗口跨越部署前后，不用于归因本次
补丁。候选模型预测误差改善、方向比较恢复和交易保护完整，均不能替代
当前策略的费后盈利验收；本轮未宣称盈利闭环或放行实盘。

## November 17 持仓估值与快风控快照一致性

`2026-11-17T01:29:22Z` 六小时权威简报确认有 15 次开仓执行，20 笔已结算
净亏损 `50.28552365 USDT`，PF `0.28663748`；窗口内同时开平的 8 笔净亏损
`5.04912181 USDT`。当前问题不能继续描述为完全不开仓，也不能宣称盈利闭环。

最近 30 笔结算记录逐笔关联 OKX 历史、准确交易所订单身份、入场决策和退出
决策。26 笔已有 240 分钟影子结果，其中 14 笔最终方向收益为正，10 笔方向
收益为正但实际结算亏损。该组四小时毛方向收益均值为 `-0.06002468%`，实际
结算合计 `-58.47614580 USDT`。影子结果使用公开市场和决策参考价，未重建
完整成交路径、资金费或挂单成交情况，不能当作可实现收益，也不能据此直接
取消止损或延长持仓。OP 决策 `832672` 约一分钟后快风控退出，记录浮亏
`13.65552 USDT`，最终权威价差亏损 `0.47415 USDT`；两者明显不同，但现有
历史快照不足以将该单的所有差异直接归因于某个补丁或某次价格跳动。

新增回归已复现并修复三处同类不一致：

- 价格刷新对仍有效的 OKX 缓存快照放弃原标记价和 UPL，改用公开特征价格
  重算浮盈亏。现在新取和缓存快照都保留配套的原生估值；缓存复用上限与
  持仓上下文统一为 30 秒，过期重新获取，不保留五分钟估值。
- 快风控用覆盖价格后的副本评估，却向执行前复核传递原始持仓。现在两次
  评估传递同一份独立快照，保留峰值及触发价格，避免共享原始数据变化造成
  结论不一致。原生 mark/UPL 配套使用，公开特征继续用于市场信号。
- 持仓复盘排序同样会覆盖原生 mark 而保留 UPL，公开价格差异会夸大风险
  占用。现在原生分片保留原值；真实止损触发价和最终执行权限继续校验。

本次不改变模型、开仓授权、仓位预算、杠杆、止盈止损距离或盈利门槛。
定向退出、同步回归 `505 passed`，额外执行链路回归 `101 passed`；全量
`6319 passed, 1 skipped`，最终 30 秒缓存边界补测 `13 passed`。全仓 Ruff
和差异检查通过。源码 `dbc1d119` 已推送并部署，533 个运行源码哈希一致，
交易服务、Dashboard、模型通道及 OKX 网络通过部署检查。

`2026-11-17T01:57:31Z` 真实验收为 16 个仓位、16 个原生保护单、待成交单零，
保护缺失、孤立、数量覆盖错误和非法单均零。ZRO、CRO、DOGE、LINK、SUSHI
五个真实仓位加入当前公开行情前后的浮盈亏、风险占用、止损越界和减仓比例
全部一致；此前五仓实测中三仓存在差异。`02:03:28Z` 容量检查覆盖完整，
当前持仓经济性缺口、待补经济性和已执行动态离场合约缺口均零。

## November 17 同笔平仓的身份与成交投影互锁

部署后完整验收发现 INJ 持仓 `15557` 缺平仓关联。逐笔核对确认：该行拥有
真实 OKX posId 和完整权威结算；另一个本地投影 `15558` 已拥有平仓成交
`4106028740307038210`，但缺 posId。补链逻辑拒绝任何已有本地所有者的成交，
去重逻辑又要求先证明同一生命周期，两者互锁，阻断 OKX 事实质量和六个
本地 AI 工具模型的训练。该问题不是成交缺失，也没有依据改变亏损金额。

新增回归在旧代码上两种镜像关联状态均失败。修复只在官方全平生命周期、
已确认成交、精确平仓时间、价格、数量、入场订单和交易方向一致，且不存在
冲突 posId 时允许同一投影补链，随后交给既有结算去重逻辑保留唯一记录。
身份、数量、入场订单和时间冲突继续拒绝。相关结算、生命周期和事实完整性
测试 `166 passed`，全仓 Ruff 和差异检查通过。

真实 INJ 记录在独立诊断进程内使用新逻辑验证准确补链，并回滚会话，无持久
写入。部署前已备份准确关联的两条持仓、两条订单和一条权威历史。源码
`aa03ac1c` 已推送并部署，533 个运行源码哈希及服务检查通过。

`2026-11-17T02:14:34Z` 线上复读确认 `15557` 已补齐平仓关联，`15558` 保留
原记录并标记 `superseded_position_residual`，权威历史也已关联准确平仓成交。
原始权威净亏损 `0.4666602 USDT` 保持不变。交易事实审计覆盖最近 500 条订单、
505 条持仓、341 条权威历史，关键和警告项均零；其余十项是保留的退役记录说明。
最近原生检查为 15 个仓位、15 个保护单，保护覆盖错误和待成交单仍零。

`02:15:16Z` 重新生成完整 OKX 对账快照，四张卡片无关键问题，开仓和训练
账本阻断均零，`okx_training_refresh_gate.allowed=true`；六个本地 AI 工具模型
已重新触发正常受控训练流程。`02:16:56Z` 三期验收已清除 OKX 事实阻断，
当前交易合约违规、经济性和离场合约缺口均零。仍未达到实盘晋升的是当前
权威结算收益不为正和 local_ml 未具备实盘盈利资格，不能据代码修复宣称盈利
完成。该滑动审计窗口为 96 笔、净亏损 `251.69939427 USDT`；窗口变化不作为
本补丁收益归因。

`2026-11-17T02:21:17Z` 六个工具模型的受控重训实际完成，生成候选
`20261117T022117879245Z-2aad7310`。训练输入含 512 条清洁影子样本、128 条
权威交易成本样本；共同检验集为 465 条，未用于本次拟合，但不属于封存的
独立盈利考试。候选多空毛收益误差、费后收益误差和亏损概率误差均未改善，
共同检验的正目标入场集合为空。系统拒绝替换，仍保留工具模型
`20261116T200517068863Z-a59fb25a`，没有取得生产交易权限。

候选收益目标均值 `0.0464562%`，经验收益下铰点 `-0.71107616%`，仍未满足
晋升条件。训练完成和阻断修复已经验收，不能把候选训练成功、回放正均值或
滚动窗口亏损缩小写成已实现稳定盈利。local_ml 是另一套模型；其自动训练
已于 `02:04:32Z` 完成，不与工具模型候选或旧冠军混写。

## November 17 模拟盘整笔成本与选中方向诊断

确认两处错误：选中方向的顶层阻断混入被拒绝反向的原因；模拟盘整笔滑点只累计
逐笔不利差额，丢弃有利差额，持续的 mark/last 基差被计为单边交易成本。
选边和执行权限保持原规则，历史展示仅修正诊断，不改写历史记录。

新增版本化 `paper_round_trip_cost`：完整权威成交的有符号差额在生命周期内求和，
再取非负成本，仅用于模拟盘成本训练与校准。保留原逐笔不利滑点、原始成交、
真实结算和整个收益标签契约。实盘、缺失或不匹配证据继续使用原保守口径；
反转订单分摊同步按数量比例分摊新增字段，不重复计算生命周期成本。

线上独立只读进程比较最近 72 小时 331 笔生命周期，逐笔真实盈亏及完整
`training_label_contract` 完全一致，99 笔成本口径发生变化。XPL 决策 `820775`
的原成本 `163.9080%` 修正为 `0.229885%`，真实净亏 `0.0100485 USDT` 不变；
INJ `832486` 从 `7.92862%` 改为 `0.54195%`，净亏 `0.4666602 USDT` 不变。
KAITO 约 `10.1324%` 的真实高成本仍保留。314 条 short 校准期望从 `1.77720%`
改为 `0.319699%`，真实平均净收益仍为 `-0.13171325%`，不能据此宣称已盈利。

全量测试 `6343 passed, 1 skipped`，最终增量定向回归 `294 passed` 和
`187 passed`；补充旧记录缺少方向证据时保留原阻断的展示回归后，相关五组测试
`263 passed`。全仓 Ruff 与差异检查通过。

`2026-11-17T06:01:18Z` 完整部署完成，533 个运行源码哈希、导入、交易服务、
Dashboard、模型通道与 OKX 网络检查通过。补充展示边界单独同步并仅重启 Dashboard，
HTTP 返回 302，交易服务保持运行。线上新决策选中侧阻断不再混入反向原因，
运行紧凑结果读取路径已携带新成本契约：XPL 仍保留原不利滑点 `163.90804598%`，
整笔校准为 `0.22988506%`。该紧凑窗口 104 条 short 的期望校准为 `0.34852629%`，
原不利口径 `2.15101656%` 同时保留；窗口与前述 331 笔比较不同，不混为同一统计。

部署后 TRX 决策 `836188` 已确认 OKX 原生开仓成交；随后只读核对为 18 个持仓、
18 个完整 reduce-only 止盈止损保护单，遗留待成交、缺失、孤立、数量覆盖错误和
非法保护单均零。`06:04:50Z` 三期检查当前执行契约违规、持仓经济性缺口、已执行
动态离场契约缺口均零；仍因当前已结算收益为负和模型未达到实盘条件而 no_go。

本轮已实际重训 local_ml，新候选 `20261117T061056272447Z-5ee3fb96` 于 `06:11Z`
通过现有冠军比较并激活为 paper canary；线上状态已读取确认该版本，实盘仍未放行。
共同未拟合检验范围 2,413 个决策组，按候选入场条件选出 5 笔，平均目标收益
`0.17328567%`、PF `2.16478285`、收益置信下界 `-0.21017807%`；旧模型在同一
范围选择 12 笔，均值 `-1.70825341%`、PF `0.22525614`。该比较并非封存的独立
盈利考试，且未完成成交、离场和容量回放，不能等同真实交易盈利。

六个工具模型本轮也已完成受控重训，候选为 `20261117T061554171700Z-00084622`，
输入 512 条影子样本和 128 条权威成本样本，共同未拟合比较 498 条。候选与冠军的
正目标入场集合均为空；影子预测比较因 `candidate_long_ordinary_loss_brier_not_improved`
未通过，保留原工具模型 `20261116T200517068863Z-a59fb25a` 的 shadow 阶段。
这是已完成但未通过的模型验收，不能写成待启动训练、成功晋升或已解决盈利。
