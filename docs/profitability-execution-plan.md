# Profitability Remediation Execution Plan

Updated: 2026-10-24, Asia/Shanghai.
Status: ACTIVE. Profitability acceptance has NOT passed.
Repository: `G:\数字量化\bb`; remote application: `/data/bb/app`.
This file is the durable recovery checkpoint for the current Codex chat.

## Objective

Complete authoritative loss attribution, validate data and signals, research
cost-aware candidates, freeze a selected policy, evaluate fresh independent
evidence, and validate actual paper fills and settlement. Finish only when
every acceptance gate below has verifiable evidence.

No strategy is guaranteed to become profitable. A failed experiment is useful
research evidence, not completed profitability acceptance. Never lower a
quality/risk gate, choose parameters on an inspected exam, or increase leverage
or trade frequency to manufacture success.

## Safety And Ownership

- Live trading stays inactive. Do not manually open, close, or cancel orders.
- Existing paper position management and verified exchange protection remain
  active. Do not disable exits or remove protective orders to change results.
- Paper experiments start read-only or shadow-only. Order-capable evaluation
  must use the existing governed paper pipeline after all prior gates pass,
  with a frozen risk budget and no direct SDK order submission.
- Preserve unrelated staged, unstaged and untracked work. Do not deploy the
  entire working tree. Commit only owned paths.
- Never store credentials, raw private API responses or raw logs in this plan,
  Git, project memory, or the automation prompt.
- Use the `bb` project memory bank only. Current verified repository/remote
  state overrides memory. Follow `AGENTS.md`; its old E: project paths are stale.
- Use bounded scalar/projection queries with statement timeouts. Avoid loading
  full Position ORM graphs, large JSON scans, unbounded history or automatic
  mutation scripts during audits.

## Recovery Procedure

1. Read this file, `AGENTS.md`, the active goal and the latest relevant project
   memory. Read `docs/profitability-remediation-2026-10-20.md` for verified v20
   history. Do not restart the investigation from scratch.
2. Check branch, HEAD and dirty paths; identify changes made since the last
   checkpoint. Verify remote service state before treating an old snapshot
   as current. Do not recreate an existing goal or duplicate automation.
3. Resume the first incomplete gate. Work in bounded, resumable batches.
   Check whether a previous remote diagnostic is still running before
   launching another copy. Do not restart completed experiments.
4. Before a long operation, record the immutable inputs, command, resource
   budget, safe output location and recovery action. Record experiment
   identity and whether any labels have already been inspected.
5. After a meaningful result, update the stage table, evidence references,
   decisions and exact next action below. Persist durable findings to `bb`.
   Store generated aggregate reports under `data/profitability_remediation/`
   or a similarly ignored directory; keep reproducibility fingerprints here.
6. For code changes, run focused tests and lint, perform staged targeted
   deployment and remote verification where applicable, then commit/push
   explicit owned files. Do not mix another operator's changes.
7. If evidence is insufficient, leave acceptance pending and identify the
   missing independent observations. If a candidate fails, record rejection
   and return to development; never retest a tuned candidate on the same exam.
8. Mark the goal complete only after every gate passes. Then pause this
   continuation automation, report evidence and residual risks, and leave
   this chat available. Do not archive the chat automatically.

## Stage Gates

| Gate | State | Deliverable | Acceptance |
| --- | --- | --- | --- |
| G0 Recovery and execution safety | PASSED | Active goal, this checkpoint, same-chat scheduled continuation | Saved automation inspected and app view verified on 2026-10-22 |
| G1 Authoritative loss attribution | PASSED | Fixed-cutoff canonical lifecycle and complete official-bill audit; evidence below | Separate full round trips, partial updates and window cash flow; reconcile price PnL, signed fees and funding without double counting; expose unresolved links and costs |
| G2 Data and label validity | BLOCKED | Eligibility, timing, cost and provenance report plus defect regressions | No future-data leakage or split-group overlap; correct price/label horizon and source identity; missing or contradictory facts quarantined, not imputed into profitability |
| G3 Cost-aware candidate development | PENDING | Preregistered bounded experiment ledger and purged chronological comparisons | Candidate family/features/horizon/threshold/exits selected only within development folds; compare no-trade and transparent baselines on identical costs and exposure |
| G4 Frozen independent evaluation | PENDING | Policy/code/data fingerprints, untouched exam and acceptance report | Predefined sample coverage, time/dependence-aware lower bound above zero, positive net mean, PF above one, predefined drawdown/tail limits, and cost stress tests pass; no post-exam tuning |
| G5 Governed paper execution | PENDING | Controlled paper fills, protection, exits and official lifecycle settlements | Prior gates passed; frozen risk budget; no unlinked fills/unprotected exposure; observed fee-after results and forecast calibration meet preregistered criteria |
| G6 Final acceptance and handoff | PENDING | Complete evidence inventory, verified deployment, regression report and monitor | G0-G5 all passed, exact deployed hashes match accepted version, independent subsequent window passes predefined criteria; remaining uncertainty disclosed |

Gate G1 may pass as an audit even when its PnL is negative. Gates G4-G6 cannot
pass on negative or insufficient profitability evidence. Do not confuse
account cash change, opportunity-return sums and executed portfolio returns.
Do not use the diagnostic per-trade downside quantile as a confidence bound
on the mean. A favorable historical test is not a guarantee of future profit.

## Experiment And Acceptance Rules

- Freeze the research question, candidate set, split/purge rules, cost model
  and compute budget before a development batch starts.
- Learn liquidity/universe filters only from information available at the
  decision time; do not select winners by their realized exam returns.
- Use the exchange account's actual fee rates and executable bid/ask/depth.
  Include funding, entry/exit fees, slippage, incomplete fills and adverse
  selection where relevant. A limit order is not automatically a maker fill.
- Align forecast horizon with the evaluated entry/exit policy. Treat a change
  of horizon, sizing or exits as a new policy requiring validation.
- Select candidates using nested chronological development folds with purged
  label boundaries. Account for simultaneous/correlated opportunities and
  overlapping holdings. Do not present naive opportunity sums as portfolio PnL.
- Before opening a fresh exam, write numerical coverage, risk, cost-stress
  and confidence-method requirements. Missing thresholds mean NOT READY.
  Requirements must be justified by the data/risk budget, not chosen after
  seeing whether they pass.
- Keep an exam ledger recording time ranges, source IDs/fingerprints, artifact
  identities and inspection status. Already-viewed labels are development or
  diagnostic data, not a fresh sealed exam.
- Negative/insufficient results reject promotion. Do not reverse a score,
  switch sides or pick a horizon using inspected exam outcomes.

## Verified Starting Point

- Commit `96c8d11` is deployed and pushed to `origin/main`.
- v20 fixes the historical-quality permission bypass at selection,
  construction, execution-cost reconciliation and final validation.
- The last verified deployment restarted paper/dashboard at 2026-10-22
  00:21:27 Asia/Shanghai; live remained inactive. Recheck before relying on it.
- Final local worktree tests for that change: 4,770 passed; changed-file Ruff
  and whitespace checks passed. This is software verification, not profit.
- Previously inspected current/challenger forward comparisons remain
  unqualified. The 225-row five-minute comparison is already inspected and
  must not be reused as a fresh exam.
- Repository already has `services/profit_attribution.py`,
  `services/return_loss_attribution.py`, `services/okx_position_ledger_view.py`
  and `scripts/inspect_online_48h_pnl.py`. Discover/reuse their contracts before
  adding a competing PnL or attribution implementation.

## Continuation Schedule

- Intended cadence: every 30 minutes in this chat.
- Automation identity: `automation`; name: `数字量化盈利整改续跑`.
- Verified saved state: `ACTIVE`, heartbeat, every 30 minutes, target chat
  `01a158e8-0329-7021-94b5-5affeb85f90a`. App view verified on 2026-10-22.
- Purpose: resume unfinished work after an idle/interrupted turn, not simply
  poll the same account numbers or duplicate an active diagnostic.
- Notify only on a meaningful milestone, completed acceptance, failure, or a
  necessary user decision. Stay quiet for unchanged/non-actionable state.
- This is a local continuation mechanism, not a guarantee of execution while
  the host/application is unavailable.

## Current Checkpoint

- Completed this turn: active goal and same-chat continuation verified.
  Existing canonical outcome, native bill, attribution and ledger paths read.
  Paper/dashboard are active with zero restarts since 2026-10-22 00:21:27
  Beijing. `bb-paper.service` is NOT the paper unit; query
  `bb-paper-trading.service`.
- In progress: bounded economics audit implemented using canonical lifecycle
  outcomes and native bill normalization. Its 11 focused regressions passed;
  the related canonical/native/authoritative suite passed 78 tests before
  online transport was added. Audit script/test uploaded without restart.
- Frozen G1 baseline: `[2026-10-20T16:55:00Z,2026-10-21T16:55:00Z)`,
  24 hours, 120 lifecycle limit, eight official-bill pages.
  Unit: `bb-profitability-economics-20261021T165500Z`.
  Remote report:
  `/data/bb/app/data/profitability_remediation/economics-20261021T165500Z.json`.
  Local report:
  `data/profitability_remediation/economics-20261021T165500Z.json`.
  Limits: 15-second SQL statements, one CPU core, 1 GiB memory, 240-second
  systemd deadline and 210-second asynchronous deadline.
- G1 baseline completed at `2026-10-21T16:57:10Z`. Six full lifecycles,
  ten partial/unknown, no lifecycle limit saturation. Full-lifecycle price
  PnL `-13.196020`, signed fees `-2.37409148`, funding `+0.064444114`,
  net `-15.505667366 USDT`; two winners/four losers. All six have complete
  canonical training evidence and holding time beyond their forecast horizon
  (maximum ratio 26.26869). This is a diagnostic mismatch, not causal exit
  attribution. Measured slippage `6.46027 USDT` is already in fill PnL.
  Lifecycle fingerprint:
  `eb1eca52399a9a0bd3bda2650e3ad53990da02553cef1b049570357414eb1961`.
- The same window has 170 official bills and complete two-page boundary
  coverage. Price PnL `-14.3243872861`, fees `-6.50549907`, funding
  `+0.104211904`, cash change `-20.7256744521 USDT`; unexplained residual
  is below `1e-12 USDT`. 115 bills link to sampled lifecycle orders.
  Remaining bills and partial lifecycle gaps stay disclosed; these two
  totals describe different populations and must not be equated.
  Bill fingerprint:
  `5ded5b7c096f685ebb9d763d098453a79b3642ace040f07035b3c2d3123582fb`.
- Immutable development-source audit v2 is complete, using exact source IDs
  in bounded pages without reading exam labels. Snapshot has
  9,191 rows/2,434 groups; cutoff remains
  `2026-09-26T22:11:51.501105Z`, SHA256
  `afced19c8876ec2fc8237b298f22248d326160e6bdc83a754df1bc431cc9cec3`.
  Existing horizon research remains rejected: only 15 nonoverlapping
  selections and naive lower bound `-0.4772335972%`. Do not rerun it unchanged.
- Timing correction: the current dirty/deployed shadow writer explicitly uses
  `DecisionOutput.timestamp`, but HEAD's legacy writer scheduled from
  `datetime.now(UTC)` and did not set `created_at`. Historical reconstruction
  preserves `AIDecision.created_at` and floors the path to a minute. Neither
  `created_at` nor `due_at - horizon` universally establishes the original
  decision clock of the September frozen rows. Do not infer historical
  timestamp semantics from today's code. Audit hashes `348950648d77...`,
  `216aa7592c9b...`, `ce1239992236...` and `cdd1fee263af...` are superseded for
  timing acceptance; the reported 521 negative feature offsets do not prove
  future-feature leakage.
- Exact source IDs and numeric values previously matched all frozen rows.
  Declared-clock purged folds have no group overlap, but do not establish
  point-in-time feature/label availability. Historical fact builders use
  candle closes with bar-open timestamps; close availability is at least one
  minute later and requires a separate validated timing contract.
- Blocking dependencies: a profitable candidate and fresh independent exam
  do not yet exist. This is not permission to skip G1-G3 or mark G4 passed.

## Current G2 Repair Batch

- Verified services on continuation: paper and dashboard running, zero
  restarts since `2026-10-21T16:21:27Z`. No active profitability/frozen audit
  unit was returned by bounded discovery. This does not independently prove
  that a guessed live unit exists or is disabled.
- Defect: historical rebuild copies numeric fee/book/funding features.
  Counterfactual supervision and shadow settlement previously accepted these
  numbers despite explicit historical/nonexecutable market provenance.
- Repair: the common execution-cost estimator rejects historical-only rows
  and explicit nonexecutable assertions in either full or compact market
  contracts. Market-opportunity research remains separate. Supervision
  version changes to `2026-10-22.separated-profit-executable-provenance.v3`,
  so artifacts from the old contract cannot authorize model influence.
- Regressions cover copied complete costs, conflicting full/compact
  provenance, shadow settlement, historical rebuilding into the training
  frame, and old artifact influence. Final local focused suite: 160 passed;
  related paper/training-boundary/queue/economics suite: 98 passed. Owned
  Python files passed Ruff; scoped whitespace check passed.
- Audit v2 records schedule offsets without calling them decision clocks,
  preserves unresolved timing as a G2 blocker, fingerprints relevant code
  dependencies, and checks for an already active identical unit.
- Remote source backup before targeted upload:
  `/data/bb/app/data/profitability_remediation/source_backups/20261021T180948Z-cost-provenance`.
  All audited dependencies already matched the local checkout except the two
  repaired services and the audit script.
- Targeted upload and five-file integrity verification completed. Remote
  focused suite: 37 passed. Only paper/dashboard were restarted, at
  `2026-10-21T18:25:36Z` / `2026-10-21T18:25:35Z`; subsequent checks found
  both active, zero automatic restarts, and dashboard health HTTP 200.
  Actual paper-process settings and persisted control both say `paper`;
  zero models are configured live. No manual orders or protection changes.
- Audit completed `2026-10-21T18:21:25.644255Z`, with 200-row pages, one DB
  connection, 15-second statements, one CPU, 1 GiB memory and 300-second
  systemd deadline. All 9,191 exact source IDs were found. The 789 historical
  rows are now excluded from cost-complete training; the remaining 8,402
  rows had no numeric mismatch. All 9,191 original decision clocks remain
  unverified. Three declared-clock folds have zero group overlap, which
  does not establish component availability. No exam labels were read.
- Audit dependency identity:
  `d921ebedb9b731b3ab9ecd596e7f98c781d90510363d26876d954575d6a4137b`.
  Source label identity:
  `3b7e3b887ed99026f9f64b3a47c910896f938222295e3c781a0bf85f56d3fd24`.
  Local report:
  `data/profitability_remediation/development-evidence-frozen-v2.json`.
  Remote report:
  `/data/bb/app/data/profitability_remediation/frozen-v1-source-r10000-d921ebedb9b7.json`.
  Dependency fingerprints include existing dirty/deployed quality/historical
  files; this audit is evidence about those exact bytes, not pristine HEAD.
- Next exact action: inspect original decision timing and feature component
  lineage for deterministic frozen source IDs using bounded JSON projections,
  then expand only if those fields can establish availability. Keep the
  original snapshot/cutoff immutable; do not rerun the completed audit
  unchanged or inspect a new exam. G2 remains blocked by timing evidence.

## Original-Clock Audit Batch

- Cost provenance repair and G1/G2 source audits committed/pushed as
  `c8d42e8`; unrelated staged/unstaged paths preserved.
- Deterministic 12-row discovery found original `raw_llm_response.timing`
  and a matching compact timing copy. These clocks can be checked directly;
  scheduling timestamps need not be substituted for original completion.
  Some original feature payloads are missing and book reception clocks are
  absent. Feature-vector construction time is not individual technical/news
  input availability. Discovery did not read return labels or place orders.
- New `scripts/audit_development_timing.py` selects only timing/identity JSON
  projections for exact frozen IDs, checks original/compact conflicts,
  feature and cost clock consistency, and reports missing component clocks.
  It never accepts complete lineage merely because construction preceded
  completion. Focused audits/provenance tests: 39 passed; Ruff passed.
- Preregistered operation: original snapshot SHA256/cutoff unchanged;
  `--max-rows 10000`, 200-row pages, one read-only DB connection,
  15-second statements, one CPU, 1 GiB, 300-second systemd deadline.
  No database return-value fields or new exam labels are selected.
  Output: `data/profitability_remediation/development-timing-frozen-v1.json`.
  Remote unit/report names include the audit dependency fingerprint;
  cached complete reports are reused and active duplicate units rejected.
- Before execution: targeted script/test upload, remote hash check and
  focused tests, no service restart. Recovery: inspect named unit/report
  before retrying. Then record full coverage and precise remaining lineage
  gaps, and specify the bounded repair/collection required by those gaps.
- Timing v1 completed `2026-10-21T18:45:37.773789Z`: all 9,191 IDs found,
  all original recorded clocks recovered with consistent compact copies,
  and all feature construction times precede recorded completion. Original
  feature clocks are absent for 1,267 rows. The 789 historical rows retain
  bar-close availability defects. This supersedes the blanket "all decision
  clocks unresolved" observation, not the G2 component-lineage blocker.
  v1 dependency identity:
  `00b7f4dd6fec2e5c327b82853817880601d16cd6ccfca39f2783e9abc8354ad4`.
  Timing evidence:
  `3bc098b27c360d01256067959fc8caf153dbcad8ee1f76773e03a4364ebdd168`.
- Follow-up found OKX funding clocks are numeric millisecond strings; v1's
  parser incorrectly reported these as missing. v2 reuses the runtime
  timestamp parser, checks overwritten fee values as well as clocks, and
  reads compact contract observation windows when wide payloads are absent.
  Do not use v1's funding-missing counts as evidence.
- Confirmed code defect: shadow settlement overwrote decision-time fee facts
  with the account fee observation obtained at settlement. Discovery showed
  later clocks with equal numerical rates; this is a provenance failure,
  not proof that every historical numerical fee was wrong.
  Repair preserves decision-time fields and records a bounded separate
  `shadow_settlement_cost_observation`, explicitly not a decision feature.
  Missing original fees stay incomplete/quarantined instead of being filled.
  Local expanded suite: 237 passed; owned-file Ruff passed.
- v2 operation keeps the same immutable IDs/cutoff and resource budget,
  output `data/profitability_remediation/development-timing-frozen-v2.json`.
  Backups before staging:
  `/data/bb/app/data/profitability_remediation/source_backups/20261022T030500-entry-cost-immutability`.
  Remote shadow service matched the preexisting staged baseline after
  newline normalization; no unrelated service changes need deployment.
  New runtime hunk must be committed independently from those staged changes.
- Timing v2 completed `2026-10-21T19:02:31.296684Z`. All 9,191 original
  completion clocks are recovered and consistent; construction precedes
  completion in every row. Original component clocks remain absent in 1,267
  rows, all 9,191 lack individual technical/sentiment availability evidence,
  and 789 historical rows remain ineligible for executable-cost training.
  There are 9,143 shadow/original fee-clock mismatches, 8,346 shadow fee
  observations after completion and 1,314 fee-value mismatches (including
  missing original values); do not equate clock contamination with a changed
  numeric fee in every row. Funding clocks are recovered for 7,924 rows.
  The other 1,267 are missing; v1's millisecond-string counts are superseded.
- In 1,699 of 8,402 compact contract windows, the recorded result observation
  precedes the scheduled label time (maximum 108.964138 seconds). This is
  a horizon-alignment blocker to investigate, not a causal claim about PnL.
  No new exam labels or database return values were read.
  v2 dependency identity:
  `67a2bc26d83e10179431a143bb9e4c3869d2dc65d562a2e3c740d6db023dda32`.
  Timing evidence identity:
  `312c7b0c5eab8cfa6c8a0e58ace74126e19378b0412655f90e629324fe88a5a4`.
  Remote report:
  `/data/bb/app/data/profitability_remediation/frozen-timing-r10000-67a2bc26d83e.json`.
  Local report:
  `data/profitability_remediation/development-timing-frozen-v2.json`.
- Targeted four-file upload passed integrity checks and 87 remote tests.
  Only paper was restarted, at `2026-10-21T19:13:44Z`. Paper/dashboard remain
  active/running with zero automatic restarts; `/health` returns HTTP 200.
  Settings loaded from the actual paper-process environment and persisted
  control both remain `paper`, with zero live-configured models.
- Bounded rollout discovery read scalar projections for the newest 128
  shadow IDs, finding 12 rows with separate settlement cost observations.
  Eight displayed examples preserve both original fee values and clocks;
  row 610666 preserves decision fee time `2026-10-21T19:10:47.795Z` while
  settlement observation is `2026-10-21T19:16:28.811Z`. No return labels were
  inspected. Final verification reselected the newest 128 IDs: all eight
  new-format rows in that moving bounded window preserve original fee values
  and clocks and mark settlement observations as nonfeatures, with zero
  failures. IDs: 610631, 610635, 610639, 610643, 610647, 610651, 610666, 610691.
  Verification evidence:
  `a071b1b7117dfcb0f84e2d4d9e07d3cc2e5cd2478f901b198d9a4ff5ab2013ec`.
  This is data integrity evidence, not profitability.
- Next exact action: isolate this runtime/test hunk plus timing
  audit/checkpoint in Git, preserving previous
  staged queue/historical changes. Then investigate result-window alignment
  and implement verifiable component-availability collection. G2 stays
  incomplete, and no frozen candidate or independent profitable exam exists.

## Historical Label Clock Repair Batch

- The prior timing report treated historical candle `open_time`/`received_at`
  as when its close price was available. That is not the candle's availability
  time. For one-minute OHLCV, the close availability lower bound is
  `open_time + 60 seconds`; actual delivery latency is not in historical data.
- Historical sample rebuild also stored `floor(decision_time, minute) +
  horizon` as `due_at`, shifting the planned horizon earlier by up to one
  minute. The intended scheduled label time is now the exact original decision
  time plus horizon. The existing minute close is accepted only when its
  availability lower bound is at/after schedule and no more than 60 seconds
  late; the label contract records the close-availability lower bound
  separately from scheduled time and effective horizon.
- Historical contract version is now
  `2026-10-22.preserved-ohlcv-close-availability.v2`. Prior v1 historical
  samples fail closed as stale until rebuilt. Training-quality validation
  requires the source availability metadata and rejects early or over-one-bar
  late labels. This does not create executable historical costs or upgrade
  historical-only data into trading evidence.
- Timing audit v3 distinguishes candle open from theoretical close
  availability; it records the close-time lower bound without claiming
  actual retrieval latency. It still reads no outcome fields. Frozen sample,
  cutoff and IDs remain unchanged.
- Initial local verification: 117 focused tests passed across historical
  rebuild, timing, training quality/quarantine and shadow collection; Ruff and
  whitespace checks passed. Online targeted upload/tests and the v3 frozen
  timing run are pending; do not use the old v2 early-result count after this
  timestamp reinterpretation.
- Commit `bfba943` (decision-time fee immutability plus timing v2 audit) is
  pushed to `origin/main`. Other staged changes remain untouched.
- Next exact action: finish staged deployment and remote focused tests for the
  historical availability contract, run timing v3 against the same immutable
  development IDs under the prior 1-CPU/1-GiB/300-second budget, record its
  corrected offsets, then commit only owned hunks. Keep G2 blocked on missing
  individual feature availability; G3-G6 remain pending and no profitability
  claim is made.

## Current Continuation Checkpoint: Label Timing And Historical Entry Isolation

- Local repair now separates scheduled label time, result source time and
  result availability time through `shadow_label_timing`. Native result facts
  carry bounded source/receipt projections in their compact contract.
  Historical OHLCV labels carry the one-minute close-availability lower bound;
  the field is named `label_maturity_lag_seconds`, not an effective horizon.
- Shadow labels now record `scheduled_label_at` separately from
  `label_timestamp`. Quality assessment rejects missing, early, contradictory,
  or over-one-minute-late result timing and rejects labels whose timestamp is
  not the recorded result availability. A late native result may remain an
  audit row, but cannot enter training.
- Historical rebuild no longer falls back to a future candle close when the
  original decision price is missing. Such a row is skipped. Its OHLCV entry
  fact is retained only as `historical_market_path_entry_fact` for bounded
  path reconstruction; it is not a decision-time `market_fact` feature.
- Focused local regression status: 104 passed across historical rebuild/path,
  shadow settlement, label timing, quality and quarantine tests; changed-file
  Ruff and `git diff --check` passed. This is lineage evidence, not profit
  evidence.
- Remote deployment and the v3 frozen timing audit are still pending for this
  exact source state. No service restart has been performed for this batch.
  Paper/live safety remains unchanged: live is off, no manual order action,
  no protection changes, and no candidate or exam promotion.
- Next action: upload only the owned timing/historical/quality files and their
  focused tests with `--skip-restart`, run the bounded remote suite, then
  activate only the required paper/dashboard services after hash and mode
  checks. Run the immutable v3 audit once, record its evidence and remaining
  component-lineage blocker, then isolate-commit/push this batch.

### Frozen Timing Audit v3 Result And Follow-Up

- v3 completed online at `2026-10-22T04:32:03.596732Z`, unit
  `bb-frozen-timing-r10000-6fce600ab3f8`, with 1 CPU, 1 GiB, a 300-second
  deadline, 15-second read statements and 200-row pages.
- The exact frozen development snapshot remains 9,191 rows / 2,434 groups,
  cutoff `2026-09-26T22:11:51.501105Z`, SHA256
  `afced19c8876ec2fc8237b298f22248d326160e6bdc83a754df1bc431cc9cec3`.
  All 9,191 IDs were found; original recorded clocks were consistent and
  feature construction preceded completion in all rows. The source-clock
  evidence SHA256 exactly matched v2:
  `312c7b0c5eab8cfa6c8a0e58ace74126e19378b0412655f90e629324fe88a5a4`.
  No outcome fields or exam labels were selected.
- v3 found 789 historical rows with missing result source clocks and
  unobserved delivery latency. It also found 1,699 of 8,402 nonhistorical
  compact result receipt clocks before `due_at` (minimum `-108.964138s`).
  Their native source timestamps did not precede schedule, so the issue is
  specifically receipt-clock/schedule inconsistency; it is not evidence that
  those labels are early from source-event time alone. They remain blocked
  until the contract requires both source-event and receipt availability to
  be no earlier than schedule.
- The historical rebuild's verified rule floors decision creation to the
  minute for the result bar and labels at that bar's open time. v4 now reads
  only the compact label timestamp clock and derives the one-minute close
  availability lower bound from that result-bar open, compared against
  original decision creation plus horizon. The derived bound is explicitly
  not an observed exchange delivery clock; missing source/feature-component
  and actual historical latency evidence remain G2 blockers.
- New local regression suite: 130 passed across audit, timing, historical
  path/rebuild, quality, quarantine and settlement; Ruff and whitespace
  checks passed. The prior deployed-source remote run passed 95 tests, but
  did not include the new untracked timing test and predates the v4 follow-up.
- v4 remote report and remote focused tests for this exact source state are
  now complete. Remote suite before adding the standalone timing test:
  120 passed in 5.02 seconds. The v4 report completed at
  `2026-10-22T04:43:17.228052Z`, unit
  `bb-frozen-timing-r10000-3acd743a722f`; dependency identity
  `3acd743a722ff9f9ac154983c324e99a29dc298351c5eaf816cbbbf3822db920`;
  source-clock evidence SHA256
  `06d492f72bfb619d7921507c8300dc6d28baf32679468a90eb4bccc1ab879edd`.
- v4 found all 9,191 frozen IDs with unchanged snapshot SHA/cutoff and zero
  missing source rows. All 789 historical label bar-open timestamps yielded
  a derived one-minute close-availability lower bound from original decision
  creation plus horizon: `0.372065s` minimum, `59.603941s` maximum,
  `30.963838s` median. None is early or more than one bar late. All 789 remain
  explicitly marked as delivery-latency-unobserved and are not G2-accepted.
- The 1,699 native receipt-before-schedule rows remain unchanged; these are
  distinct from source-event times, and the runtime/quality gate now also
  rejects receipt clocks before schedule. Overall G2 remains
  `blocked_by_component_availability_evidence`: 9,191 technical/sentiment
  component availability clocks are absent; 1,267 original feature payloads
  are missing; 9,191 book receipt clocks are missing; historical actual
  delivery latency is unobserved; and fee/funding lineage mismatches remain.
- Read-only platform diagnosis at `2026-10-22T04:43Z` showed settings mode
  `paper`, the paper executor log reported `mode=paper,demo=true`, all six
  configured models use the local paper route, paper/dashboard were active,
  and both had zero automatic restarts. This diagnosis output is not a
  profitability claim.

### Final v4 Deployment Verification

- On 2026-10-22, the exact six-file focused suite
  (`test_audit_development_timing`, historical path/rebuild,
  shadow backtest/label timing, and training quality) passed locally and
  remotely: 122 passed in each environment. No exam labels were read.
- A targeted 15-path sync with `--skip-restart --skip-secret-file-purge`
  found the 14 source/test files already identical online; only this
  checkpoint document required upload. Remote source integrity reported
  `source-integrity-ok:14`. The completed v4 unit
  `bb-frozen-timing-r10000-3acd743a722f` was inactive; its report was not
  rerun or modified.
- Read-only verification after sync: paper and Dashboard were active with
  `NRestarts=0`, both last started at `2026-10-22 04:48:30 UTC`; paper
  executor journal says `mode=paper,demo=true`; runtime settings say
  `trading_mode=paper` and `live_trading_enabled=false`. Persisted control is
  `mode=paper, paused=false`. All six configured model slots route locally.
  Dashboard `/health` returned HTTP 200. Since the exact code was already
  deployed and both services had started after that deployment, no duplicate
  restart was needed.
- The bounded recent journal projection contained a
  `shadow sample recovery scan failed` `TimeoutError` at
  `2026-10-22T05:00:00.834979Z`; nearby shadow maintenance still completed
  25 rows. Recurrence/count and cause were not established. Investigate this
  bounded recovery path before relying on its maintenance coverage.
- G2 remains blocked: per-component technical/news and book availability
  clocks are still absent in the frozen rows; historical delivery latency is
  unobserved; and prior fee/funding lineage gaps remain. G3-G6 remain pending.
  The next action is to bound and diagnose the recovery-scan timeout, then
  design and test point-in-time availability capture for each newly collected
  feature component. Do not rerun the frozen audit unchanged, and do not open
  or tune against a new exam.

### Shadow Recovery Batch Repair

- The bounded online diagnosis on 2026-10-22 found that the recovery query
  itself took about `0.10-0.27s`, while four separate horizon inserts took
  about `2.4-4.5s` under PostgreSQL contention. The old per-decision timeout
  (`2.5s`) therefore produced false recovery failures even without a lock
  wait. The inspected rows had raw payloads around `162-166 KiB`; no outcome
  labels or orders were read by this diagnosis.
- Repair: `MemoryRepository.create_shadow_backtests` now performs one bounded
  existing-row read and one savepoint/flush for all horizons, retaining the
  established uniqueness/idempotency fallback for concurrent writers.
  `ShadowBacktestService.create` uses that batch contract, while the legacy
  single-row method remains the fallback for small repository fakes.
  Recovery records per-decision elapsed time and defers a candidate when the
  remaining pass budget cannot accommodate a full decision batch, avoiding
  cancellation during a database flush. The per-decision budget is `6s` and
  the whole pass remains `12s`; this maintenance path is read-only with
  respect to trading and cannot place orders.
- Local focused verification after the repair: `401 passed`, Ruff clean, and
  scoped whitespace check clean. Remote focused verification of the uploaded
  shadow queue/service sources: `38 passed`; the unrelated full boundary run
  still has one pre-existing source/test mismatch in the unuploaded trading
  timeout test and is not used as repair evidence.
- Targeted upload completed with source integrity checks; only
  `bb-paper-trading.service` was restarted at `2026-10-22 07:12:42 UTC`.
  `bb-dashboard.service` remained active, both services reported
  `NRestarts=0`, runtime mode remained `paper`, and no live orders or
  protection actions were performed.
- Post-restart bounded online evidence at `2026-10-22T07:27:27Z`:
  406 recent paper market decisions had `incomplete_decisions=0` and
  `minimum_horizons=4`; shadow maintenance had `success_count=20`,
  `failure_count=0`, `last_error=null`, and the latest recovery log was
  `scanned=1,recovered=1,failed=0` in `2.811s`. This proves recovery
  coverage, not profitability. The remaining G2 blockers are unchanged:
  missing per-component availability clocks, unobserved historical delivery
  latency, and unresolved fee/funding lineage gaps.
- Next exact action: design and implement point-in-time availability capture
  for technical, news/sentiment, order-book and cost/funding components on
  newly collected samples, with component-level quarantine tests. Do not
  promote a candidate or reopen the inspected exam.

## Point-In-Time Feature Availability Capture Batch

- Local implementation is staged across 50 owned source/test paths on
  `main` at `093f704`; three untracked audit helpers remain outside the index.
  The only changes staged during this checkpoint were the two already-owned
  test-file trailing-blank-line cleanups. No unrelated path was added.
- The batch records an immutable pre-model feature fingerprint and freeze
  time, and carries component-level clocks/fingerprints for the actual
  technical K-line response, contributing news/social inputs, ticker,
  order-book, mark/index prices, funding, account fee and open interest.
  Training snapshots preserve frozen inputs without resealing or filling
  missing facts; training quality excludes snapshots with missing,
  contradictory, late, post-freeze, or changed component evidence. This
  evidence gate does not weaken entry or risk controls and does not establish
  profitability.
- Local focused verification for feature capture, compaction, quality,
  sentiment, shadow quarantine/settlement and queue paths:
  `243 passed, 1 skipped`. Ruff across the owned Python paths passed;
  staged and unstaged whitespace checks passed.
- The targeted source/test sync completed without restarting either service.
  Remote Python source integrity reported `source-integrity-ok:888`.
  Remote focused suite with a unique, initially absent `/tmp/bb-g2-tests.*`
  pytest base passed `243 passed, 1 skipped` in 17.24 seconds. The initial
  sync used the complete tracked test set because the helper accepts
  `--include-tests`; it uploaded 69 changed tracked paths and reported one
  stale remote audit helper removed. Restore and hash-verify that helper from
  the preserved local untracked copy before the next deployment. Future syncs
  for this batch must use explicit `--only` filters to avoid pruning other
  remote paths.
- Read-only online check around `2026-10-22T14:00Z`: paper and Dashboard
  active, `NRestarts=0`; paper process `TRADING_MODE=paper`; persisted runtime
  state `mode=paper`, `running=true`, `paused=false`; Dashboard health HTTP
  200. No matching profitability/frozen/G2 evaluation unit or system job was
  active. This check preceded loading the new code into the running process.
- A bounded 3-hour coverage projection found 92 recent market decisions,
  three with incomplete shadow horizons and a minimum of zero horizons. The
  recent 15-minute queue projection showed 1,868 due pending rows, with
  horizon counts 347/379/477/665, and 40 shadow rows updated to completed in
  the preceding five minutes. Queue work is progressing but coverage is not
  yet complete. No G4 exam was opened.
- G2 remains blocked on legacy samples without component availability,
  verified post-deploy capture for newly collected decisions, historical
  delivery latency, and unresolved fee/funding lineage. G3-G6 remain pending;
  no profitability conclusion is supported.
- The first post-restart projection (seven market decisions created after
  `2026-10-22T14:12:38Z`) confirmed the sealed contract on all seven, but
  only four had all selected scalar fields present and only four had observed
  technical delivery. News clock errors were nonzero because RSS publication
  dates use RFC-2822 strings that the original parser rejected; some rows also
  reused database-only K-lines without an observed network receipt. These are
  lineage defects, not profitability evidence.
- Repair in the follow-up working tree: `availability_timestamp` now accepts
  timezone-bearing RFC-2822 dates; an indicator snapshot built only from
  cached K-lines is accepted as fresh only when its technical availability
  contract proves observed delivery, otherwise the service performs a bounded
  exchange refresh and fails closed if that refresh cannot provide receipts.
  Existing cached rows remain available for diagnostics but cannot silently
  authorize a model input.
- New local regression status after the repair: `145 passed` across the
  availability, sentiment/news and data-service security suites; Ruff and
  whitespace checks passed. The broader prior local/remote batch remains
  `243 passed, 1 skipped` before this follow-up repair.
- Next exact action: stage only the repair, its tests and this checkpoint;
  upload with explicit `--only` filters and no restart, run the same remote
  focused suite in a unique initially-absent pytest base directory, then
  restart only `bb-paper-trading.service` after read-only paper-mode checks.
  Verify a fresh bounded decision sample has zero RFC-date clock errors and
  no database-only technical snapshot is treated as observed. Keep all
  incomplete samples quarantined; G2 remains pending until this evidence and
  the historical/fee lineage gaps are resolved.

### Compact Storage Fingerprint Parity Diagnostic (2026-10-23)

- Recovery confirmed the point-in-time availability batch is still the first
  incomplete gate; the active goal and 30-minute continuation remain intact.
  Local `main` and `origin/main` are both `093f704`; unrelated staged and
  unstaged paths remain present and were not reset or broadly staged.
- Local focused regression group passed `174 passed, 1 skipped`; the skipped
  test is the PostgreSQL temporary-transaction parity check because local
  `BB_TEST_POSTGRES_URL` is unset. Ruff passed for the affected availability,
  market-fact, snapshot, data-service and test modules; `git diff --check`
  passed.
- Read-only remote inventory found paper, Dashboard and model-tunnel services
  active; no profitability/frozen-feature audit service was active. A
  `bb-phase3-market-data-warmup.service` start job was present at the time of
  inventory and must be checked before adding database work. No service was
  restarted.
- Remote hashes for `core/feature_availability.py`,
  `core/market_facts.py`, `core/training_feature_snapshot.py` and
  `models/learning.py` matched the working tree. The diagnostic test file was
  stale. Two targeted SFTP uploads were reset during file close and left the
  remote test path at zero bytes; that file was restored via an SSH-streamed
  temporary file and atomic rename, then verified against local SHA-256
  `18a8544908a1d5f15ff3f06344186b9bb63997a864f51bb511937a8bb58dbdf8`
  (36,302 bytes). The targeted sync still skipped all service restarts.
- The parity test now sets a 15-second transaction-local statement timeout
  and a 5-second lock timeout before creating temporary objects. This limits
  the remote check to one connection, temporary functions/table, scalar
  fingerprint/size comparisons and transaction rollback. No outcome labels
  or trading rows are selected or mutated.
- Current result: the local behavior suite passes, but SQL/Python compact
  fingerprint parity remains unverified online; do not activate a code
  service or claim this repair is complete. Next: confirm the warm-up unit is
  no longer running, then run only
  `test_postgres_and_python_snapshot_compaction_match_in_temporary_functions`
  under a unique transient unit with one CPU, 1 GiB, 300-second deadline,
  transaction-local statement/lock timeouts and a unique temporary pytest
  directory. Record the three diagnostic fingerprints and full comparison
  result; repair the smallest source discrepancy and repeat focused tests
  before considering any service restart.

### Compact DDL Installation Recovery And Fresh Capture Check (2026-10-23)

- Root cause was confirmed in the running paper service: the PostgreSQL
  migration loop sent one prepared statement containing three `CREATE
  FUNCTION` commands. `asyncpg` rejected it with
  `cannot insert multiple commands into a prepared statement`, leaving
  `bb-paper-trading.service` in an automatic restart loop (`NRestarts=26`
  during the first diagnosis).
- `postgres_training_feature_snapshot_functions()` now returns four
  independent DDL statements in dependency order: storage bytes, JSONB
  normalization, market-fact compaction, and training-snapshot compaction.
  A regression asserts exactly one `CREATE OR REPLACE FUNCTION` per returned
  statement. Local focused verification passed `43 passed, 1 skipped`;
  Ruff, compilation, and whitespace checks passed.
- The owned source/test paths were uploaded with explicit `--only` filters
  and no broad worktree sync. The remote isolated PostgreSQL parity test
  passed `1 passed`; the remote combined DDL/compaction suite passed
  `44 passed`. The test used one connection, temporary functions, a
  transaction-local 15-second statement timeout and 5-second lock timeout.
- After the upload, the paper process recovered without a duplicate restart:
  `bb-paper-trading.service` stayed `active/running` with the same main PID
  and `NRestarts=72` across a 30-second bounded check. Runtime settings were
  `mode=paper`, `paper=True`, `live=False`, and no new prepared-statement
  error appeared after the 2026-10-22 19:01:38 UTC process start.
- Fresh post-recovery capture from the immutable boundary
  `[2026-10-22T19:01:38Z, current check]` contained 36 market decisions.
  Python compaction had maximum frozen size `96,902 B` against the `98,304 B`
  budget, with zero over-budget rows and zero seal errors. The 45-minute
  diagnostic that included pre-recovery rows is not used as fresh evidence;
  its legacy seal errors remain quarantined.
- The same fresh boundary had 38 shadow decisions with the new availability
  contract and storage contract. Twenty-two passed component-availability
  checks; sixteen remained quarantined. The dominant reasons were missing or
  unobserved index/mark receipt clocks and missing order-book facts. A
  bounded source projection showed valid REST fallback facts on many rows,
  while other rows had explicit zero/empty exchange responses; no defaults
  were treated as observed data.
- Full local repair/regression group passed `713 passed, 1 skipped`; no
  profitability labels were opened by this batch. The exact same 23-file
  remote focused group then passed `714 passed` in 99.05 seconds with one
  CPU, 1 GiB memory and a 300-second deadline. The remote run used
  `DATABASE_MAX_OVERFLOW=0` without forcing pool size to one, preserving the
  pool-concurrency regression's intended precondition. Remote source
  integrity passed for all 51 explicit paths; only
  `tests/test_feature_availability_compaction.py` needed an upload.
- Final read-only runtime check: paper and Dashboard are active; paper main
  PID `2530425` and `NRestarts=72` were stable; runtime and persisted control
  both report paper mode, live false, running true and unpaused. No prepared
  statement error has appeared since the recovered process start, and no
  profitability/frozen/G2 audit unit remains active. No order or protection
  action was taken.
- G2 remains `blocked_by_component_availability_evidence`; G3-G6 remain
  pending. Next exact action: repair or instrument the remaining
  index/mark/order-book receipt paths, then collect a fresh bounded post-fix
  window and require zero contract errors plus complete component clocks
  before any candidate research or exam is opened.

### Reference-Route Follow-Up (2026-10-23)

- After the second paper-only restart at `2026-10-22T20:46:45Z`, the first
  seven fresh market decisions had order-book receipts `7/7` and frozen
  payloads between `91702` and `96642` bytes (`0/7` over the 96 KiB budget).
  However, mark receipts were `5/7` and index receipts `4/7`; three rows had
  empty reference responses under burst contention. The rows stayed
  quarantined and no order was opened.
- An isolated bounded public-route probe for `PEPE/USDT`, `OP/USDT` and
  `IRYS/USDT` completed with positive mark/index prices and independent
  receipt clocks for all three. This distinguishes route contention/latency
  from exchange endpoint unavailability.
- Repair: increase the bounded reference-route join window from `1.5s` to
  `3.0s` while retaining single-flight requests and fail-closed facts. The
  earlier fallback repair also retries the authoritative ticker batch when
  the short ticker route returns only a background marker. Local focused
  verification after these changes passed `146` tests; Ruff and whitespace
  checks passed. The next step is targeted upload, remote focused verification,
  paper-only restart, and a fresh sample requiring mark/index/order-book
  receipt coverage before considering this sub-gate complete.

### Recovery Continuation: Receipt Diagnostics And Budget Recheck (2026-10-23)

- Local focused verification for the current working tree passed
  `202 passed, 1 skipped`; the skipped case is the PostgreSQL temporary
  transaction parity check because `BB_TEST_POSTGRES_URL` is not configured
  locally. Ruff, compilation and `git diff --check` passed.
- A bounded online entry/recovery audit at
  `2026-10-22T20:28:05Z` covered the preceding three hours. All 40 recent
  market decisions were `insufficient_evidence`/HOLD; no order or protection
  action was taken. Paper mode and `live=false` remained in force. The
  execution gate remains disabled because the authoritative local-ML evidence
  is fee-after-return negative (`average_return=-0.311071%`,
  `return_lcb=-0.441736%`, `profit_factor=0.549345`), while other sources
  have no accepted fee-after-return sample.
- The same bounded report found quarantines dominated by missing or
  unobserved `orderbook`, `mark_price` and `index_price` facts. Pending
  shadow rows remain disclosed and are not treated as valid labels. This is
  the current G2 blocker, not a reason to relax gates or force trades.
- A 60-minute frozen-payload projection at `2026-10-22T20:27:10Z` found
  `23/64` raw decision snapshots over the 96 KiB frozen budget
  (maximum `100708` bytes). The largest contributor was the full
  `market_fact.source_consistency` projection. This is a storage/eligibility
  defect to resolve before training; it is not profitability evidence.
- Added a bounded-cache regression for the 256-symbol native 1m evidence
  cache. The next action is explicit-path online verification of the current
  availability/market-fact/data-service batch, followed by a fresh
  post-deploy projection. G2 remains blocked until newly captured rows have
  complete component receipts and the payload budget is clean; G3-G6 remain
  pending.

### Order-Book Receipt Follow-Up (2026-10-23)

- The current availability/compaction/reference batch was uploaded with
  explicit paths and verified remotely. The local focused group passed
  `150 passed, 1 skipped`; the same remote group passed `150 passed,
  1 skipped`. The PostgreSQL skip is expected because the local dedicated
  `BB_TEST_POSTGRES_URL` is unset.
- Only `bb-paper-trading.service` was restarted after the targeted upload.
  At `2026-10-22T21:14:03Z` it was `active/running` with `NRestarts=0`;
  runtime mode remained `paper` and live execution stayed disabled.
- A new bounded budget audit distinguishes the raw decision snapshot from the
  actual training snapshot. In the post-deploy window ending
  `2026-10-22T21:16:05Z`, the training column had 64 rows, maximum frozen
  payload `96,636 B`, and `0/64` over the `98,304 B` limit. The raw
  `AIDecision.feature_snapshot` column still had `13/37` rows over the limit
  (maximum `100,014 B`); this is retained for audit/execution context and is
  not accepted as training evidence. The training payload budget therefore
  passes, while raw decision storage remains disclosed.
- The fresh availability projection after the first restart contained 8
  market decisions from `2026-10-22T21:16:51Z` through `21:20:11Z`: all 8
  were sealed, 6 were clean, and 2 were quarantined for missing order-book,
  mark, and index facts. A later 11-row slice had 9 clean and 2 with
  `orderbook:fact_missing`; no row was authorized for entry. These are
  component-lineage results, not profitability evidence.
- The remaining order-book route was allowed a bounded `4.0 s` single-flight
  join window (below the existing six-second derivatives deadline), and
  timeout results now carry an explicit `orderbook_timeout` marker. Local
  regressions passed `204 passed, 1 skipped`; the remote order-book/reference
  group passed `51 passed`. A second paper-only restart completed at
  `2026-10-22T21:30:31Z` with `active/running` and `NRestarts=0`.
- G2 is still blocked. The next evidence boundary is the fresh sample after
  `2026-10-22T21:30:31Z`; require zero order-book/mark/index availability
  errors, complete receipts, and a clean training payload before considering
  this sub-gate passed. Negative fee-after-return evidence remains a separate
  execution blocker; G3-G6 stay pending.

### Post-Restart Receipt Audit (2026-10-23)

- The paper unit entered `active/running` at
  `2026-10-22T21:30:31Z`; a read-only check at `21:48:52Z` found the same
  active process, `NRestarts=0`, no matching profitability/frozen/G2 audit
  unit, and no running market-data warm-up. No order or protection operation
  was performed. The runtime audit continued to report paper-only decisions
  and disabled model execution permission.
- A bounded projection over `[2026-10-22T21:30:31Z, 21:53:29Z)` returned
  56 market decisions, all `hold` and not executed. Forty-five passed the
  component-availability check; eleven were quarantined. Reasons included
  nine missing order-book facts, seven missing mark facts, seven missing
  index facts (the mark/index gaps overlap), one zero index-price receipt/
  clock contradiction, and one stale sentiment availability contract.
  These are input-integrity findings, not strategy outcomes.
- The earlier 42-row route projection showed six order-book timeouts and
  four reference-price timeouts, with three rows timing out on both routes;
  one additional row returned an empty index fact while the wrapper route
  appeared completed. This confirms that route completion alone is not proof
  of usable facts. Existing per-instrument order-book single-flight means
  ticker and derivatives callers may join the same task; duplicated network
  requests are not established as the cause.
- A fresh 15-minute training-snapshot budget projection (64 rows, after the
  restart) had no seal errors and no rows above the `98,304 B` limit.
  Frozen payload sizes were `89,980-96,872 B`, with p50 `96,018 B` and p95
  `96,872 B`. This passes storage budget only, not component validity or
  profitability.
- The bounded execution audit still found `120/120` recent market decisions
  blocked as `insufficient_evidence`. Authoritative local-ML fee-after-return
  evidence remains negative: 261 samples, mean `-0.311071%`, LCB
  `-0.441736%`, and PF `0.549345`; model influence and execution remain
  disabled. A read-only historical requalification dry-run scanned 110,709
  rows and found zero eligible rows because the historical rebuild/label
  availability contract is stale or missing.
- A one-time host sample showed the paper process at 100% of one CPU and
  `2,292,172 KiB` RSS, with CPU-pressure `some avg10=2.72`; this is a
  contention lead, not proof that CPU caused the route misses. Filtered
  privileged journal queries returned no matching route-timeout diagnostics,
  so request queue-wait/provider-duration evidence is still missing. Do not
  increase route deadlines or concurrency from this snapshot alone.
- Current local `HEAD` is `92f8f4a`, one commit ahead of `origin/main`;
  unrelated `db/session.py` remains modified and untouched. No code change was
  made in this checkpoint. G2 remains blocked, and G3-G6 remain pending.
- Next exact action: instrument or otherwise obtain bounded request queue-wait
  and provider-duration measurements for ticker, order-book and reference
  routes without logging credentials/raw responses. Then correct the measured
  bottleneck with focused regressions and a paper-only deployment if a code
  defect is demonstrated. Continue fresh component projections; require zero
  availability errors and the existing payload budget before G2 can pass.
  Keep historical labels quarantined and do not open/tune an exam.

### Latency Evidence And Fresh Boundary (2026-10-23)

- The owned latency diagnostics are present online byte-for-byte for
  `data_feed/okx_rest_client.py`, `services/trading_params.py` and their two
  focused test files. The paper process entered `active/running` at
  `2026-10-22T22:15:18Z` and remained at `NRestarts=0`; persisted control is
  still `mode=paper`, `paused=false`.
- A bounded journal projection from that restart boundary through
  `2026-10-22T22:26:59Z` contained 620 route events. The route summaries were:
  `fetch_order_book` 167 calls (queue p50/p95/max
  `0.0/3389.8/10536.0 ms`, provider p50/p95/max
  `1616.0/8241.5/12628.6 ms`, 10 timeouts);
  `publicGetMarketIndexTickers` 212 calls (queue
  `134.0/4836.5/8354.3 ms`, provider
  `1409.8/7604.0/13121.9 ms`, 9 timeouts);
  `publicGetMarketTicker` 67 calls (queue
  `606.4/4165.3/5514.9 ms`, provider
  `2401.4/8016.2/11262.9 ms`, 3 timeouts); and
  `publicGetPublicMarkPrice` 174 calls (queue
  `0.0/4803.8/8014.8 ms`, provider
  `998.4/5441.7/12480.1 ms`, 2 timeouts). This establishes both provider
  latency and public-route contention; it does not justify increasing
  concurrency or route deadlines without a controlled comparison.
- A 10-second cgroup sample during the same active paper process measured
  approximately `1.02` CPU cores used against a two-core quota,
  `2,235,359,232` bytes current memory, and zero `avg10` CPU/memory PSI
  pressure. This short sample does not rule out earlier contention, but it
  does not show current host saturation as the primary explanation for the
  route delays.
- The immutable fresh availability boundary
  `[2026-10-22T22:15:18Z, 2026-10-22T22:27:58Z)` contained 21 market
  decisions, all HOLD/not executed. Fourteen were availability-clean and
  seven were quarantined: four each for missing order-book, mark and index
  facts, with two additional index receipt/clock ambiguity cases. No row was
  authorized for entry.
- In the same boundary, the training snapshot audit covered 64 rows with
  zero seal errors and zero rows above the `98,304 B` training budget
  (p50 `93,882 B`, p95/max `96,482 B`). The raw decision snapshot remained
  over budget for 6/21 rows (maximum `99,872 B`); raw context is not accepted
  as training evidence and remains disclosed.
- The authoritative fee-after quality gate is unchanged and negative:
  `n=261`, mean `-0.311071%`, lower bound `-0.441736%`, profit factor
  `0.549345`; model influence remains disabled. Historical requalification was
  not rerun; the prior bounded `110,709`-row scan with zero eligible rows is
  still the current evidence. G2 therefore remains
  `blocked_by_component_availability_evidence`; G3-G6 remain pending.
- Local targeted tests passed `44/44`, Ruff and whitespace checks passed, and
  remote source hashes match the owned working-tree files. The next exact
  action is to keep collecting a sufficiently large post-restart sample and
  isolate endpoint latency from gate queueing before changing concurrency or
  timeouts; do not promote a candidate, open a fresh exam, or relax
  quality/risk gates.
- Owned source/tests and this checkpoint were committed locally as
  `fef731f`. The targeted sync confirmed all four remote hashes and uploaded
  zero changed files, with restart skipped. `git push origin main` then
  failed three times with a GitHub HTTPS connection timeout; local `main` is
  three commits ahead of `origin/main`. The unrelated pre-existing
  `db/session.py` modification remains unstaged and untouched. Retry the push
  when GitHub connectivity is available.

### Compact Budget Repair And Fresh Post-Restart Boundary (2026-10-23)

- The next incomplete G2 sub-gate was the training-snapshot compaction
  contract. `compact_market_fact()` previously returned an already-contracted
  object unchanged, so a verbose old storage contract could remain above the
  frozen budget. Recompaction also dropped
  `missing_open_times_count`/`missing_open_times_fingerprint` when the raw
  millisecond list had already been removed. Both Python and PostgreSQL
  compaction now always re-project the native fact and preserve those bounded
  metadata fields. The provenance fallback reason is capped at 160 characters
  in both implementations to prevent duplicated diagnostic text from
  consuming the model-input budget; full source identity remains in the
  existing fact and source-consistency fingerprints.
- The online frozen-budget audit now reports
  `training_feature_compaction_error` in addition to availability seal errors.
  This prevents a failed training snapshot from being misreported as a
  budget-clean row with a 32-byte error object.
- Local focused tests passed `58 passed, 1 skipped`; the skip is the local
  PostgreSQL temporary-function check because `BB_TEST_POSTGRES_URL` is unset.
  The targeted remote group passed `45 passed`, including the PostgreSQL
  Python/SQL parity test with the runtime database URL translated to an
  `asyncpg`-compatible scheme. Ruff, compilation and whitespace checks passed.
  Only the four owned source/test paths were uploaded; no broad worktree sync
  or service restart occurred during verification.
- `bb-paper-trading.service` was then restarted once at
  `2026-10-23T00:17:47Z` UTC and remained active with `NRestarts=0`; paper
  mode and live execution remained disabled. The strict boundary
  `[2026-10-23T00:17:47Z, 2026-10-23T00:23:46Z)` contained three market
  decisions, all sealed and availability-clean, with zero executions. Twelve
  training snapshots in the same boundary had no compaction errors and frozen
  sizes `65,490-66,812 B`, all below `98,304 B`. This is a positive repair
  result but not a completed G2 acceptance sample.
- A wider eight-minute diagnostic after deployment covered 64 training rows:
  all 64 had no seal or compaction errors, with frozen sizes
  `95,606-97,250 B` and zero over-budget rows. The wider market projection
  still mixed pre-restart rows and showed 19 market decisions with 11 clean
  and 8 quarantined; the strict post-restart boundary above is the evidence
  for the repaired code.
- The remaining G2 requirement is a larger independent post-restart market
  sample. Preregistered for this repair boundary: at least 30 consecutive
  market decisions spanning at least 20 distinct symbols, every row sealed
  and with zero order-book/mark/index receipt or clock errors; every training
  snapshot in the same boundary must have zero compaction/seal errors and
  remain at or below `98,304 B`. Continue collecting it before any candidate
  research or exam. The fee-after quality gate remains negative (`n=261`,
  mean `-0.311071%`, LCB `-0.441736%`, PF `0.549345`), so model influence and
  order-capable paper execution remain disabled; G3-G6 stay pending.

### Optional Diagnostic Budget Repair And New Boundary (2026-10-23)

- The first post-deploy budget audit exposed a real distinction: four
  `IRYS/USDT` rows had frozen inputs below the 96 KiB model-input budget
  (`97,010 B`) but their non-frozen diagnostic fields pushed the wider
  128 KiB snapshot budget over the limit. The old implementation rejected the
  entire training snapshot as `snapshot_budget_exceeded`, even though the
  sealed training inputs were valid.
- Repair: `compact_training_feature_snapshot()` and the PostgreSQL
  `bb_compact_training_feature_snapshot()` function now iterate optional
  diagnostic keys in stable lexical order and retain only candidates that fit
  the remaining 128 KiB budget. Frozen fields still fail closed on a genuine
  frozen-input overflow or storage-fingerprint mismatch; no frozen value is
  truncated or rewritten. A regression covers a large optional-diagnostic
  payload and requires deterministic Python output.
- Local focused verification passed `183 passed, 1 skipped`; remote Python/
  PostgreSQL parity and availability group passed `197 passed, 1 skipped`.
  Targeted source upload and remote integrity verification passed. The paper
  service restarted at `2026-10-23T01:43:58Z` UTC with `NRestarts=0`,
  `active/running`, paper mode and live execution disabled.
- The 10-minute audit ending `2026-10-23T01:45:00Z` still included two
  pre-restart rows (`01:35:47` and `01:37:31`) marked with the old
  `snapshot_budget_exceeded`; those rows are explicitly excluded from the
  fresh boundary. No post-restart market row had yet accumulated by the
  check time. A strict fresh audit must start at
  `2026-10-23T01:43:58Z` UTC and require the existing preregistered
  30-row/20-symbol market sample plus zero training seal or compaction errors.
- Current execution evidence remains negative and unchanged:
  local-ML fee-after-return `n=261`, mean `-0.311071%`, LCB `-0.441736%`,
  PF `0.549345`; model influence and order-capable paper execution remain
  disabled. G2 is still blocked pending the fresh component-availability
  boundary; G3-G6 remain pending.
- Next exact action: collect only rows created after
  `2026-10-23T01:43:58Z` UTC, audit the post-restart training and market
  projections, and retain the gate as pending unless all component receipts,
  clocks and budgets pass. Do not open a candidate exam or manually trade.

### Receipt And Funding-Horizon Recheck (2026-10-23)

- The follow-up repair batch fixed the PostgreSQL optional-field compaction
  loop's `item` record-name collision by using `optional_item`, removed
  nonessential source-consistency diagnostics from the frozen projection, and
  added a bounded sentiment startup warm-up. The targeted online verification
  reported 46 passed and 1 skipped; the paper service restarted at
  `2026-10-23T02:19:31Z`. Local focused retest of compaction, market-fact and
  data-service security regressions passed `128 passed, 1 skipped`.
- A read-only 10-minute online coverage audit at `2026-10-23T02:30:34Z`
  found paper and dashboard services active with zero restarts, persisted
  `mode=paper`, and `paused=false`. It covered 23 market decisions across 23
  symbols; all were incomplete with `funding_evidence_unavailable`, and none
  produced a directional signal or execution. Another 25 candidates were
  deferred for `feature_unavailable`. The observation window was too short
  for the broader coverage-readiness assessment.
- A bounded projection at `2026-10-23T02:37:58Z` covered 37 post-restart
  market HOLD decisions, all unexecuted and sealed. Twenty-five passed
  component availability; twelve remained quarantined: five missing
  order-book facts, three missing mark facts, three missing index facts, six
  ambiguous mark receipts/clocks, eight ambiguous index receipts/clocks, and
  one stale technical availability contract. Counts overlap by row.
- The same 37-row projection found every direction-aware funding projection
  incomplete because no prediction horizon was selected. All 37 reported
  `paper_prediction_horizon_unavailable`; the inspected source-evidence
  counters included 74 `model_strategy_blueprint_paper_permission_missing`
  and 148 `return_distribution_profit_supervision_version_mismatch`
  observations. These counts are evidence-row counts, not unique models or
  decisions. The permission blocker must not be bypassed; the supervision
  version mismatch and source/horizon lineage need investigation before
  another candidate study.
- A read-only 15-minute training-snapshot audit covered 64 rows: zero seal or
  compaction errors, maximum frozen payload `97,582 B`, and zero rows over the
  `98,304 B` limit. The storage-budget sub-gate passes for this sample, but
  does not offset component-availability failures or establish profitability.
- G2 remains blocked; G3-G6 remain pending. Authoritative local-ML
  fee-after-return evidence remains negative (`n=261`, mean `-0.311071%`,
  LCB `-0.441736%`, PF `0.549345`). No order was submitted and no execution
  permission was enabled. Next action: trace the return-distribution
  supervision-version mismatch to its producer/artifact contract and inspect
  why otherwise observed model output lacks an authorized paper prediction
  horizon; preserve all quality and risk gates, then collect a fresh
  post-fix component-clean boundary before G3.
- The owned source/test/plan paths are committed locally and deployed with
  targeted online tests already recorded above. Push to `origin/main` failed
  because the local host could not connect to `github.com:443`; retry when
  network connectivity returns. The unrelated `db/session.py` modification
  remains unstaged and excluded.

### Return-Distribution Source Follow-Up (2026-10-23)

- A further bounded read-only projection at `2026-10-23T02:44:56Z` covered
  32 recent market decisions. Every decision still lacked an eligible
  prediction horizon. Across source/side evidence rows, `server_profit` and
  `timeseries` each supplied 64 contracts tagged
  `2026-07-24.separated-profit-supervision.v2`; the current code requires
  `2026-10-22.separated-profit-component-availability.v4`. The 64 `local_ml`
  source/side rows had no return-distribution supervision version and were
  blocked by `model_strategy_blueprint_paper_permission_missing`.
- This confirms the no-horizon finding is downstream of stale or
  unauthorized return-distribution evidence; it is not a missing-cost
  permission that can be repaired by accepting the old version. Keep these
  sources observation-only. They need valid current-contract artifacts and
  independent fee-after evidence before they can provide an eligible horizon.
- G2 remains the first incomplete gate because 12 of the 37 audited decisions
  still had component-receipt/clock or technical-availability errors.
  Do not start or tune G3 using the currently blocked data/artifacts. First
  obtain the preregistered consecutive clean availability sample and
  error-free, in-budget training snapshots; then retrain/re-evaluate only
  within the governed development protocol and preserve a new independent
  exam.

### Preregistered Route-Wait Intervention (2026-10-23)

- Read-only evidence boundary: paper process start
  `2026-10-23T02:19:31Z` through `02:54:09Z`; 33 recent market decisions
  were all HOLD/unexecuted, with 16 availability-clean and 17 invalid.
  A separate 15-minute training audit covered 64 rows with zero seal or
  compaction errors and maximum frozen size `97,582 B` (budget `98,304 B`).
- A bounded privileged-journal projection selected at most 1,000 request
  latency events and 143 partial-snapshot events. It showed order-book route
  p95 queue wait/provider duration of `3,847/4,574 ms`; mark-price p95
  `5,617/3,606 ms`; index-price p95 `5,534/3,724 ms`. Partial snapshots
  recorded 130 order-book and 32 reference-price caller timeouts. The request
  SDK still has an 8-second hard timeout, while the inner reference/order-book
  waits are 3.0/4.0 seconds and the outer required-derivatives snapshot cap is
  6 seconds. The latency-event query hit its 1,000-row cap; report these
  percentiles as a capped sample, not a full population.
- Preregistered intervention: change only the reference-price and order-book
  single-flight join waits to `5.5 s`, preserving the existing `6.0 s` outer
  snapshot cap, `8.0 s` provider timeout, concurrency, fail-closed fact and
  timestamp checks, execution authorization, and risk limits. Add a regression
  proving both inner waits remain below the outer cap. This tests whether
  callers were persisting incomplete facts before already-queued requests
  finished; it is an availability experiment, not profitability tuning.
- Post-deploy acceptance boundary begins at the exact paper-service restart
  time. Require at least 30 consecutive market decisions across at least 20
  symbols, every row sealed with zero order-book/mark/index receipt or clock
  errors, plus zero training seal/compaction errors and all same-boundary
  snapshots at or below `98,304 B`. Keep G2 blocked on any failure. Compare
  request/partial-route aggregates against the capped baseline; do not treat
  this as a strategy exam or read return labels.

### Live Recheck And Route-Wait Patch (2026-10-23)

- Read-only coverage check at `2026-10-23T03:04:01Z` covered
  `[02:49:01Z,03:04:01Z)`. The paper, dashboard, and model-tunnel services
  were active with zero restarts; persisted control was unpaused `paper`.
  Thirty-five market decisions across 31 symbols were incomplete with
  `funding_evidence_unavailable`; the entry funnel had zero signals and zero
  executions. The coverage window was not ready. This is a current direct
  blocker to entries, distinct from route timeouts.
- A same-time bounded training-snapshot check covered 64 rows. All had zero
  seal/compaction errors; frozen payloads were `94,362-97,582 B` with zero
  above the `98,304 B` budget. This passes storage budget only.
- Implemented the preregistered local route-wait intervention:
  reference-price and order-book waits are now `5.5 s`, each strictly below
  the unchanged `6.0 s` outer snapshot cap. Concurrency, provider timeout,
  fact/timestamp validation, execution permissions, and risk limits are
  unchanged. Regression asserts the exact wait values and both inequalities.
  Focused local suite passed `42` tests; Ruff and `git diff --check` passed.
- Local source/test SHA256 values are
  `7e408f7ad3d585869a6d7a8f47d183ff7706e0f3bee345ecbfad7ef2dc659e33` and
  `719523ef3ba895be9ad388f9e3fecb80934a3b2bc9e7d12ee152d32f4ca49ac6`.
- Staged targeted upload completed for only the source and test paths; remote
  integrity validation passed and the focused remote suite passed `42` tests.
  The split-service deployment restarted the paper, dashboard, and model
  tunnels at `2026-10-23T03:10:41Z` (tunnels at `03:10:30Z`). The first
  post-restart read-only check at `03:11:21Z` confirmed paper/unpaused state,
  active services, no automatic service restarts, and zero open paper positions.
  Its one market decision was incomplete with
  `funding_evidence_unavailable`; there were zero signals/executions. The
  one-minute training query had zero rows, so there is not yet any post-restart
  training-budget evidence. This sample is explicitly too small to evaluate
  the preregistered route intervention.
- Baseline HEAD for this batch was `9d5b903`. The pre-existing
  `db/session.py` modification remains unrelated, unstaged, and untouched.
  The targeted source/test deployment did not include unrelated working-tree
  files.
- Next: collect only rows at or after the exact trading restart time above.
  Require 30 consecutive market decisions across at least 20 symbols with
  zero order-book/mark/index receipt or clock errors, and same-boundary
  training snapshots with no seal/compaction errors and maximum frozen size
  `98,304 B`. Separately trace `funding_evidence_unavailable`, stale
  return-distribution supervision contracts, missing authorized paper
  horizons, and missing independent fee-after evidence. Keep G2 blocked unless
  the component sample passes; do not treat the route adjustment as evidence
  of profitability or relax the model permission gate.
- Code-path trace confirms the current funding blocker is not repaired by the
  route-wait change: `services/entry_direction_competition.py` can derive a
  paper funding horizon only from a selected eligible prediction cohort, an
  authorized horizon, or an eligible paper-training cohort. If all are
  absent, it requires a funding projection without inventing a horizon;
  `ai_brain/ensemble_coordinator.py` then marks the analysis incomplete when
  funding evidence is not complete. The last per-source audit at
  `2026-10-23T02:44:56Z` found server-profit/timeseries contracts on stale v2
  supervision and local-ML contracts without paper permission. The immediate
  post-restart row still has `funding_evidence_unavailable`. Do not whitelist
  stale contracts or grant local-ML permission; resolve those upstream only
  with a current-contract artifact and separate fee-after evidence.
- Profitability remains unqualified: fee-after-return evidence is still
  `n=261`, mean `-0.311071%`, LCB `-0.441736%`, PF `0.549345`. G3-G6 remain
  pending; this route change is not a profitability fix or trade authorization.

### Fresh Read-Only Check (2026-10-23)

- A bounded eight-minute coverage audit completed at
  `2026-10-23T03:30:42Z`, after the route-wait deployment restart at
  `03:10:41Z`. Paper remains unpaused, trading/dashboard/model-tunnel services
  are active with zero automatic restarts, and there are zero open paper
  positions. The audit observed 18 market decisions across 18 symbols; all 18
  were incomplete with `funding_evidence_unavailable`, with zero signals and
  zero executions. The decision-count/symbol preregistration (30/20) is not
  yet met, so this is not route-intervention acceptance.
- The same coverage check found 35 deferred candidates (33
  `feature_unavailable`, two `shortlist_capacity`) and marked the observation
  assessment not ready. This is additional current evidence that the entry
  funnel is blocked upstream of order execution, not evidence that entries
  would be profitable if forced through.
- A same-window training-snapshot audit completed at
  `2026-10-23T03:30:57Z`: 64 rows, zero seal errors, maximum frozen payload
  `97,590 B`, and zero rows above the `98,304 B` budget. The storage-budget
  sub-gate passes; it does not offset the market/funding blockers.
- The last authoritative fee-after quality evidence remains negative
  (`n=261`, mean `-0.311071%`, LCB `-0.441736%`, PF `0.549345`). No strategy
  profitability or paper-entry readiness is established. G2 remains blocked;
  G3-G6 remain pending. Next: resolve the funding-evidence/horizon and current
  supervision-contract path without granting unauthorized model influence,
  then collect the preregistered post-restart market sample. Keep live
  execution disabled and do not submit orders.

### Fresh Status Verification (2026-10-23)

- Read-only 30-minute online coverage audit completed at
  `2026-10-23T03:55:37Z`, covering
  `[2026-10-23T03:25:37Z,2026-10-23T03:55:37Z)`. Paper, dashboard and model
  tunnel services were active/running with zero automatic restarts; persisted
  mode was `paper`, `paused=false`, and there were zero open paper positions.
- The window contained 82 market decisions across 43 symbols. All 82 were
  incomplete; the audit's bounded first 20 incomplete-analysis samples
  reported `funding_evidence_unavailable`. The entry funnel had zero
  directional signals and zero executions. Sixty-two candidates were
  deferred (60 `feature_unavailable`, two `shortlist_capacity`), with no
  overdue symbols. The coverage monitor assessment was `ready=true` for
  monitoring continuity only; it is not the G2 component-availability,
  entry-authorization, or profitability acceptance gate.
- A same-window frozen training-snapshot projection inspected 32 rows: zero
  seal/compaction errors, maximum frozen payload `94,160 B` against the
  `98,304 B` limit. This storage-budget subgate passes for the inspected
  rows only.
- Read-only `GET /models/status` on the Phase 3 model host found no
  `training_process.json` runtime marker, a current artifact pointer at
  `20260919T201515268231Z-7dab2e19` with lifecycle `canary`, no candidate
  pointer, and challenger `20261021T130459127095Z-79a46eb5`.
  `live_ml_ready=false`. The API reports
  `2026-07-24.separated-profit-supervision.v2`, while the repository's
  authoritative contract is
  `2026-10-22.separated-profit-component-availability.v4`. The API's latest
  recorded training time is `2026-10-21T13:04:59Z`; no new qualifying or
  promoted artifact is visible in the current pointer state.
- G1 remains passed as an attribution audit with negative net outcomes; G2
  remains blocked and G3-G6 remain pending. The fee-after quality evidence
  remains `n=261`, mean `-0.311071%`, LCB `-0.441736%`, PF `0.549345`.
  The local repository is at `2397e0c`, equal to `origin/main`; unrelated
  `db/session.py` and this checkpoint's existing uncommitted documentation
  change are preserved.
- Next: align the Phase 3 API's supervision contract to v4 with focused
  regressions that continue to reject old v2 artifacts; keep any new v4
  artifacts shadow-only until independently qualified. Separately resolve
  the missing authorized prediction horizon and collect a fresh
  component-clean G2 boundary. Do not treat monitoring readiness, storage
  budget, a successful training process exit, or a canary pointer as evidence
  of profitability or order authorization.

### Status And Remote Regression Update (2026-10-23)

- The Phase 3 API source now has the v4 supervision contract and keeps
  candidate promotion shadow-only until independent-exam promotion is
  explicitly enabled. Local source SHA256
  `a3a1423441c4c2fdd2a2c76b23fcda18de95c397c1da4e7d1e6b21079340e072`
  matched `/data/bb/app/scripts/phase3_quant_api_service.py`. The online test
  file was initially empty after a failed transfer; only
  `tests/test_local_ai_tools_deploy_service.py` was then uploaded with
  `--skip-restart`, and remote integrity verification passed. The focused
  remote regressions for shadow-only promotion and v4/legacy-v2 rejection
  passed `3 passed` in `3.74s`; the complete local test file passed
  `99 passed`. No trading service was restarted for this test-file repair.
- Read-only online coverage audit at `2026-10-23T04:42:58Z` covered
  `[04:12:58Z,04:42:58Z)`. Persisted mode remained `paper`, unpaused; trading,
  dashboard and model tunnels were active with zero restarts since
  `03:10:41Z`/`03:10:30Z`. There were 29 market decisions across 24 symbols;
  all 29 analyses were incomplete, the bounded first 20 incomplete samples
  reported `funding_evidence_unavailable`, and the funnel had zero directional
  signals and zero executions. There were 124 deferred candidates, including
  110 `feature_unavailable`; five symbols were overdue and the coverage window
  was not met. No paper positions were open.
- A bounded feature-budget audit at `2026-10-23T04:44:23Z` inspected six
  decision snapshots: all were sealed, but one exceeded the `98,304 B` frozen
  budget; maximum size was `102,774 B`. A separate training-snapshot audit at
  `2026-10-23T04:46:58Z` covered 16 rows with zero seal errors and a maximum
  frozen size of `90,978 B`; this passes only the training-snapshot budget
  sub-gate and does not cancel the decision-snapshot failure.
- G1 remains passed as an attribution audit with negative economics. The
  latest authoritative fee-after quality evidence remains `n=261`, mean
  `-0.311071%`, LCB `-0.441736%`, PF `0.549345`; no profitability or paper
  entry readiness is established. G2 remains blocked on funding/horizon and
  feature-availability evidence, including the decision-snapshot budget
  breach. G3-G6 remain pending. Local HEAD and `origin/main` remain
  `2397e0c`; the Phase 3 source, test and this plan have uncommitted changes,
  while the unrelated `db/session.py` modification remains untouched.
- Next: identify and correct the decision-snapshot over-budget component
  without altering frozen inputs; trace why eligible current-contract
  prediction evidence still does not provide a paper funding horizon; then
  collect a fresh preregistered component-clean sample. Keep model influence
  and order-capable paper execution gated until independent fee-after
  profitability acceptance passes. Do not submit orders or claim the current
  candidate is profitable.

### Budget Audit Correction And Live Blocker Trace (2026-10-23)

- Corrected the frozen-payload audit to call the production
  `compact_frozen_feature_payload()` and `frozen_feature_payload_size()`
  functions. The old audit counted verbose pre-compaction `market_fact`
  diagnostics and therefore reported a false frozen-input budget breach; the
  frozen inputs themselves were not modified or truncated. A regression now
  uses a large source-consistency diagnostic payload that compacts below the
  budget and asserts the online audit calls the production projection.
- Focused local verification passed `29 passed, 1 skipped`; Ruff and
  `git diff --check` passed. The core projection, corrected audit script, and
  focused test were uploaded with `--skip-restart`; source integrity passed
  for all three files. No service was restarted for this audit-only change.
- Corrected read-only online budget audit at
  `2026-10-23T05:10:23Z`, last 30 minutes, limit 30 per source:
  decision snapshots `30/30` sealed, `0` seal/compaction errors, maximum
  compact frozen payload `95,726 B`; training snapshots `30/30`, zero errors,
  maximum `93,202 B`; both are below `98,304 B`. This supersedes the
  `2026-10-23T04:44:23Z` decision-budget breach. The storage-budget subgate
  passes for this bounded sample only.
- Read-only coverage at `2026-10-23T05:11:28Z` covered 48 market decisions
  across 37 symbols. All 48 were incomplete with
  `funding_evidence_unavailable`; there were zero directional signals, zero
  executions, zero open paper positions, and zero model failures. The
  candidate monitor still had 8 overdue symbols and the coverage window was
  not met; 10 candidates were deferred for `feature_unavailable` and 11 for
  `shortlist_capacity`.
- Source-level horizon audit covered 50 market decisions, IDs `689131-689180`,
  from `2026-10-23T04:59:08Z` through `05:19:34Z`. All were HOLD with
  `funding_evidence_unavailable`; no authorized, selected, or training horizon
  existed. Both funding sides failed with `prediction_horizon_missing`.
  Across side/source evidence rows: `local_ml` had no horizon or supervision
  version and was blocked by
  `model_strategy_blueprint_paper_permission_missing` (100 rows);
  `server_profit` and `timeseries` had 10-minute display horizons but were
  `signal_unavailable`, ineligible, and had no current supervision version
  (100 rows each). Do not treat display horizons as eligible forecasts or
  substitute them into funding estimates.
- Read-only strategy/training status at `2026-10-23T05:22:37Z` confirms the
  paper/dashboard/model-tunnel services are operating and the OKX
  reconciliation gate allows new entries, but no strategy currently authorizes
  one: local ML readiness is disabled, the blueprint is `unversioned` /
  `unregistered` with no eligible paper sides, the paper champion is
  unavailable, and all 73 scheduled strategy candidates are rejected (zero
  governed). The local profit-quality artifact is `promotion_blocked` with
  negative/stability, incomplete market-fact, and tail-evidence blockers.
  Six local AI models remain canary with non-positive empirical lower-hinge
  blockers. These are qualification failures, not exchange/order-routing
  failures.
- The training schedulers have fresh heartbeats and are not stuck, but the
  latest profitability/return-distribution training attempts were skipped:
  `local_ml_profit_quality` reported
  `authoritative_execution_cost_distribution_unavailable`; local AI
  distribution training reported
  `cost_complete_training_distribution_unavailable`. The current artifact
  therefore cannot supply an authorized horizon, while the models cannot
  produce a replacement from the currently eligible cost-complete rows. The
  last accepted fee-after evidence remains negative (`n=261`, mean
  `-0.311071%`, LCB `-0.441736%`, PF `0.549345`).
- The same read-only strategy check found one orphan paper protection order
  for `TRX/USDT` short with no corresponding open position. It was not
  modified. Keep this under operator review; do not cancel it as part of this
  remediation.
- G1 remains an attribution audit with negative economics; G2 remains blocked
  on current component availability and a usable, authorized current-contract
  prediction horizon; G3-G6 remain pending. The branch remains at `2397e0c`,
  equal to `origin/main`; unrelated `db/session.py` remains untouched.
- Next exact action: trace the bounded current-epoch training rows and
  authoritative cost-distribution builder to identify which required fill,
  fee, funding, slippage, entry-feature, or component-receipt fields are absent
  and why the training probe counts zero new eligible decision groups. Repair
  only a demonstrated producer/join/eligibility defect, add regressions, and
  rerun the preregistered fresh component sample. Do not authorize local ML,
  promote canary artifacts, substitute the display-only 10-minute horizon,
  or open orders to create training data.

### Fresh Online Status And Training Blocker (2026-10-23 18:35 Asia/Shanghai)

- A bounded read-only coverage check completed at
  `2026-10-23T10:34:59Z` (`18:34:59` Asia/Shanghai), covering the preceding
  30 minutes. The persisted mode remains `paper`, `paused=false`; trading,
  dashboard, and model-tunnel services are active with zero automatic restarts.
  There are zero open paper positions.
- The window contains 96 market decisions across 61 symbols. All 96 analyses
  are incomplete with `funding_evidence_unavailable`; the entry funnel has
  `signal_count=0` and `executed_count=0`. The coverage monitor itself is
  healthy (`coverage_window_met=true`); this confirms the no-entry condition is
  an authorization/data condition, not a scheduler stall.
- The bounded horizon audit at `2026-10-23T10:35:04Z` reports no authorized,
  selected, or training horizon for all 96 rows. Both funding sides fail with
  `prediction_horizon_missing`. `local_ml` is blocked by
  `model_strategy_blueprint_paper_permission_missing`; `server_profit` and
  `timeseries` expose only a display horizon of 10 minutes but are
  `signal_unavailable`, have no current supervision version, and are ineligible.
  A display interval is not a usable forecast and must not be substituted into
  the funding projection.
- The current Phase 3 model status remains `artifact_unavailable` /
  `live_ml_ready=false`: no candidate, current, or rollback local-AI artifact
  is registered under the v4 contract; the API reports incompatible legacy
  artifact contracts and no trained local-AI model bundle. Specialist model
  files being present on disk is not evidence that the governed return
  distribution is available.
- A bounded authoritative-training audit at `2026-10-23T10:29:39Z` inspected
  300 recent paper outcomes. All 300 are complete as trade lifecycles and have
  OKX fee/slippage/funding sources plus supervision version
  `2026-10-22.separated-profit-component-availability.v4`, but all 300 are
  excluded because their linked decision feature snapshot lacks
  `feature_availability_contract`; `trainable_count=0`. The wider summary
  over 500 rows reports the same 500/500 exclusion. This is insufficient
  evidence to rewrite or impute historical feature contracts; the rows must
  remain quarantined unless a verified original sealed snapshot can be joined.
- The corrected frozen-feature audit at `2026-10-23T10:28:57Z` found 32/32
  decision snapshots and 32/32 training snapshots within the `98,304 B` frozen
  budget, with no seal/compaction errors. Decision snapshots were only 25/32
  availability-valid (seven had missing/ambiguous order-book or mark/index
  facts); training snapshots were 32/32 valid. The storage sub-gate passes,
  while the current component-validity gate remains blocked.
- G1 remains passed as attribution only, with negative fee-after quality
  evidence (`n=261`, mean `-0.311071%`, LCB `-0.441736%`, PF `0.549345`).
  G2 remains blocked; G3-G6 remain pending. No profitability or paper-entry
  readiness is established, and no order was submitted.
- Exact next action: determine whether the 500 excluded outcomes are
  irrecoverably pre-contract lifecycles or whether a bounded, provenance-safe
  join to an original sealed `AIDecision.feature_snapshot` exists. In parallel,
  keep collecting the post-restart component-validity sample and resolve the
  missing current v4 cost-complete training artifact. Do not grant model
  permission, use the 10-minute display horizon, or open orders to manufacture
  training data.

### Entry Funnel And Training Recheck (2026-10-23 18:35 Asia/Shanghai)

- A second bounded entry-recovery audit completed at
  `2026-10-23T10:35:01Z`. It found 97 market decisions in the last 30 minutes,
  all `hold` and unexecuted; all had `insufficient_evidence`. The latest
  97-row availability projection had 47 clean and 50 invalid snapshots, with
  the invalid rows dominated by missing/ambiguous order-book and mark/index
  facts. This is an additional G2 blocker even though the coverage scheduler is
  healthy.
- Direction selection explicitly rejected the recent candidates for
  `direction_support_expected_net_not_positive`,
  `direction_support_objective_net_missing`,
  `direction_support_quality_observation_loss_probability_too_high`,
  `direction_support_quant_evidence_missing`, and
  `direction_support_prediction_horizon_missing`. No entry permission was
  granted.
- Current online model-quality evidence in this audit was `n=267`,
  average fee-after return `-0.303832%`, LCB `-0.431590%`, and profit factor
  `0.549713`; local-ML paper permission is false. Specialist profit/time-series
  sources each have zero fee-after shadow samples and no paper permission.
  Six local-AI training schedulers are alive but their most recent runs were
  skipped for `authoritative_execution_cost_distribution_unavailable`;
  local-ML profit-quality training was skipped for
  `cost_complete_training_distribution_unavailable` despite reporting 1,769
  newly observed samples. These are data/label readiness failures, not stuck
  scheduler processes.
- The online historical-requalification dry run inspected 110,709 rows,
  reported complete scan coverage but zero eligible or requalified rows.
  Its dominant blockers are missing/stale historical rebuild contracts and
  unverified historical label-availability provenance. Therefore these
  already-inspected rows cannot be reused as a fresh exam or retroactively
  sealed to make them trainable.
- Current conclusion: paper services and candidate monitoring are operational,
  but the system is not presently entry-ready or profitability-qualified. It
  has opened no current paper position; no profitable strategy is established.
  The next work is to verify whether any exact linked trade decision has a
  genuinely original sealed feature snapshot, then repair only a proven
  snapshot producer/join defect. If the old rows predate the contract, retain
  them as quarantined and rely only on newly observed, valid data for later
  development; keep the current unprofitable model and all unauthorized
  sources out of the entry path.

### Executable Quote Cache Repair (2026-10-23 19:40 Asia/Shanghai)

- The no-entry path was repaired at the producer/consumer boundary instead of
  loosening the entry gate. The OKX WebSocket client now subscribes to and
  stores the distinct `index-tickers` channel, and dynamic subscribe,
  resubscribe, unsubscribe and reconnect cleanup cover tickers, books5,
  mark-price and index-tickers together.
- WebSocket books now preserve price-weighted contract depth as raw
  `*_depth_quote_contracts`; `DataService` converts it to executable USDT
  depth only after a complete native instrument specification is available.
  This removes the prior unit error that treated swap contracts as USDT
  quantities.
- `DataService` projects only fresh, source/identity-bound WebSocket
  orderbook/mark/index facts into `fetch_derivatives_snapshot`. The REST
  adapter accepts that projection only when all required facts have valid
  receipt clocks, are no older than 10 seconds, match the expected SWAP/index
  identities and channels, and pass the existing completeness contract.
  Missing, stale, misbound or ticker-derived facts still fall back to the
  existing REST path and remain fail-closed on timeout.
- Verification: the focused market/data suite passed `189` tests; changed
  Python files passed Ruff and `git diff --check`. No online deployment or
  profitability claim has been made yet. The local worktree still contains
  unrelated pre-existing modifications, including `db/session.py`, which
  remains untouched.
- Next exact action: upload only the two OKX data-feed files, the data service,
  and their focused tests; restart only paper/dashboard through the governed
  split-service deploy; then run a bounded read-only check for fresh
  component-valid decisions. Keep model permission, live execution and
  profitability acceptance blocked until the independent fee-after gates pass.

### Targeted Quote-Cache Deployment And Fresh Evidence (2026-10-23 20:58 Asia/Shanghai)

- The pre-deployment bounded check found paper mode, `paused=false`, zero open
  positions, all three existing services active, and no remaining instances of
  the two previously tracked diagnostic PIDs. Local focused verification for
  the quote-cache patch passed `169 passed, 1 skipped`; Ruff and
  `git diff --check` passed.
- Only the three owned runtime files and their three focused test files were
  uploaded. Remote source integrity passed for all six files. The bounded
  remote regression unit ran with one CPU, 1 GiB memory and a 180-second
  deadline; it passed `150 passed` in 43.94 seconds. No stale remote source
  was removed.
- Paper trading and Dashboard were restarted at
  `2026-10-23T12:45:35Z` / `12:45:34Z`. The model tunnel was not restarted.
  Both services were active afterward, the Dashboard returned HTTP 302, the
  persisted mode remained `paper` and unpaused, and there were zero open paper
  positions. This was a deployment check, not an entry or profitability pass.
- The post-restart decision-snapshot audit at `12:57:46Z` covered 23 rows in
  the preceding 15 minutes. All 23 had sealed availability contracts and
  stayed below the 98,304-byte budget; 20 were component-valid and three
  invalid. The one pre-restart invalid row was decision `690155`. The two
  post-restart invalid rows were `690176` (reference-price route timed out)
  and `690177` (orderbook and reference-price routes timed out). The first 20
  post-restart decisions, IDs `690156-690175`, had valid native orderbook,
  mark and index facts. Bounded inspection of decisions `690171-690177`
  showed all four required derivatives routes completed for five rows; the
  last two rows still require fallback/timeout investigation.
- Despite the data improvement, all 15 post-restart market decisions through
  `12:50:41Z` remained HOLD with zero directional signals or executions:
  their actual funding facts were present, but both sides' funding projections
  were ineligible solely because no authorized prediction horizon was
  available (`prediction_horizon_missing`). This repair therefore reduces
  missing component evidence; it does not authorize a model, signal or order.
- A bounded authoritative training-eligibility audit at `12:58:03Z` inspected
  120 recent paper outcomes. All 120 have official OKX fills, fees, slippage
  and funding plus supervision v4, but all are excluded because their linked
  original feature snapshots lack `feature_availability_contract`. A
  three-row exact decision-ID projection dated October 15-16 confirmed those
  original snapshots have zero/absent mark/index source clocks and no sealed
  availability contract. Do not reconstruct or backfill those historical
  contracts from current code or current market data.
- The same-window snapshot audit inspected 23 decision and 40 training rows.
  All were sealed and within the storage budget. The training rows were
  32/40 component-valid; those rows predate this deployment and do not qualify
  as fresh post-restart entry evidence.
- G2 remains blocked; G3-G6 remain pending. The latest accepted fee-after
  evidence remains negative (`n=267`, mean `-0.303832%`, LCB `-0.431590%`,
  PF `0.549713`). The deployed code has not established profitable trading.
  Next: investigate the two new derivative-route timeouts and continue the
  preregistered post-restart component-valid sample; separately obtain a
  v4-compatible, provenance-complete return-distribution artifact without
  importing quarantined pre-contract outcomes or substituting a display-only
  horizon. Live trading remains disabled; no manual orders or protection
  changes were made.

### Compact Contract v2 Repair And Current Entry Blocker (2026-10-23)

- Recovered the in-progress compact-contract repair and preserved all other
  staged/unstaged paths. The prior regression had added only 60 characters
  after selecting a payload near the legacy/v2 boundary, so it did not
  reliably cross the frozen-input budget. It now adds a larger deterministic
  payload, compacts it through the production projection, asserts the measured
  size exceeds `98,304 B`, then verifies fail-closed availability rejection.
  Ruff import ordering is fixed.
- Focused local regression group passed `124 passed, 1 skipped`; Ruff and
  `git diff --check` passed. The skipped case is the PostgreSQL parity test:
  `BB_TEST_POSTGRES_URL` is not configured in this environment. The compact
  contract v2 core and its one regression file were uploaded using explicit
  `--only` paths and `--skip-restart`; remote source integrity passed and the
  remote focused regression passed `22 passed, 1 skipped`. No service restart
  was needed for this storage-contract-only change.
- A subsequent targeted restart of paper trading and Dashboard found both
  active with zero automatic restarts and Dashboard HTTP 302. Runtime remains
  `paper`, running and unpaused; database projection reports zero open
  positions in either mode. The service timestamps advanced to
  `2026-10-23T14:10:37Z` / `14:10:36Z` for paper/Dashboard; model tunnels were
  not restarted. This restart validates startup, not entry or profit.
- Latest bounded market-hold audit at `2026-10-23T14:12:38Z` covered 31
  decisions in the preceding 15 minutes; all 31 were HOLD, with no authorized,
  selected or training horizon. Each lacked a valid local-ML paper permission;
  `server_profit` and `timeseries` exposed only unavailable 10-minute display
  output with no current supervision version. Both funding sides therefore
  failed `prediction_horizon_missing`; there were no signals, executions, or
  open positions. This independently confirms the current entry blocker is
  model-artifact/readiness and authorization, not exchange order routing.
- The same decision audit found 23/31 availability-valid snapshots and 8/31
  invalid. All 31 fit below the frozen `98,304 B` budget (maximum `97,438 B`).
  Invalid reasons were missing mark/index facts (seven each), missing account
  fee receipt/availability clocks (four each), and missing orderbook facts
  (three). A bounded projection of the latest 12 decisions shows recent rows
  with complete orderbook/mark/index and fee clocks mixed with recent route
  timeouts, so component availability is still variable and needs a clean
  post-restart sample.
- A separate 32-row recent training-snapshot audit found no seal errors or
  frozen-budget breaches (maximum `97,438 B`) but only 28/32 availability-valid.
  In a separate authoritative sample, 120/120 recent paper outcomes have
  official OKX fills, fees, funding and slippage plus supervision v4, but
  0/120 are trainable because each linked original decision snapshot lacks
  `feature_availability_contract`. The existing negative historical quality
  sample remains `n=267`, mean `-0.303832%`, LCB `-0.431590%`, PF `0.549713`.
- Shadow data collection and settlement are operating, not wedged: recent
  read-only queue counts showed completed and pending 5/15/60/240-minute
  observations, zero overdue pending rows, and 343/343 market decisions with
  the expected four shadow horizons. However, newly completed historical
  OHLCV labels are explicitly `historical_ohlcv_only`; they are research path
  labels, not executable quote evidence and cannot repair missing authorized
  model predictions or be passed off as fills.
- Current acceptance remains G1 passed as negative-economics attribution, G2
  blocked by component availability, quarantined pre-contract trade features,
  and absent authorized current-contract prediction horizon; G3-G6 pending.
  No profitable model, paper-entry readiness or fee-after acceptance exists.
  Do not promote/authorize legacy or canary artifacts, substitute display-only
  horizons, requalify the 120 pre-contract snapshots, or place orders to create
  evidence.
- Next: (1) diagnose why each model endpoint returns unavailable/legacy
  supervision and find the v4 cost-complete model-data path without persisting
  an unqualified artifact; (2) verify the interval from new sealed market
  features through shadow settlement and immutable label availability, while
  keeping historical OHLCV in research-only scope; (3) collect a preregistered
  fresh post-restart component-valid window. Run the dedicated PostgreSQL
  Python/SQL compaction-parity test against a dedicated test database when
  `BB_TEST_POSTGRES_URL` is available. Keep live trading disabled and existing
  exchange protections unchanged.

### Fresh Shadow-Settlement Routing Repair (2026-10-23 22:48 Asia/Shanghai)

- A bounded read-only online audit completed at `2026-10-23T14:48:38Z`.
  Paper mode was running and unpaused; the trading heartbeat and market/
  position error fields were current and clear. No non-HOLD paper decisions
  were recorded in the preceding six hours. The shadow queue had pending rows
  across all four horizons; one recent maintenance run reported `TimeoutError`
  and another was still running when the audit sampled status.
- Both completed shadow rows in the preceding 15 minutes were tagged
  `historical_ohlcv_only`: XRP/USDT 15-minute (scheduled result at
  `14:37:53Z`, updated `14:43:33Z`) and RESOLV/USDT 240-minute (scheduled at
  `14:34:59Z`, updated `14:35:31Z`). Their preserved-candle fingerprints
  verified, but `executable_market_fact_verified=false`; they are market-path
  research evidence, not fee-after executable observations.
- Root cause confirmed in `ShadowBacktestService.update_due`: when the
  historical provider was configured, every matured row entered historical
  OHLCV collection before the native current-fact branch, so even a freshly
  matured row with a usable native observation could not create a native
  result contract. The repository queue already interleaves oldest and newest
  due rows, so old history work does not exclude fresh rows, but it was
  consuming the collection budget before the fresh-row path.
- Local repair: rows whose `due_at` is within the shared 60-second source-lag
  contract now use native current market facts and native price-path
  verification first. Early or late native facts remain pending/quarantined
  under the existing timing contract and are never relabeled as valid. Older
  rows continue through the explicitly non-executable historical OHLCV path,
  after fresh work has used its bounded collection budget. Historical rows
  remain excluded from execution-cost labels; no old feature contract or
  outcome was backfilled.
- Verification: 117 focused tests passed across native/historical settlement,
  label timing, quarantine, training quality and queue behavior. Changed
  Python files passed Ruff and `git diff --check`. A historical-rebuild
  test fixture now seals valid synthetic decision-time inputs for its
  downweighted market-opportunity case and separately asserts that a missing
  original availability contract stays excluded.
- This is a label-production repair, not entry authorization or profit
  evidence. G1 remains passed only as a negative-economics attribution audit;
  G2 remains blocked, and G3-G6 remain pending. No strategy was promoted and
  no order was submitted.
- Targeted deployment completed at 2026-10-23 23:08 Asia/Shanghai. Only the
  two runtime files and four regression/fixture files were selected; five
  changed files uploaded, no stale files removed, and six selected source
  files passed remote integrity checks. The bounded remote regression passed
  `117` tests in `5.87s`. Paper trading and Dashboard were restarted directly;
  both are active, paper remains unpaused, and Dashboard returned HTTP 302.
  Model tunnels were not restarted.
- Fresh post-deploy verification at `2026-10-23T15:10:28Z` showed shadow
  maintenance processing a 25-row batch with no current error. A subsequent
  bounded exact-ID projection found native result facts were collected for
  recent rows, proving fresh rows no longer automatically take the historical
  provider. However, representative new labels remain quarantined because the
  original decision-time snapshots themselves are invalid: SAND/DOT/ARB had
  zero bid/ask depth and missing native orderbook/mark facts; TRX/SUI had
  unreconciled quote sources, no recent one-minute path, and observed prices
  outside that path. Their result facts were clean, but executable label
  eligibility is false. Old historical rebuild rows also remained explicitly
  non-executable and some were quarantined for their pre-contract provenance;
  they were not rewritten.
- Independent entry audit at `2026-10-23T15:12:37Z` covered 33 market
  decisions in the preceding 15 minutes. All 33 were HOLD with
  `prediction_horizon_missing`; local ML lacks governed paper permission,
  while server-profit and time-series outputs are unavailable display-only
  10-minute results with no current supervision version. The zero-entry cause
  therefore remains model/artifact authorization as well as inconsistent
  decision-time market components, not exchange order routing. No positions
  were opened and no profitability result was established.
- Exact next action: trace the two demonstrated producer failures separately:
  (1) missing/zero orderbook and mark facts in the decision feature assembly,
  including the cache-only market screening path and required-route receipts;
  (2) native quote/path temporal reconciliation failures. Repair only proven
  source/merge defects without relaxing completeness, identity, or freshness
  checks. In parallel, identify why the v4 fee-after distribution trainer has
  no eligible rows/artifact without reusing quarantined pre-contract trade
  features. Keep all model permissions blocked until fresh independent
  profitability evidence passes; keep live trading disabled.

### Bounded Training Query Repair And Online Gate Recheck (2026-10-24 01:41 Asia/Shanghai)

- Recovered the bounded shadow-training input repair in
  `core/training_contracts.py`, `services/shadow_backtest_service.py`,
  `services/ml_training_dataset.py`, and
  `tests/test_ml_signal_training_quality.py`. Candidate identity selection
  and time-stratified decision-group sampling now happen in SQL rather than
  streaming all candidate IDs into application memory. The query hard-caps
  the group budget at 5,000, limits returned IDs to the supported 5/15/60/240
  minute horizons and current label version, and fails closed if rows exceed
  the four-horizons-per-group budget. The row and group counters use the same
  candidate contract.
- Verification for this change: local focused ML training-quality tests
  passed `88`; Ruff and `git diff --check` passed. The prior targeted online
  regression for the selected training files passed `124` tests, and the
  selected sources were deployed. This is a query/resource and sample-contract
  fix only; it does not produce valid missing labels, a model artifact, entry
  permission, or profitability evidence.
- Current online state was read-only checked at
  `2026-10-23T17:37Z-17:39Z` (2026-10-24 01:37-01:39 Asia/Shanghai). Paper mode
  is running and unpaused, heartbeat is current, reconciliation permits entry
  and training refresh, and there are zero open positions and zero executed
  entries in the checked recent window. In the preceding 30 minutes, all
  `13/13` market decisions were HOLD: no authorized, selected, or training
  prediction horizon exists. Local ML has no paper permission; server-profit
  and time-series each expose only unavailable 10-minute display results.
- The current local-ML result still reports `2,040` completed samples but
  `cost_complete_sample_count=0`; its chronological training partition has
  zero eligible groups. The artifact is unregistered/unversioned and paper
  execution is disabled. Existing model-quality blockers include missing
  profit supervision, stale training-data version, nonpositive fee-after
  lower bounds/profit factors, and failed walk-forward/regime/symbol stability.
  Training scheduler heartbeats were fresh, but the latest local-ML attempt
  skipped for `cost_complete_training_distribution_unavailable`; the
  specialist training run had also been interrupted. No duplicate training
  job was started.
- A bounded authoritative trade sample (`200` paper outcomes) found official
  OKX fills, fee, slippage and funding facts for all 200, but `0/200` are
  trainable because every linked original decision feature snapshot lacks
  `feature_availability_contract`. These historical snapshots remain
  quarantined; do not reconstruct their decision-time facts or seal them with
  current data.
- A bounded new-decision snapshot sample (`14` rows, last 30 minutes) was
  within the `98,304 B` frozen budget with no seal errors, but only `7/14`
  passed component availability. Seven failed on missing orderbook and/or
  mark/index receipts; four also lacked unambiguous account-fee receipt and
  availability clocks. This is a live feature-producer/data-evidence defect,
  separate from the model authorization blocker.
- The shadow worker is alive and making progress, but the last five-minute
  settlement projection counted `117` quarantined observations and only `3`
  completed historical OHLCV-only observations. Historical OHLCV labels are
  not executable cost-complete labels. A protection inventory also reported
  one orphan `TRX/USDT` short-side order key while no paper position existed;
  it was not modified. Resolve its authoritative exchange/database status
  read-only before any future paper execution validation.
- Gate status remains G1 passed for attribution only (historical fee-after
  evidence is negative), G2 blocked, G3-G6 pending. Current net-profit
  acceptance has not passed, and normal paper entry readiness is not
  established. No strategy or model was authorized and no order/protection
  mutation was performed.
- Next exact action: diagnose the seven invalid decision snapshots by comparing
  their recorded required-route receipts with the native WebSocket/REST
  producer and cache identity for those exact decisions. Fix only a confirmed
  producer/merge defect; collect a new bounded post-fix window and require
  component-valid decision-time contracts before using its matured labels.
  Separately inspect the training frame's bounded exclusion-reason distribution
  and reconcile the orphan protection record without mutating it. Keep the
  model permission gates unchanged until cost-complete development data and
  fresh independent fee-after acceptance both pass.
