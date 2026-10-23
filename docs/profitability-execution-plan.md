# Profitability Remediation Execution Plan

Updated: 2026-10-23, Asia/Shanghai.
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
  sample with zero order-book/mark/index receipt errors and no training
  compaction errors. Continue collecting it before any candidate research or
  exam. The fee-after quality gate remains negative (`n=261`, mean
  `-0.311071%`, LCB `-0.441736%`, PF `0.549345`), so model influence and
  order-capable paper execution remain disabled; G3-G6 stay pending.
