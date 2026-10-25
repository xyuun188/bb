# Profitability Remediation Execution Plan

Updated: 2026-10-25, Asia/Shanghai.
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
- Follow-up bounded audits at `2026-10-23T17:45Z-17:50Z` refined the source
  failure. A newer 30-minute decision sample contained `29` rows, of which
  `19` passed and `10` failed component availability; all fit the frozen size
  budget. Failed route receipts included orderbook/reference-price timeouts
  (including one reference-price failure), while several snapshots explicitly
  recorded `orderbook_depth_invalid`. A read-only projection of the latest
  `200` eligible training candidates found `200/200` used
  `historical_market_path_only`: the two-sided market-opportunity task was
  eligible, but the counterfactual execution-cost task was ineligible for all
  200 and `build_training_frame` produced zero rows. Thus the current-epoch
  sample count does not represent cost-complete training data; additional
  training retries or larger row limits cannot resolve this provenance gap.
- In a separate bounded projection of `120` shadow rows updated in the last
  30 minutes, `118` were quarantined and two completed. Quarantine reasons
  included `61` historical market-fact contracts that were missing, stale,
  or unverifiable; `35` missing shadow-label contracts; `8` invalid native
  market facts; and `7` invalid orderbook-depth records. The paired training
  and decision snapshots showed the same component-availability failures:
  missing orderbook facts were most common, followed by missing mark/index
  facts and ambiguous account-fee clocks. Two completed labels were
  `historical_ohlcv_only`, not executable observations. Historical contract
  or label gaps remain quarantined and must not be backfilled from present
  market data.
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
- Next exact action: use only post-repair decisions to trace required
  `orderbook`/`reference_prices` receipts, cache identity, and route completion
  through the native shadow result for each affected decision group. A route
  timeout or invalid book must remain unavailable; only change a timeout or
  merge policy if measured route latency or a reproducible producer defect
  justifies it under the existing request/resource budget. Collect a bounded
  fresh component-valid window and confirm its matured native labels retain
  the executable quote, fee, and funding provenance needed for the cost task.
  Separately reconcile the orphan protection record read-only before any G5
  paper execution. Keep all model permission gates unchanged until
  cost-complete chronological development data and fresh independent
  fee-after acceptance both pass.

### WebSocket Subscription Budget Repair (2026-10-24 02:15 Asia/Shanghai)

- A bounded online audit at `2026-10-23T18:00:44Z` covered the latest 24
  decisions in a 15-minute window: 15 had valid component availability and
  nine were invalid; all 35 market decisions in the window were HOLD and
  unexecuted. The service was paper/unpaused, heartbeat current, and had no
  automatic restart. The latest 24 rows had no model version; local ML paper
  permission remained false with negative fee-after evidence
  (`n=267`, mean `-0.303832%`, LCB `-0.431590%`, PF `0.549713`).
- The current process started its WebSocket at `17:33:36Z` and subscribed to
  1,992 channel arguments for 498 symbols in 20 batches of 100, spaced about
  0.1 seconds apart. The server consumer remained connected, but subscription
  acknowledgements and errors were only debug-logged and absent from the
  production log, so "sent" was not evidence that all channels were accepted.
  The official per-IP WebSocket operation limit is three requests per second;
  the startup burst exceeded it. This is a confirmed subscription-budget
  violation and a plausible cause of the missing native book/reference cache
  coverage. Do not claim individual subscription rejection without its
  acknowledgement/error evidence.
- A bounded 30-minute journal aggregation found 160 partial derivatives
  snapshots: 118 orderbook, 133 reference-price, 34 funding, and 35
  open-interest route timeouts (routes can overlap within one snapshot).
  Public REST request instrumentation showed queue-wait p95 of 7.2-8.6 seconds
  for orderbook, mark, and index routes; provider-duration p95 ranged from
  2.6 to 4.1 seconds. The required quote routes have 5.5-second deadlines.
  Thus REST fallback is also saturated; increasing the route deadline alone
  would not remove the measured queue backlog.
- Runtime source hashes for `data_feed/okx_rest_client.py`,
  `data_feed/okx_ws_client.py`, and `services/data_service.py` matched the
  local files. The fresh-decision WS cache and contract-depth conversion patch
  was already deployed, so the remaining observed weakness is not explained
  by a missing upload of those three files.
- Local repair now rate-limits every subscribe/unsubscribe frame to one per
  0.5 seconds under a per-connection lock and preserves the 100-channel frame
  cap across initial subscribe and dynamic resubscribe paths. Subscription
  rejection events now produce error-level diagnostics; runtime WS stats count
  requests, acknowledgements, errors and messages per channel. WebSocket URLs
  now use the default TLS port 443 instead of port 8443.
- Focused local regression (`test_okx_ticker_volume.py`,
  `test_okx_rest_client_symbols.py`, `test_data_service_security.py`):
  `152 passed`; changed-file Ruff and `git diff --check` passed. No online
  upload or restart has occurred for this repair yet.
- Gate state remains G1 attribution-only with negative fee-after economics,
  G2 blocked, G3-G6 pending. The subscription repair can improve native market
  evidence and reduce REST fallback load; it does not provide an authorized
  model, profitable strategy, entry permission or paper fill.
- Next exact action: upload only `services/okx_perpetual_sdk.py`,
  `data_feed/okx_ws_client.py` and `tests/test_okx_ticker_volume.py` without
  restart; verify remote hashes and run the selected remote regression.
  Then restart only `bb-paper-trading.service`, confirm WS acknowledgements
  and per-channel messages, and collect a fresh bounded component-valid
  decision window plus matured native labels. Keep entry/model/profitability
  gates unchanged and retain G2 as blocked until its preregistered boundary
  and executable-label evidence pass.

### Post-Deployment Runtime And Feature Audit (2026-10-24 02:37 Asia/Shanghai)

- The next continuation recovered that the WebSocket repair had already been
  uploaded and the paper service restarted, despite the previous section still
  describing deployment as pending. Read-only service and runtime projections
  reconcile the state: `bb-paper-trading.service` is active/running with zero
  restarts, and the trading process started at `2026-10-23T18:21:35Z`
  (`2026-10-24 02:21:35` Asia/Shanghai). Paper mode is running and unpaused.
- The live WebSocket status at `2026-10-23T18:31:53Z` showed a connected
  socket, 498 tracked symbols, zero reconnects, 20 subscription requests,
  1,685 aggregate acknowledgements, and zero subscription errors. All four
  streams had current traffic: tickers 19,930 messages, index-tickers
  100,786, mark-price 229,997, and books5 78,635. This verifies channel-level
  traffic after the throttle repair, but the current aggregate ACK metric
  cannot prove per-instrument subscription coverage; the difference between
  acknowledgements and requested channel arguments remains unresolved. Do not
  infer rejection from this difference without matching per-argument events.
- The bounded 15-minute coverage audit at `2026-10-23T18:29:08Z` covered
  35 market decisions across 31 symbols, all incomplete, with zero directional
  signals, zero executions, and zero open paper positions. Service continuity
  for a full 15-minute post-restart window was not yet established. The
  bounded horizon audit covered 37 decisions, all HOLD with
  `paper_prediction_horizon_unavailable`: authorized, selected, and training
  horizons were missing in all 37; both funding sides therefore reported
  `prediction_horizon_missing`. Local ML still lacks paper permission;
  server-profit and time-series each exposed only unavailable display-only
  10-minute results with no current supervision version. These are not
  forecast horizons and must not be substituted into funding projections.
- The feature-availability audit at `2026-10-23T18:30:29Z` covered 38 sealed
  decision snapshots. All fit the frozen 98,304-byte budget (maximum 97,200 B)
  with no seal errors; 29/38 were component-valid and 9/38 remained invalid.
  The main observed gaps were missing order-book facts (7), index facts (6),
  mark facts (5), plus ambiguous mark/index receipts and one stale technical
  availability contract. This is an improvement in observed coverage, not a
  passing G2 sample.
- A bounded shadow-pipeline audit at `2026-10-23T18:28:46Z` found all 341
  recent decisions had the expected four shadow horizons, but 62 observations
  were quarantined in the preceding five-minute update window. The three
  recently completed result labels inspected were all explicitly
  `historical_ohlcv_only` with
  `executable_market_fact_verified=false`; their fingerprints verified, but
  they are research path data, not executable-cost labels. The latest bounded
  30-row shadow projection at `18:37:21Z` consisted of still-pending future
  observations and therefore added no matured label evidence.
- No actionable model candidate or prediction horizon exists yet. The
  currently verified blocker remains the absence of eligible current-contract,
  cost-complete training evidence and an authorized model artifact, not order
  routing. G1 remains passed only as negative-economics attribution; G2 remains
  blocked; G3-G6 remain pending. Do not force a paper entry, grant model
  permission, or treat WebSocket traffic as profitability evidence.
- Next exact action: continue a bounded post-restart G2 sample once the fresh
  five- and fifteen-minute shadows have matured; record component-validity and
  native executable-label provenance separately. In parallel, audit the
  v4 training eligibility path without starting a duplicate trainer, and
  determine whether the new clean decision snapshots can produce
  cost-complete shadow labels. If ACK coverage is needed to isolate remaining
  missing market facts, first add bounded per-channel/per-instrument requested
  versus acknowledged accounting; preserve the current aggregate counters.
  Keep the active model and paper entry permissions unchanged until independent
  fee-after acceptance passes.

### Continuation G2/Entry Audit (2026-10-24)

- Read-only frozen-feature audit at `2026-10-23T18:58:34Z` UTC covered the
  newest 32 decisions and 32 corresponding training snapshots in a 45-minute
  window. All 64 were sealed without compaction errors and under the `98,304 B`
  budget (decision maximum `96,856 B`, training maximum `96,856 B`). The
  decision sample had 29/32 component-valid rows; all 32 training snapshots
  passed component availability. The decision sample still fails the
  preregistered G2 boundary because three rows have missing or ambiguous native
  quote facts.
- Exact bounded projection of those three rows showed CRO and SHIB had
  orderbook and reference-price route timeouts; DOGE had an orderbook timeout,
  a reference-price failure, and zero observed bid/ask depth. These remain
  quarantined entry observations, not reasons to substitute or infer prices.
  The existing crypto-only instrument filter is already applied in
  `_load_usdt_swap_markets`; the suspected duplicate symbol-discovery gap was
  therefore not a confirmed defect and was not changed.
- Read-only entry/training audit at `2026-10-23T18:59:53Z` UTC covered 333
  market decisions from the preceding three hours: all were HOLD and
  unexecuted. The newest bounded 80-row projection had 74 component-valid
  snapshots and 6 invalid; every row lacked an available model version.
  The selection and entry-permission contracts blocked direction because
  current expected-net, objective-net, quality-loss, quantitative-family,
  and authorized prediction-horizon evidence were missing. Local ML paper
  execution remains disabled on negative authoritative fee-after evidence
  (`n=267`, mean `-0.303832%`, LCB `-0.431590%`, PF `0.549713`); other model
  families have no fee-after evidence.
- A bounded authoritative training sample at `2026-10-23T18:59:33Z` UTC found
  0/200 trainable trade outcomes. All 200 have official OKX fill, fee,
  slippage, and funding sources, but their linked pre-contract decision
  features lack `feature_availability_contract`. They remain excluded; do
  not backfill those historical feature facts from current data. The current
  local-AI trainers continue to skip because no authoritative execution-cost
  distribution is available, and local ML skips because no cost-complete
  training distribution exists.
- Shadow maintenance was active and advancing, but the bounded latest
  settlement projection remained dominated by quarantines. The two recently
  completed labels in the inspected shadow-pipeline window were both
  `historical_ohlcv_only` with
  `executable_market_fact_verified=false`; they cannot satisfy executable
  cost supervision. The protection inventory still reports one orphan
  `TRX/USDT` short-side order key with no paper position; it was left
  untouched pending authoritative read-only reconciliation.
- No code change was justified by this audit. A broad dashboard health report
  was also found to produce an unbounded multi-megabyte payload; subsequent
  checks must use bounded field projections. No order, protection, model
  permission, or risk-parameter mutation was performed.
- G1 remains passed only as negative-economics attribution; G2 remains
  blocked; G3-G6 remain pending. Exact next action: continue tracing the
  request-queue/native-cache causes of missing orderbook and reference-price
  receipts using bounded route telemetry, while collecting a fresh
  preregistered component-valid decision window and matured native shadow
  labels. Separately inspect the model/training input contract for a safe
  current-epoch path to cost-complete observations. Do not reuse pre-contract
  trade features, promote the negative artifact, or authorize entries before
  all gates pass.

### Required Shadow-Contract Compaction Repair (2026-10-24)

- Reproduced the training-snapshot defect from the prior continuation: a
  completed shadow row can retain its native `training_market_fact_contract`
  in `feature_snapshot` while the optional-object budget omits it from
  `training_feature_snapshot`. Quality validation then quarantines the row as
  `market_fact_contract_missing_or_stale`, even though the source contains the
  complete native market/label evidence.
- A focused regression also exposed that a large earlier-sorted optional
  diagnostic set can evict `training_label_contract`. Both contracts are now
  projected before optional diagnostics; the market contract uses a compact
  provenance projection, and the already-compact label contract is retained
  under its own object and total-snapshot budgets. Impossible required-contract
  budgets fail closed rather than silently dropping provenance.
- The required contracts remain outside `FROZEN_TRAINING_FEATURE_KEYS` and
  `FROZEN_FEATURE_FIELDS`; they do not enter the sealed decision-time input
  fingerprint. Snapshot schema version is now 5, and training-data-quality
  version is now `2026-10-24.component-feature-availability.v9`.
- PostgreSQL parity is implemented with
  `bb_compact_training_market_fact_contract`; its generated DDL is installed
  with the existing shadow snapshot functions. Schema initialization performs
  one bounded newest-first re-projection batch of at most 500 rows whose full
  source still has a required label contract absent from the stored training
  view. This changes only the derived training snapshot, not shadow status,
  labels, model permission, orders, or protective orders.
- Regressions cover required native-contract projection despite oversized
  optional diagnostics, contract inclusion in the resulting eligible sample,
  frozen fingerprint invariance, fail-closed missing-contract behavior,
  snapshot idempotence, and Python/PostgreSQL parity fixtures. Local focused
  suite passed `160`; one PostgreSQL integration test skipped because
  `BB_TEST_POSTGRES_URL` is not configured. Ruff, compileall, and
  `git diff --check` passed.
- Deployment and online PostgreSQL verification have not yet occurred.
  G1 remains passed only as negative-economics attribution; G2 remains
  blocked; G3-G6 remain pending. This repair restores missing provenance only;
  it does not establish a profitable model, authorize an entry, or prove that
  existing quarantined rows should be requalified.
- Next exact action: stage and upload only the compaction/schema files and
  their focused tests without restart; run the remote focused regression,
  including the PostgreSQL parity test if the isolated DB is available. Then
  restart only paper trading and Dashboard so schema functions load and the
  bounded derived-snapshot re-projection runs. Verify deployed file hashes,
  service health, bounded re-projection results, and fresh native shadow
  eligibility. Keep all entry/model permissions unchanged and retain G2 as
  blocked until its independent evidence gate passes.

### Required Shadow-Contract Deployment And Fresh G2 Evidence (2026-10-24 04:15 Asia/Shanghai)

- Recovered and completed the preceding deployment checkpoint. The market-data
  warm-up start job had finished successfully before database verification;
  its 5-minute warm-up reported 120 rows and its timer remained active. No
  matching profitability audit was running. Local and remote integrity checks
  matched all five compaction/schema/test files, so the staged sync uploaded
  zero changed files and restarted no service.
- Local focused verification for compaction and DB initialization passed
  `53 passed, 1 skipped`; Ruff passed. The same focused suite ran remotely in
  transient unit `bb-shadow-compaction-regression-20261024t0001` with one CPU,
  1 GiB memory and a 300-second deadline: `53 passed, 1 skipped`, unit result
  `success`. The skipped case is the PostgreSQL temporary-function parity
  integration test because no dedicated `BB_TEST_POSTGRES_URL` is configured.
  The production DB is not a test DB and was not used for parity.
- Restarted only `bb-paper-trading.service` and `bb-dashboard.service` at
  `2026-10-23T20:03:28Z` / `20:03:27Z`. Both are active/running with zero
  automatic restarts; Dashboard returned HTTP 302. Live mode, model
  authorization, orders and protective orders were not changed.
- The startup re-projection is bounded to 500 newest source rows. A read-only
  projection of the newest 500 shadow rows found 492 retain their source
  `training_market_fact_contract` in the training view and 200 retain their
  `training_label_contract`. The remaining eight have explicit
  `frozen_input_budget_exceeded` compaction errors because the frozen model
  input exceeds its 96 KiB limit; the required contracts are intentionally
  not used to make those samples eligible. No row status, label, order or
  protection was modified by the audit.
- A fresh 15-minute feature audit at `2026-10-23T20:08:48Z` covered 32 paper
  decisions and 32 training snapshots. All fit the 98,304-byte frozen budget
  (decision max 98,164 B; training max 95,568 B) with no seal/compaction
  errors. Decision component validity was only 21/32; training component
  validity was 12/32. Missing or ambiguous orderbook, mark and index receipts
  plus stale technical availability contracts keep G2 blocked.
- A bounded mature-shadow audit covering the preceding 45 minutes found
  36 completed rows: 21 passed the current sample-quality assessment across
  20 decision groups, including 12 native executable-market-fact contracts
  with `fee_after_complete=true` across 12 distinct decision groups. Their
  horizons were 5m (6), 15m (4), 60m (10) and 240m (1) among all 21
  eligible rows. Fifteen historical-market-path rows remained quarantined;
  nine additional historical-path observations remain distinguished from
  the 12 executable contracts. This establishes that the repaired compact
  contract now survives into eligible recent native samples, not that the
  candidate has positive net returns.
- The latest 15-minute decision-horizon audit covered 36 decisions, all HOLD,
  with no authorized/selected/training prediction horizon. Local ML lacked
  `model_strategy_blueprint_paper_permission`; server-profit and time-series
  outputs were unavailable. The latest market funnel had 240 scanned symbols,
  48 feature fetches (47 valid), 18 ranked candidates but zero tradable
  candidates; three were secondary candidates. There were zero directional
  signals, executions or open paper positions. Do not relax market filters or
  grant model permission to manufacture activity.
- G1 remains passed only as negative-economics attribution; G2 remains
  blocked; G3-G6 remain pending. No profitability acceptance has passed.
  Next: (1) use bounded post-restart telemetry to trace why completed market
  routes still lack some native orderbook/mark/index receipts and maintain a
  clean preregistered decision window; (2) continue collecting mature native
  fee-after labels until a sufficiently broad, time-diverse development set
  exists, without mixing historical OHLCV-only labels into executable-cost
  supervision; (3) only then resume preregistered G3 nested chronological
  cost-aware research. The independent PostgreSQL parity check remains
  outstanding and must run only against a dedicated test database.

### Partial-Snapshot Refresh Backoff And Queue Recheck (2026-10-24 04:56 Asia/Shanghai)

- Read-only route projection at `2026-10-23T20:38:57Z` covered 45 recent
  decisions. Order-book timed out for 11, reference prices timed out for 9,
  and reference prices failed for 5. The bounded 15-minute request-log
  aggregate had 585 latency events: order-book queue-wait p95 `7,606.7 ms`,
  mark-price p95 `8,719.0 ms`, and index-price p95 `8,438.3 ms`, all beyond
  the existing `5,500 ms` quote-route deadline. Provider-duration p95 was
  `3,820.5 ms`, `2,369.6 ms`, and `3,117.9 ms`, respectively; do not respond
  by merely increasing the caller timeout.
- The route and cache code showed that cache-only market scans rescheduled a
  background derivatives refresh whenever a recent cached snapshot lacked
  required book/reference facts. The incomplete snapshot correctly remained
  unusable, but there was no cooldown between completed partial refreshes.
  This could repeatedly add REST quote work while the public route queue was
  already saturated.
- Repair: cache-only reads now suppress another refresh for 15 seconds after
  a recent incomplete snapshot, expose `derivatives_refresh_suppressed` and a
  bounded retry delay, and explicitly keep
  `derivatives_required_data_available=false`. Fresh/complete snapshots retain
  the existing update interval, and a partial snapshot becomes retryable
  after the backoff. No quote deadline, quality threshold, funding rule,
  model permission, entry gate, risk budget, or live setting was loosened.
- Local regression suite for data-service, market-fact and trading boundaries:
  `480 passed`; changed-file Ruff, `compileall`, and `git diff --check` passed.
  The remote focused `tests/test_data_service_security.py` run passed
  `102` tests with one CPU, 1 GiB memory and a 300-second transient-unit
  deadline.
- Targeted upload included only `services/data_service.py` and
  `tests/test_data_service_security.py`; remote integrity verification passed
  for both. Before restart, paper state was unpaused with zero open paper
  positions and zero nonterminal paper orders. The market round completed
  without error. Restarted only `bb-paper-trading.service` at
  `2026-10-23T20:55:58Z`; it is active/running with zero automatic restarts,
  paper mode unpaused, a fresh heartbeat, and no round error. No order or
  protective order was modified.
- The current post-restart sample is not yet mature. Do not count it as G2
  evidence until a full new 15-minute window has elapsed. At this checkpoint
  G1 remains attribution-only with negative economics; G2 remains blocked;
  G3-G6 remain pending. This repair only bounds repeated quote requests and
  does not establish an executable profitable strategy.
- Exact next action: after `2026-10-23T21:11:00Z`, collect one bounded
  post-restart 15-minute decision/training sample and the same bounded
  per-route queue-latency aggregate. Check whether retry suppression is
  present in incomplete rows, whether quote-route p95 is within the existing
  deadline, and whether component availability improves without weakening
  any contract. Continue the frozen G2 sample until its preregistered size,
  coverage and zero-core-error boundary passes; then resume G3 only if
  eligible native fee-after training evidence is sufficient. Keep the model
  disabled for entries and live trading inactive.

### Fresh G2, Entry-Gate, And Label Recheck (2026-10-24 05:46 Asia/Shanghai)

- Recovered the live state before further work. `bb-paper-trading.service`
  and `bb-dashboard.service` are active with zero automatic restarts; the
  paper service started at `2026-10-23T20:55:58Z`. Runtime mode is paper,
  unpaused, with a current heartbeat and no round/market/position error.
  SHA-256 hashes for the live data-service, REST, WebSocket, SDK, and
  quant-service files match the local working tree exactly. No order or
  protection was changed.
- The live WebSocket is connected for 498 symbols. Its bounded persisted
  counters show 20 subscription requests for 1,992 expected channel
  arguments, 1,685 acknowledged, zero rejected, and 307 without ACK. Per
  channel, missing ACKs are books5 75/498, index-tickers 77/498,
  mark-price 75/498, and tickers 80/498. Aggregate message counts are
  nonzero on all four channels, but the current snapshot has no
  per-instrument message timestamps; do not assume the missing ACKs caused
  particular quote failures.
- Fresh post-restart, read-only G2 audit at `2026-10-23T21:34:10Z` covered
  the newest 64 decisions in a 30-minute window. All 64 sealed snapshots fit
  the 98,304-byte frozen budget (maximum 96,984 B) with no seal errors.
  Only 52/64 decisions passed component availability; 12/64 remained invalid:
  nine order-book fact misses, seven mark and seven index fact misses,
  three ambiguous mark/index receipts and source clocks, and two decisions
  with no account-fee observation clock. The matching 64 training snapshots
  had 44 valid and 20 invalid; repeated horizon rows are not independent
  decision groups.
- A bounded projection of the invalid decisions confirmed actual
  `orderbook`/`reference_prices` route timeouts for XPL, PEPE, SHIB, XRP,
  MAGIC and ZRO, and order-book timeouts plus zero-clock/zero-price
  reference failures for WLD, DOT and INJ. ENA and DOGE had usable market
  facts but no account-fee source or receipt clock. Recent snapshots mark
  cached derivatives data stale and background refresh active. The optional
  `derivatives_refresh_suppressed` diagnostic is not retained in the frozen
  decision snapshot, so this projection cannot prove whether its cooldown
  fired.
- A separate 30-minute decision audit at `2026-10-23T21:41Z` covered 64
  decisions from `21:18:13Z` through `21:40:18Z`. All 64 were `HOLD` for
  `insufficient_evidence`; all lacked authorized, selected, and training
  prediction horizons, and all funding projections were incomplete with
  `prediction_horizon_missing`. The local model has no available version and
  lacks `model_strategy_blueprint_paper_permission`; server-profit and
  time-series outputs were unavailable/display-only. Existing local-model
  governance blockers remain negative average fee-after return, negative
  lower bound, profit factor below break-even, and missing fee-after quality
  evidence. This is the direct current entry blocker; market-data fixes alone
  cannot authorize a trade.
- Read-only authoritative training-eligibility audit of 200 recent paper
  outcomes at `2026-10-24T05:41:38+08:00` found `0/200` trainable. Every
  outcome has authoritative OKX fill/fee/funding sources, but all 200 linked
  pre-contract decision snapshots lack `feature_availability_contract` and
  are correctly excluded. These historical inputs cannot be repaired by
  copying current features or relabeled for training.
- The 45-minute shadow-pipeline projection at `2026-10-24T05:42:17+08:00`
  counted 27 completed rows and 112 quarantined rows across the four
  horizons. The latest five-minute settlement pass completed 13 and
  quarantined 40. The shadow worker is active, read-only, and reports no
  failures or dropped queued samples. The bounded recent sample contained
  historical-OHLCV-only rows and three native executable-price-path examples
  whose evidence-fingerprint verifier returned false; none of those examples
  is counted as accepted training evidence. No executed entry evidence was
  found in the audit window.
- Focused local verification of the current changed audit/data/market tests:
  `279 passed`; Ruff, `compileall`, and `git diff --check` passed. The earlier
  remote `test_data_service_security.py` regression remains `102 passed`.
  The 15-minute request-latency aggregate could not be read: the `bb` SSH
  account lacks journal access and `sudo -n journalctl` is denied. No API
  refresh, service restart, order, or protection mutation was attempted in
  this audit.
- Gates remain G1 passed for negative-economics attribution only, G2 blocked,
  and G3-G6 pending. No profitability acceptance or independent positive
  net-return evidence exists.
- Exact next action: add bounded per-instrument WebSocket last-message
  coverage to the existing runtime summary so missing-ACK instruments can
  be distinguished from ACK omissions with actual stream traffic; add a
  bounded route-cooldown diagnostic that survives into audit-only status
  rather than the frozen model inputs. Verify against the nine confirmed
  route-failure symbols after a fresh, post-restart window. In parallel,
  determine why three otherwise native shadow examples fail their evidence
  fingerprint verifier and count eligible native, fee-after decision groups
  without inspecting labels for candidate selection. Keep G2 blocked, all
  entry/model permissions unchanged, and live trading inactive until these
  provenance and profitability gates pass.

### Fresh Route And Entry Blocker Check (2026-10-24 06:09 Asia/Shanghai)

- Re-ran the local focused source/audit suite after recovering the workspace:
  `275 passed`; the new bounded route-diagnostic audit tests add `25 passed`.
  Changed-file Ruff, compileall, and `git diff --check` pass. The first
  attempted `rtk pytest` invocation could not find a global pytest executable;
  tests were run through the repository's `.venv\Scripts\python.exe -m pytest`.
- Read-only online decision audit at `2026-10-24T06:04:15+08:00` queried
  64 market decisions from the preceding 30-minute window. All 64 were HOLD;
  all lacked authorized/selected/training prediction horizon and had
  `paper_prediction_horizon_unavailable`. Local ML had no paper permission;
  server-profit and time-series artifacts/signals were unavailable. The
  sampled decision still passed all four expert calls, but analysis was
  blocked by `funding_evidence_unavailable`; direction evidence and net
  return estimates were absent. This is not an exchange order-placement
  failure.
- The matching 30-minute frozen-feature projection covered 64 decisions:
  all contracts present, all frozen sizes below the 98,304-byte limit
  (maximum 97,562 B), but only 36/64 component-valid. A fresh 15-minute,
  32-row projection was worse: 13/32 component-valid and 19/32 invalid.
  The latter still had zero over-budget rows and zero seal/compaction errors.
  Invalid rows showed 17 orderbook fact misses, 12 each for mark and index,
  plus seven ambiguous mark/index source clocks/receipts.
- Extended `scripts/audit_online_frozen_feature_budget.py` to retain, for at
  most 12 invalid examples, bounded `derivatives_route_status`, unavailable
  reason, and source identity/clock fields for orderbook/mark/index. It does
  not read outcome labels or add diagnostics to frozen model inputs. Its
  focused regressions passed (`25` tests across this and neighboring audit
  modules), Ruff and compileall passed. The 15-minute online projection
  confirmed recurring orderbook/reference timeouts; several REST fallback
  reference facts carried zero source timestamps and correctly failed
  availability. The new diagnostics need to be committed and pushed before
  they are durable.
- A bounded authoritative training-eligibility audit at
  `2026-10-24T06:06:06+08:00` examined 100 most recent paper trade outcomes:
  `0/100` trainable. All have official OKX fill/fee/funding sources and v4
  supervision, but every linked old decision lacks
  `feature_availability_contract`; do not reconstruct these old inputs from
  current values. No candidate or profitability acceptance exists.
- G1 remains passed only for negative-economics attribution; G2 is blocked;
  G3-G6 remain pending. Entry-model permissions, risk settings, and live
  trading were not changed. No order or protective order was touched.
- Exact next action: collect route-cache diagnostic evidence keyed to the
  recent failure symbols, with a strict bounded runtime summary outside the
  frozen feature input; distinguish absent/stale/misbound WebSocket facts from
  public REST route timeout before changing market code. Then fix the
  demonstrated cause, run focused tests, stage targeted deployment, and
  verify a fresh decision/training window. Separately count matured native,
  executable, fee-after shadow decision groups and inspect their fingerprint
  failures without using already inspected labels to choose a candidate.
  Keep all model/entry permissions unchanged until G2 and later gates pass.

### Per-Instrument WebSocket Coverage Deployment (2026-10-24 06:56 Asia/Shanghai)

- Added bounded per-instrument last-message timestamps to the existing
  WebSocket subscription summary. The diagnostic is keyed by exact
  `(channel, inst_id)` arguments and is outside frozen model inputs. It reports
  message coverage separately from ACK coverage, so an ACK omission cannot be
  mistaken for an absent stream.
- Added the bounded native-derivatives cache diagnostic to the persisted
  trading runtime summary. The diagnostic retains only recent route status,
  source identity, receipt age and source age; it does not alter entry,
  quality, model, risk, order or protection decisions.
- Local verification passed `10` WebSocket tests and `468` trading-service
  boundary tests; Ruff and `git diff --check` passed. Targeted upload of
  `data_feed/okx_ws_client.py`, `services/trading_service.py` and
  `tests/test_okx_ticker_volume.py` completed with remote
  `source-integrity-ok:3` and no service restart during upload. The remote
  WebSocket regression passed `10`; the broader remote trading-boundary run
  was not accepted because the online pytest environment contains
  root-owned temporary directories that cause permission failures during
  teardown, so it is not evidence of a code failure or a pass.
- Paper and Dashboard were restarted at `2026-10-23T22:53:56Z`; both are
  active with zero automatic restarts. Runtime remains `paper`, unpaused,
  and live trading is inactive.
- The first post-restart runtime projection at
  `2026-10-23T22:56:15Z` showed 498/498 ticker instruments with messages.
  Books5, mark-price and index-tickers had messages for 153/498, 184/498 and
  165/498 instruments respectively. Native cache diagnostics showed books and
  mark facts available for several sampled symbols while many index facts were
  `source_stale`; two sampled symbols had all three facts missing. The
  connection was healthy (`reconnect_count=0`, ticker stream not stale), so
  missing ACKs are not by themselves a sufficient root-cause explanation.
  The entry model remains unauthorized (`model_strategy_blueprint_paper_permission`
  absent; no positive cost-complete quality evidence), and no new entry was
  authorized or executed.
- G1 remains passed only as negative-economics attribution; G2 remains
  blocked; G3-G6 remain pending. This deployment is diagnostic only and does
  not establish a profitable candidate.
- Next exact action: collect a full bounded post-restart window after the
  process has produced mature decisions; compare route diagnostics for the
  confirmed failure symbols against per-instrument WebSocket coverage and
  REST fallback status. Then fix only the demonstrated route/availability
  defect, retest and redeploy in a targeted batch. Separately continue
  counting mature native executable fee-after decision groups without
  inspecting labels for candidate selection. Keep all entry/model permissions
  unchanged and live trading inactive.

### Derivatives Stream Backlog Repair (2026-10-24 07:07 Asia/Shanghai)

- Root-cause evidence: after the diagnostic-only deployment, the connection
  received 172,453 messages in roughly six minutes, while books5, mark-price
  and index-tickers facts were received with source clocks about 249--251
  seconds old and were therefore classified as
  `stale_or_ticker_misaligned`/`source_stale`. Tickers were current. The
  underlying SDK coalesced only tickers; every derivative update was queued
  individually, allowing a historical-message backlog to delay current
  executable facts.
- Repair: `OkxPublicWebSocketSdkStream` now coalesces only the latest payload
  per `(channel, inst_id)` for `books5`, `mark-price` and `index-tickers`,
  while preserving subscribe/error/control events and the existing ticker
  rate limit. This is a transport freshness repair; it does not alter
  quality thresholds, model permissions, sizing, exits, risk or live mode.
- Local focused verification passed `20` tests (SDK stream plus OKX WS client),
  Ruff, compileall and diff checks. Targeted upload of the SDK and its test
  passed `source-integrity-ok:2`; the same 20 tests passed remotely. Paper and
  Dashboard were restarted at `2026-10-23T23:06:56Z`; both are active with
  zero automatic restarts and paper mode remains unpaused.
- This repair is not yet accepted as G2 evidence until a fresh mature window
  confirms source-clock freshness and complete route contracts for the
  previously failing symbols. No entry was authorized or manually created.
- Exact next action: after the new bounded window completes, compare
  per-instrument message coverage and source/receipt ages before and after the
  repair, then run the frozen-feature and entry-gate audits. If freshness
  recovers, keep G2 blocked until independent native fee-after samples meet
  the preregistered evidence gate; if it does not, trace the remaining REST
  fallback or instrument-contract path without weakening the gate.

### Native Shadow Lineage Audit Repair And Fresh Evidence (2026-10-24 08:10 Asia/Shanghai)

- The online shadow audit had a verifier defect: it always hashed the
  historical `price_path.ohlcv_evidence` object. Native executable-market
  contracts do not contain that object; their authoritative lineage is
  `entry_fact_id + result_fact_id + path_fingerprint`, with `data_fingerprint`
  equal to the canonical hash of those three values. The old verifier
  therefore falsely marked native rows as fingerprint failures.
- Repaired `scripts/audit_online_shadow_pipeline.py` to keep historical OHLCV
  verification and native lineage verification separate. Added bounded native
  fee-after group counts and a 200-row lineage check. Local focused regression
  passed `3`; Ruff, compileall and diff checks passed. Only the audit script
  was targeted-uploaded with source integrity verified; no service restart or
  runtime mutation occurred.
- Read-only online audit at `2026-10-24T00:08:11Z` UTC found, within the last
  six hours, `176` completed native rows with `fee_after_complete=true`,
  spanning `149` distinct decision groups and all four horizons. Native
  lineage verification passed `176/176`; no fingerprint failure remains in
  the checked set. Historical OHLCV rows are separate (`51` rows / `39`
  decision groups) and are not executable-cost supervision.
- The same window still has no executed paper entry. Entry recovery reports
  `176` market decisions, all `HOLD`/`insufficient_evidence`; local ML paper
  permission remains false because authoritative fee-after economics are
  negative (`n=267`, mean `-0.303832%`, LCB `-0.431590%`, PF `0.549713`), and
  the other model families have no qualified evidence.
- Fresh frozen-feature checks show the transport repair is improving
  availability but not yet a clean acceptance window: latest 30-minute
  decision snapshots are `61/64` component-valid and training snapshots
  `64/64` valid; the latest 10-minute `16/16` decision and training snapshots
  are valid. All are below the `98,304 B` budget with zero seal errors. The
  remaining older invalid decision rows contain missing index facts or a stale
  technical availability contract and remain quarantined.
- G1 remains passed only as negative-economics attribution; G2 remains
  **BLOCKED**; G3-G6 remain pending. The native sample volume now supports
  bounded cost-aware research preparation, but profitability has not been
  demonstrated and no model/entry permission may be granted.
- Exact next action: collect a fresh preregistered clean decision/training
  window after the repaired stream, keep quarantined rows out of research,
  then run the frozen cost-aware development ledger on native fee-after
  groups only. Do not mix the historical rows, tune on the inspected window,
  or manufacture orders.

### Post-Repair Route Diagnostic (2026-10-24 08:18 Asia/Shanghai)

- The bounded runtime projection now includes the persisted WebSocket and
  native-cache diagnostics. At `2026-10-24T00:17:50Z` UTC the connection was
  healthy with no reconnects, `1,131,819` messages received, `1,992/1,992`
  subscription acknowledgements, and `498/498` message-seen coverage for each
  of `books5`, `mark-price`, `index-tickers`, and `tickers`.
- The previously failing PEPE route is currently healthy: the native cache
  diagnostic shows executable book, mark and index facts from
  `okx_ws_public`, with the index fact on `PEPE-USDT` about 14 seconds old.
  The latest market round had `11/11` valid feature snapshots and no market
  feature failures. The historical invalid PEPE rows in the 30-minute frozen
  sample are therefore retained as old quarantined observations, not evidence
  that the current stream repair failed.
- The current market funnel remains intentionally conservative: four ranked
  candidates were evaluated, but `rank_tradable_candidates=0` because the
  existing notional, ADX, volatility and 24-hour change quality contracts
  rejected them. This is an evidence/risk result, not an exchange execution
  failure; no threshold, leverage, sizing, model permission or risk veto was
  changed.
- Shadow maintenance is currently succeeding (`last_error=null`, latest
  successful pass recorded) although its cumulative failure counter still
  includes earlier timeout/quarantine attempts. No current service error,
  dropped sample, order or protection mutation was observed.
- G2 remains **BLOCKED** until a new, mature, preregistered clean window
  excludes the old missing-index snapshots. G3-G6 remain pending; the native
  fee-after sample is evidence for research preparation only and has not
  demonstrated positive net returns. Exact next action: let the repaired
  stream produce a full independent window, then re-run the frozen-feature,
  entry-gate and native-label audits before any candidate research or
  permission change.

### Late Native Cache Recheck And Development Baseline (2026-10-24 09:40 Asia/Shanghai)

- Repaired the derivatives refresh join: after the bounded REST/reference
  wait returns, the service now performs one read-only identity/freshness
  check of the native WebSocket cache before persisting a partial snapshot.
  Only complete order-book/reference payloads with real source clocks and
  receipts can complete the route; incomplete or stale facts remain blocked.
  The helper also clears the partial flag when all required routes are
  complete. Local focused verification passed `174` related tests before the
  repair and `22` direct late-cache/derivatives regressions after it; Ruff,
  compileall and diff checks passed. The online selected regression passed
  `108` tests, source integrity passed for the three runtime/test files, and
  paper/Dashboard were restarted at `2026-10-24 09:09:01` Asia/Shanghai with
  `NRestarts=0`. Live trading and protections were untouched.
- The first complete post-restart component window covered `44` decision
  snapshots across `27` symbols and `64` training snapshots. All were
  availability-valid, sealed, and within the `98,304 B` frozen budget
  (`maximum=96,990 B`); no route, receipt, source-clock, seal, or compaction
  errors remained in that boundary. A later 30-minute mixed boundary retained
  a `51`-row valid streak but included older/pre-restart rows with missing
  index or account-fee receipts; those rows remain quarantined and are not
  counted as acceptance evidence.
- Fresh shadow settlement now contains `245` completed native rows across
  `204` decision groups in the last six hours, all with
  `fee_after_complete=true`; the bounded lineage verifier passed `200/200`
  checked rows with zero fingerprint failures. The current queue is
  completing post-restart 5/15/60/240-minute rows without a current
  maintenance error. This establishes usable native labels for development,
  not profitability.
- A preregistered read-only cost-aware baseline ledger ran on `259` native
  fee-after rows / `214` decision groups from the six-hour development
  window. The candidate set was limited to no-trade, always-long,
  always-short, and a hindsight best-action upper bound for diagnostics;
  no independent exam was opened and no candidate was promoted. The live
  decision action was `hold` for all `259` rows. Development/validation
  results are regime-sensitive: 15-minute always-short was positive in the
  development fold (`+0.0511%`, PF `1.283`) but negative in validation
  (`-0.4128%`, PF `0.112`); 60-minute always-short was positive in both
  folds (`+0.1410%` / `+0.1085%`) but only `67` rows; the 240-minute result
  was positive but only `59` rows and is not an independent acceptance
  sample. Always-long was negative in development at 5/15/60/240 minutes.
  These baselines are rejected for promotion because they are too small,
  horizon/regime dependent, and have no fresh independent exam or risk-bound
  evidence.
- The entry audit after the repair still found `64/64` HOLD decisions with
  `insufficient_evidence`, no model version, and no authorized prediction
  horizon. Local-ML paper permission remains false because authoritative
  fee-after evidence is negative (`n=267`, mean `-0.303832%`, LCB
  `-0.431590%`, PF `0.549713`). This is the direct reason there is still no
  paper entry; it is not an exchange order-placement failure.
- G2's post-restart component-validity subgate is now evidenced, but the
  overall gate remains **BLOCKED** until current-contract cost/horizon
  supervision is sufficient for a nested candidate study and the historical
  pre-contract outcomes remain excluded. G1 remains passed only as negative
  attribution; G3-G6 remain pending. No profitability acceptance exists.
- Exact next action: keep collecting native fee-after groups under the same
  frozen development ledger, add a cost-aware feature candidate only inside
  chronological development folds, and require a new untouched exam with
  predefined lower-bound, profit-factor, drawdown, and stress criteria before
  any paper permission. Do not turn the positive 60/240-minute baseline into
  an order rule, and do not use the inspected development window as G4.

### Preregistered Current-Native Nested Development Run (2026-10-24)

- Before execution, freeze a bounded development-only study ending at
  `2026-10-24T02:27:00Z`, covering the preceding six hours. Query only
  completed paper shadow rows whose compact market contract identifies the
  native executable 1m path and whose fee-after label is complete; select at
  most 1,000 rows in chronological completion order with a 15-second SQL
  statement timeout.
- Reuse `services.ml_nested_development` unchanged: primary horizon remains
  five minutes, the Ridge alpha set remains 1/10/100, and predicted-net entry
  thresholds remain 0/0.05/0.1 percent. The only permissible fitting and
  selection are inside existing purged chronological development folds. This
  sample is development data, not G4, even if a fold result is positive.
- A new read-only runner records its input/dependency hashes and a compact
  aggregate report under the ignored
  `data/profitability_remediation/` directory. It creates no model artifact,
  does not mutate the database or runtime policy, and submits no orders.
  Remote budget is 25% CPU, 1 GiB memory, 300 seconds, one DB connection and
  no overflow. The report is keyed by its exclusive cutoff; an existing
  report or active same-cutoff unit must be reused/checked, not duplicated.
- Recovery action: inspect the named transient unit and its saved report
  before any retry. A timeout or insufficient development fold leaves G3
  pending; it does not authorize model permission or an independent exam.

### Six-Hour Nested Run Result And Bounded Extension (2026-10-24)

- The fixed six-hour query completed at `2026-10-24T02:46:16Z`: all 413
  selected native fee-after rows built into the training frame, across 304
  decision groups (5m 125, 15m 113, 60m 104, 240m 71). Input fingerprint:
  `4da9ddd4ea0a2f66e7f619ff4e7af3e0dcc2f97b7c5b25fba6eabfdad00f1ee8`.
- The outer chronological partition was ready and disjoint, with 124 training
  groups / 143 rows and 152 holdout groups / 226 rows after purging 28
  candidate training groups. The nested researcher nevertheless returned
  `insufficient_development_data`, no outer folds and zero selected policies;
  no exam labels were evaluated. This is a blocked G3 study, not a loss or
  profitability result. The exact report is ignored locally at
  `data/profitability_remediation/online-nested-development-20261024T022700Z.json`.
- The report lacked per-fold coverage detail. The runner now adds an outcome-
  independent outer/inner primary-horizon coverage summary. A separate,
  preregistered extension uses a fixed 24-hour development window ending
  `2026-10-24T02:48:00Z`, still limited to 1,000 native fee-after rows and
  the same model/search/split requirements. It overlaps the already-inspected
  six-hour development subset and therefore cannot be treated as G4.
- Online paper process CPU was observed near one full core during the
  preflight. The extension uses `Nice=10`, a 25% CPU quota, 1 GiB memory,
  one DB connection, 15-second SQL timeout and 300-second runtime limit to
  minimize paper-service contention. It remains read-only and must not change
  any execution permission or risk rule. Recovery is to inspect the unique
  cutoff unit/report before retry; if folds remain insufficient, G3 stays
  pending and more mature native evidence is required.

### Fixed-Horizon Purge Correction Preregistration (2026-10-24)

- The 24-hour v1 report exposed an over-purge in the fixed-five-minute nested
  researcher: its model and inner selection fit only five-minute labels, but
  `_folds` used each decision group's maximum label end across 5/15/60/240
  minutes. This left outer primary-horizon training counts `1/4/16` and inner
  counts between `1` and `6`; every fold failed the unchanged minimum of 16.
- The fix is scoped to `ml_nested_development`: derive split availability from
  the fixed primary-horizon rows for both outer and nested boundaries, while
  continuing to assign whole decision groups to train/validation.
  `ml_horizon_development` retains all-horizon purging because it searches
  across horizons.
- The v2 retry kept the exact 430-row / 315-group source fingerprint, but
  still produced zero selected policies. Per-fold audit showed only 58
  primary-horizon groups overall; inner chronological training chunks had
  8-16 rows and the first two outer folds lacked two qualified inner folds.
  The runner was also limiting raw rows, which can truncate some horizons of
  the most recent decision groups.
- The next preregistered run changes the dataset selector to rank and cap
  distinct decision groups, then include every qualifying horizon row for
  each selected group. It will use a 7-day upper window ending at
  `2026-10-24T03:30:00Z`, at most 1,000 groups, unchanged native-path and
  fee-after-complete provenance, same primary 5m model/search and
  qualification rules, one CPU quota fraction, one DB connection, 1 GiB and
  300 seconds. The report must compute the legacy all-horizon development
  fingerprint and match
  `c3d71adc4178e4ad389cb54fdea5d41b87e7b6f52a3f536a1510dd66dac2f813`;
  otherwise it aborts rather than claiming a same-window comparison.
- This extension is still development-only and includes previously inspected
  labels, so it cannot serve as G4. Insufficient folds or negative evidence
  keep G3 blocked; no thresholds or entry permissions change.

### Nested Study Regression Fix And Verification (2026-10-24 11:43 Asia/Shanghai)

- Corrected `test_nested_exam_changes_cannot_affect_report`: the mutation
  previously changed `horizon_minutes`, which is present in `FEATURE_KEYS`
  but also defines the primary-horizon partition. The regression now mutates
  model inputs only, leaving that partition key intact.
- Focused nested, horizon-development, diagnostic and online-runner tests:
  `25 passed`; sync-tool selected tests: `20 passed`. Ruff, compileall and
  `git diff --check` passed. This is a test-fixture correction; it does not
  change the strategy or demonstrate profitability.
- The existing 24-hour v2 result contains 430 rows / 315 groups, with 132
  primary-horizon rows. After the fixed-horizon purge correction it forms
  three disjoint outer folds (16/19/23 primary training rows), but all three
  inner selections report `insufficient_inner_folds`; no policy was selected.
- Next exact action: stage only the five already-owned plan/runner/researcher/
  regression paths, targeted-upload those sources and tests with restart
  skipped, verify remote hashes and focused tests, then check for the exact
  v3 unit/report before running the preregistered seven-day development
  extension ending `2026-10-24T03:30:00Z`. Require the legacy all-horizon
  development fingerprint
  `c3d71adc4178e4ad389cb54fdea5d41b87e7b6f52a3f536a1510dd66dac2f813`.
  Do not interpret this overlapping development window as G4 or change any
  permission based on its result.

### Seven-Day Nested Development Result (2026-10-24 11:48 Asia/Shanghai)

- Targeted upload completed for the runner, fixed-horizon researcher and two
  regression modules; remote source verification returned `source-integrity-ok:4`
  and no service restart was performed. The bounded remote regression returned
  `11 passed in 47.70s`.
- The preregistered seven-day run (`bb-nested-dev-v3-20261024T033000Z`) read
  `430` native fee-after rows across `315` decision groups (`132` primary
  five-minute rows). Its legacy all-horizon development fingerprint matched
  `c3d71adc4178e4ad389cb54fdea5d41b87e7b6f52a3f536a1510dd66dac2f813`.
  The current primary-development fingerprint is
  `a5f5cb9787c3fbd72edc2081a0c5a24e50721aecb7f550d153630b8df3e7f362`.
- The fixed-horizon outer partition is chronological and disjoint
  (`62` training / `66` holdout primary rows after purging `4` groups).
  Outer fold zero still lacks a qualified pair of inner folds; folds one and
  two are computable, but all `18` searched alpha/threshold combinations
  have negative fee-after mean and/or profit factor below one. Therefore
  `selected_policy_fold_count=0`, `selected_count=0`, and no policy can be
  frozen or promoted.
- This is a complete negative/insufficient G3 development result, not G4:
  no exam labels were evaluated, no artifact was persisted, no database or
  runtime policy changed, and no orders were submitted. G2 remains blocked,
  G3 remains pending, G4-G6 remain pending, and local-ML paper permission
  stays false. Continue accumulating fresh native fee-after groups and
  preregister the next development batch; do not convert this result or the
  positive historical baselines into an entry rule.

### Native Path Retry Timing Guard (2026-10-24 13:08 Asia/Shanghai)

- Recovery confirmed the active goal remains `ACTIVE`; branch `main` was at
  `f6a047e` / `origin/main`, with only
  `services/data_service.py`, `services/shadow_backtest_service.py`, and
  `tests/test_shadow_backtest_service.py` dirty before this batch.
- Root cause in the in-progress shadow retry change: an incomplete native
  1m path was kept pending before checking whether its result fact was already
  outside the one-minute label-source tolerance. A stale label could
  therefore retry indefinitely instead of reaching the existing quarantine
  flow.
- Repair: only a timing-valid native label may remain pending for an
  incomplete/missing 1m path. Late labels continue through the existing
  completion/quarantine path and are not promoted into training. The native
  path fetch now reuses the bounded per-symbol cache/single-flight helper and
  explicitly anchors the result candle for adapters whose `after` cursor is
  exclusive.
- Added a regression for late labels with a missing path and corrected the
  existing session-boundary fixture to use consistent fixed source clocks.
  Focused local suite `tests/test_shadow_backtest_service.py` plus
  `tests/test_data_service_security.py`: `143 passed`. Changed-file Ruff,
  compileall and `git diff --check` passed.
- This is a settlement/data-pipeline repair only. It does not create
  executable signals, permit entries, or show profitability. G1 remains
  attribution-only with negative economics; G2 remains **BLOCKED**; G3-G6
  remain pending; live trading remains inactive.
- Online state has not yet been changed or rechecked in this batch. Next:
  targeted upload only of the two services and shadow regression test with
  restart skipped; verify remote source hashes and run the same bounded
  regression remotely. Only then restart `bb-paper-trading.service` (not
  Dashboard, tunnels, positions or protections), verify service health and
  paper mode, and collect fresh native path/fee-after evidence. Do not
  authorize entries from these pipeline fixes.

### Post-Deployment Native Settlement Recheck (2026-10-24 13:18 Asia/Shanghai)

- Targeted upload of the two service files and one regression module completed
  with `source-integrity-ok:3`; the bounded remote regression ran as
  `bb-shadow-path-regression-20261024t1308.service` under 25% CPU/1 GiB and
  passed `143` tests in `56.92s`.
- Only `bb-paper-trading.service` was restarted at approximately
  `2026-10-24 13:08` Asia/Shanghai. It remained `active/running` with
  `NRestarts=0`; `bb-dashboard.service` and `bb-model-tunnels.service` stayed
  active. The runtime shadow audit reports `mode=paper`, `running=true`,
  `paused=false`, no market/position/round error, and no live entry.
- The 120-minute recovery audit after restart recorded `252/252` market
  decisions as `HOLD`/`no_candidate`; every sampled entry permission remained
  false. The authoritative local-ML evidence is unchanged and negative:
  `n=267`, mean fee-after return `-0.303832%`, lower bound `-0.431590%`,
  profit factor `0.549713`. Other model families still have no qualified
  fee-after evidence. This is the current reason for zero orders, not an
  exchange rejection.
- The settlement audit at `2026-10-24T05:17:27Z` found successful maintenance
  (`success_count=8`, `failure_count=0`), `130` newly completed and `48`
  quarantined rows in the observed throughput interval, `295` native
  fee-after rows across `217` decision groups, and native lineage verification
  `200/200`. Historical-path rows (`120` rows / `112` groups) remain
  non-executable and excluded. No OKX position-history entries or executed
  paper entries were present.
- This materially confirms the path/settlement repair is functioning, but it
  does not create a profitable candidate or satisfy G4. G1 remains negative
  attribution-only; G2 remains **BLOCKED**; G3 remains pending; G4-G6 remain
  pending. The new development window must be analyzed without changing
  entry permissions or reusing it as an exam.
- Commit `7b618b9` contains the scoped repair and plan evidence. Push to
  GitHub was attempted but failed with a transient `curl 28` connection
  timeout; local `main` is one commit ahead of `origin/main`. Retry push
  before final handoff. No unrelated work was changed.

### Fresh Native Nested Development Result (2026-10-24 13:22 Asia/Shanghai)

- Preregistered read-only development run:
  `audit_online_nested_current_development.py --limit 1000 --hours 24
  --cutoff 2026-10-24T05:20:00Z --revision v3`. The immutable interval was
  `[2026-10-23T05:20:00Z, 2026-10-24T05:20:00Z)`, selecting at most 1,000
  distinct decision groups and only completed paper labels with native
  executable path provenance and `fee_after_complete=true`.
- Exact data read: `465` rows / `341` decision groups; 5-minute primary
  horizon `137` rows. Development input fingerprint
  `e65564fc2b8abc37d8388d1bd29b7323cb06651bac78e6694b866c78503c364a`;
  source-frame fingerprint
  `ae9f4599e931fef8dae4a3da88943f10a96f362e79b24483d63fb7d7a4091e36`.
  Nested model/search/split dependencies matched local and online.
- The chronological outer partition was disjoint (zero group overlap) and
  label-time purged; it had 62 training and 69 holdout primary rows after
  purging six groups. Fold 0 lacked adequate inner primary-horizon coverage.
  Folds 1-2 evaluated all nine fixed Ridge/threshold combinations; zero of
  nine qualified. Their inner fee-after means ranged from `-0.3720%` to
  `-0.0504%` in fold 1 and `-0.2124%` to `-0.0731%` in fold 2; all profit
  factors were below 1 (`0.074`-`0.624`). Across outer folds, zero policies
  were selected from 31 diagnostic opportunities.
- Safety report: exam labels not evaluated; no artifact, DB mutation,
  promotion, orders or runtime-policy change. This fresh G3 result rejects
  the current frozen search family; it is not G4 and provides no basis to
  enable an entry. Do not retune on this inspected interval. G1 remains
  negative attribution-only, G2 blocked, G3 pending candidate redesign,
  G4-G6 pending.
- Result artifact is ignored local development evidence:
  `data/profitability_remediation/online-nested-development-v3-20261024T052000Z.json`.
  Next: inspect the development-only feature/label diagnosis and cost
  decomposition without opening an exam, define a materially distinct
  candidate family on a future development window, and preregister it before
  execution. If no valid candidate emerges, keep paper entry permission false.

### Current-Native Transparent Baseline Diagnostic (2026-10-24 13:23 Asia/Shanghai)

- A separate read-only 24-hour ledger used `477` native fee-after rows /
  `352` decision groups and no independent exam. All recorded live decision
  actions were `hold` (`0` long / `0` short) because the entry gate is
  evidence-blocked.
- Transparent baselines are regime-dependent and do not authorize a policy:
  5-minute always-long was `-0.0116%` development / `-0.1614%` validation;
  15-minute always-long was `-0.0849%` / `-0.2225%`; both sides were
  negative at 15 minutes. The 60-minute always-long baseline was positive
  (`+0.1883%` / `+0.9059%`, PF `1.402` / `6.578`) on only `85`/`37` rows,
  while 240-minute always-short was positive (`+2.2392%` /
  `+0.0850%`, PF `63.057` / `1.134`) on only `58`/`25` rows.
- These positive baselines are not promotable: they are single-side,
  horizon-specific, low-sample, regime-sensitive, and were inspected during
  development. They are not a frozen candidate, not a fresh exam, and not
  proof that current signals can predict those directions. No thresholds,
  horizon, sizing or quality gate was changed.
- The consistent result across the nested study and transparent baselines is
  that the current five-minute Ridge family cannot cover trading costs; the
  immediate research blocker is candidate/feature/horizon quality, not order
  placement. G3 remains pending and must continue with a preregistered,
  materially distinct development experiment; G4-G6 remain pending.

### Horizon-Complete Candidate Check (2026-10-24 13:29 Asia/Shanghai)

- A bounded, read-only horizon-aware nested check used the same native
  fee-after source contract, 24-hour window
  `[2026-10-23T05:30:00Z, 2026-10-24T05:30:00Z)`, 1,000-group cap, one DB
  connection, 25% CPU and 1 GiB memory. It read `513` rows but only `4`
  decision groups contained all four required horizons (`16` rows); `373`
  groups were incomplete.
- The existing horizon policy correctly returned
  `insufficient_development_data` without folds or selection. This is not a
  model loss result and not permission to drop incomplete horizons: doing so
  would select on a biased subset. No exam labels, artifact, DB mutation,
  policy change or order occurred.
- The remaining G2/G3 blocker is now concrete: native labels are arriving,
  but complete four-horizon groups are too sparse for an unbiased
  horizon-aware candidate, while the fixed five-minute candidate family is
  negative after costs. Continue collecting current-contract groups and
  repair/measure horizon completeness before any candidate promotion. Keep
  entry permission false and paper-only mode unchanged.

### Horizon Maturity And Quarantine Breakdown (2026-10-24 13:35 Asia/Shanghai)

- A bounded latest-created projection (2,500 paper rows, 625 decision groups,
  created between `2026-10-24T00:43:39Z` and `2026-10-24T05:29:38Z`) separated
  normal maturity from data defects. At the `05:30Z` cutoff, 240-minute rows
  were still pending for `520` rows, so most groups were not yet expected to
  be complete. This is a wait-for-maturity condition, not permission to use
  partial groups.
- Among matured/all-four groups, only four groups had all four native
  fee-after rows completed and clean; other groups were split by native
  quarantine, historical fallback, or missing labels. Dominant quarantine
  reasons were native market-fact invalidity, missing path labels, and
  historical-path provenance. Historical rows remain excluded and are never
  backfilled into executable-cost training.
- The earlier old-row projection that showed mostly historical rows was
  affected by its bounded ascending selection; it was not used as acceptance
  evidence. The latest-created projection confirms current native completion
  is increasing (`108` 5m, `98` 15m, `61` 60m, `26` 240m completed native
  fee-after rows in the bounded slice), while 240m maturity is still catching
  up.
- No code, permission, threshold, sizing, leverage or runtime policy changed
  from this audit. Continue the same frozen data contract until enough
  matured, all-native groups exist for a preregistered horizon study; G2/G3
  remain blocked/pending and no entries are authorized.

### Live Decision And Training Preflight Recheck (2026-10-24 13:55 Asia/Shanghai)

- Read-only online settlement audit at `2026-10-24T05:39:51Z` found `362`
  native fee-after rows / `259` decision groups across the latest six hours;
  native lineage passed `200/200`; settlement success/failure counters were
  `51/0`. The latest three-hour recovery projection had four horizon rows for
  each of `379` decisions, but this does not imply every horizon is a clean,
  completed native fee-after training label. No paper entry or OKX position
  history appeared in the bounded six-hour query.
- The bounded three-hour market-decision audit at `05:40:05Z` found `378/378`
  actions were `hold`, with no executed decision. The authorized model horizon
  was 5 minutes, but every selected cohort was missing. The aggregate blocker
  was `paper_prediction_horizon_unavailable`; all 756 local-ML side contracts
  reported `current_prediction_contract_incomplete`, while server-profit and
  time-series families had no available artifacts. Exact recent decision
  projections showed `distribution_members_missing`, null fee-after objective,
  and no eligible quant evidence. This is an evidence/model-contract hold,
  not an exchange rejection.
- A fresh transparent baseline ledger at `05:43:08Z` read `527` native
  fee-after rows / `383` groups (horizons: `154` x 5m, `139` x 15m, `135` x
  60m, `99` x 240m). It selected no candidate. The 60m always-long diagnostic
  was positive on only `94` development / `41` validation rows
  (`+0.2141%` / `+0.5796%`; PF `1.492` / `3.266`); it remains inspected,
  single-side development evidence, not a policy or exam. The 240m always-short
  result reversed from `+2.0743%`, PF `65.953` on 69 development rows to
  `-0.3877%`, PF `0.382` on 30 validation rows, showing material regime
  instability.
- Normal gated local-ML training preflight completed at `05:54:50Z` after its
  earlier attempt correctly declined to overlap an active training lease.
  It loaded 2,612 rows and built 527 eligible rows / 383 groups. Purged
  chronological training used 160 rows / 135 groups; holdout used 295 rows /
  192 groups; 72 training rows / 56 groups were purged, with zero group
  overlap. The resulting 5m regressor was in the low-sample Ridge path, which
  emits no empirical model members and therefore cannot create an eligible
  standardized prediction distribution. More importantly, its ranking
  evidence was adverse: the highest-scored long bucket averaged `-0.1785%`
  (LCB `-0.2845%`, PF `0.082`), and the highest-scored short bucket averaged
  `-1.4022%` (LCB `-3.5052%`, PF `0.027`); the lower-scored long bucket
  averaged `+1.2916%`. Authoritative executed-trade samples were zero. Repeated
  logistic convergence warnings were also emitted. The run was diagnostic
  only: no artifact persisted, no promotion, no policy change and no orders.
  Training data fingerprint: `1988bfe7c732f2f65fe6fb76defe9a54d094d0068db6a6861281a6fe2eaa98e1`.
- These findings reject the current 5m fitted candidate; they do not justify
  score reversal, changing the estimator threshold on this inspected sample,
  or opening entries. G2 remains **BLOCKED** and G3 remains pending; G4-G6
  remain pending.

### Horizon-Aware Current Development Preregistration (2026-10-24 13:57 Asia/Shanghai)

- Before the next read-only run, freeze the experiment as
  `horizon-current-v1-20261024T055700Z`: cutoff-exclusive
  `2026-10-24T05:57:00Z`, prior 24 hours, at most 1,000 decision groups,
  completed paper labels with native executable one-minute path and
  `fee_after_complete=true` only. This is development data; it is not an
  independent exam and cannot pass G4.
- Use the existing `ml_horizon_development` search unchanged:
  horizons 5/15/60/240 minutes, existing-vs-market-only feature sets, Ridge
  alpha 1/10/100, thresholds 0/0.05/0.1%, two purged inner folds and three
  chronological outer folds. Common four-horizon groups only; compare all
  opportunities and one-position non-overlapping outcomes. Keep the
  existing minimum 16 selected rows, positive fee-after mean/LCB, PF > 1 and
  positive means in both inner folds. No candidate-set or gate edits.
- One read-only DB connection, 15-second statement timeout, 25% CPU, 1 GiB
  RAM, 300-second runtime; output only aggregate counts, exact source/frame
  fingerprints and compact fold metrics under ignored
  `data/profitability_remediation/`. No DB writes, artifact writes,
  promotions, runtime-policy changes, orders, or service restarts.
- If there is insufficient complete-group coverage, keep G3 pending and
  continue collecting native maturity. If a candidate is selected, it remains
  development-only; preregister a new untouched future interval before G4.

### Online Maturity And Settlement Throughput Recheck (2026-10-24 14:35 Asia/Shanghai)

- The completed preregistered horizon report
  `online-horizon-development-v1-20261024T055700Z.json` is a
  development-only negative/insufficient result: `527` native fee-after rows,
  `383` decision groups, only `4` groups with all four horizons, no folds,
  no selected policy, no artifact, and no orders. It cannot pass G4.
- A bounded online coverage comparison at `2026-10-24T06:15:29Z` falsified the
  first suspected cause that the four-horizon shortage was only an
  `updated_at` window truncation. The decision-created 24-hour window and the
  legacy updated-time 24-hour window both produced `383` eligible groups with
  the same coverage distribution: `275` groups with one horizon, `76` with
  two, `28` with three, and `4` complete groups. Partial/historical/quarantined
  rows remain excluded by contract.
- The bounded post-restart settlement projection at `2026-10-24T06:27:54Z`
  found overdue paper rows still pending: `82` at 5m, `87` at 15m, `79` at
  60m and `90` at 240m; the oldest were roughly `55-57` minutes overdue.
  Runtime maintenance was healthy but its last completion count was zero.
  This is a data-pipeline throughput defect, not exchange rejection or a reason
  to lower entry gates.
- Likely root cause identified in the local code, pending online throughput
  confirmation: the market-only background worker
  selected up to `25` due rows while `ShadowBacktestService.update_due`
  has a `20s` total collection budget and only `2` native path requests in
  flight. The batch can cancel unfinished path work before any durable
  completion, leaving due rows pending and starving mature four-horizon groups.
- Repair made in owned paths: set
  `SHADOW_BACKTEST_MARKET_BACKGROUND_UPDATE_LIMIT = 4`, matching the existing
  bounded recovery batch; keep foreground update and all quality, cost,
  prediction, risk, leverage and live-mode gates unchanged. Added a regression
  asserting the bounded batch size. Local targeted suite: `38 passed`;
  Ruff, compileall and `git diff --check` passed.
- This repair is not a profitability result and has not yet been deployed or
  rechecked online. Next exact action: upload only
  `services/trading_service.py` and `tests/test_trading_service_boundaries.py`
  plus the existing development-audit paths with restart skipped; run the
  bounded remote tests, then restart only `bb-paper-trading.service` and
  measure fresh overdue counts, completion throughput, native fee-after groups,
  and entry decisions. Keep paper mode and entry permission unchanged.

### Bounded Settlement Batch Deployment Recheck (2026-10-24 14:54 Asia/Shanghai)

- Targeted sync for the four owned source/test paths returned
  `source-integrity-ok:4` and `Uploaded 0 changed files`: the online checkout
  already matched the intended local bytes. No unrelated file was uploaded.
- Read-only pre-restart checks at `2026-10-24T06:46:xxZ` confirmed
  `bb-paper-trading.service` active/running, `NRestarts=0`, persisted and
  runtime mode `paper`, `paused=false`, `running=true`. The bounded settlement
  audit showed no executed entries or OKX position-history rows. Only
  `bb-paper-trading.service` was restarted; it returned active/running with
  `NRestarts=0` and `ExecMainStartTimestamp=2026-10-24 06:47:58 UTC`.
- Three remote service-boundary regressions passed individually:
  background maintenance scheduling, batch/recovery limit equality (`4`),
  and reuse of an existing background task. The new horizon-audit command
  regression also passed (`1 passed`). A broader remote run of the full
  boundary file was not clean: one existing boundary failed and pytest later
  reported temporary-directory cleanup errors; this is not counted as a
  passing suite. The local focused suite remains `38 passed` with Ruff,
  compileall and diff checks passing.
- At `2026-10-24T06:46:55Z`, before restart, the pending-due counts were
  `124/128/117/139` for 5/15/60/240m (`508` total). At
  `2026-10-24T06:54:17Z`, the bounded queue projection showed `88/95/90/110`
  pending (`383` total), alongside `81` completed and `109` quarantined rows
  updated in the last five minutes. The latest native fee-after set increased
  from `248` to `274` decision groups; 200/200 sampled native lineage checks
  passed. Maintenance was active in paper mode with no surfaced round error,
  last completed count `21`, cumulative success/failure counters `11/1`.
- This is evidence that the deployed batch is making progress, but not proof
  that the queue backlog is resolved: pending work remains, a maintenance
  failure was counted, and many late/invalid rows are quarantined. Inspect the
  failure/quarantine reasons and monitor one additional bounded interval
  before calling the throughput repair stable.
- No paper entries or OKX position-history rows appeared after restart.
  Decision permissions, prediction requirements, risk limits, leverage and
  live mode were unchanged. The zero-entry blocker remains model evidence:
  the recent decisions stay `HOLD`; this pipeline repair does not create a
  profitable signal. G1 remains negative attribution-only, G2 **BLOCKED**,
  G3 pending, and G4-G6 pending.
- Next: extract only bounded maintenance/quarantine reason aggregates and a
  fresh all-native horizon-coverage projection; do not read new exam labels or
  retune the already rejected five-minute family. Continue with G2/G3 only if
  the current-contract data supports it.

### Horizon-Current Development Six-Hour Preregistration (2026-10-24 14:57 Asia/Shanghai)

- Development-only experiment identity:
  `horizon-current-v1-6h-20261024T065600Z`. The audit CLI's protocol choice
  remains `v1`; no unsupported protocol revision is being introduced.
  Cutoff-exclusive UTC timestamp: `2026-10-24T06:56:00Z`; lookback: six
  hours; group cap: 1,000.
- Input contract is unchanged: paper rows with `status=completed`,
  native executable one-minute path source
  `shadow_entry_result_and_okx_native_1m_path`, and
  `fee_after_complete=true`. This is a new development interval after the
  throughput repair, not an independent exam and cannot pass G4.
- Search and acceptance rules remain unchanged from v1: common
  5/15/60/240-minute groups only; existing-vs-market-only features; Ridge
  alpha `1/10/100`; net thresholds `0/0.05/0.1%`; purged chronological inner
  and outer folds; minimum selected rows `16`; positive fee-after mean/lower
  bound; PF above one; positive means in both inner folds. No candidate,
  threshold, sizing, risk or entry-gate edits are allowed.
- Resource budget: one read-only connection, 15-second SQL statement timeout,
  25% CPU, 1 GiB RAM, 300-second runtime; aggregate metrics and fingerprints
  only under ignored `data/profitability_remediation/`. No artifact, database
  mutation, promotion, policy change, order, or service restart.

### Horizon-Current Development Six-Hour Result (2026-10-24 15:02 Asia/Shanghai)

- The preregistered read-only run completed at
  `2026-10-24T07:01:36.789194Z` with exact cutoff
  `2026-10-24T06:56:00Z`. It read `389` native fee-after rows across `286`
  decision groups: `118` x 5m, `109` x 15m, `104` x 60m and `58` x 240m.
  Only `4` groups had all four horizons, so no nested folds were estimable and
  no policy was selected (`insufficient_development_data`).
- Development input SHA256:
  `9dbee51f4c8da9e523f3ca3d3b98c20b079fee5e463d206a5ce9e132c9b50777`;
  snapshot SHA256:
  `1bcd0c0f1424c20837cb0621f6d7bdb4af2a4a9b93698f977faf61380be87ac9`.
  Full aggregate report:
  `data/profitability_remediation/online-horizon-development-v1-20261024T065600Z.json`.
  The report confirms `exam_labels_evaluated=false`, no artifact, no database
  mutation, no promotion, no orders and no policy change.
- This confirms that reducing the settlement batch now drains stale pending
  work but has not yet yielded enough complete four-horizon groups for the
  preregistered horizon search. It is not a candidate rejection or an exam
  result. G2 remains **BLOCKED** and G3 remains pending; G4-G6 remain pending.
- Next exact action: use one bounded read-only grouped projection over the
  same recent decision cohort to separate still-immature 240m labels from
  quarantine, provenance and missing-row causes. Preserve the current search
  family and entry permissions; do not create or inspect an exam.

### Post-Repair Current-Stream And Entry Blocker Recheck (2026-10-24 15:14 Asia/Shanghai)

- A fresh bounded projection at approximately `2026-10-24T07:14Z` found
  `404` current native fee-after rows across `330` decision groups in the
  latest six hours. In the last 15 minutes, `36/36` paper decisions were
  `HOLD` and not executed. The paper service remains active/running in paper
  mode, `NRestarts=0`, with `ExecMainStartTimestamp=2026-10-24 06:47:58 UTC`.
- There were zero new native quarantines for missing shadow path labels in
  that 15-minute window. This is only an observation about the post-restart
  stream; no new retry guard was deployed or inferred from it, and the
  window is not sufficient to claim G2 complete.
- A bounded read-only exact-source recheck of four previously quarantined
  native rows found their stored result clocks were within 1 minute of the
  scheduled label time (10-26 seconds); each original stored path had
  `bar_count=0` and no MFE/MAE path labels. Fetching the same exact native
  source interval now returned eight 1m bars and re-verified six expected bars
  per row as `clean`, with both entry and result prices reachable. No database
  row was changed or requalified. This shows those older missing-path labels
  are now reconstructible, but why their original updates bypassed the current
  timely-path pending guard remains unresolved; do not use them as acceptance
  evidence until a governed repair/replay contract is reviewed.
- The separate three-hour entry-horizon audit completed at
  `2026-10-24T07:08:32Z` read 300 decisions, all `HOLD`. All 300 local-ML
  primary-horizon contracts were `current_prediction_contract_incomplete`,
  with no selected or training horizon; the aggregate blocker was
  `paper_prediction_horizon_unavailable`. Server-profit and time-series
  sources were `signal_unavailable`. No exchange entry was attempted. This is
  the direct current explanation for zero positions: eligible prediction
  evidence is absent, not an exchange order rejection.
- No profitability candidate has passed. G1 remains negative
  attribution-only; G2 **BLOCKED**; G3 pending due to only four all-native
  four-horizon groups in the preregistered six-hour window; G4-G6 pending.
  Keep paper-only mode and every quality, cost, prediction and risk gate
  unchanged.
- Next: continue observing the current post-restart stream until clean,
  matured four-horizon coverage supports nested development; separately
  reconcile why the four older timely-clock rows were completed with empty
  native paths. Any requalification requires an explicit reproducible,
  auditable repair and cannot turn the already inspected interval into an
  independent exam.

### Checkpoint Commit And Source Integrity (2026-10-24 15:18 Asia/Shanghai)

- Owned changes are committed and pushed as `7701151`
  (`Bound paper shadow maintenance and record profitability evidence`).
  The commit contains only the bounded background batch, its regression and
  the bounded development audit/test plus this execution-plan evidence.
- A post-push staged source check returned `source-integrity-ok:4` with zero
  changed files uploaded and no service restart. The remote paper process
  therefore continues running the already verified bytes.
- Local verification after the checkpoint: `401 passed` across the shadow,
  trading-boundary and horizon-audit tests; Ruff, compileall and diff checks
  passed. This verifies software behavior, not profitability.

### Latest Bounded Native Projection (2026-10-24 15:51 Asia/Shanghai)

- The latest six-hour bounded projection at `2026-10-24T07:51:58Z` found
  native fee-after rows by horizon of `107` (5m), `101` (15m), `90` (60m)
  and `64` (240m). No decision group had all four horizons; `236` groups had
  one horizon and `63` had two. Recent decisions remained `64 HOLD` with no
  executed entries, and the paper service remained healthy.
- This is a maturity/coverage observation only. It does not establish a
  profitable candidate, a valid current prediction contract, or an
  independent exam. G2 remains **BLOCKED** and G3-G6 remain pending.
- The local branch is one commit ahead of `origin/main` at `d2de71a`; the
  previous push attempt failed on transient GitHub HTTPS connectivity.
  Retry the push of this documentation-only checkpoint before starting any
  new experiment. Do not rerun the completed six-hour study unchanged.

### Horizon-Current Development Recheck Preregistration (2026-10-24 16:06 Asia/Shanghai)

- Development-only identity: `horizon-current-v1-20261024T080400Z`.
  Cutoff-exclusive UTC timestamp: `2026-10-24T08:04:00Z`; six-hour lookback;
  at most 1,000 decision groups. Freeze the source to completed paper labels
  with native executable one-minute paths and `fee_after_complete=true`.
- Keep the existing v1 search unchanged: common 5/15/60/240-minute groups,
  existing-vs-market-only features, Ridge alpha 1/10/100, thresholds
  0/0.05/0.1%, chronological purged inner/outer folds, minimum 16 selected
  rows, positive fee-after mean and lower bound, PF above one, and positive
  means in both inner folds. This overlapping window is development evidence,
  not a fresh exam.
- Budget: one read-only database connection, 15-second statement timeout,
  25% CPU, 1 GiB memory, 300-second runtime; aggregate metrics and exact
  fingerprints only under ignored `data/profitability_remediation/`. The
  runner must confirm dependency hashes and reject an already active
  identical unit. No database mutation, artifact, promotion, policy change,
  order or service restart.

### Latest Online Entry And Settlement Recheck (2026-10-24 16:49 Asia/Shanghai)

- A bounded read-only recovery audit covered the latest three hours
  (`2026-10-24T05:49:31Z` through `2026-10-24T08:49:31Z`). It found `391`
  paper market decisions, all `hold` and none executed. This is an evidence
  and model-permission hold, not an OKX order rejection.
- The recent funnel classified `292` decisions as `no_candidate` and `8` as
  `insufficient_evidence`. Recent examples consistently report missing
  positive expected net return, missing objective net, missing quant evidence
  and missing prediction horizon. Feature availability was clean for
  `298/300` sampled rows; the two invalid rows remain disclosed and were not
  used to authorize entries.
- Local-ML paper permission remains explicitly disabled by authoritative
  executed-trade evidence: sample count `267`, fee-after mean
  `-0.303832%`, lower confidence bound `-0.431590%`, and profit factor
  `0.549713`. Server-profit, time-series and sentiment sources still have no
  qualifying fee-after evidence. The current artifact remains `learning_only`
  / degraded, with live and paper model influence disabled. No artifact,
  threshold, leverage, risk gate or entry permission was changed.
- Settlement is still a separate pipeline blocker. The same bounded audit
  observed overdue pending rows of `81` (5m), `76` (15m), `81` (60m) and
  `77` (240m), plus new pending rows that are not yet mature. Quarantine
  reasons were dominated by historical-path availability/provenance
  violations (`88` sampled reasons), with additional native market-fact and
  source-consistency failures. The latest native executable set had `298`
  completed rows across `274` decision groups; this is not yet enough
  complete, independent four-horizon development coverage.
- The OKX public latency diagnostic now covers the native consistency route in
  addition to quote routes. Focused local verification passed (`45` tests,
  Ruff, compileall and diff check). This instrumentation is diagnostic only;
  it does not change trading behavior or authorize entries.
- Acceptance state remains unchanged: G1 passed as negative attribution-only,
  G2 **BLOCKED**, G3 pending, and G4-G6 pending. The next action is to keep
  the paper service fail-closed while improving native settlement coverage
  and collecting enough clean, mature fee-after rows for a preregistered
  development search. A positive independent candidate is still required
  before any controlled paper execution can be authorized.

### Fresh Online Entry And Backlog Audit (2026-10-24 17:42 Asia/Shanghai)

- The bounded shadow/settlement audit completed at
  `2026-10-24T09:34:26Z`. Paper runtime was active, unpaused and reporting no
  round or market error. The bounded evidence contained no executed-entry rows,
  no OKX position-history rows, and no current open positions.
- The same projection found `593` due `pending` shadow rows:
  `150` at 5m, `148` at 15m, `154` at 60m and `141` at 240m. The oldest due
  timestamps were between `2026-10-24T07:52Z` and `08:02Z`, about 92-102
  minutes overdue at audit time. It also found `298` native executable
  fee-after rows across `274` groups; the latest accepted native update was
  `2026-10-24T08:40:29Z`. Later completed rows included `45` entries sourced
  from `preserved_market_klines_historical_path`, which are historical-only
  and cannot qualify as native executable training evidence. The bounded
  fingerprint check verified `200/200` sampled native rows.
- A selected fresh market decision (`692862`, SUSHI/USDT), audited at
  `2026-10-24T09:42:34Z`, was `hold`, `was_executed=false`, with no order or
  position. Its entry permission was denied as `no_direction`; blockers were
  non-positive/missing expected net return, missing objective net, excessive
  loss probability for quality-observation entry, missing quantitative
  evidence, and missing prediction horizon. The profit model returned
  `trained_profit_model_unavailable`; both long and short after-cost return
  distributions were absent. The most recent 30-decision summary likewise
  contained only `hold` decisions. This is an evidence/model-permission hold,
  not a current exchange order rejection. The reconciliation snapshot still
  reported `can_open_new_entries=true`, which means the account-wide switch is
  not the immediate blocker; no direction-qualified candidate exists.
- A separate high-granularity entry-recovery audit lost its SSH channel and
  produced no evidence; it was not retried. The successful bounded audits above
  were read-only and did not inspect exam labels, alter policy, or place orders.
- The latest exact entry reason explains why there is no new paper trade, but
  does not establish profitability. G1 remains passed only as negative
  attribution; G2 remains **BLOCKED**; G3-G6 remain not passed. Five of the six
  profitability gates therefore remain incomplete: one blocked data/label
  gate and four downstream research, independent-evaluation, paper-execution
  and final-acceptance gates. The prior local-ML executed-trade evidence
  (`n=267`, fee-after mean `-0.303832%`, LCB `-0.431590%`, PF `0.549713`)
  remains adverse and was not refreshed by this entry audit.
- Next exact action: add a bounded due-age projection by horizon and inspect
  aggregate native-route wait/provider latency to determine why mature native
  rows stopped updating after `08:40Z`. Confirm the queue-selection/freshness
  interaction before changing scheduling or concurrency. Keep all entry and
  risk permissions unchanged; no profitability claim or independent exam is
  available yet.

### Native Evidence Queue Repair (2026-10-24 18:20 Asia/Shanghai)

- The bounded remote journal aggregation covered the prior 30 minutes and
  found `1,175` `fetch_native_consistency_ohlcv` latency events. Queue wait was
  `20,557.2 ms` at the median, `25,791.1 ms` at P95 and `26,412.6 ms` at the
  maximum; provider duration was `0.0 ms` at the median, P95 and maximum.
  There were `51` shadow price-path batch timeouts. This pattern identifies
  request-gate starvation/cancellation before the provider, not an OKX
  rejection or a slow candle response.
- The code path confirmed the cause: cache-only feature scans started a
  non-blocking native request for each rotating symbol, while
  `DataService._fetch_native_consistency_bars` had no total background-task
  bound. Those tasks accumulated behind the isolated two-request native lane.
  Foreground path verification also used the same low-priority lane, so valid
  settlement work could wait behind diagnostic refreshes.
- Repair implemented in owned paths: cache-only native refreshes now stop
  creating work after four pending background tasks; foreground/native path
  verification is submitted as a priority request; compatibility fallbacks
  preserve older test adapters; cancellation diagnostics are recorded as
  `outcome=cancelled` instead of `outcome=ok`. No entry threshold, cost rule,
  risk limit, leverage or live-mode setting changed.
- Local verification passed: `155` focused data-service/REST tests and `435`
  related feature/SDK/trading-boundary tests; Ruff, compileall and whitespace
  checks passed. The source was staged to the online checkout with
  `source-integrity-ok:4` and no service restart. The repair is therefore not
  yet active in the long-running paper process.
- Before rollout, the paper unit must be read-only checked for mode `paper`,
  `paused=false`, zero positions and no active orders. Restart only
  `bb-paper-trading.service`; then measure native queue wait, completed native
  rows, due counts and HOLD reasons over one bounded interval. Do not promote
  any model or requalify historical rows based on this throughput repair.

### Priority Isolation Follow-Up (2026-10-24 18:34 Asia/Shanghai)

- The first post-deployment interval verified native data progress (fee-after
  rows increased from `298` to `309`, decision groups from `274` to `285`, and
  the latest native update advanced to `2026-10-24T10:29:47Z`). Maintenance
  completed `25` rows, with no surfaced runtime error; `200/200` sampled native
  fingerprints remained valid. However, the pending queue continued growing,
  so the throughput fix is not yet accepted.
- Post-deployment journal evidence still showed a native request queue-wait
  median of `20,027.6 ms` / P95 `25,993.6 ms` over 197 requests, with 8 shadow
  path-batch timeout events. Code review found the first rollout had also
  placed delayed shadow path verification into the priority lane. That would
  let maintenance compete with current market facts, contrary to the required
  execution ordering.
- Follow-up correction keeps real-time current-fact requests priority-enabled
  but restores delayed shadow path verification to normal/low priority. Added
  a regression asserting this separation. Four focused regressions passed
  locally; the changed bytes still require targeted online upload, remote
  regression, paper-only restart and another bounded latency/throughput check.
- The four-task background cap and priority separation are transport/data
  availability fixes only. They do not supply missing prediction distributions
  or positive after-cost expected return; the model remains unauthorized and
  entry/HOLD conclusions must be refreshed after rollout.

### Priority-Isolated Rollout Recheck (2026-10-24 18:52 Asia/Shanghai)

- After the second paper-only restart at `2026-10-24T10:46:48Z`, the bounded
  window through `2026-10-24T10:49Z` completed `96` rows and quarantined `24`
  rows with no maintenance failure. Pending due rows fell from `804` in the
  prior window to `668` (`139/145/126/128` for 5/15/60/240m); the oldest
  pending rows were around `09:11-09:14Z`, so backlog remains material but is
  moving in the correct direction.
- Native executable fee-after coverage increased from `309` rows / `285`
  groups to `398` rows / `347` groups. The bounded lineage check remained
  `200/200` verified. Historical-only completions (`322` rows / `288` groups)
  remain quarantined from native training and are not counted as profitability
  evidence.
- Five-minute journal aggregation found no shadow price-path batch timeout.
  Native low-priority requests had queue-wait P50/P95/max of
  `313.9/1,575.7/1,940.4 ms`; current-fact priority requests had
  `101.6/992.2/992.2 ms`. This is a substantial improvement over the earlier
  `20-26s` queue starvation pattern and validates the priority separation.
- The latest market decision (`693009`, PEPE/USDT) was still `hold` with no
  order/position. Its exact entry blockers remained expected net return not
  positive, objective net missing, loss probability too high for observation
  entry, quantitative evidence missing and prediction horizon missing. Both
  profit and time-series local model contracts reported
  `trained_*_model_unavailable` with incomplete return distributions. The
  account-level reconciliation still allowed new entries, so no exchange
  rejection is involved.
- This rollout fixes a real evidence-collection bottleneck but does not pass
  G2 or establish a profitable strategy. G1 remains negative-attribution-only;
  G2 **BLOCKED**; G3-G6 remain pending. Continue bounded collection until
  native four-horizon development coverage is sufficient, then run only the
  preregistered development search. Do not use historical-only rows, inspect
  an exam, or lower entry/risk gates to force a trade.

### Horizon-Current Development Recheck Preregistration (2026-10-24 18:55 Asia/Shanghai)

- Development-only identity: `horizon-current-v1-20261024T105500Z`.
  Cutoff-exclusive UTC timestamp: `2026-10-24T10:55:00Z`; six-hour lookback;
  maximum 1,000 decision groups. The source is limited to completed paper
  rows with `shadow_entry_result_and_okx_native_1m_path` and
  `fee_after_complete=true`.
- Keep the existing v1 family unchanged: common 5/15/60/240-minute groups;
  existing-vs-market-only feature policies; Ridge alpha `1/10/100`; net
  thresholds `0/0.05/0.1%`; purged chronological inner and outer folds;
  minimum selected rows `16`; positive fee-after mean/lower bound; PF above
  one; and positive means in both inner folds. No candidate-set, threshold,
  sizing, risk or entry-gate edits.
- Budget remains one read-only DB connection, 15-second statement timeout,
  25% CPU, 1 GiB RAM, 300-second runtime; aggregate metrics, source IDs and
  fingerprints only under ignored `data/profitability_remediation/`. No
  artifact, database mutation, promotion, policy change, order, service
  restart or exam-label read.

### Horizon-Current Development Recheck Result (2026-10-24 18:57 Asia/Shanghai)

- The preregistered development-only run completed at
  `2026-10-24T10:57:16Z` with the exact cutoff
  `2026-10-24T10:55:00Z`. It read `429` native fee-after rows across `371`
  decision groups: `96` x 5m, `111` x 15m, `117` x 60m and `105` x 240m.
- The common four-horizon group count was still `0`; therefore no purged
  nested folds were estimable, no policy was selected, and no artifact or
  model permission was produced. The report status is
  `insufficient_development_data`; `exam_labels_evaluated=false`,
  `artifact_persisted=false`, `database_mutated=false`,
  `promotion_authorized=false` and `orders_submitted=false`.
- Development input SHA256:
  `840fa19450104ff54beda499dcd610ebd955d6c5a2d00746877c2c56b73c48a3`;
  snapshot SHA256:
  `48356ac4c579a3647b9810b52fdeeafd186f44be6eebae2c75e955def80a6578`.
  Full aggregate report:
  `data/profitability_remediation/online-horizon-development-v1-20261024T105500Z.json`.
- This is not a candidate rejection and not an independent exam result; it
  means the repaired stream still has no complete four-horizon decision
  groups. G2 remains **BLOCKED** and G3-G6 remain pending. Continue collecting
  clean native labels and do not enable entries merely because throughput
  improved.

### Fresh Online Entry And Queue Audit (2026-10-24 19:35 Asia/Shanghai)

- Two bounded read-only online audits completed at approximately
  `2026-10-24T11:35Z`. The paper runtime was active and unpaused, with no
  runtime, market or position-loop error. In the preceding three hours,
  `394/394` market decisions were `hold` and none was executed; the bounded
  OKX position-history query found no entry lifecycle in the preceding six
  hours. This is not evidence of an exchange order rejection.
- The last 60 decisions were all classified `no_candidate`. A representative
  complete analysis was held because direction support lacked positive
  fee-after expected return, objective net, quantitative evidence and a
  prediction horizon; quality-observation loss probability was also too high.
  Account-level entry availability was not reported as the blocker in this
  audit. Feature availability was sealed and clean for `59/60`; one row had a
  missing/stale technical availability contract.
- Model authorization remains intentionally fail-closed. Local ML's
  authoritative executed-trade evidence is `n=267`, mean fee-after return
  `-0.303832%`, lower bound `-0.431590%`, and PF `0.549713`; execution
  permission is false. Server-profit, time-series and sentiment sources each
  have zero qualifying fee-after samples. The latest artifact is still
  `learning_only` / degraded with influence disabled. Recent model training
  uses shadow data (`trade=0` in its sample cursor), so it has not replaced
  the negative authoritative execution evidence.
- The repaired settlement stream is progressing: the preceding six-hour
  bounded projection contained `528` native executable, fee-after-complete
  rows across `388` decision groups and all four horizons; the newest native
  update was `2026-10-24T11:34:25Z`. In the preceding five minutes,
  `28` rows completed and `10` were quarantined. The 200-row lineage sample
  remained `200/200` valid. At audit time, `10` pending rows were due across
  horizons, most just reaching their scheduled time; the much larger pending
  totals include labels not yet mature and must not be described as overdue.
- Data-quality quarantine remains material: recent causes include invalid
  native market facts, missing/unverified historical label availability,
  absent shadow-label contracts, and source-consistency/path disagreements.
  No quarantined or historical-only rows were counted as native training
  evidence or modified by these audits.
- Acceptance remains G1 passed as negative attribution only, G2 **BLOCKED**,
  and G3-G6 not passed. Five gates remain incomplete: G2, G3, G4, G5 and G6.
  The immediate entry blocker is missing/insufficient predictive evidence and
  failed historical fee-after quality, not an account-wide open switch or
  exchange rejection. This explains no new positions; it does not establish
  why any future strategy would profit.
- Next development-only run preregistration: identity
  `horizon-current-v1-20261024T113500Z`, cutoff-exclusive
  `2026-10-24T11:35:00Z`, six-hour window, maximum 1,000 decision groups.
  Freeze source to completed paper rows with native executable 1m paths and
  `fee_after_complete=true`. Keep the previously preregistered v1 research
  family unchanged: common 5/15/60/240-minute groups; existing-vs-market-only
  feature policies; Ridge alpha `1/10/100`; net thresholds `0/0.05/0.1%`;
  purged chronological inner/outer folds; minimum 16 selected rows; positive
  fee-after mean and lower bound, PF above one, and positive means in both
  inner folds. This is development, not an exam or promotion. Retain the
  existing resource budget (one read-only DB connection, 15-second statement
  timeout, 25% CPU, 1 GiB memory, 300-second runtime); no artifact, database
  mutation, model permission, policy change, order or service restart.

### Horizon-Current Development Recheck Result (2026-10-24 19:40 Asia/Shanghai)

- The preregistered development-only run completed at
  `2026-10-24T11:39:58Z` with the exact cutoff
  `2026-10-24T11:35:00Z`. It read `528` native fee-after rows across
  `388` decision groups: `127` x 5m, `133` x 15m, `134` x 60m and `134`
  x 240m.
- The common four-horizon group count was still `0`, so no purged nested
  folds were estimable, no candidate was selected, and no artifact or model
  permission was produced. The report status is
  `insufficient_development_data`; `exam_labels_evaluated=false`,
  `artifact_persisted=false`, `database_mutated=false`,
  `promotion_authorized=false` and `orders_submitted=false`.
- Development input SHA256:
  `1433768634865c21066f3ecb2a2078132dda17e3e3e8fcb8e0f8f3c5042f7dc5`;
  snapshot SHA256:
  `8ce94558f09436da446538c786648b0b052d2dec59d6cb98021bbdbbb63325c8`.
  Full aggregate report:
  `data/profitability_remediation/online-horizon-development-v1-20261024T113500Z.json`.
- This is insufficient data, not a candidate rejection and not an
  independent exam result. G2 remains **BLOCKED** and G3-G6 remain pending.
  The next investigation must explain the missing same-decision four-horizon
  closure (including group identity, scheduling and quarantine causes) before
  another identical development run is started.

### Mature Horizon-Group Coverage Root Cause (2026-10-24 20:04 Asia/Shanghai)

- Added the read-only bounded diagnostic
  `scripts/audit_online_horizon_group_coverage.py` and regression
  `tests/test_audit_online_horizon_group_coverage.py`. It uses one
  repeatable-read/read-only DB transaction, a 10-second SQL timeout, a
  12-hour lookback, current label version, four explicit horizons and at most
  eight compact group examples. It does not fetch full feature snapshots or
  mutate rows.
- The online snapshot at `2026-10-24T12:03:57Z` found `1,051` market
  decision groups older than 242 minutes in the 12-hour window. Each had all
  four horizon rows present, so incomplete row creation is not the dominant
  cause of zero nested-development coverage. However only `4/1,051` groups
  had all four rows completed from the native executable 1m path with
  `fee_after_complete=true`; at least one horizon failed that eligibility
  contract for the other groups. The four fully eligible groups were created
  around `2026-10-24T01:18Z-01:23Z` and their labels matured outside the
  development run's six-hour `updated_at` window. Therefore the six-hour
  run's `0` common groups is consistent with current source and cutoff rules,
  not evidence that the research runner dropped complete rows.
- A separate bounded sample of the latest 100 quarantine notes in the prior
  three hours was dominated by native market-fact invalidity (`40`) and
  historical-path provenance/label-availability failures (`48` across two
  reason combinations). Other observed causes were source-consistency/path
  disagreements (`5`), missing shadow-label contract (`3`), stale technical
  availability (`1`), price-vs-indicator divergence (`1`), and missing or
  ambiguous index-price availability (`2`). Historical-path rows are not
  eligible substitutes for decision-time native executable evidence.
- Local verification for the new diagnostic: compileall and Ruff passed;
  focused related coverage tests passed. The audit was read-only and created
  no online data changes. No trading rule, model artifact, threshold, entry
  permission, order or service was changed.
- This sharpens, but does not resolve, the original blocker: G1 remains
  negative attribution only; G2 is **BLOCKED** by insufficient clean native
  fee-after label quality; G3 has no estimable four-horizon nested candidate;
  G4 has no frozen independent exam; G5 has no accepted candidate to take
  through governed paper fills; G6 cannot pass. Five gates remain incomplete.
- Next exact action: trace the top live quarantine reason
  `native_market_fact_invalid` to its validation fields and the point-in-time
  capture path, then separately inspect the five source-consistency/path
  disagreements. Repair only a reproducible present-time defect; never
  requalify already quarantined historical rows or loosen provenance and
  fee-after gates. After a focused fix, verify on newly created samples, then
  start a newly cutoff-bound development run only after its complete-group
  coverage is sufficient.

### Fresh Root-Cause And Coverage Recheck (2026-10-24 21:11 Asia/Shanghai)

- Online service state was checked at `2026-10-24T13:09:40Z`: paper trading
  and dashboard units were active, with zero service restarts; runtime mode
  was `paper`, unpaused, with fresh heartbeat and no round/market/position
  error. This verifies a running decision loop, not trading eligibility or
  profitability.
- The bounded 30-minute funnel at `2026-10-24T12:54:38Z` contained `79/79`
  market decisions as `hold`; the latest 30 detailed decisions were `30/30`
  `no_candidate`. All 30 had sealed, clean feature availability and required
  derivatives routes completed. The representative blockers were missing
  positive fee-after expected return, objective-net evidence, quantitative
  evidence and forecast horizon, plus excessive observation loss probability.
  This points to missing qualified signal/economics, not a stopped loop or
  observed exchange order rejection.
- The strategy/model permission evidence remains adverse: local ML has `267`
  authoritative executed samples with mean fee-after return `-0.303832%`,
  lower bound `-0.431590%`, and profit factor `0.549713`. Its execution
  permission remains false. Other specialist models have zero qualifying
  authoritative samples; current training attempts for their fee-after
  objectives report `authoritative_execution_cost_distribution_unavailable`.
  The recent local-model training cursor still uses shadow samples and zero
  new trade samples; it does not supersede the negative executed evidence.
- A new 12-hour, read-only coverage query at `2026-10-24T13:08:43Z` found
  `1,036/1,036` mature groups had all four horizon rows present, but only
  `4/1,036` had every 5/15/60/240-minute row completed from the native
  executable path with fee-after costs complete. Per-horizon eligible rows
  were 389 (5m), 385 (15m), 328 (60m), and 290 (240m); marginal row totals
  are not common complete groups. Thus the stream is producing data, but
  cross-horizon eligibility remains far too sparse for the preregistered
  nested search. This confirms G3 was not merely waiting for missing row
  creation.
- A separate 30-minute reason-code projection at `2026-10-24T13:07:39Z`
  found three decision groups / five rows with no 1m path and unreconciled
  executable quote sources (quote spans observed at 2.7-50.7 seconds), and
  one decision group / two rows with an incomplete path. The sampled
  incomplete path had 300 bars and one missing minute; the missing minute
  was the latest minute in the captured window. These are evidence-collection
  defects or unavailable source windows to investigate, not grounds to
  accept the quarantined labels. Historical OHLCV-only rows continue to lack
  verified decision-time label availability and are not eligible substitutes.
- Acceptance remains G1 passed only as negative attribution, G2 **BLOCKED**,
  and G3-G6 not passed. Five evidence gates remain incomplete. The direct
  reason for no new entries is that current decisions do not have an eligible
  positive after-cost candidate; clearing the data-collection issues alone
  will not make a candidate profitable.
- Next exact action: instrument and trace native 1m fetch outcomes for newly
  created decision IDs, distinguishing provider empty/timeout, background
  queue saturation, cursor-window coverage and the expected open-minute gap.
  Make a narrowly scoped collection repair only if it is reproducible, then
  verify on fresh rows and recompute common four-horizon coverage. Keep model
  influence, entry permissions, execution thresholds and risk limits
  unchanged. Start the frozen development search only when sufficient common
  eligible groups exist; do not claim a profit result or reuse inspected
  observations as an independent exam.

### Fresh Entry And Quarantine Recheck (2026-10-24 22:40 Asia/Shanghai)

- The read-only entry recovery audit completed at
  `2026-10-24T14:40:29Z`. The three-hour window contained `430/430` market
  decisions as HOLD, with no executed decision. In the latest 80 decisions,
  `42` were `no_candidate` and `38` were shadow-training selections; neither
  category had entry permission granted. Feature availability was sealed on
  `80/80` and clean on `79/80`; the one invalid row had a missing/stale
  technical availability contract. Required derivatives routes completed on
  the representative rows except isolated funding timeouts, not a broad
  market-feed outage.
- The immediate entry blocker is unchanged and directly observed:
  representative shadow candidates lacked positive expected net after costs
  and quantitative evidence, while the local ML execution gate remained
  false. Its authoritative executed-trade evidence was `n=267`, mean
  fee-after return `-0.303832%`, lower bound `-0.431590%`, and PF `0.549713`.
  Server-profit, time-series and sentiment sources still had zero qualifying
  authoritative fee-after samples. The latest artifact remained
  `shadow` / `learning_only`, influence disabled, with training cursor
  `trade=0`; the latest local training check reported `new_sample_count=215`
  but `not_due`. Those shadow rows do not replace executed-trade evidence.
  This is a fail-closed strategy/economics decision, not an observed exchange
  rejection or globally disabled paper entry switch.
- The bounded 12-hour per-horizon audit completed at
  `2026-10-24T14:37:51Z`: among `1,602` decision groups, `1,024` were mature
  across all four horizons, but only `4` had all four horizons natively
  executable and fee-after complete. Marginal eligible rows were `414/429/430/360`
  for 5/15/60/240 minutes, respectively; those marginal counts are not
  common groups. Most other mature groups include historical-only labels
  whose point-in-time availability cannot be verified. Recent native
  quarantines also include unreconciled ticker/orderbook quotes, missing or
  incomplete 1-minute paths, and observed prices outside the captured path.
  A sampled quote-source span reached `33.5s` while the validator requires
  source reconciliation within `2s`; the requirement was not relaxed.
- This separates two blockers that must not be conflated: insufficient
  after-cost predictive evidence blocks entries, while incomplete provenance
  and sparse common native labels block research and fresh validation. Fixing
  label collection alone will not authorize a trade or prove profit.
- Stage status is still G0 passed, G1 passed only as negative attribution,
  G2 **BLOCKED**, and G3-G6 not passed: five gates remain incomplete. No
  position, order, model permission, threshold, policy, or risk setting was
  changed by these audits.
- Immediate next action: use a bounded time/symbol-scoped journal projection
  for the newly observed native 1-minute fetches to distinguish empty provider
  responses, timeouts, cursor-window misses, and in-flight background fetches.
  Then repair only a reproducible collection defect and verify on newly
  created labels. Do not rerun broad status audits unchanged, count
  shadow-training selections as trades, lower entry gates, or use inspected
  outcomes as the independent exam.

### Native Path Capacity Repair And Development Preregistration (2026-10-24 23:07 Asia/Shanghai)

- Reproducible defect: the native executable 1-minute route had only two
  isolated request slots while shadow maintenance could queue a four-horizon
  path batch. In the preceding three-hour journal window this produced native
  path `TimeoutError` events and repeated `shadow result price path batch
  timed out` events. This was a bounded collection bottleneck, not a reason to
  weaken the market-fact or entry-quality contract.
- Repair: `PUBLIC_MARKET_NATIVE_CONSISTENCY_CONCURRENCY` was changed from `2`
  to `4` in `data_feed/okx_rest_client.py`. The native route remains on its
  own gate, separate from ticker/orderbook capacity; no quote-reconciliation
  tolerance, fee model, prediction threshold, risk limit, leverage or trading
  mode changed. Local focused tests passed (`155`), Ruff and compileall passed.
  The changed source was uploaded alone, remote `py_compile` passed, remote
  REST-client tests passed (`46`), and only `bb-paper-trading.service` was
  restarted at `2026-10-24T15:04:03Z`; mode remained `paper`.
- Fresh post-restart observation through `2026-10-24T15:07:08Z` showed no new
  native path timeout or shadow path batch timeout in the last two minutes,
  with `21` successful native consistency requests. The bounded 12-hour
  coverage snapshot at `2026-10-24T15:07:07Z` showed `24` mature groups with
  all four horizons qualified, up from `4` before the repair. This validates
  collection progress only; it is not a profitability result.
- Development-only run preregistration: identity
  `horizon-current-v1-20261024T150800Z`; cutoff-exclusive UTC
  `2026-10-24T15:08:00Z`; 12-hour lookback; maximum 1,000 decision groups.
  Source is restricted to completed paper rows with native executable 1m path
  and `fee_after_complete=true`. Candidate family remains unchanged:
  common 5/15/60/240-minute groups; existing-vs-market-only feature policies;
  Ridge alpha `1/10/100`; net thresholds `0/0.05/0.1%`; purged chronological
  inner/outer folds; minimum selected rows `16`; positive fee-after mean and
  lower bound, PF above one, and positive means in both inner folds. This is
  development evidence only: no exam labels, artifact persistence, model
  permission, database mutation, order or service restart.

### Horizon Development Result After Capacity Repair (2026-10-24 23:10 Asia/Shanghai)

- The preregistered development run completed at
  `2026-10-24T15:10:46Z` using the exact cutoff
  `2026-10-24T15:08:00Z`. It read `1,726` native fee-after rows across the
  first `1,000` decision groups; per-horizon rows were `419/443/453/411`
  for 5/15/60/240 minutes and `20` groups had all four horizons.
- The run returned `insufficient_development_data` before candidate scoring:
  the fixed three outer folds cannot meet the minimum primary training
  coverage with only 20 common groups. No policy was selected and no
  profitability metrics were produced. This is a sample-size result, not a
  candidate rejection or an independent exam.
- Input SHA256:
  `9b3da3dfe2f958807fdee2d003d8ab2c1ed92684e4b5a7bb21470b5a31d25bfb`;
  snapshot SHA256:
  `41b80e89883e93398d75a591266e3a7f328f56119a9750a38a98d5830084e204`.
  Report:
  `data/profitability_remediation/online-horizon-development-v1-20261024T150800Z.json`.
  Safety fields remained false for exam evaluation, artifact persistence,
  database mutation, promotion and orders.
- G2 remains **BLOCKED** because native and historical provenance defects still
  exist; G3 remains pending because the preregistered folds are not yet
  estimable; G4-G6 remain pending. The next development run must use a new
  cutoff after enough clean common groups accumulate (at least the fixed-fold
  minimum), not a replay of this same cutoff.

### Fresh Common-Group Coverage Recheck And Development Registration (2026-10-24 23:29 Asia/Shanghai)

- A new bounded read-only coverage query completed at
  `2026-10-24T15:24:46Z` using the 12-hour window. It found 33 mature
  decision groups with all four horizons (`5/15/60/240`) completed from the
  native executable path with `fee_after_complete=true`; 463 other mature
  groups had at least one horizon quarantined or otherwise ineligible. The
  33-group count is above the fixed 16-observation primary training minimum,
  so a new development run is now estimable. This is a data-coverage result,
  not a profitability result.
- The same query showed marginal native rows of `457/479/485/387` for
  `5/15/60/240` minutes and 565 market decisions in the latest four-hour
  entry window. Marginal counts are not common groups and do not authorize
  entries.
- Preregistered next development identity:
  `horizon-current-v1-20261024T152900Z`; cutoff-exclusive UTC
  `2026-10-24T15:29:00Z`; 12-hour lookback; maximum 1,000 decision groups.
  Inputs are restricted to completed paper rows from the native executable
  1-minute path with `fee_after_complete=true`. The candidate family,
  purged chronological folds, minimum selected rows `16`, positive
  fee-after mean and lower bound, PF above one, and positive means in both
  inner folds remain unchanged. This run is development-only: no exam labels,
  artifact persistence, model permission, database mutation, order, or
  service restart.
- Entry recovery at `2026-10-24T15:25:20Z` still showed `565/565` market
  decisions as HOLD and no executed entries. In the latest 80 decisions,
  38 were `no_candidate`, 38 were `shadow_training`, and 4 were
  `insufficient_evidence`; all entry permissions remained false. The direct
  blockers were non-positive expected net after costs, missing quantitative
  evidence, and excessive observation-loss probability. Local ML
  authoritative evidence remains `n=267`, mean fee-after return
  `-0.303832%`, lower bound `-0.431590%`, PF `0.549713`, so its execution
  permission remains false.

### Horizon Development Result At Fresh Cutoff (2026-10-24 23:33 Asia/Shanghai)

- The preregistered run `horizon-current-v1-20261024T152900Z` completed at
  `2026-10-24T15:32:41Z` with the exact cutoff
  `2026-10-24T15:29:00Z`. It read 1,746 native fee-after rows across the
  1,000 most recent decision groups, but only 25 groups contained all four
  horizons. The fixed three outer folds require at least 31 groups because
  the first half of groups must provide the 16-row primary training minimum;
  therefore the runner correctly returned `insufficient_development_data`
  before scoring any candidate.
- No policy, artifact, model permission, database mutation or order was
  produced. This is insufficient coverage, not a candidate rejection and not
  an independent exam result. G2 remains **BLOCKED** and G3-G6 remain
  pending.
- The 12-hour cutoff is still too short for the fixed fold protocol despite
  the 33-group created-time coverage snapshot. Development eligibility is
  bounded by `updated_at`, while the coverage diagnostic is grouped by
  `created_at`; older completed rows therefore fall outside the development
  window. The next run must use a newly preregistered longer development
  window, not reuse this cutoff or lower the fold minimum.

### Longer Development Window Registration (2026-10-24 23:35 Asia/Shanghai)

- New development-only identity:
  `horizon-current-v1-20261024T153500Z-24h`; cutoff-exclusive UTC
  `2026-10-24T15:35:00Z`; lookback `24` hours; maximum `1,000` decision
  groups. The source filter, model family, alpha/threshold grid, purged
  chronological folds, minimum `16` selected observations and all
  fee-after acceptance criteria are unchanged. The longer window is
  preregistered solely to obtain enough current `updated_at` coverage for the
  fixed protocol; it is not an exam and cannot authorize trading.

### Updated-Time Coverage Diagnosis And Bounded Re-registration (2026-10-24 23:45 Asia/Shanghai)

- A separate read-only `updated_at` projection completed at
  `2026-10-24T15:44:53Z`. Across the full 24-hour scope it found 54 mature
  groups with all four native fee-after-complete horizons. This confirms the
  data stream contains enough common groups for the fixed folds.
- The 24-hour run with `decision_group_limit=1,000` still saw only 4 common
  groups because its selected-group query orders `max(updated_at)` ascending
  before applying the limit. The oldest eligible groups, including stale
  quarantined/rebuilt activity, consume the bound and exclude newer complete
  groups. This is a bounded research-input selection defect, not a reason to
  lower entry gates or accept quarantined labels.
- New preregistered identity:
  `horizon-current-v1-20261024T154700Z-24h-2500`; cutoff-exclusive UTC
  `2026-10-24T15:47:00Z`; 24-hour lookback; maximum `2,500` decision groups.
  Candidate family, fold rules, costs, and acceptance criteria remain
  unchanged. The run is development-only and cannot persist an artifact,
  authorize model influence, mutate the database, or submit orders.

### Horizon Development Result With Expanded Group Bound (2026-10-24 23:49 Asia/Shanghai)

- The run `horizon-current-v1-20261024T154700Z-24h-2500` completed at
  `2026-10-24T15:48:58Z`. It read 2,392 native fee-after rows across 1,377
  decision groups, with 56 common four-horizon groups. The result remained
  `insufficient_development_data` before candidate scoring.
- A bounded fold-coverage projection on the exact frozen frame explains why:
  the first outer validation boundary starts at `2026-10-24T11:08:48Z`, but
  the 240-minute purge leaves only 4 primary training groups before that
  boundary. All three outer folds therefore fail the fixed minimum of 16
  primary training observations. The 56 groups are concentrated in roughly
  `2026-10-24T01:18:22Z` through `2026-10-24T11:46:22Z`; group count alone
  does not replace chronological separation.
- No candidate was scored, no return metric was produced, and no policy,
  artifact, permission, database row or order was changed. This is
  insufficient chronological coverage, not a profitability pass or a
  candidate rejection. G2 remains **BLOCKED**; G3-G6 remain pending.
- The next exact action is to continue collecting newly created native
  fee-after-complete groups across a longer time span until the fixed
  240-minute purged folds have at least 16 primary training groups. Do not
  shorten the purge, reuse this window as an exam, or lower the entry gates.

### New Development Registration After Current Coverage (2026-10-25 00:50 Asia/Shanghai)

- A fresh bounded entry/coverage check at `2026-10-24T16:46:35Z` still showed
  `398/398` paper market decisions as `hold`, with `17` shadow-training
  selections and `13` no-candidate decisions in the latest 30. No entry
  permission was granted. The direct blockers remain non-positive expected net
  return, missing quantitative evidence, and excessive observation-loss
  probability; the authoritative local-ML evidence remains `n=267`,
  mean fee-after return `-0.303832%`, lower bound `-0.431590%`, and PF
  `0.549713`.
- The same coverage window contains `85` mature four-horizon groups with
  native executable 1-minute paths and `fee_after_complete=true`. This is
  sufficient to attempt a development-only nested run, but it is not an
  independent exam and cannot authorize execution.
- Preregistered development identity:
  `horizon-current-v1-20261024T165000Z-24h-2500`; cutoff-exclusive UTC
  `2026-10-24T16:50:00Z`; 24-hour lookback; maximum `2,500` decision groups.
  Inputs are restricted to completed paper rows from the native executable
  1-minute path with `fee_after_complete=true`. The candidate family, Ridge
  alpha grid, net thresholds, purged chronological folds, minimum selected
  rows, fee-after cost model, and positive mean/lower-bound/PF requirements
  remain unchanged. No exam labels, artifact persistence, permission change,
  database mutation, order, or service restart is allowed.

### Development Result And Cache-Path Repair (2026-10-25 01:08 Asia/Shanghai)

- The registered development run completed at
  `2026-10-24T16:53:34Z` with `2,585` native fee-after rows across `1,438`
  decision groups and `88` common four-horizon groups. It returned
  `insufficient_development_data` before candidate scoring: the fixed
  chronological/purged folds still cannot supply the required primary
  training coverage after the 240-minute label boundary. No candidate,
  artifact, permission, database mutation, exam evaluation, or order was
  produced. Report:
  `data/profitability_remediation/online-horizon-development-v1-20261024T165000Z.json`.
- The online source hash before deployment differed from the workspace only
  by the uncommitted cache-path guard. The 2-second stale-WebSocket comparison
  fix was already online; it was not the missing deployment. The new guard
  was verified locally with `111` focused tests, Ruff, and compileall, uploaded
  alone, remotely compiled and tested (`110` existing remote tests passed), then
  `bb-paper-trading.service` was restarted once. Remote service state after
  restart was `active/running`, `NRestarts=0`, paper mode unchanged.
- A post-restart read-only window through `2026-10-24T17:08:33Z` observed
  `18` market decisions, all `hold`: `8` shadow-training, `8` no-candidate,
  and `2` insufficient-evidence. The model/economics permission remained
  false with authoritative local-ML `n=267`, mean fee-after return
  `-0.303832%`, lower bound `-0.431590%`, and PF `0.549713`. This verifies
  the service is running and the guard is active; it does not establish
  profitability.
- The cache repair is a data-integrity correction, not a strategy relaxation:
  cache-only shadow settlement now requires a non-empty native 1-minute path,
  a current clean source-consistency contract, and a clean market fact before
  short-circuiting the authoritative refresh. Fresh one-hour quarantine
  samples still include missing/incomplete native paths, so G2 remains
  **BLOCKED**. G1 remains passed only as negative attribution; G3 remains
  blocked by chronological coverage, and G4-G6 remain pending.
- Exact remaining work: continue collecting clean native groups across a
  wider time span, fix only reproducible path-collection defects, then run a
  new development search. Only a candidate with positive fee-after mean,
  lower bound, PF and fold stability can proceed to a frozen independent exam
  and governed paper fills. Do not lower gates or count shadow selections as
  trades.

### Post-Deployment Path Observation (2026-10-25 01:17 Asia/Shanghai)

- The first bounded one-hour post-deployment quarantine projection completed
  at `2026-10-24T17:17:29Z`. It observed `27` qualified 5-minute rows and
  `26` qualified 15-minute rows, while newer 60/240-minute rows were still
  pending. Remaining native invalidity was still present: 5-minute rows had
  `15` `recent_one_minute_price_path_missing`/quote-reconciliation groups
  and `15` incomplete-path groups; 15-minute rows had the same `15/15`
  pattern. This confirms the cache guard is active but does not yet repair
  the underlying provider/path availability problem.
- No historical quarantined row was requalified. G2 therefore remains
  **BLOCKED**, and the negative fee-after model evidence continues to block
  entries independently of data collection. The next repair must instrument
  the native fetch outcome (empty response, timeout, cursor-window miss, or
  path continuity gap) on newly created decision IDs before changing any
  collection behavior.

### Fresh Online Status Recheck (2026-10-25 01:30 Asia/Shanghai)

- A new bounded market-hold economics projection covered 500 paper market
  decisions from `2026-10-24T13:42:44Z` through `2026-10-24T17:27:27Z`.
  All 500 final actions were `hold`; there were no executed entries or open
  positions. Of these, 460 analyses completed with four experts and 40 were
  `insufficient_evidence` because the shared Qwen3.8-27B batch circuit breaker
  deferred the expert call after a prior incomplete response.
- The completed rows are not blocked by a scheduler or exchange rejection.
  They are rejected by the economics/quality contract: the local-ML quality
  permission was false for all 500 projections because authoritative
  fee-after evidence remains `n=267`, mean `-0.303832%`, lower bound
  `-0.431590%`, and profit factor `0.549713`. The required break-even
  contract is positive mean, positive lower bound, and PF above one. Server
  profit and time-series sources remain unavailable, so they cannot provide an
  independent qualifying family.
- A direct compact projection of decision `693914` shows the exact current
  veto: long expected net `-0.1011786%` and short expected net
  `-0.1585154%`, with no quantitative evidence family accepted; entry
  permission is `false` with `no_direction`. This is the concrete reason the
  latest online scan did not open a position.
- Native data collection has improved but is not yet a clean research feed.
  The six-hour quarantine projection found 65 mature groups with all four
  native fee-after horizons qualified, while many other rows remain
  quarantined for historical-only availability, quote-source
  non-reconciliation, missing/incomplete one-minute paths, or observed prices
  outside the captured path. The 24-hour coverage projection found 107 fully
  qualified four-horizon groups; this is development evidence only and does
  not authorize trading.
- Current gates remain G0 passed, G1 passed only as negative attribution, G2
  **BLOCKED**, and G3-G6 pending. The remaining work is therefore two
  separate tracks: (1) obtain a stable positive fee-after candidate through
  preregistered nested development on newly qualified native groups, and
  (2) independently close the remaining provenance/quarantine and expert
  batch reliability defects. No quality gate, risk limit, leverage, or live
  trading setting was relaxed.
- Next exact action: finish one bounded current-window development diagnostic,
  then preregister any subsequent run's immutable cutoff and inputs before
  execution. In parallel, trace the expert batch circuit-breaker timeout on
  fresh decision IDs; do not count deferred expert rows as trades or use them
  to justify an entry.

### Nested Development Result At Fresh Cutoff (2026-10-25 01:32 Asia/Shanghai)

- A bounded 24-hour native development diagnostic completed at
  `2026-10-24T17:31:22Z` with cutoff-exclusive
  `2026-10-24T17:28:57.682776Z`, 2,702 fee-after-complete rows across 1,484
  decision groups, input SHA256
  `5678f8528412eb87e39c0768ffedb130cd4ca5d226669a72bdd6d4758d22dd93`,
  and source-frame SHA256
  `1e7079769615d7de9ed5506ac17b63f85b34adee0c65ba68795bd8412a2eb4e5`.
  Report: `data/profitability_remediation/online-nested-development-v1-20261024T172857Z.json`.
  Its cutoff was generated at execution rather than recorded as a distinct
  preregistration entry; keep this run development/diagnostic only and do not
  treat it as G3 acceptance evidence. All three chronological outer folds and
  their inner folds were estimable with zero group overlap and purged label
  boundaries.
- No policy qualified in any fold. The aggregate opportunity count was 164,
  but selected count was zero because every candidate failed the inner
  positive fee-after mean/lower-bound/PF requirements. Representative inner
  results remained negative (for example fold 0 mean `-0.1070%`, PF
  `0.3629`; fold 1 best mean `0.0618%` still had lower bound `-0.1192%`;
  fold 2 best mean `0.1740%` still had lower bound `-0.1880%`). No artifact,
  promotion, permission change, database mutation or order was produced.
- This is a completed negative development diagnostic, not an
  insufficient-data result and not an independent exam. G3 therefore remains
  not passed; G4-G6 cannot start until a new preregistered candidate passes
  development and the remaining G2 provenance requirements are satisfied.
  Any future run must preregister a new cutoff and may not tune on this result.

### Live Service And Collection Reliability Recheck (2026-10-25 01:44 Asia/Shanghai)

- A fresh 30-minute online projection through `2026-10-24T17:41:20Z`
  still showed only `hold` market decisions and zero executed entries/open
  positions. `bb-paper-trading.service` is `active/running` with
  `NRestarts=0`; dashboard is also active with zero restarts. This rules out a
  stopped service as the current entry cause.
- The latest retained-log tail spans `2026-10-24T12:00:34Z` through
  `2026-10-24T17:44:40Z`. It contains 48 explicit native-path
  `TimeoutError` events, 14 shadow path-batch timeout events and 47 Qwen
  batch-circuit-breaker warnings. A newer clean tail from
  `2026-10-24T17:34:36Z` through `17:41:54Z` contains 79 successful native
  one-minute requests, no native errors, zero queue wait, provider-duration
  p95 `3.91s` and maximum `4.94s`. This shows the capacity repair is working
  in the latest minutes but has not erased intermittent failures across the
  retained window; deferred expert rows remain fail-closed and are not counted
  as trades.
- No code or runtime policy was changed by this recheck. The remaining
  blockers are still the negative fee-after economics, incomplete provenance
  on part of the native label stream, and intermittent model/path reliability;
  none justifies lowering the entry or risk contracts.
- The validated `services/data_service.py` hash matches the remote application
  exactly: `918613a008a88ffbb146853dc4ae3a0a470a3c9a6c3d956dad0b401f0cb169a6`.
  The owned source, regression tests and this checkpoint are committed locally.
  Two `git push origin main` attempts failed because the connection to
  `github.com:443` failed/reset; do not redeploy for this transport failure.
  Next Git action is to verify `origin/main` and retry the push when network
  access is available.

### Current Entry And Profitability Recheck (2026-10-25 02:00 Asia/Shanghai)

- The local branch and `origin/main` are both at `4fab404`; the worktree is
  clean. The current goal remains active. This turn made no online mutations,
  orders, position changes, or service restarts.
- A bounded read-only online paper audit (`--limit 200 --minutes 180`) covered
  decision IDs `693797-693996`, from `2026-10-24T16:29:56Z` through
  `2026-10-24T17:57:03Z`. All 200 final actions were `hold`. Four experts
  completed on 195 rows; five rows had a shared Qwen batch failure or its
  circuit-breaker deferral. In this snapshot the expert statuses were one
  batch failure and four deferred rows. The latest 395-decision projection,
  generated at `2026-10-24T17:57:17Z`, also showed all actions `hold` and
  `was_executed=false`; its latest 200 funnel was 114 `no_candidate`, 83
  `shadow_training`, and 3 `insufficient_evidence`.
- The dominant execution veto is still return quality, not an exchange/order
  rejection. Authoritative local-ML evidence is `n=267`, mean fee-after return
  `-0.303832%`, lower bound `-0.431590%`, and PF `0.549713`; its paper
  permission is false. Server-profit and time-series families have zero
  qualifying samples. For both sides, the latest projection records failures
  of the positive mean, positive lower-bound, and PF-above-one contract.
  Directional/shadow selections do not override that permission.
- The development model is still `shadow` / `learning_only`, with influence
  disabled. Its top-ranked long return is `-0.1274%` with PF `0.2047`; the
  top-ranked short return is `-0.0594%` with PF `0.2853`. This is additional
  negative development evidence, not an execution candidate.
- Data/label collection remains a separate material blocker. In the same
  bounded recovery report, only 192/200 recent sealed facts were clean and
  8 were invalid. The historical requalification dry run scanned 110,709
  rows and found zero eligible rows, primarily because historical label
  availability and path provenance are missing or stale. Five local-AI
  profitability trainers were skipped because
  `authoritative_execution_cost_distribution_unavailable`; local ML had 863
  new samples but was not yet due to train. These counts do not mean the
  scheduler is stopped.
- The nested development run recorded above remains the latest candidate
  search: all folds were estimable, but zero policies passed the positive
  fee-after mean/lower-bound/PF criteria. No frozen candidate exists, so G4
  independent evaluation and G5 governed paper fills have not started.
- Correction to the earlier Qwen hypothesis: production batch prompting uses
  the V3 ultra-compact action vector (`{"a":[...]}`), not four full expert
  JSON objects. The 32-token output cap is therefore not, by itself, proof of
  truncation. Batch failures are currently surfaced as an incomplete-response
  reason without persisted `finish_reason`, output-token count, or response
  length, so the cause is unverified. Do not increase the cap until fresh
  provider telemetry demonstrates `finish_reason=length` or equivalent.
- Focused local model/batch contract tests passed: 44. Current gates remain
  G0 passed, G1 passed only as negative attribution, G2 blocked, G3 not
  passed, and G4-G6 pending. The next work must prioritize a reproducible
  native-path/provenance defect and capture safe batch response diagnostics
  if the Qwen failure recurs; do not promote the negative model, relax quality
  or risk gates, or treat these already inspected rows as an independent exam.

### Bounded Native-Path And Batch-Diagnostic Repair (2026-10-25 02:18 Asia/Shanghai)

- Reproducible collection evidence showed low-priority
  `fetch_native_consistency_ohlcv` calls timing out at the provider's bounded
  eight-second request limit. The isolated native lane already had a separate
  gate, but `_ccxt_call` returned the first `TimeoutError` without retrying.
  This left otherwise recoverable native 1-minute paths empty and caused
  `native_market_fact_invalid` quarantines.
- Narrow repair: low-priority native consistency requests retry once after
  50ms; authoritative/high-priority native requests do not retry and keep
  their fail-closed behavior. No entry threshold, cost, leverage, risk,
  funding, or model permission changed. The repair is bounded to one extra
  read-only provider call and remains within the existing maintenance budget.
- Narrow Qwen observability repair: a target batch parse/incomplete failure now
  includes only bounded provider metadata (`finish_reason`,
  `completion_tokens`, `content_chars`, `reasoning_only`, `truncated`) in the
  error reason. It never records model response content. The 32-token cap was
  not changed because the V3 action-vector contract has not shown proven
  length truncation.
- Local verification: native REST client tests `48` passed; batch/model runtime
  tests `45` passed; Ruff and `git diff --check` passed. No online upload or
  restart has occurred yet for this repair. Next action is targeted upload of
  `data_feed/okx_rest_client.py`, `ai_brain/llm_agent.py`, and their tests,
  remote compile/focused tests, then a controlled paper-service restart and
  fresh path/entry observation.

### Controlled Deployment And Fresh Post-Restart Observation (2026-10-25 02:27 Asia/Shanghai)

- The four owned source/test files from the repair batch were verified on the
  remote application with matching SHA256 hashes. Remote compilation passed
  and the focused suite passed `84` tests. The upload reported zero changed
  files because the same bytes were already present remotely; no unrelated
  source was uploaded.
- Only `bb-paper-trading.service` was restarted at
  `2026-10-24T18:24:56Z` UTC. It returned `active/running` with `NRestarts=0`;
  dashboard was not restarted, the persisted mode remained `paper`, no model
  was configured as `live`, and no order/position/protection mutation occurred.
- A bounded post-restart window through `2026-10-24T18:27:10Z` covered four
  new decisions (`694063-694066`). All four were complete analyses with all
  four experts returned and all four final actions `hold`; there were no
  executed entries. The latest row's direct veto was
  `direction_support_expected_net_not_positive` plus missing quantitative
  evidence; its long raw return was `+0.011260%`, but uncertainty and tail
  penalties produced objective return `-1.391434%`, fee-after expected net
  `-0.114881%`, and loss probability `76.47%`. This is a valid fail-closed
  outcome, not an exchange rejection or scheduler stop.
- Since the restart, the bounded service log contained `14` successful native
  consistency requests and one unrelated high-priority public-index timeout;
  no new Qwen batch-incomplete event occurred in the sampled interval. The
  retry/diagnostic repair is therefore deployed and observable, but it has not
  produced a qualifying profitable candidate.
- Acceptance status is unchanged: G0 passed; G1 passed for negative
  attribution only; G2 remains **BLOCKED** by incomplete native provenance and
  intermittent provider availability; G3 remains not passed because the last
  nested development search selected zero policies; G4-G6 remain pending.
  Next action is to continue collecting clean native groups and run a newly
  preregistered development search only after chronological coverage is
  sufficient. Do not relax the positive net mean/lower-bound/PF or risk gates.

### Preregistered Current Native Development Run (2026-10-25 02:31 Asia/Shanghai)

- Development-only identity: `horizon-current-v1-20261024T183000Z-24h-2500`.
  The immutable cutoff-exclusive UTC time is `2026-10-24T18:30:00Z`;
  lookback is `24` hours and the bounded selected-group limit is `2,500`.
- Inputs are restricted to completed paper shadow labels from the native
  executable 1-minute path with `fee_after_complete=true`. The existing
  candidate family, Ridge alpha grid, net-return thresholds, chronological
  outer/inner folds, 240-minute purge, minimum training/selection coverage,
  cost model and positive mean/lower-bound/PF requirements are unchanged.
- This run is registered only to measure current development coverage and
  candidate stability. It cannot inspect exam labels, persist a policy
  artifact, mutate the database, change model influence or permissions, or
  submit orders. The report must remain development evidence even if a
  candidate appears positive; a later separately preregistered untouched
  sample is required for G4.

### Current Online Cause And Exact Fold Coverage (2026-10-25 02:59 Asia/Shanghai)

- A bounded read-only online economics projection covered decision IDs
  `693919-694118` through `2026-10-24T18:52:24Z` (Beijing
  `2026-10-25 02:52:24`). All 200 final actions were `hold`; 186 analyses
  completed with four experts and 14 were `insufficient_evidence`. There was
  no execution attempt or exchange rejection in this window.
- The current entry veto is reproducible on decision `694118`:
  the selected short direction has fee-after expected return `+0.227500%`,
  but risk-adjusted objective return `-0.615266%` and loss probability
  `46.37%`. The authoritative local-ML permission remains false at
  `n=267`, mean `-0.303832%`, lower bound `-0.431590%`, and profit factor
  `0.549713`; the required positive mean, positive lower bound and PF above
  one are all unmet. The same projection reports 60 side-level quality
  blockers and 120 missing-quality-evidence blockers across 30 fresh rows.
  This is a quality/economics veto, not a scheduler, sizing or OKX order
  failure.
- Added `scripts/audit_online_horizon_fold_coverage.py`, a bounded read-only
  projection that selects only group timing metadata and does not read
  returns or mutate the database. At the registered cutoff
  `2026-10-24T18:30:00Z`, it found 2,872 native fee-after rows and 115
  complete four-horizon groups. Reproducing the production fold rule shows
  58 groups in the initial half, but the 240-minute purge leaves only 4
  primary training groups for each outer fold; the fixed minimum is 16.
  Each fold therefore misses 12 groups and the run correctly returns
  `insufficient_development_data` before candidate scoring.
- This confirms the remaining development blocker is effective chronological
  coverage, not a missing candidate threshold. Do not shorten the purge,
  lower the 16-group minimum, or reuse this inspected window as an exam.
  Continue collecting clean native groups across a wider time span, then
  preregister a new cutoff and rerun development only after all outer folds
  meet the fixed coverage rule. G2 remains **BLOCKED**, G3 is not passed, and
  G4-G6 remain pending.

### Preregistered Fixed-Primary Nested Development (2026-10-25 03:12 Asia/Shanghai)

- Development-only identity:
  `nested-current-v1-20261024T191034Z-24h-2500`; cutoff-exclusive UTC
  `2026-10-24T19:10:34Z`, 24-hour lookback and at most 2,500 decision groups.
- Inputs are completed paper shadow labels from the native executable 1-minute
  path with `fee_after_complete=true`. The existing fixed-primary (5-minute)
  Ridge search, alpha values `1/10/100`, net thresholds `0/0.05/0.1%`,
  chronological outer/inner folds, 240-minute primary-label purge, minimum
  16 selected observations, positive fee-after mean, positive lower bound and
  PF-above-one requirements are unchanged.
- This run is diagnostic/development only. It cannot inspect a sealed exam,
  persist or promote an artifact, mutate the database, change model influence
  or permissions, or submit an order. A positive development result would
  require a separately preregistered untouched exam for G4.

### Fixed-Primary Nested Result (2026-10-25 03:14 Asia/Shanghai)

- The preregistered run completed at `2026-10-24T19:13:48Z` with 2,984
  native fee-after rows, 1,611 decision groups and 745 primary 5-minute
  groups. All three outer folds and all six inner folds were estimable:
  outer primary training counts were `180/239/303`, with validation counts
  `61/61/61`; group overlap was zero and label periods were disjoint.
- No policy qualified in any outer fold (`selected_policy_fold_count=0`).
  Fold 0's strongest small slice (alpha `100`, threshold `0.05%`) had
  selected count `7`, mean `+0.1690%`, PF `3.3914`, but lower bound
  `-0.1237%` and therefore failed the minimum-16 and positive-LCB rules.
  Fold 1's best positive slice (alpha `10`, threshold `0.1%`) had count
  `19`, mean `+0.0806%`, PF `1.6184`, but lower bound `-0.0990%`.
  Fold 2's best positive slice (alpha `100`, threshold `0.1%`) had count
  `4`, mean `+0.0769%`, PF `1.4329`, but lower bound `-0.4758%`.
  These are development diagnostics, not deployable or independent
  profitability evidence.
- The aggregate outer evaluation selected zero observations and produced no
  return estimate. No artifact was persisted, no permission or runtime policy
  changed, and no order was submitted. This closes the current
  `insufficient_development_data` explanation for the fixed-primary protocol:
  with adequate chronological coverage, the candidate family still fails the
  preregistered fee-after stability gates. G3 remains **NOT PASSED**; G4-G6
  cannot start.

### Latest Data Quarantine And Entry Funnel (2026-10-25 03:15 Asia/Shanghai)

- A bounded rolling 24-hour coverage projection completed at
  `2026-10-24T19:07:13Z`. Within the query's 20,000-row cap it found 2,665
  mature groups with all four horizons observed; 119 groups qualified at all
  four horizons as completed native-path, fee-after-complete labels. These
  counts are an audit snapshot, not the full lifetime database.
- A more detailed 4,000-row quarantine projection covered 1,000 recent
  decision groups. Of 466 mature groups with all four horizons observed, 58
  were fully qualified. Dominant reasons included
  `historical_label_availability_missing`,
  `historical_label_availability_basis_unverified`,
  `historical_market_path_not_clean`, and
  `native_market_fact_invalid`. In the sampled native-invalid rows, recurring
  reasons were non-reconciled executable quote sources, observed prices
  outside the captured native path, and missing/incomplete 1-minute paths.
  Reason counts overlap because one row may carry multiple quarantine codes.
- The same 4,000-row reason projection was attempted at a 20,000-row limit but
  its JSON-heavy SQL projection hit the configured 10-second statement
  timeout. The bounded 4,000-row result is the accepted current sample; do not
  interpret the timed-out query as a zero count or a complete census.
- The fresh market-entry projection at `2026-10-24T18:55:49Z` covered the
  latest 30 decisions (`694096-694125`): all were `hold`, all 30 had four
  experts complete, and there were zero execution attempts. The funnel was
  22 `no_candidate` and 8 `shadow_training`. Nine long and five short
  directions had positive fee-after expected mean, but all 30 sides had
  nonpositive lower bounds; the fixed authoritative ML permission still fails
  mean, lower-bound and PF checks. Therefore none was an approved candidate.
- The previously registered native-path timeout retry is deployed and the
  low-priority request lane remains bounded. The current quarantine sample
  does not prove that every path failure is a provider timeout; quote
  disagreement, price reachability and historical timing failures are
  separate causes. Do not increase candle windows or retry authoritative
  requests without fresh evidence that isolates a recoverable timeout.
- This turn changed only the local read-only audit helper, its contract test
  and this plan. A staged `--skip-restart` sync matched no files because those
  operator-only audit paths are outside the application upload allowlist;
  nothing was uploaded and no service was restarted.
- Current stage status: G0 passed; G1 passed as negative attribution only;
  G2 **BLOCKED** by historical timing/provenance and incomplete native
  executable paths; G3 **NOT PASSED** because the adequately covered
  fixed-primary nested search selected zero policies; G4, G5 and G6 remain
  pending. No profitability guarantee or completion claim is supported.
- Next work: keep the paper path and live-trading shutdown unchanged; collect
  new native fee-after labels, quarantine only reproducible source defects,
  and preregister a new development protocol before exploring a materially
  different model family. A candidate must pass the existing positive mean,
  positive lower-bound, PF-above-one, sample, tail and cost-stress rules before
  any untouched G4 exam or governed paper round trip can begin.

### Current User Status Recheck (2026-10-25 03:25 Asia/Shanghai)

- Fresh bounded online checks covered the six hours ending
  `2026-10-24T19:25:31Z`. The paper and dashboard systemd units are active;
  the persisted trading control is `paper`, `paused=false`. No
  `bb-live-trading.service` unit appeared in the systemd inventory. This
  confirms analysis is running, but does not authorize live trading.
- All 811 paper market decisions in that six-hour aggregate were `hold` and
  `was_executed=false`. In the newest bounded 200-row analysis, the funnel
  was 127 `no_candidate`, 62 `shadow_training`, and 11
  `insufficient_evidence`; all 200 final actions were `hold`. No exchange
  order rejection is present because no order was attempted.
- The immediate entry cause is eligibility, not a stopped scheduler or
  exchange failure. The local-ML authoritative permission is false at 267
  realized samples: mean fee-after return `-0.303832%`, lower confidence
  bound `-0.431590%`, and profit factor `0.549713`. Each misses its
  preregistered requirement. The latest 200 rows also contain seven missing
  index-price facts and six ambiguous/missing index-price receipts/clocks;
  186 rows pass the feature-availability check and 14 do not. Several
  directional selections are explicitly `paper_training_entry` with
  `entry_permission.granted=false`: they are observations, not order
  intents.
- Collection has improved but has not qualified the candidate. The local-ML
  trainer reported 1,075 new samples and `not_due`; six local-AI trainers
  remain skipped for `authoritative_execution_cost_distribution_unavailable`.
  A 48-hour group projection now contains 120 decision groups with all four
  horizons independently qualified on native fee-after facts, versus 119 in
  the prior checkpoint. The same window includes 4,425 mature groups with all
  four horizons observed but none qualified at all four horizons in that
  population; historical-label rebuild dry-run remains 0 eligible of 110,709
  scanned because the original timing/path lineage is absent or stale.
- The last adequately covered nested development search still selected zero
  policies. No frozen candidate or independent exam exists, and no governed
  paper round trip has passed. Therefore the request is not "one switch away":
  G2 is blocked, G3 has no qualified candidate, and G4-G6 remain
  unstarted/pending. A positive profit guarantee is not supported by these
  results. Current mode is paper, so real-money profit cannot accrue in this
  state; paper entries are also correctly withheld because no candidate
  currently meets the existing fee-after and risk contracts.
- This recheck made no application-code changes, uploads, service restarts,
  orders, position changes, or risk/quality threshold changes. The
  analysis-coverage helper failed to decode a remote JSON report, so this
  checkpoint does not claim a fresh full-window service-continuity or market
  coverage assessment from that helper.
- Next action remains G2/G3: repair only reproducible decision-time feature
  receipt defects; preserve new clean native fee-after observations; then
  preregister a materially different development candidate family, including
  fixed cutoff, feature set, cost model, purged folds and compute budget.
  Do not rerun the already inspected nested Ridge search as though it were
  new evidence, and do not open G4 until a development candidate first passes.

### Fresh Entry-Blocker Recheck (2026-10-25 03:45 Asia/Shanghai)

- A new bounded read-only market-hold economics projection covered decision
  IDs `694174-694246`, from `2026-10-24T19:16:03Z` through
  `2026-10-24T19:45:48Z` (Beijing `2026-10-25 03:16-03:45`). All 73 final
  actions were `hold`: 46 `no_candidate`, 27 `shadow_training`, and zero
  execution attempts. All 73 analyses were complete and decision-eligible;
  all four experts completed on every row. This rules out a stopped analysis
  loop, incomplete model responses, and exchange order rejection as the
  immediate cause of zero entries in this sample.
- The actual veto remains profitability evidence. The authoritative local-ML
  permission is false at 267 settled samples: fee-after mean `-0.303832%`,
  lower confidence bound `-0.431590%`, and profit factor `0.549713`. Each
  misses the fixed positive-mean, positive-lower-bound, PF-above-one contract.
  Across both sides of all 73 rows, none of the 146 side observations had a
  positive return lower bound and none was production-eligible. Some
  individual directions had positive point estimates (23 long-side and 16
  short-side rows), but their lower bounds remained negative; point estimates
  are not sufficient evidence to authorize entries.
- A separate 30-minute service/coverage projection generated at
  `2026-10-24T19:44:44Z` found paper, dashboard and model-tunnel services
  active, control mode `paper` and `paused=false`, 72 market analyses, no
  model failures, and market coverage ready. It found zero directional
  signals, zero executed entries and no open positions. The two projections
  use slightly different cutoffs; both independently confirm healthy
  analysis and no entry/order activity.
- This fresh check identifies no new scheduler, queue-priority or execution
  transport defect. The previously repaired native-evidence priority
  isolation is not the current entry veto. It does not resolve the separate
  G2 historical timing/path provenance blocker, and the already-inspected
  nested Ridge candidate family still fails G3. No threshold, risk, leverage,
  model permission or trading mode was changed; no order, position,
  protection, restart or application source was modified.
- Gate count: G0 and the G1 attribution audit are passed (G1's measured
  economics remain negative); G2 is blocked; G3 has no qualifying policy;
  G4 has no frozen candidate or untouched exam; G5 has no authorized
  controlled paper round trip; G6 is pending. Thus five of six gates are
  not fully complete for the overall profitability objective. This is not a
  guarantee that an eventual profitable policy exists, and paper mode cannot
  generate real-money profit.
- Next action remains G2/G3: preserve fresh, fully sourced native fee-after
  groups, repair only reproducible point-in-time data defects, and preregister
  a materially different development family with its fixed features,
  chronological/purged evaluation, cost model, and compute budget. Keep the
  existing quality and risk gates fixed. Start G4 only if development
  evidence first passes; otherwise retain the failed result and continue
  research without promoting or forcing trades.

### Entry Contract Confirmation (2026-10-25 03:49 Asia/Shanghai)

- A second bounded read-only entry-contract report through
  `2026-10-24T19:49:29Z` confirmed `can_open_new_entries=true`,
  `paper_entry_requires_historical_fee_after_quality_permission=true`,
  zero contract violations and zero executed entries in the latest
  30-minute window. The paper position list is empty.
- The latest decisions (`694226-694255`) remain `hold`; no order was created
  in that window. Older rejected order rows (decision IDs around `688243-688666`,
  2026-10-22 to 2026-10-23) are historical records and are not the cause of
  the current no-entry interval. Do not conflate those past exchange outcomes
  with today's pre-entry quality veto.

### Fresh Native Quarantine Recheck (2026-10-25 03:51 Asia/Shanghai)

- A bounded 24-hour quarantine projection (`row_limit=4000`) completed at
  `2026-10-24T19:51:15Z` without database mutation, exam-label inspection or
  order submission. It covered 1,000 decision groups; `453` groups were
  mature at all four horizons and `44` were qualified at all four. Per horizon,
  qualified rows were `391/384/326/165` for 5/15/60/240 minutes; quarantined
  rows were `584/571/518/284`. This is a rolling sample, not a lifetime
  census.
- The dominant remaining G2 defects are still provenance, not model
  scheduling: historical label availability/basis is unverified and the
  historical market path is not clean. Among fresh native-invalid rows, the
  recurring executable-path failures are `recent_one_minute_price_path_missing`,
  `observed_price_outside_recent_native_path` and
  `executable_quote_sources_not_reconciled`; several rows also have missing or
  ambiguous index-price receipts/clocks. A representative recent row had
  `bar_count=0` and failed one-minute-path verification even though native
  identity and reference-price checks passed.
- This explains why collection is progressing but the clean four-horizon
  development pool remains small. The quarantines are correctly fail-closed;
  treating them as valid labels would fabricate training evidence and could
  make the system trade on unverifiable costs/prices. The next repair must
  target the native one-minute path/quote-reconciliation lineage and original
  historical availability basis, not relax quarantine rules.

### Current User Blocker Recheck And Request-Lane Repair (2026-10-25 04:20 Asia/Shanghai)

- A fresh read-only `audit_online_market_hold_economics.py --limit 100
  --minutes 30` projection covered decision IDs `694259-694327` through
  `2026-10-24T20:20:21Z` (Beijing `2026-10-25 04:20:21`). All 69 decisions
  were `hold`; all 69 analyses completed with all four experts. The funnel
  contained 40 `no_candidate` and 29 `shadow_training`; this sample had no
  production-eligible entry.
- Across all 138 side observations, none had a positive return lower bound.
  The authoritative local-ML permission remains false at 267 settled samples:
  fee-after mean `-0.303832%`, lower bound `-0.431590%`, and profit factor
  `0.549713`. All three fixed requirements fail. There are occasional positive
  fee-after point estimates, but the lower bounds remain negative and the
  production permission is false; those estimates cannot authorize an order.
  The latest hold is therefore still a pre-entry quality/economics veto, not a
  scheduler stop or exchange rejection.
- The separate G2 blocker is data quality/coverage: the latest bounded
  quarantine sample remains 1,000 groups with only 44 fully qualified across
  all four horizons. Recurrent exclusions are missing/unverified historical
  label availability, missing or incomplete native one-minute paths, and
  unreconciled executable quote sources. The fixed-primary nested Ridge
  development run had adequate fold coverage but selected zero policies.
  Thus better collection does not itself create a profitable candidate.
- A local QoS repair was completed in the request client: background and
  final authoritative ticker/orderbook/reference-price refreshes now use
  separate single-flight keys and explicit queue priority, so a slow
  background request cannot make the final validation join that lower-priority
  task. This is a collection-latency improvement only; it does not loosen
  entry gates or solve negative expectancy. Four touched source/test files
  pass 162 focused OKX/DataService tests plus 40 market-receipt/feature
  availability tests; Ruff, compileall and `git diff --check` pass. The first
  local run exposed six stale expectations/compatibility defects; those were
  corrected and the final focused reruns passed.
- The QoS patch is uploaded to `/data/bb/app` without restart. Remote hashes
  match the four touched source/test files, and the same focused remote suite
  passed `203` tests. The current paper process was intentionally not
  restarted: the read-only paper-resume preflight at `2026-10-24T20:32:23Z`
  reports the paper service already active and four Phase 3 quant child
  endpoints unavailable, so restarting would violate the preflight contract
  and would not improve entry eligibility. The preflight also confirms
  paper mode, live routing disabled, no local/OKX positions or orders needing
  repair, and no protection mutation. Commit `89ed1b3` was pushed to
  `origin/main`; a concise project-memory retention was accepted by the `bb`
  Hindsight bank. The service process has not loaded the uploaded code because
  it has not been restarted.
- Gate status is unchanged: G0 passed; G1 attribution audit passed with
  negative measured economics; G2 **BLOCKED**; G3 **NOT PASSED**; G4, G5 and
  G6 pending. Five stage gates remain incomplete for the profitability
  objective. No claim of profitability or positive-profit guarantee is
  supported.
- Next actions: keep the existing paper service and live shutdown state
  unchanged while the four Phase 3 quant child endpoints are unavailable. Once
  the preflight blockers clear, verify paper-only runtime state and then
  consider a paper-only restart to load the verified request-lane patch.
  Separately trace and repair only reproducible native path/quote receipt
  defects, keep the negative Ridge family rejected, and register a materially
  new development protocol only after enough clean chronological coverage
  exists.

### Current User Root-Cause Recheck (2026-10-25 04:53 Asia/Shanghai)

- Fresh bounded read-only market audit completed at `2026-10-24T20:52:29Z`
  (Beijing `2026-10-25 04:52:29`) over 100 decisions (`694306-694405`).
  All 100 analyses completed with all four experts, all final actions were
  `hold`, and there were zero order attempts. The funnel was 58
  `no_candidate` and 42 `shadow_training`; no exchange rejection explains the
  no-entry interval.
- Each of the 100 long-side and 100 short-side return lower bounds was
  non-positive: the best observed long lower bound was `-0.525998%` and the
  best short lower bound was `-0.170804%`. Positive point estimates existed
  for 21 long and 31 short observations, but none had a positive lower bound.
  The governing local-ML permission remains false at 267 authoritative
  settled samples: fee-after mean `-0.303832%`, lower bound `-0.431590%`,
  profit factor `0.549713`. Therefore entry is withheld by the profitability
  contract, not a stopped scheduler, missing expert response, or exchange
  rejection.
- Bounded shadow-pipeline audit at `2026-10-24T20:52:46Z` confirmed the paper
  runtime was running, unpaused, and heartbeating at `20:52:41Z`; the last
  six-hour decision projection had no non-hold actions. Shadow settlement was
  active: 47 rows completed and 23 quarantined in the latest five minutes.
  The rolling six-hour aggregate held 752 native fee-after decision groups
  across four horizons. These are training/label observations, not executed
  trades or proof of profitability.
- The latest bounded 24-hour quarantine sample (`row_limit=4000`) covered
  1,000 groups: 445 were mature at all four horizons and 24 qualified at all
  four. Per-horizon native-market invalid reasons included missing one-minute
  paths (203/200/186/99), incomplete paths (68/68/63/41), and unreconciled
  executable quotes (210/207/190/102) for 5/15/60/240 minutes. Historical
  rebuilt rows separately remain quarantined for missing/unverified
  availability basis; they must not be promoted into executable training
  evidence. These samples are bounded projections, not lifetime censuses.
- Read-only paper-resume preflight at `2026-10-24T20:49:17Z` reported the paper
  service already active and four Phase 3 quant child endpoints unavailable.
  The model server and quant API parent were reachable, but those child
  contracts had no ready registered artifact. Live routing remained disabled,
  and the authoritative account sync showed no open positions or orders to
  repair. This preflight is a release/restart gate, not the direct veto in the
  100-decision entry sample; do not restart the running paper service or
  reinterpret parent liveness as child-model readiness.
- Remaining acceptance state: G0 passed; G1 passed only as a negative loss
  attribution audit; G2 remains blocked; G3 remains not passed because the
  adequately covered fixed-primary nested Ridge search selected zero
  policies; G4 has no frozen candidate or untouched exam; G5 has no governed
  profitable paper round trip; G6 is pending. Thus five of the six
  profitability gates G1-G6 remain incomplete or failed. Real-money profit
  cannot accrue while execution mode is paper, and no result supports a
  guarantee of future profit.
- Next exact action: instrument and directly probe the bounded native 1-minute
  endpoint on the online runtime, distinguishing timeout/error, empty response,
  cursor/window coverage miss, and cache-only decision timing without storing
  raw quotes. Only after a reproducible subcause is isolated, repair it and
  collect a fresh boundary. Then preregister a materially different
  cost-aware development family on eligible native labels. Keep all entry,
  quality, cost, leverage, and risk gates unchanged; do not open G4 until a
  candidate passes G3. No application source, deployment, service restart,
  order, position, protection, or account mutation occurred in this recheck.

### Native Route Probe And Current Entry Recheck (2026-10-25 05:22 Asia/Shanghai)

- The read-only probe was completed after bounding the sample lookup to the
  latest 4,000 IDs and bounding each OKX native 1-minute request to 12
  seconds. Local focused tests for the probe contract passed `4`; Ruff and
  compileall passed. The probe never writes the database, submits orders, or
  changes positions/protection.
- Eight newest quarantined native rows were checked against the complete
  entry-to-result window, using the stored result timestamp when available.
  All eight returned `complete_window`; observed coverage was 7/7, 7/7, 7/7,
  9/9, 8/8, 8/8, 16/16 and 7/7 bars, with no request timeout or empty
  response. Therefore the current public OKX 1-minute route is reachable for
  these windows. This does not promote the rows: their decision-time
  contracts remain immutable evidence.
- The stored entry-time quarantine reasons are reproducible and distinct from
  the later probe result. Representative reasons were
  `executable_quote_sources_not_reconciled`,
  `recent_one_minute_price_path_missing` or
  `recent_one_minute_price_path_incomplete`, and
  `observed_price_outside_recent_native_path`. A later successful REST fetch
  cannot prove that the original decision-time quotes/path were available or
  executable. The remaining G2 blocker is therefore capture-time provenance
  and source reconciliation, not a blanket current REST outage.
- A fresh bounded market-hold economics audit covered decision IDs
  `694368-694467` through `2026-10-24T21:18:11Z`. All 100 analyses completed
  and all 100 final actions were `hold`; the funnel was 55 `no_candidate` and
  45 `shadow_training`, with zero order attempts. Across both sides, all 200
  observations were not production eligible. Twenty-five long and 36 short
  sides had positive point estimates, but every such side had a non-positive
  lower bound; the latest lower-bound maxima remained negative
  (long `-0.666982%`, short `-0.170804%`). Quality blockers remained
  `average_fee_after_return_not_positive`,
  `fee_after_return_lcb_not_positive`,
  `fee_after_profit_factor_not_above_break_even`, and missing fee-after
  quality evidence. The direct no-entry cause is still the fixed
  profitability permission, not scheduler inactivity or exchange rejection.
- Runtime remained paper-only, running and unpaused, with no executed entries.
  Shadow settlement continued to produce completed and quarantined rows. The
  compact audit output was reduced to bounded runtime/economics/probe fields
  so future checks do not dump raw market diagnostics.
- Acceptance is unchanged: G0 passed; G1 is passed only as negative loss
  attribution; G2 is blocked by decision-time provenance/reconciliation;
  G3 has no qualifying policy; G4 has no frozen candidate or untouched exam;
  G5 has no authorized governed paper round trip; G6 is pending. Five
  profitability gates remain incomplete or failed. No code change has altered
  entry thresholds, model permission, risk, leverage, execution mode, or
  exchange behavior.
- Next exact action: trace the capture-time source-consistency path from the
  market-analysis feature snapshot into the shadow row, and add a focused
  regression only if the missing-path/quote-reconciliation defect is
  reproducible in code. Preserve all quarantined rows and do not requalify
  them from the later probe. In parallel, keep collecting fresh native
  fee-after groups and preregister a materially different candidate family
  only after enough clean chronological labels exist. G4 remains closed until
  G3 passes.

### Capture-Time Native Refresh Repair (2026-10-25 06:14 Asia/Shanghai)

- The capture-time path was reproducible in the market-analysis flow:
  `_market_feature_fetch_options()` intentionally uses cache-only ticker and
  derivative reads for the broad discovery pool. `_fresh_feature_vector_for_analysis()`
  then accepted a complete priced fallback with a `native_market_fact_invalid`
  observation issue, even when its reason codes were
  `recent_one_minute_price_path_missing`,
  `recent_one_minute_price_path_incomplete`,
  `observed_price_outside_recent_native_path`, or
  `executable_quote_sources_not_reconciled`. The resulting AI/shadow row could
  therefore be persisted without decision-time native-path evidence. A later
  REST probe cannot retroactively establish that the original snapshot was
  executable.
- Repair in `services/trading_service.py`: these four capture-time reason
  families now suppress the observation-only fallback and force one bounded
  authoritative native refresh before AI analysis. The refresh also requests
  the derivative facts needed to rebuild the full native consistency contract.
  Other observation-only quality issues retain the prior bounded fallback
  behavior. Refresh timeout/error still leaves the row isolated; no quality,
  return, risk, leverage, or execution threshold changed.
- Regression coverage added in
  `tests/test_trading_service_boundaries.py` for a missing native path at the
  fallback boundary. Verification: the full boundary module passed
  `365 passed`; the focused refresh subset passed `7 passed`; the focused
  market-data quality module passed `18 passed`; the native ticker/data-service
  subset passed `18 passed`; Ruff, compileall and `git diff --check` passed.
- Targeted source/test upload completed with
  `source-integrity-ok:2` and no service restart. The read-only online
  paper-resume preflight at `2026-10-24T22:13:39Z` still reports
  `bb-paper-trading.service` already active and all four Phase 3 quant child
  endpoints unavailable, so the uploaded process has not loaded this patch;
  restarting is intentionally deferred until the existing preflight contract
  is clear. Paper mode remains active, live routing remains disabled, and
  there are no positions or orders needing repair.
- G2 remains BLOCKED pending fresh post-load capture-time evidence. G1 is
  passed only as a negative attribution audit; G3 has no qualifying candidate;
  G4, G5 and G6 remain pending. The next exact action is to clear/verify the
  existing paper restart preflight, load this patch in paper only, then audit
  a new decision boundary for native-path capture success and fee-after
  eligibility. Existing quarantined rows remain immutable and are not
  retroactively promoted.

### Current Status And Phase 3 Artifact Recheck (2026-10-25 06:28 Asia/Shanghai)

- A fresh bounded online entry-recovery projection completed at
  `2026-10-24T22:21:21Z` (Beijing `2026-10-25 06:21:21`). It covered
  134 paper market decisions in the preceding hour; every final action was
  `hold`, with 69 `no_candidate` and 31 `shadow_training` in the latest
  100-row detail (`694509-694608`). Analysis completed for all sampled rows,
  and no order attempt or exchange rejection occurred.
- The direct authorization veto is unchanged and independently visible in the
  same report: local-ML paper permission is `false` at `n=267`, fee-after
  mean `-0.303832%`, return lower bound `-0.431590%`, and profit factor
  `0.549713`. The blockers are
  `average_fee_after_return_not_positive`,
  `fee_after_return_lcb_not_positive`, and
  `fee_after_profit_factor_not_above_break_even`. Selected shadow-training
  rows therefore remain non-executing observations, not missed orders.
- Read-only paper-resume preflight at `2026-10-24T22:21:22Z` confirms exchange
  truth, account equity, target single-model runtime, Dashboard and tunnel
  services are healthy. It remains blocked by the four Phase 3 child endpoint
  contracts being unavailable and by `bb-paper-trading.service` already being
  active, so it must not be used to restart the process.
- A direct read-only Phase 3 API check explains the child-endpoint blocker:
  `bb-phase3-quant-api.service` is active, but reports
  `artifact_unavailable`, `model_bundle_available=false`,
  `candidate_artifact_not_registered`, and current/challenger/rollback pointers
  rejected for an incompatible historical `profit_supervision_version`.
  Downloaded specialist model files (`10` downloaded, `9` validated) do not
  constitute a registered cost-aware bundle; all four child routes correctly
  remain unavailable.
- A no-persist online local-AI training preflight completed at
  `2026-10-24T22:27:06Z`: `432` shadow samples were readable but the
  authoritative execution-cost sample count was `0`; the trainer returned
  `authoritative_execution_cost_distribution_unavailable` and persisted no
  artifact. This is consistent with the quarantined pre-contract rows and
  must not be “fixed” by promoting the incompatible bundle or by treating
  shadow/price-only labels as fee-after fills.
- Current acceptance remains: G0 passed; G1 passed only as a negative loss
  attribution audit; G2 blocked by decision-time native provenance and
  executable-cost lineage; G3 has no qualifying cost-aware policy; G4 has no
  frozen candidate or untouched exam; G5 has no governed paper round trip;
  G6 is pending. Five profitability gates remain incomplete or failed.
- Next exact action: keep live routing disabled and do not restart the active
  paper process while the Phase 3 artifact/preflight contract is blocked.
  Continue collecting only newly captured, native, fee-after-complete groups;
  inspect the current cost-contract registration path for a reproducible
  defect without writing an unqualified artifact; then preregister a new
  chronological, purged, cost-aware development family only after clean
  coverage exists. Existing quarantined rows and the negative `n=267`
  evidence remain immutable and cannot be reused as a fresh exam.

### Preregistered New Candidate Family: Fixed Random Forest (2026-10-25 06:45 Asia/Shanghai)

- New development-only identity:
  `native-fee-after-tree-v1-20261024T224500Z-24h-2500`; cutoff-exclusive UTC
  `2026-10-24T22:45:00Z`; lookback `24` hours; maximum `2,500` decision
  groups. This is a new candidate family and is not a rerun or retuning of the
  inspected Ridge experiment.
- Inputs are restricted to completed paper rows whose market contract source is
  `shadow_entry_result_and_okx_native_1m_path` and whose label contract has
  `fee_after_complete=true`. The model is a fixed random-forest regression
  over the existing feature keys: `n_estimators=220`, `max_depth=8`,
  `min_samples_leaf=8`, `random_state=42`, single-threaded research worker.
  The only searched entry thresholds are `0`, `0.05%`, and `0.1%` net.
- Selection remains nested and chronological: three purged outer folds,
  two purged inner folds, primary five-minute horizon, decision-group
  separation, 240-minute label-end purge, minimum 16 selected rows, positive
  fee-after mean and lower bound, PF above one, and positive mean in both
  inner folds. No exam labels, artifact persistence, permission change,
  database mutation, order, or service restart is allowed.
- The new source/test/audit code passed the focused seven-test suite, Ruff,
  compileall, and `git diff --check`. It will be uploaded with
  `--skip-restart`; the running paper service is not required for this
  development-only reader.

### Fixed Random-Forest Development Result (2026-10-25 06:53 Asia/Shanghai)

- The preregistered run
  `native-fee-after-tree-v1-20261024T224500Z-24h-2500` completed at
  `2026-10-24T22:53:01Z` with the exact fixed cutoff
  `2026-10-24T22:45:00Z`. It read `3,582` native executable,
  fee-after-complete rows across `1,885` decision groups; horizon counts were
  `899/906/909/868` for `5/15/60/240` minutes. The development input
  fingerprint is
  `12e4e96594d6ac01d51ea21f3a83c4238e0868b342546bcfac026d35149a94ea`.
  Report:
  `data/profitability_remediation/online-nested-development-tree-v4-20261024T224500Z.json`.
- All three outer folds and all six inner folds were estimable with zero
  decision-group overlap and chronological label separation. The fixed tree
  family selected **zero policies in every outer fold** (`0/3`), with
  `selected_count=0` and no aggregate return metric because no inner policy
  met the preregistered positive fee-after mean/lower-bound/PF and fold
  requirements.
- Representative inner evidence was negative: outer fold 0 threshold `0.0%`
  had mean `-0.081478%`, lower bound `-0.191482%`, PF `0.4474`; fold 1
  threshold `0.0%` had mean `-0.121086%`, lower bound `-0.217260%`, PF
  `0.2707`; fold 2 threshold `0.0%` had mean `-0.428333%`, lower bound
  `-0.717095%`, PF `0.0540`. Higher thresholds did not satisfy the same
  gates.
- Safety evidence is explicit: no exam labels were evaluated, no artifact was
  persisted, no model permission or runtime policy changed, no database row
  was mutated, and no order was submitted. This is a completed negative G3
  experiment, not a candidate and not an independent exam.
- G3 therefore remains **NOT PASSED**. G4 cannot open an exam, and G5/G6
  remain pending. The next research action must be a separately preregistered
  candidate family or a proven data/provenance repair; do not retune this tree
  result, promote it, or lower any entry/risk/cost gate.

### Current Entry And Readiness Recheck (2026-10-25 07:07 Asia/Shanghai)

- Read-only online entry recovery completed at `2026-10-24T23:02:47Z` for
  `[2026-10-24T21:32:47Z, 2026-10-24T23:02:47Z)`. There were 198 market
  decisions; all final actions were `hold`. The latest 120 detailed rows
  comprised 76 `no_candidate` and 44 `shadow_training`, with 119 sealed
  feature-availability contracts clean and one missing `index_price` fact.
  All analyses completed; none had an executed order or granted paper-entry
  permission. Shadow-training selections are labels only, not paper fills.
- The direct current veto is unchanged: local-ML paper execution permission is
  false at 267 authoritative settled samples, with fee-after mean
  `-0.303832%`, return lower bound `-0.431590%`, and profit factor
  `0.549713`. The latest model still has `artifact_lifecycle=shadow`,
  `readiness_state=degraded`, `live_ml_ready=false`, and disabled influence.
  No model has a positive, sufficiently supported fee-after permission.
- A separate read-only Phase 3 go/no-go at `2026-10-24T23:03:56Z` returned
  `no_go`; its one direct blocker was `local_ml_not_live_ready`. The
  authoritative paper-resume preflight at `2026-10-24T23:06:50Z` was also
  blocked by four unavailable quant child-endpoint contracts and by the paper
  service already being active. The service was not restarted. Exchange sync
  was clean in that preflight: zero recent positions, orders, fills, or
  reconciliation issues; live routing remained disabled.
- Recent shadow labels continue to queue/complete, but this has not yielded a
  registered deployable artifact or authoritative executed return
  distribution. The latest local-ML scheduler check was `not_due`; all
  specialist training jobs last reported
  `authoritative_execution_cost_distribution_unavailable`. The separate
  online nested tree study still selected zero of three outer-fold policies.
- Exact user-facing answer: entries are currently withheld because the
  observed fee-after evidence is negative and quality/quant evidence is
  missing, not because OKX rejected an order. Profit is not verified because
  there have been no new governed paper fills and no validated candidate.
  Shadow-only `paper_training_entry` rows are not trades.
- Acceptance count is unchanged: G0 passed; G1 passed as attribution only
  (negative economics); G2 blocked by point-in-time native/executable-cost
  provenance; G3 not passed (zero qualified policies in the preregistered
  Ridge and fixed-tree studies); G4 has no frozen candidate or untouched exam;
  G5 has no governed, settled profitable paper round trip; G6 is pending.
  Thus five of the six profitability gates are still incomplete or failed.
- Do not restart the active paper service while the read-only resume preflight
  is blocked; do not register a model from the shadow rows or enable routing.
  Next: diagnose why the cost-aware child endpoint contracts and authoritative
  cost training input are unavailable, then verify the uploaded native refresh
  repair only after a governed paper-service reload path is safe. Separately
  preregister a new development family only on clean eligible labels. No code
  restart, account mutation, order, or protection change occurred in this
  recheck.

### Authoritative Trade Training Eligibility Recheck (2026-10-25 07:14 Asia/Shanghai)

- The bounded read-only audit
  `audit_online_trade_training_eligibility.py --limit 500 --mode paper`
  completed at `2026-10-24T23:13:50Z`. It inspected 500 paper authoritative
  outcomes and found `trainable_count=0`, `included_count=0`,
  `excluded_count=500`. Every row was excluded for
  `feature_availability_contract_missing`.
- These 500 rows do contain authoritative OKX fill/fee/funding source lineage
  and non-null fee, slippage and funding values; however, the feature
  availability contract is missing, so the cost supervision is not eligible
  and those outcomes cannot safely train/promote a model. The sampled source
  supervision version is
  `2026-10-22.separated-profit-component-availability.v4`.
- Code-path inspection verified that authoritative outcome reconstruction
  projects the original `AIDecision.feature_snapshot` for linked order
  `decision_id`s; training quality then rejects a missing availability seal.
  This does not yet establish whether the absent contracts are unrecoverable
  legacy decisions or a link/projection defect for any newer decisions.
  Never fabricate or backfill the seal using settlement-time or current facts.
- This is a second, independent explanation for the absence of a qualified
  cost-aware artifact: prior executed outcomes are not trainable under the
  current point-in-time feature contract, while current shadow observations
  are not trades and have not produced a profitable selected policy.
- Next exact action: perform a small bounded cross-check of recent
  `decision_id`s from the already audited eligible-trade projections against
  their original decision snapshots and linked order facts, using only the
  availability-contract presence/version and source timing fields. If these
  are legacy rows, preserve their exclusion and validate the capture-to-order
  path only on future governed paper fills after preflight is clear. If recent
  rows have a seal in `AIDecision` but lose it during projection, fix that
  deterministic transport defect with focused tests. Do not create orders or
  start/restart the active service to obtain test data.

### Recent Trade Lineage And New-Snapshot Capture Check (2026-10-25 07:24 Asia/Shanghai)

- A follow-up bounded read-only join checked exactly the 12 newest
  authoritative paper trade `decision_id`s from the preceding outcome query.
  All 12 source `AIDecision` rows were found. Their creation times run from
  `2026-10-21T11:49:01Z` through `2026-10-21T16:02:56Z`; all 12 have no
  availability contract, freeze timestamp, or original decision timestamp.
  This confirms these are pre-capture historical trades, not evidence that a
  post-capture trade snapshot loses its seal in the authoritative projection.
  No backfill or mutation was performed. This check supersedes the uncertainty
  about whether the 500-row exclusion sample was caused by contract loss in
  the current code path.
- A separate read-only 240-minute frozen-feature audit inspected the newest
  64 market decisions and 64 shadow-training snapshots. Every decision row
  had the v1 availability contract, freeze timestamp, and original decision
  timestamp; 57/64 passed point-in-time checks and 7/64 were quarantined:
  five for missing account-fee receipt/availability and two for missing or
  ambiguous index-price facts/clocks. Decision frozen payloads stayed under
  the 96 KiB limit.
- Among the 64 shadow-training rows, 40 passed and 24 failed availability
  checks. This is row-level coverage, not 64 independent decision groups.
  The latest clean decision streak was still only four rows when the audit
  completed. Therefore the new capture contract is being written, but
  completeness and clean streak coverage are not yet stable enough to claim
  a fully verified post-fix pipeline.
- Current direct entry status remains unchanged: 120/120 recent decisions were
  `hold`, with no granted entry permissions or exchange rejections; 76 had no
  candidate and 44 were shadow-training selections only. The direct ML
  profitability gate is still false on negative authoritative evidence. The
  3,582-row fixed-tree development experiment remains a negative result
  (`0/3` outer policies).
- Refined next action: first identify why one market-analysis batch lacks the
  account-fee receipt and why two snapshots lack usable index-price clocks;
  repair only reproducible source capture defects, then observe a longer fresh
  clean streak and verify native executable-path contracts. Do not change
  quality thresholds to accept missing clocks. Only after G2 evidence and a
  separately preregistered G3 policy qualifies may G4 use an untouched exam.
  There is still no profitable candidate to freeze, no exam, and no governed
  paper profitability acceptance.

### Preregistered Third G3 Candidate Family: Fixed Histogram Gradient Boosting

- Development-only identity:
  `native-fee-after-hgb-v1-20261024T233000Z-48h-2500`, with exclusive UTC
  cutoff `2026-10-24T23:30:00Z`, a 48-hour lookback, and at most 2,500
  decision groups. This family is preregistered before its labels are read and
  is not a retune of the rejected Ridge or random-forest families.
- The fixed estimator is
  `HistGradientBoostingRegressor(max_iter=160, learning_rate=0.05,
  max_leaf_nodes=15, min_samples_leaf=24, l2_regularization=1.0,
  early_stopping=False, random_state=42)`, with median imputation and the
  existing feature-key set only. The only entry thresholds are `0`, `0.05%`,
  and `0.1%` net.
- It uses the same primary five-minute label, three chronological purged outer
  folds, two chronological purged inner folds, decision-group separation,
  240-minute label-end purge, minimum 16 selections, positive fee-after mean
  and lower bound, PF above one, and positive mean in both inner folds.
  No exam labels, model artifact, permission change, database mutation, order,
  or service restart is permitted. A failed result must remain rejected.
- Recovery checkpoint at 2026-10-25 07:39 Asia/Shanghai: the fixed HGB
  estimator module existed locally but the shared online nested-development
  audit did not yet expose it as a selectable family, so the preregistered run
  could not be launched. Added the HGB dependency hash, explicit `--family hgb`
  dispatch, and isolated family coverage tests. Local focused verification:
  12 tests passed across HGB, remote audit and fixed-tree suites; Ruff,
  compileall and `git diff --check` passed.
- Exact next operation: upload only
  `services/ml_nested_development_hgb.py`,
  `scripts/audit_online_nested_current_development.py`,
  `tests/test_ml_nested_development_hgb.py`, and
  `tests/test_audit_online_nested_current_development.py` with
  `--skip-restart`; verify hashes and run the bounded remote focused tests.
  Then run
  `scripts/audit_online_nested_current_development.py --family hgb --revision v1 --limit 2500 --hours 48 --cutoff 2026-10-24T23:30:00Z`.
  The report path is
  `data/profitability_remediation/online-nested-development-hgb-v1-20261024T233000Z.json`.
  Resource limits are one read-only DB connection, 15-second statement
  timeout, 25% CPU, 1 GiB memory and 300 seconds. Recovery is to inspect the
  named remote unit/report before any retry. The operation reads development
  labels only and must not restart services, persist an artifact, change
  permissions or submit orders.

### Fixed HGB Development Result (2026-10-25 07:54 Asia/Shanghai)

- The preregistered HGB run completed at `2026-10-24T23:53:46Z` using the
  exact `[2026-10-22T23:30:00Z, 2026-10-24T23:30:00Z)` window. It read
  3,815 native executable, fee-after-complete rows across 2,028 decision
  groups; the primary five-minute horizon had 951 rows. All three outer and
  six inner chronological purged folds were estimable with ready coverage.
- Development input fingerprint:
  `9603873eb2752a6c8dda1349f4ffd3528d0eddc71a31c1312156f3163a535df6`.
  Source frame fingerprint:
  `616c41582e4d119f33e9940e0563989be9f3e998530d2b300c7a32c79ba1656a`.
  The complete bounded report is
  `data/profitability_remediation/online-nested-development-hgb-v1-20261024T233000Z.json`.
- The fixed family selected 0 policies in 3 outer folds. Every outer-fold
  inner selection reason was `no_inner_policy_qualified`; all three tested
  thresholds failed in all folds. Representative inner fee-after means were
  -0.06263% to -0.14599%, lower bounds -0.15206% to -0.21388%, and profit
  factors 0.1941 to 0.5976. The selected gross edge was generally only about
  0.00%-0.07% against roughly 0.126%-0.136% measured execution cost.
- Safety fields confirm no exam labels evaluated, no artifact persisted, no
  DB mutation, no promotion, no order, and no runtime policy change. Remote
  focused tests passed 10; the paper/dashboard services were not restarted.
  The research code-path gap found during this work (HGB omitted from the
  remote dependency fingerprint) was fixed and reverified before accepting
  the report.
- Result: this is a completed negative G3 development experiment, not a
  profitability or trade authorization. G3 remains **NOT PASSED**; G4 has
  no candidate/exam, G5 has no governed profitable paper round trip, and G6
  remains pending. Do not retune on this inspected window.
- Next exact action resumes the earlier G2 checkpoint: trace the five recent
  account-fee receipt/availability misses and two index-price fact/clock
  misses using their source IDs and bounded provenance projections. Repair
  only a reproducible capture defect; do not infer missing clocks or promote
  the quarantined rows. Then record fresh clean decision-group coverage and
  verify the paper-only service reload preflight before any restart.

### Current Runtime And Entry Blocker Recheck (2026-10-25 08:42 Asia/Shanghai)

- A bounded read-only entry recovery covering
  `2026-10-24T20:25:21Z` through `2026-10-25T00:25:21Z` found `559` market
  decisions. Every final action was `hold`; there were no order attempts or
  exchange rejections. The latest `120` rows split into `81 no_candidate` and
  `39 shadow_training`; selected shadow rows remain labels only.
- The direct permission veto is unchanged and is the primary no-entry cause:
  local ML paper permission is false at `n=267`, fee-after mean
  `-0.303832%`, lower bound `-0.431590%`, and profit factor `0.549713`.
  Recent selected rows also report
  `direction_support_expected_net_not_positive` and
  `direction_support_quant_evidence_missing`; several short rows additionally
  fail the loss-probability quality gate. This is an internal authorization
  decision, not an OKX rejection.
- The same audit confirms shadow maintenance is live and settling. A
  subsequent bounded native pipeline check at `2026-10-25T00:40:39Z` counted
  `1,166` completed native fee-after rows across `745` decision groups and
  verified `200/200` sampled native lineage fingerprints. It also showed
  `418` recent decision groups with four horizons, one still incomplete.
  Historical-path rows remain separate and quarantined.
- Runtime transport is healthy: paper process is running and unpaused,
  WebSocket is connected, all `1,992/1,992` subscription arguments are
  acknowledged with zero errors, and all `498/498` instruments have messages
  on books5, mark-price, index-tickers and tickers. This rules out a
  disconnected WebSocket as the current broad no-entry explanation.
- A fresh 240-minute frozen-feature audit at `2026-10-25T00:36:03Z` found
  all `64` decision and `64` training snapshots sealed and under the
  `98,304`-byte budget. Decisions were `62/64` availability-valid; the two
  invalid rows were one missing index fact caused by a reference-price
  timeout (`ADA/USDT`) and one stale technical contract (`KAITO/USDT`).
  The training view was `60/64` valid because the four invalid rows shared the
  same ADA decision group. The newest clean decision streak reached `14`;
  this is improved coverage, but not the preregistered sustained G2 gate.
- Runtime diagnostics show the remaining reference-price weakness is
  freshness, not subscription loss: some index receipts reached
  `66-75` seconds while the contract correctly rejects index facts older than
  `65` seconds. No threshold was widened. Account-fee receipts were present
  in the fresh boundary, so the earlier fee-receipt defect is not reproduced
  in this window.
- Acceptance remains: G0 passed; G1 passed only as negative attribution; G2
  blocked; G3 not passed after the fixed Ridge, random-forest and HGB
  development families selected `0/3` outer policies; G4 has no frozen
  candidate or untouched exam; G5 has no governed paper round trip; G6 is
  pending. Five of six gates therefore remain incomplete or failed.
- This recheck changed no model permission, threshold, risk, leverage,
  service mode, order, or protection. The bounded frozen-feature audit now
  carries the linked `decision_id` for each invalid row, so future tracing
  can distinguish a capture defect from a derived-snapshot projection issue.
- Next exact action: use the linked `decision_id`s to inspect the ADA
  reference-route timeout and KAITO technical-contract source path with
  bounded provenance only; preserve both rows as quarantined. Continue
  collecting new native fee-after groups, and preregister a materially
  different chronological candidate only after specifying its horizon and
  untouched evaluation window. Do not register the incompatible Phase 3
  artifacts, widen the 65-second freshness rule, or authorize entries from
  the negative `n=267` history.

### Current User-Requested Status Recheck (2026-10-25 09:08 Asia/Shanghai)

- A fresh bounded frozen-feature audit at `2026-10-25T00:50:53Z` inspected
  the newest 64 market decisions and 64 shadow-training snapshots. All 128
  rows carried the v1 availability contract, freeze timestamp and original
  decision timestamp, and every frozen payload stayed below the 98,304-byte
  budget. Decision rows were `61/64` availability-valid and training rows
  were `60/64` valid. The three decision invalidations were two missing
  technical contracts (CRV) and one missing index fact caused by an ADA
  reference-price timeout; the training invalidations were the four CRV
  rows in the same decision group. The newest valid streak was only 7
  decision rows and 28 training rows, so G2 is still not accepted.
- The bounded source projection identified the exact capture behavior:
  CRV (`decision_id=694951`) had
  `indicator_snapshot_reason=indicator_cached_source_delivery_unobserved`;
  its derivatives orderbook/reference routes completed but funding/open
  interest timed out. ADA (`decision_id=694909`) had complete technical
  inputs but its reference-price route timed out and therefore no
  `index_price_fact`. KAITO (`decision_id=694895`) had all derivative
  routes completed in the stored row, but its indicator snapshot was
  unavailable because the underlying kline delivery receipt was not
  observed. These are correctly quarantined provenance/availability
  failures, not reasons to synthesize clocks or widen the 65-second
  freshness contract.
- A fresh entry-flow projection through `2026-10-25T01:07:12Z` found 64
  newest market decisions, all final actions `hold`, with no order attempt,
  exchange rejection, open position or fill. The repeated permission
  decision is explicit on every row:
  `average_fee_after_return_not_positive`,
  `fee_after_return_lcb_not_positive`, and
  `fee_after_profit_factor_not_above_break_even`.
  The shared authority remains `n=267`, average fee-after return
  `-0.303832%`, lower bound `-0.431590%`, PF `0.549713`, gross profit
  `99.035696`, gross loss `180.158844`, worst return `-12.34323%`, and
  maximum drawdown `92.014573%`. The model service is not the blocker:
  the read-only Phase 3 readiness audit at `2026-10-25T01:06:52Z` found the
  target endpoint and GPU service ready, while live routing and the
  specialization remain intentionally shadow-only.
- The authoritative outcome audit at `2026-10-25T01:06:52Z` found 500
  complete paper outcomes, but `0` trainable outcomes because all 500
  historical decisions lack the point-in-time feature availability
  contract. This is why the existing negative authority cannot be safely
  refreshed into a new candidate and why the shadow rows cannot be treated
  as trades. No backfill, order, permission change, restart, leverage
  change or protection change was performed.
- User-facing root cause: the system is currently not opening because the
  governed profitability gate is correctly closed on measured negative
  fee-after economics; a smaller secondary blocker is intermittent
  point-in-time data capture (CRV/ADA/KAITO examples). It is not currently
  an OKX submission failure. Profitability is still **not achieved**.
- Remaining acceptance work is unchanged and explicit: G2 needs a sustained
  clean native/executable feature-contract window; G3 needs a separately
  preregistered candidate that passes nested chronological cost-aware
  development (the fixed Ridge, random forest and HGB families each
  selected `0/3` outer policies); G4 needs a frozen policy and untouched
  independent exam; G5 needs governed paper fills with official settlement;
  G6 needs subsequent independent-window acceptance. Five gates remain
  incomplete or failed. The next action is to preserve quarantined rows,
  keep collecting clean native fee-after groups, and only then run a new
  preregistered development family; no gate or cost rule will be weakened
  to manufacture trades.

### Preregistered Fourth G3 Development Family: Fixed Extra Trees

- Development-only identity:
  `native-fee-after-extra-trees-v1-20261024T233000Z-48h-2500`, with exclusive
  UTC cutoff `2026-10-24T23:30:00Z`, a 48-hour lookback, and at most 2,500
  decision groups. This exact development window was already label-inspected
  by the fixed Ridge/tree/HGB development studies; this ExtraTrees run is
  another development comparison on known development data, never a fresh
  sample or independent profitability claim. G4 remains sealed.
- Fixed estimator:
  `ExtraTreesRegressor(n_estimators=220, max_depth=10, min_samples_leaf=16,
  max_features=0.7, random_state=42, n_jobs=1)`, with median imputation and
  the existing feature-key set only. The only entry thresholds are `0`,
  `0.05%`, and `0.1%` net. No parameter search or feature search is allowed.
- Evaluation is the existing primary five-minute target, three chronological
  purged outer folds, two chronological purged inner folds, decision-group
  separation and primary-label-end purge, minimum 16 selections, positive fee-after
  mean and lower bound, PF above one, and positive mean in both inner folds.
  The query admits only completed native-path, fee-after-complete paper
  labels; `build_training_frame` must additionally pass the shared shadow
  sample quality and feature-availability supervision contracts.
- Exact command:
  `scripts/audit_online_nested_current_development.py --family extra_trees
  --revision v1 --limit 2500 --hours 48 --cutoff 2026-10-24T23:30:00Z`.
  Expected report:
  `data/profitability_remediation/online-nested-development-extra_trees-v1-20261024T233000Z.json`.
- Before launch: focused local tests and Ruff must pass; upload only the
  estimator, dispatch script, and their focused tests with `--skip-restart`;
  verify dependency hashes and run the bounded remote focused tests. The
  audit is read-only, one DB connection, 15-second SQL statements, 25% CPU,
  1 GiB memory, and a 300-second systemd deadline. It must not persist an
  artifact, evaluate an exam, mutate the database, change runtime permission,
  restart services, or submit orders. Inspect the named unit/report before
  any retry.
- Decision rule: a zero-qualified result is recorded as a rejection and no
  additional model-family sweep will be used to claim progress. Even if
  development qualifies, do not promote it: it must first be frozen and then
  evaluated against a separately preregistered untouched independent sample.
- Current status at registration: no result exists. G2 remains blocked by
  incomplete point-in-time availability coverage; G3 has no qualified policy;
  G4-G6 remain unpassed. Next action is the single bounded run above, followed
  by returning to the documented G2 capture gaps and finite acceptance plan.

### Fixed Extra Trees Result And Time-Bound Replan (2026-10-25 09:37 Asia/Shanghai)

- The preregistered development-only run completed at
  `2026-10-25T01:30:14Z` for the exact known window
  `[2026-10-22T23:30:00Z, 2026-10-24T23:30:00Z)`. It read 3,815
  native/executable fee-after-complete rows across 2,028 decision groups;
  the primary five-minute horizon had 951 rows. This input fingerprint is
  identical to the already-inspected HGB study:
  `9603873eb2752a6c8dda1349f4ffd3528d0eddc71a31c1312156f3163a535df6`;
  source frame fingerprint:
  `616c41582e4d119f33e9940e0563989be9f3e998530d2b300c7a32c79ba1656a`.
  Report:
  `data/profitability_remediation/online-nested-development-extra_trees-v1-20261024T233000Z.json`.
- The fixed ExtraTrees family selected **0 of 3** outer-fold policies.
  All inner thresholds failed the registered fee-after rules. Across folds,
  selected inner mean returns were negative (representative ranges
  `-0.0668%` to `-0.5156%`), with PF `0.2586` to `0.5841`; measured selected
  execution costs were approximately `0.114%` to `0.158%`. No artifact,
  exam evaluation, permission change, database mutation, service restart or
  order occurred. This rejects the family; it is not profitability evidence.
- The same one-hour online recovery at `2026-10-25T01:32:38Z` found 137
  market decisions, all `hold` and unexecuted. In the newest 80, 43 were
  `no_candidate` and 37 were `shadow_training`; none granted entry. Local-ML
  paper permission remains false at 267 authoritative samples:
  mean fee-after return `-0.303832%`, lower bound `-0.431590%`, PF `0.549713`.
  Its three direct blockers remain non-positive mean, non-positive lower
  bound and PF not above break-even. The latest shadow artifact remains
  `shadow`/`degraded`; its influence is disabled. The visible model-service
  readiness is not an order-submission failure.
- A fresh 240-minute availability audit at `2026-10-25T01:37:12Z` found:
  decision snapshots `60/64` valid, newest clean streak `6` across `6`
  symbols; training snapshots `52/64` valid, newest clean streak `20` across
  `5` symbols. All 128 snapshots had contracts/seals, no seal/compaction
  errors and were under the 98,304-byte budget (decision max 97,550 B,
  training max 95,326 B). The four invalid decisions and twelve invalid
  training rows were tied to missing/ambiguous index-price facts/clocks;
  observed examples include reference-price timeouts and REST index facts
  with source timestamp `0`. The preregistered G2 boundary is still not met:
  at least 30 consecutive clean market decisions across at least 20 symbols,
  with clean corresponding training snapshots.
- Profitability status is unchanged: G0 passed; G1 passed as negative
  authoritative attribution; G2 blocked; G3 **NOT PASSED** after fixed Ridge,
  random-forest, HGB and ExtraTrees families each selected `0/3`; G4 has no
  frozen candidate or untouched exam; G5 has no governed profitable paper
  round trip; G6 remains pending. Thus five profitability gates remain
  incomplete or failed. New shadow rows and training scheduler heartbeats are
  not trades and do not improve realized PnL.
- Corrective decision: stop estimator-family sweeps on this already-inspected
  window. The next work is one bounded trace of the index-price capture path
  for the recent invalid decision IDs, correlating route status/source clocks
  with the existing request-queue telemetry. Within 24 hours of this checkpoint,
  record either (a) a reproducible local defect with a focused fix, tests,
  targeted no-restart upload and verified fresh G2 boundary, or (b) evidence
  that the remaining failure is external/provider/queue behavior and its
  explicit mitigation/owner. Do not extend timeouts or loosen freshness.
- No honest completion date for profitability can be given: it depends on a
  candidate producing genuinely positive fee-after evidence, then passing an
  untouched exam and subsequent governed paper settlements. A 24-hour
  infrastructure decision checkpoint is not a promise of profitability or
  entry authorization. Keep live trading disabled and permissions unchanged.

### Current User Recheck And WebSocket Reuse Repair (2026-10-25 10:36 Asia/Shanghai)

- The latest bounded entry recovery through `2026-10-25T02:13:46Z` covered
  `147` market decisions. All final actions were `hold`; there were no order
  attempts, exchange rejections, open positions or fills. The direct entry
  veto remains the authoritative fee-after permission at `n=267`: mean
  `-0.303832%`, lower bound `-0.431590%`, PF `0.549713`. The newest rows also
  remain blocked by non-positive expected net return and missing quantitative
  evidence. Shadow-training rows are not trades.
- The same report shows the paper service and training heartbeat are active.
  Recent component availability was `141/147` clean; the six invalid rows were
  index-price fact/receipt/clock failures. Historical requalification remains
  dry-run with `0` eligible rows because its point-in-time availability basis
  is still unverified. No model permission, threshold, risk, leverage, mode,
  order or protection state was changed.
- A fresh bounded public-market latency audit at
  `2026-10-25T02:08:29Z` read journald only (one 2 MiB tail, so this is
  diagnostic evidence rather than a complete window census). It found
  priority-route queue p95 waits of `23,280.6 ms` for index tickers,
  `26,795.6 ms` for mark prices, `12,188.9 ms` for order books and
  `17,736.6 ms` for single tickers; provider p95 durations were materially
  lower (`1,993.2/3,015.5/3,108.7/4,342.0 ms`). This confirms local request
  fan-out/queueing as a secondary G2 capture blocker, not an OKX submission
  rejection and not evidence that profitability is positive.
- Repair added a fail-closed WebSocket fast path in
  `services/data_service.py`: an authoritative refresh may reuse a WebSocket
  snapshot only when order book, mark, index, native 1m path, source
  consistency assertions and clean market-fact quality are all present. Any
  missing/stale/ambiguous component still uses the existing strict REST
  refresh. This removes duplicate high-priority REST requests without
  widening freshness or quality contracts.
- Local verification after the repair: `481` focused data/latency/trading
  boundary tests passed, including the new complete-WS reuse regression and
  the existing fail-closed authoritative-refresh regression. Ruff,
  compileall and `git diff --check` passed.
- Current gate state is unchanged: G0 passed; G1 passed only as negative
  attribution; G2 blocked; G3 not passed after four fixed development
  families selected `0/3`; G4 has no frozen candidate/exam; G5 has no
  governed profitable paper round trip; G6 pending. There is still no honest
  profitability completion date. Next action is targeted no-restart upload,
  remote hash/focused-test verification, then a new bounded post-load
  availability/latency boundary. Do not restart or authorize entries until the
  paper-only preflight is clear.

### WebSocket Reuse Targeted Deployment Verification (2026-10-25 10:42 Asia/Shanghai)

- Targeted sync completed with `--skip-restart`; only
  `services/data_service.py` and `tests/test_data_service_security.py` changed
  on the remote checkout. The three public-latency audit files were already
  byte-identical. Remote integrity verification returned `source-integrity-ok:5`.
- Remote focused validation passed `116` tests in `61.19s` for
  `tests/test_data_service_security.py` and
  `tests/test_public_market_latency_audit.py`. No database mutation, order,
  position, protection, permission or model-routing change occurred.
- The read-only paper-resume preflight at `2026-10-25T02:41:32Z` remains
  blocked by the four Phase 3 quant child endpoint contracts being unavailable
  and by `bb-paper-trading.service` already being active. Model server,
  account truth, OKX integrity, Dashboard and tunnel checks are healthy.
  Therefore the process was not restarted; the new WebSocket reuse code is
  uploaded but not yet loaded by the long-running paper process.
- Acceptance remains unchanged and honest: the direct no-entry cause is still
  negative fee-after permission, while G2 has a secondary queue/freshness
  blocker. Next action is to wait for a clear paper-only restart preflight,
  then restart only the paper service and measure a new bounded component,
  queue-latency and entry-funnel window. Keep live routing disabled and do not
  treat any shadow row as a trade or profitability result.
