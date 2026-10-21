# Profitability Remediation Execution Plan

Updated: 2026-10-22, Asia/Shanghai.
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
| G2 Data and label validity | IN PROGRESS | Eligibility, timing, cost and provenance report plus defect regressions | No future-data leakage or split-group overlap; correct price/label horizon and source identity; missing or contradictory facts quarantined, not imputed into profitability |
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
