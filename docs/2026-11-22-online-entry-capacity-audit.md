# Online Paper Entry Capacity Audit

Checked: 2026-11-22 06:21-06:23 Asia/Shanghai

## Findings

- The online service was active and producing market decisions. The four-hour
  paper audit loaded 1,202 decisions and found 72 normal-paper candidates, 70
  valid authorized paper-training contracts, and 8 executed entries.
- All 8 executed entries in the recent execution report were long. The latest
  60-minute audit showed 8 entries, 4 exits, 3 authoritative closed positions,
  and net realized PnL of +0.0420574 USDT; 2 of the 3 closed positions lost.
  This is too small and directionally concentrated to establish profitability.
- There are currently 8 open paper position groups and 8 verified protection
  orders. Missing, orphaned, split, mismatched, and invalid protection counts
  were all zero. There were no dynamic-exit contract gaps.
- The paper-training review-group limit is 8. New paper-training candidates
  are being rejected with `paper_training_review_capacity_full` because all
  eight review groups are occupied. This is distinct from the general dynamic
  position-capacity observation, which has no hard position-count limit.
- The currently evaluated model permits paper execution only on the short
  side; its long evaluation is negative (average return -1.254%, return LCB
  -1.870%, profit factor 0.342). Short evaluation is positive in that report,
  but global live readiness remains false. The recent executed entries were
  all long, so the observed sample does not support a profitability claim.
- The normal-paper contract version in recent online decisions is
  `2026-11-22.normal-paper-strategy-trade.v26`. This version permits a bounded
  paper-training route with negative lower-bound return, while requiring
  positive size-aware expected return, finite downside, and keeping production
  permission false. It is a sample-collection route, not a profitability fix.

## Decision

Do not raise the eight-group training limit or bypass return and risk gates.
The current single-trade training risk cap is 0.5%, and the execution contract
does not apply a portfolio risk limit. More concurrent training positions
would therefore increase exposure without demonstrating positive expectancy.
Allow existing positions to complete their normal exit and settlement flow,
then reassess candidate direction, fee-after outcomes, and aggregate risk.

## Verification

- Focused entry, sizing, capacity, execution-contract, and auto-entry tests:
  327 passed.
- Ruff and `git diff --check` passed for the changed entry and sizing code.
- Online protection inventory: 8 positions, 8 protection orders, zero coverage
  blockers.
