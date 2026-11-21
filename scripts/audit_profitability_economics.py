"""Bounded read-only lifecycle economics and independent account-bill audit."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import shlex
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.exceptions import ExchangeAPIError  # noqa: E402
from db.session import close_db, get_read_session_ctx  # noqa: E402
from models.decision import AIDecision  # noqa: E402
from models.trade import OkxPositionHistory  # noqa: E402
from services.authoritative_trade_outcome import load_authoritative_trade_outcomes  # noqa: E402
from services.okx_native_facts import (  # noqa: E402
    _account_bill_from_row,
    _oldest_account_bill_pagination_cursor,
)
from services.paper_training_settlements import _settled_model_versions  # noqa: E402
from services.strategy_learning import _return_metrics  # noqa: E402

BILL_PAGE_INTERVAL_SECONDS = 0.5
BILL_RATE_LIMIT_BACKOFF_SECONDS = (1.0, 2.0, 4.0)


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("audit cutoff must have an explicit timezone")
    return result.astimezone(UTC)


def _single_entry_decision_id(row: dict[str, Any]) -> int | None:
    decision_id = row.get("decision_id")
    if (
        type(decision_id) is int and decision_id > 0
        and row.get("entry_decision_ids") == [decision_id]
    ):
        return decision_id
    return None


async def _load_entry_evidence(
    outcomes: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    ids = sorted({
        decision_id for row in outcomes if row.get("close_status") == "full"
        and (decision_id := _single_entry_decision_id(row)) is not None
    })
    result = {}
    snapshot = AIDecision.decision_learning_snapshot
    raw = AIDecision.raw_llm_response
    contract = snapshot["normal_paper_trade"]
    current = contract["reconciliation"]["current"]

    def tool_version(route):
        return func.coalesce(
            snapshot["local_ai_tools"][route]["model_version"].as_string(),
            raw["local_ai_tools"][route]["model_version"].as_string(),
        )

    async with get_read_session_ctx(
        statement_timeout_ms=15_000, transaction_isolation="REPEATABLE READ READ ONLY",
    ) as session:
        for offset in range(0, len(ids), 50):
            statement = select(
                AIDecision.id,
                contract["quant_evidence_families"].label("families"),
                contract["selection_reason"].as_string().label("selection_reason"),
                contract["side"].as_string().label("side"),
                contract["prediction_horizon_minutes"].label("horizon"),
                contract["recent_side_loss_guard"]["training_model_versions"].label("versions"),
                snapshot["ml_signal"]["model_version"].as_string().label("local_version"),
                tool_version("profit_prediction").label("profit_version"),
                tool_version("time_series_prediction").label("timeseries_version"),
                contract["direction_objective_net_return_pct"].label("direction_objective"),
                current["risk_adjusted_net_return_pct"].label("quote_objective"),
                current["return_lcb_pct"].label("quote_lower_bound"),
                current["expected_net_return_pct"].label("quote_expected_net"),
                current["gross_expected_return_pct"].label("quote_expected_gross"),
                current["execution_cost_pct"].label("quote_cost"),
            ).where(AIDecision.id.in_(ids[offset : offset + 50]))
            rows = (await session.execute(statement)).mappings()
            result.update({row["id"]: dict(row) for row in rows})
    return result


def _entry_signal_economics(
    rows: list[dict[str, Any]], evidence: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    cohorts = defaultdict(list)
    for row in rows:
        decision_id = _single_entry_decision_id(row)
        side = str(row.get("side") or "unknown")
        route = str(row.get("strategy_selection_reason") or "unknown")
        entry = evidence.get(decision_id) if decision_id else None
        direction_objective = quote_objective = quote_lower_bound = None
        families: tuple[str, ...] = ()
        versions: tuple[tuple[str, str], ...] = ()
        horizon = None
        if entry and entry["selection_reason"] == route and entry["side"] == side:
            direction_objective = _finite(entry.get("direction_objective"))
            quote_objective = _finite(entry.get("quote_objective"))
            quote_lower_bound = _finite(entry.get("quote_lower_bound"))
            horizon = _finite(entry.get("horizon"))
            if isinstance(entry["families"], list):
                families = tuple(sorted({
                    family for family in entry["families"] if isinstance(family, str) and family
                }))
                versions = tuple(sorted(
                    (family, version)
                    for family, version in _settled_model_versions(
                        list(families), entry.get("versions"), entry.get("local_version"),
                        entry.get("profit_version"), entry.get("timeseries_version"),
                    ).items() if isinstance(version, str) and version
                ))
        else:
            entry = {}
        cohorts[
            side, route, _sign(direction_objective), _sign(quote_objective),
            _sign(quote_lower_bound), families, versions, horizon,
        ].append((row, entry))
    summaries = []
    for key, pairs in sorted(cohorts.items(), key=lambda item: str(item[0])):
        side, route, direction_sign, quote_sign, lower_sign, families, versions, horizon = key
        cohort = [row for row, _ in pairs]
        pnl = [
            row["realized_net_pnl_components"]["reported_realized_net_pnl_usdt"] for row in cohort
        ]
        returns = [
            value for row in cohort if (value := _finite(row.get("net_return_after_all_cost_pct")))
            is not None
        ]
        gross_loss = -math.fsum(value for value in pnl if value < 0)
        equity = peak = drawdown = 0.0
        dated_pnl: dict[datetime, list[float]] = defaultdict(list)
        for row, value in zip(cohort, pnl, strict=True):
            if row.get("label_timestamp"):
                dated_pnl[_utc(row["label_timestamp"])].append(value)
        for _, values in sorted(dated_pnl.items()):
            equity += math.fsum(values)
            peak = max(peak, equity)
            drawdown = max(drawdown, peak - equity)
        notional = [
            value for row in cohort
            if (value := _finite(row.get("notional"))) is not None and value > 0
        ]
        risk = [
            (budget, stress, excess) for row in cohort
            if (budget := _finite(row.get("execution_risk_budget_usdt"))) is not None
            and budget > 0
            and (stress := _finite(row.get("execution_planned_stressed_loss_usdt"))) is not None
            and stress >= 0
            and (excess := _finite(row.get("execution_actual_over_budget_loss_usdt"))) is not None
            and excess >= 0
        ]
        summaries.append({
            "side": side,
            "entry_selection_reason": route,
            "direction_objective_sign": direction_sign,
            "quote_objective_sign": quote_sign,
            "quote_lower_bound_sign": lower_sign,
            "quant_evidence_families": list(families),
            "driving_model_versions": dict(versions),
            "prediction_horizon_minutes": horizon,
            "count": len(cohort),
            "realized_net_pnl_usdt": math.fsum(pnl),
            "fee_after_return_count": len(returns),
            "mean_fee_after_return_pct": math.fsum(returns) / len(returns) if returns else None,
            "profit_factor": math.fsum(value for value in pnl if value > 0) / gross_loss
            if gross_loss > 0 else None,
            "settled_pnl_drawdown_usdt": drawdown
            if sum(map(len, dated_pnl.values())) == len(pnl) else None,
            "entry_notional_count": len(notional),
            "mean_entry_notional_usdt": math.fsum(notional) / len(notional) if notional else None,
            "risk_evidence_count": len(risk),
            "mean_planned_risk_budget_usdt": math.fsum(item[0] for item in risk) / len(risk)
            if risk else None,
            "mean_planned_stressed_loss_usdt": math.fsum(item[1] for item in risk) / len(risk)
            if risk else None,
            "realized_loss_over_budget_count": sum(item[2] > 0 for item in risk),
            "realized_loss_over_budget_usdt": math.fsum(item[2] for item in risk)
            if risk else None,
            "forecast_comparison": _forecast_comparison(pairs),
        })
    return {
        "rows": summaries,
        "scope": "observed_closed_lifecycles_only",
        "return_confidence_bound": None,
        "counterfactual_execution_or_portfolio_equity_evaluated": False,
    }


def _sign(value: float | None) -> str:
    return "missing" if value is None else "positive" if value > 0 else "non_positive"


def _forecast_comparison(pairs: list[tuple[dict, dict]]) -> dict[str, Any]:
    forecasts = defaultdict(list)
    for row, entry in pairs:
        notional = _finite(row.get("notional"))
        net = _finite(row.get("net_return_after_all_cost_pct"))
        if not notional or notional <= 0 or net is None:
            continue
        price = row["realized_net_pnl_components"]["gross_pnl_usdt"] * 100 / notional
        for component, expected_key, actual in (
            ("net", "quote_expected_net", net),
            ("gross", "quote_expected_gross", price),
            ("cost", "quote_cost", price - net),
        ):
            expected = _finite(entry.get(expected_key))
            if expected is not None:
                forecasts[component].append((expected, actual))
    return {
        name: {
            "paired_count": len(values),
            "mean_forecast_pct": math.fsum(expected for expected, _ in values) / len(values),
            "mean_realized_pct": math.fsum(actual for _, actual in values) / len(values),
            "mean_forecast_minus_realized_pct": (
                math.fsum(expected - actual for expected, actual in values) / len(values)
            ),
        }
        for name, values in forecasts.items()
    }


async def _load_settlement_clocks(outcomes: list[dict[str, Any]]) -> dict[str, datetime]:
    history = OkxPositionHistory
    async with get_read_session_ctx(
        statement_timeout_ms=15_000, transaction_isolation="REPEATABLE READ READ ONLY",
    ) as session:
        rows = (await session.execute(select(
            history.row_identity, history.settlement_facts_known_at,
        ).where(
            history.mode == "paper",
            history.row_identity.in_([row["lifecycle_key"] for row in outcomes]),
        ))).all()
    return {key: known for key, known in rows if known is not None}


def _observed_entry_filters(
    rows: list[dict[str, Any]], evidence: dict[int, dict[str, Any]],
    settlement_clocks: dict[str, datetime], *, start: datetime, until: datetime,
) -> dict[str, Any]:
    groups = defaultdict(list)
    exclusions = Counter()
    for row in rows:
        opened = row.get("decision_timestamp")
        closed = row.get("label_timestamp")
        known = settlement_clocks.get(row["lifecycle_key"])
        if not opened or not closed or not start <= _utc(opened) <= _utc(closed) < until:
            continue
        entry = evidence.get(_single_entry_decision_id(row)) or {}
        if (
            row.get("close_status") != "full"
            or row.get("trade_fact_trusted") is not True
            or row.get("outcome_complete") is not True
            or entry.get("selection_reason") != "paper_training_entry"
            or entry["selection_reason"] != row.get("strategy_selection_reason")
            or entry.get("side") != row.get("side")
            or (row.get("realized_net_pnl_components") or {}).get("formula_consistent") is not True
        ):
            exclusions["not_trusted_exactly_attributed_training"] += 1
            continue
        if known is None or not _utc(closed) <= known < until:
            exclusions["settlement_version_unavailable_at_cutoff"] += 1
            continue
        keys = ("direction_objective", "quote_objective", "quote_lower_bound")
        scores = {key: _finite(entry.get(key)) for key in keys}
        net = _finite(row.get("net_return_after_all_cost_pct"))
        pnl = _finite(row["realized_net_pnl_components"]["reported_realized_net_pnl_usdt"])
        if net is None or pnl is None:
            exclusions["missing_net_settlement"] += 1
            continue
        groups[_single_entry_decision_id(row)].append((
            closed, row["lifecycle_key"], {
                "timestamp": closed,
                "net_return_after_all_cost_pct": net,
                "net_pnl_after_all_costs_usdt": pnl,
            }, scores,
        ))
    exclusions["repeated_entry_decision"] = sum(len(group) for group in groups.values() if len(group) > 1)
    cases = sorted(group[0] for group in groups.values() if len(group) == 1)
    evaluated = {}
    for name, selected in (
        ("observed_policy", [case[2] for case in cases]),
        ("positive_direction_objective", [
            case[2] for case in cases if _sign(case[3]["direction_objective"]) == "positive"
        ]),
        ("positive_quote_objective", [
            case[2] for case in cases if _sign(case[3]["quote_objective"]) == "positive"
        ]),
        ("positive_quote_lower_bound", [
            case[2] for case in cases if _sign(case[3]["quote_lower_bound"]) == "positive"
        ]),
        ("no_trade", []),
    ):
        evaluated[name] = {
            **_return_metrics(selected),
            "coverage": len(selected) / len(cases) if cases else None,
            "realized_net_pnl_usdt": (
                math.fsum(row["net_pnl_after_all_costs_usdt"] for row in selected)
            ),
        }
    return {
        "scope": "retrospective_abstention_on_observed_training_fills_only",
        "counterfactual_portfolio_replay": False,
        "return_uncertainty_method": "minimum_contiguous_block_mean",
        "entry_window_start": start.isoformat(),
        "settlement_availability_cutoff": until.isoformat(),
        "exclusions": dict(exclusions),
        "missing_prediction_counts": {
            key: sum(case[3][key] is None for case in cases)
            for key in ("direction_objective", "quote_objective", "quote_lower_bound")
        },
        "policies": evaluated,
    }


def summarize_lifecycles(
    outcomes: list[dict[str, Any]], *, since: datetime,
    entry_evidence: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize canonical facts; do not invent a second settlement formula."""
    unique: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for row in outcomes:
        key = str(row.get("lifecycle_key") or "")
        if not key or row.get("event_type") != "AuthoritativeTradeOutcome":
            raise ValueError("canonical lifecycle identity is required")
        if key in unique:
            if unique[key] != row:
                raise ValueError("conflicting duplicate lifecycle")
            duplicates += 1
        unique[key] = row
    rows = list(unique.values())
    full = [row for row in rows if row.get("close_status") == "full"]
    partial_rows = [row for row in rows if row.get("close_status") != "full"]
    fields = {
        "gross_pnl_usdt": "gross_pnl_usdt",
        "fees_signed_usdt": "official_fee_signed_usdt",
        "funding_signed_usdt": "funding_fee_usdt",
        "liquidation_penalty_signed_usdt": "liquidation_penalty_usdt",
        "realized_net_pnl_usdt": "reported_realized_net_pnl_usdt",
    }
    valid = []
    for row in full:
        components = row.get("realized_net_pnl_components") or {}
        if components.get("formula_consistent") is True and all(
            _finite(components.get(key)) is not None for key in fields.values()
        ):
            valid.append(row)
    totals = {
        key: math.fsum(row["realized_net_pnl_components"][source] for row in valid)
        for key, source in fields.items()
    }
    by_side_and_entry: dict[str, dict[str, list[float]]] = {}
    for row in valid:
        side = str(row.get("side") or "").lower()
        if side not in {"long", "short"}:
            side = "unknown"
        entry = str(row.get("strategy_selection_reason") or "unknown")
        net_pnl = row["realized_net_pnl_components"]["reported_realized_net_pnl_usdt"]
        by_side_and_entry.setdefault(side, {}).setdefault(entry, []).append(net_pnl)

    def pnl_stats(values: list[float]) -> dict[str, Any]:
        wins = [value for value in values if value > 0]
        losses = [value for value in values if value < 0]
        gross_profit = math.fsum(wins)
        gross_loss = -math.fsum(losses)
        return {
            "count": len(values),
            "profitable_count": len(wins),
            "losing_count": len(losses),
            "winning_net_pnl_usdt": gross_profit,
            "losing_net_pnl_usdt": math.fsum(losses),
            "realized_net_pnl_usdt": math.fsum(values),
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        }

    gap_counts = Counter(
        str(gap) for row in rows for gap in set(row.get("outcome_evidence_gaps") or [])
    )
    duration_ratios = []
    for row in valid:
        horizon = _finite((row.get("normal_paper_trade_evidence") or {}).get(
            "prediction_horizon_minutes"
        ))
        holding = _finite(row.get("holding_minutes"))
        if horizon is not None and horizon > 0 and holding is not None and holding >= 0:
            duration_ratios.append(holding / horizon)
    slippage = [
        value for row in valid
        if (value := _finite(row.get("execution_slippage_usdt"))) is not None
    ]
    return {
        "source": "services.authoritative_trade_outcome",
        "unique_lifecycle_count": len(rows),
        "identical_duplicate_count": duplicates,
        "fully_closed_count": len(full),
        "partial_or_unknown_count": len(partial_rows),
        "settlement_consistent_full_count": len(valid),
        "full_excluded_from_totals_count": len(full) - len(valid),
        "totals_are_lifecycle_pnl_not_window_or_portfolio_return": True,
        "full_lifecycles_opened_before_window_count": sum(
            bool(row.get("decision_timestamp"))
            and _utc(row["decision_timestamp"]) < since for row in valid
        ),
        "full_lifecycle_totals": totals,
        "entry_signal_economics": _entry_signal_economics(valid, entry_evidence or {}),
        "full_lifecycles_by_side": {
            side: {
                **pnl_stats([value for values in entries.values() for value in values]),
                "by_entry_selection_reason": {
                    entry: pnl_stats(values) for entry, values in entries.items()
                },
            }
            for side, entries in by_side_and_entry.items()
        },
        "profitable_full_count": sum(
            row["realized_net_pnl_components"]["reported_realized_net_pnl_usdt"] > 0
            for row in valid
        ),
        "losing_full_count": sum(
            row["realized_net_pnl_components"]["reported_realized_net_pnl_usdt"] < 0
            for row in valid
        ),
        "positive_gross_but_negative_net_count": sum(
            row["realized_net_pnl_components"]["gross_pnl_usdt"] > 0
            and row["realized_net_pnl_components"]["reported_realized_net_pnl_usdt"] < 0
            for row in valid
        ),
        "complete_training_evidence_count": sum(
            row.get("outcome_complete") is True for row in full
        ),
        "gap_counts": dict(gap_counts.most_common()),
        "measured_slippage": {
            "complete_lifecycle_count": len(slippage),
            "signed_execution_slippage_usdt": math.fsum(slippage),
            "already_embedded_in_fill_price_pnl": True,
            "subtracted_again_from_realized_pnl": False,
        },
        "holding_vs_forecast_horizon": {
            "comparable_count": len(duration_ratios),
            "missing_horizon_or_duration_count": len(valid) - len(duration_ratios),
            "holding_exceeds_forecast_count": sum(value > 1 for value in duration_ratios),
            "maximum_ratio": max(duration_ratios, default=None),
            "causal_exit_loss_inferred": False,
        },
        "source_fingerprint": _fingerprint(sorted(
            (row["lifecycle_key"], row.get("outcome_fingerprint")) for row in rows
        )),
    }


def summarize_bills(
    raw_rows: list[dict[str, Any]], *, outcomes: list[dict[str, Any]]
) -> dict[str, Any]:
    order_ids = {
        str(order_id)
        for row in outcomes
        for order_id in [*(row.get("entry_order_ids") or []), *(row.get("close_order_ids") or [])]
    }
    seen: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        key = str(row.get("billId") or "")
        if not key:
            raise ValueError("official bill identity is required")
        if key in seen and seen[key] != row:
            raise ValueError("conflicting duplicate bill")
        seen[key] = row
    buckets: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    matched = 0
    excluded_currency = 0
    for row in seen.values():
        if str(row.get("ccy") or "").upper() != "USDT":
            excluded_currency += 1
            continue
        bill = _account_bill_from_row(row)
        if bill is None:
            raise ValueError("invalid native bill")
        funding = bill.bill_sub_type in {"173", "174"}
        kind = "funding" if funding else "trade" if bill.bill_type == "2" else "other"
        item = buckets[kind]
        item["count"] += 1
        item["cash_change_usdt"] += bill.balance_change
        item["price_pnl_usdt"] += 0.0 if funding else bill.pnl
        item["fees_signed_usdt"] += bill.fee
        item["funding_signed_usdt"] += bill.funding_fee
        if str(row.get("ordId") or "") in order_ids:
            matched += 1
    for item in buckets.values():
        item["unexplained_cash_residual_usdt"] = (
            item["cash_change_usdt"] - item["price_pnl_usdt"]
            - item["fees_signed_usdt"] - item["funding_signed_usdt"]
        )
    return {
        "source": "okx_account_bills_with_canonical_native_normalization",
        "unique_bill_count": len(seen),
        "identical_duplicate_count": len(raw_rows) - len(seen),
        "excluded_non_usdt_count": excluded_currency,
        "by_kind": dict(buckets),
        "trade_and_funding_cash_change_usdt": math.fsum(
            buckets[kind].get("cash_change_usdt", 0.0) for kind in ("trade", "funding")
        ),
        "bills_with_order_in_sampled_lifecycles": matched,
        "not_an_equity_or_completed_round_trip_return": True,
        "source_fingerprint": _fingerprint(sorted(seen.items())),
    }


async def fetch_bills(
    exchange: Any, *, since: datetime, until: datetime, max_pages: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read only; make pagination exhaustion explicit instead of assuming coverage."""
    rows = []
    cursor = ""
    cursors: set[str] = set()
    since_ms, until_ms = int(since.timestamp() * 1000), int(until.timestamp() * 1000)
    for page_number in range(1, max_pages + 1):
        params = {"instType": "SWAP", "ccy": "USDT", "limit": "100"}
        if cursor:
            params["after"] = cursor
        if page_number > 1:
            await asyncio.sleep(BILL_PAGE_INTERVAL_SECONDS)
        for delay in (*BILL_RATE_LIMIT_BACKOFF_SECONDS, None):
            try:
                response = await exchange.privateGetAccountBills(params)
            except ExchangeAPIError as exc:
                if exc.code != "50011":
                    raise
                response = {"code": exc.code}
            if not isinstance(response, dict) or str(response.get("code")) != "50011":
                break
            if delay is None:
                return rows, {
                    "complete": False, "pages": page_number - 1, "reason": "okx_rate_limited",
                }
            await asyncio.sleep(delay)
        if not isinstance(response, dict) or str(response.get("code")) != "0":
            raise ValueError("official bill request failed")
        page = response.get("data")
        if not isinstance(page, list):
            raise ValueError("invalid official bill page")
        timestamps = [int(row["ts"]) for row in page]
        if any(value <= 0 for value in timestamps):
            raise ValueError("invalid official bill timestamp")
        rows.extend(
            row for row, timestamp in zip(page, timestamps, strict=True)
            if since_ms <= timestamp < until_ms
        )
        reason = (
            "window_boundary_crossed" if timestamps and min(timestamps) < since_ms
            else "end_of_history" if len(page) < 100 else ""
        )
        if reason:
            return rows, {"complete": True, "pages": page_number, "reason": reason}
        cursor = _oldest_account_bill_pagination_cursor(page)
        if not cursor or cursor in cursors:
            return rows, {"complete": False, "pages": page_number, "reason": "cursor_stalled"}
        cursors.add(cursor)
    return rows, {"complete": False, "pages": max_pages, "reason": "page_budget_exhausted"}


def _opened_since(
    outcomes: list[dict[str, Any]], *, cutoff: datetime, until: datetime,
) -> list[dict[str, Any]]:
    # Canonical OKX outcomes use the position history open time for this field.
    return [
        row for row in outcomes
        if row.get("decision_timestamp") and cutoff <= _utc(row["decision_timestamp"]) < until
    ]


async def audit(
    *, until: datetime, hours: int, limit: int, bill_pages: int,
    opened_since: datetime | None = None,
) -> dict[str, Any]:
    from executor.okx_executor import OKXExecutor

    since = until - timedelta(hours=hours)
    outcomes = await load_authoritative_trade_outcomes(
        mode="paper", since=since, until=until, limit=limit + 1, compact=True,
        session_factory=partial(
            get_read_session_ctx, statement_timeout_ms=15_000,
            transaction_isolation="REPEATABLE READ READ ONLY",
        ),
    )
    canonical_rows_loaded = len(outcomes)
    lifecycle_limit_reached = canonical_rows_loaded > limit
    # Keep only dated settlement labels, even if history lookup fell back to open time.
    outcomes = [
        row for row in outcomes
        if row.get("label_timestamp") and since <= _utc(row["label_timestamp"]) < until
    ]
    outcomes = outcomes[-limit:]
    entry_evidence = await _load_entry_evidence(outcomes)
    executor = OKXExecutor(mode="paper", load_markets_on_initialize=False)
    try:
        exchange = await executor._get_ccxt()
        bills, coverage = await fetch_bills(
            exchange, since=since, until=until, max_pages=bill_pages
        )
    finally:
        await executor.shutdown()
    report = {
        "audit_version": "profitability-economics.v3",
        "generated_at": datetime.now(UTC).isoformat(),
        "window_start_utc": since.isoformat(),
        "window_end_utc_exclusive": until.isoformat(),
        "read_only": True,
        "production_permission": False,
        "profitability_acceptance": "not_evaluated",
        "lifecycle_limit": limit,
        "canonical_rows_loaded_before_cutoff_filter": canonical_rows_loaded,
        "lifecycle_limit_reached": lifecycle_limit_reached,
        "lifecycle_window_complete": not lifecycle_limit_reached,
        "lifecycles": summarize_lifecycles(outcomes, since=since, entry_evidence=entry_evidence),
        "bill_coverage": coverage,
        "bills": summarize_bills(bills, outcomes=outcomes),
    }
    if opened_since is not None:
        cohort = _opened_since(outcomes, cutoff=opened_since, until=until)
        report["opened_since_cohort"] = {
            "opened_at_utc_inclusive": opened_since.isoformat(),
            "unknown_open_time_count": sum(not row.get("decision_timestamp") for row in outcomes),
            "lifecycle_window_complete": not lifecycle_limit_reached,
            **summarize_lifecycles(cohort, since=opened_since, entry_evidence=entry_evidence),
        }
        if not lifecycle_limit_reached:
            report["observed_entry_filters"] = _observed_entry_filters(
                cohort, entry_evidence, await _load_settlement_clocks(cohort),
                start=opened_since, until=until,
            )
    return report


def audit_online(args: argparse.Namespace) -> dict[str, Any]:
    from core.remote_ssh import connect_remote_ssh, run_remote_text

    identity = args.until.strftime("%Y%m%dT%H%M%SZ")
    remote_output = f"/data/bb/app/data/profitability_remediation/economics-{identity}.json"
    command = [
        "systemd-run", "--quiet", "--wait", "--pipe", "--collect",
        f"--unit=bb-profitability-economics-{identity}",
        "--property=WorkingDirectory=/data/bb/app", "--property=User=bb",
        "--property=EnvironmentFile=-/data/bb/app/.env",
        "--property=EnvironmentFile=/etc/bb/bb-runtime.env",
        "--property=CPUQuota=100%", "--property=MemoryMax=1G",
        "--property=RuntimeMaxSec=240", "--property=TimeoutStopSec=10",
        "--property=KillMode=control-group",
        "--setenv=DATABASE_POOL_SIZE=1", "--setenv=DATABASE_MAX_OVERFLOW=0",
        "--setenv=OMP_NUM_THREADS=1", "--setenv=OPENBLAS_NUM_THREADS=1",
        "/data/bb/app/.venv/bin/python", "scripts/audit_profitability_economics.py",
        "--until", args.until.isoformat(), "--hours", str(args.hours),
        "--limit", str(args.limit), "--bill-pages", str(args.bill_pages),
        "--output", remote_output,
    ]
    if args.opened_since is not None:
        command.extend(("--opened-since", args.opened_since.isoformat()))
    # Capture diagnostic logs on the host; transport only the aggregate JSON.
    runner = (
        "import pathlib,subprocess,sys; "
        "result=subprocess.run(sys.argv[2:],capture_output=True); "
        "assert result.returncode == 0, 'bounded economics audit failed; inspect its named unit'; "
        "import json; print(json.dumps(json.loads(pathlib.Path(sys.argv[1]).read_bytes()), "
        "separators=(',', ':'), allow_nan=False))"
    )
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        output = run_remote_text(
            ssh, shlex.join(["/data/bb/app/.venv/bin/python", "-c", runner, remote_output, *command]),
            timeout=270, max_output_chars=60_000,
        )
    finally:
        ssh.close()
    report = json.loads(output)
    report["remote_report_path"] = remote_output
    report["remote_unit"] = command[5].removeprefix("--unit=")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until", required=True, type=_utc)
    parser.add_argument("--hours", type=int, choices=range(1, 169), default=24)
    parser.add_argument("--opened-since", type=_utc, help="Report a separate OKX open-time cohort.")
    parser.add_argument("--limit", type=int, choices=range(1, 501), default=120)
    parser.add_argument("--bill-pages", type=int, choices=range(1, 21), default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    if not datetime.now(UTC) - timedelta(days=6) <= args.until <= datetime.now(UTC):
        parser.error("cutoff must be within the past six days")
    if args.until - timedelta(hours=args.hours) < datetime.now(UTC) - timedelta(days=7):
        parser.error("window must fit the current-bills seven-day history")
    if args.opened_since is not None and not (
        args.until - timedelta(hours=args.hours) <= args.opened_since < args.until
    ):
        parser.error("--opened-since must be inside the frozen audit window")

    async def run() -> dict[str, Any]:
        try:
            return await asyncio.wait_for(
                audit(
                    until=args.until, hours=args.hours, limit=args.limit, bill_pages=args.bill_pages,
                    opened_since=args.opened_since,
                ),
                timeout=210,
            )
        finally:
            await close_db()

    report = audit_online(args) if args.online else asyncio.run(run())
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
