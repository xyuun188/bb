"""Same-model feedback for paper training entries, using completed OKX lifecycles."""

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from math import isclose, isfinite

WINDOW = timedelta(days=1)
MIN_LIFECYCLES = 4
ROW_LIMIT = 500


def _key(version, side, horizon, family="local_ml"):
    try:
        minutes = float(horizon)
    except (TypeError, ValueError):
        return None
    if (family not in {"local_ml", "local_ai_tools"} or not isinstance(version, str)
            or not version.strip() or "|" in version or side not in {"long", "short"}
            or not isfinite(minutes) or minutes <= 0):
        return None
    prefix = "local_ai_tools|" if family == "local_ai_tools" else ""
    return f"{prefix}{version}|{side}|{minutes:g}"


def build_training_profiles(rows, *, now):
    groups = defaultdict(dict)
    for row in rows:
        key = _key(row["version"], row["side"], row["horizon"], row.get("family", "local_ml"))
        if (key and row["symbol"] and now - WINDOW < row["closed"] <= now
                and row["pnl"] is not None and isfinite(row["pnl"])):
            groups[key][row["id"]] = row
    profiles = {}
    for key, unique in groups.items():
        by_symbol = defaultdict(float)
        for row in unique.values():
            by_symbol[row["symbol"]] += row["pnl"]
        profiles[key] = {
            "sample_count": len(unique), "symbol_count": len(by_symbol),
            "losing_symbols": sum(pnl < 0 for pnl in by_symbol.values()),
            "net_usdt": sum(by_symbol.values()),
        }
    return {"status": "complete", "as_of": now.isoformat(), "profiles": profiles}


def training_feedback(snapshot, *, version, side, horizon, family="local_ml"):
    key = _key(version, side, horizon, family)
    available = bool(key and snapshot and snapshot.get("status") == "complete"
                     and isinstance(snapshot.get("profiles"), dict))
    row = snapshot["profiles"].get(key, {}) if available else {}
    loss = (row.get("sample_count", 0) >= MIN_LIFECYCLES and row["net_usdt"] < 0
            and row["losing_symbols"] >= 2
            and 2 * row["losing_symbols"] > row["symbol_count"])
    return {
        "cohort": key, "available": available, **row, "blocked": not available or loss,
        "reason": "training_settlements_unavailable" if not available else
        "cross_symbol_training_loss" if loss else "no_cross_symbol_training_loss",
    }


def _settled_model_versions(families, versions, local_version, profit_version, timeseries_version):
    """Use frozen selection lineage, with a bounded fallback for older single-family entries."""
    if not isinstance(families, list):
        return {}
    if isinstance(versions, dict):
        return {
            family: version for family, version in versions.items()
            if family in families and _key(version, "long", 1, family) is not None
        }
    if families == ["local_ml"]:
        return {"local_ml": local_version}
    if families == ["local_ai_tools"] and profit_version and profit_version == timeseries_version:
        return {"local_ai_tools": profit_version}
    return {}


async def load_paper_training_profiles():
    from sqlalchemy import func, select

    from db.session import get_read_session_ctx
    from models.decision import AIDecision as D
    from models.trade import OkxPositionHistory as H
    from models.trade import Order

    now = datetime.now(UTC)
    async with get_read_session_ctx(
        statement_timeout_ms=4000, transaction_isolation="REPEATABLE READ READ ONLY",
    ) as session:
        histories = (await session.execute(select(
            H.id, H.symbol, H.side, H.updated_at_okx.label("closed"), H.realized_pnl.label("pnl"),
            H.entry_order_ids, H.pnl.label("gross"), H.fee, H.funding_fee,
            H.raw_row["liqPenalty"].as_string().label("penalty"),
        ).where(
            H.mode == "paper", H.sync_status == "synced", H.close_status == "full",
            H.updated_at_okx > now - WINDOW, H.updated_at_okx <= now,
        ).limit(ROW_LIMIT + 1))).all()
        if len(histories) > ROW_LIMIT:
            return {"status": "incomplete"}
        single = {
            h.id: str(h.entry_order_ids[0]) for h in histories
            if isinstance(h.entry_order_ids, list) and len(h.entry_order_ids) == 1
        }
        if not single:
            return build_training_profiles([], now=now)
        snapshot = D.decision_learning_snapshot
        raw = D.raw_llm_response
        contract = snapshot["normal_paper_trade"]

        def recorded_tool_version(route):
            return func.coalesce(
                snapshot["local_ai_tools"][route]["model_version"].as_string(),
                raw["local_ai_tools"][route]["model_version"].as_string(),
            )

        entries = (await session.execute(select(
            Order.exchange_order_id, D.symbol, D.action, contract["prediction_horizon_minutes"],
            snapshot["ml_signal"]["model_version"].as_string(),
            contract["quant_evidence_families"],
            contract["recent_side_loss_guard"]["training_model_versions"],
            recorded_tool_version("profit_prediction"),
            recorded_tool_version("time_series_prediction"),
            Order.symbol, Order.side,
        ).join_from(Order, D, D.id == Order.decision_id).where(
            Order.execution_mode == "paper", D.is_paper.is_(True), Order.status == "filled",
            Order.exchange_order_id.in_(set(single.values())),
            contract["selection_reason"].as_string() == "paper_training_entry",
            contract["authorized"].as_boolean().is_(True),
        ))).all()
    by_order = defaultdict(list)
    for entry in entries:
        by_order[entry[0]].append(entry)
    rows = []
    for h in histories:
        candidates = by_order.get(single.get(h.id), [])
        if len(candidates) != 1:
            continue
        (_, symbol, side, horizon, local_version, families, versions,
         profit_version, timeseries_version, order_symbol, order_side) = candidates[0]
        if (symbol != h.symbol or side != h.side or order_symbol != h.symbol
                or {"long": "buy", "short": "sell"}.get(side) != order_side):
            continue
        try:
            penalty = float(h.penalty or 0)
        except (TypeError, ValueError):
            continue
        if not isclose(h.gross + h.fee + h.funding_fee + penalty,
                       h.pnl, rel_tol=1e-5, abs_tol=1e-6):
            continue
        # A combined trade cannot establish which model family drove its PnL.
        if not isinstance(families, list) or len(families) != 1:
            continue
        for family, version in _settled_model_versions(
            families, versions, local_version, profit_version, timeseries_version,
        ).items():
            rows.append(dict(id=h.id, symbol=symbol, side=side, closed=h.closed,
                             pnl=h.pnl, horizon=horizon, version=version, family=family))
    return build_training_profiles(rows, now=now)
