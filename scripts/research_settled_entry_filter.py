"""Research-only keep/abstain filter trained on observed, fully settled paper entries."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from core.feature_availability import (  # noqa: E402
    availability_timestamp,
    feature_availability_reasons,
)
from scripts.research_abstaining_direction import FEATURES, digest  # noqa: E402

VERSION = "observed-side-settled-net.v1"
INPUTS = ("side_sign", "aligned_change_24h", "aligned_returns_5", "aligned_returns_20", *FEATURES[3:])


def sample(row, mirror_time):
    if (row.get("execution_mode") != "paper" or row.get("close_status") != "full"
            or row.get("trade_fact_trusted") is not True or row.get("outcome_complete") is not True
            or row.get("strategy_entry_supervision_eligible") is not True
            or (row.get("realized_net_pnl_components") or {}).get("formula_consistent") is not True):
        return None, "incomplete_or_untrusted_settlement"
    decision = row.get("decision_id")
    if (not decision or row.get("entry_decision_ids") != [decision] or not row.get("lifecycle_key")
            or row.get("side") not in ("long", "short")):
        return None, "ambiguous_entry"
    opened, closed, receipt = map(availability_timestamp, (
        row.get("decision_timestamp"), row.get("label_timestamp"), mirror_time,
    ))
    if not opened or not closed or not receipt or closed < opened:
        return None, "invalid_settlement_clocks"
    features = row.get("features") or {}
    if feature_availability_reasons(features, decision_at=opened):
        return None, "entry_features_not_sealed"
    side = 1 if row["side"] == "long" else -1
    try:
        trend, r5, r20, volume, volatility, spread = (float(features[k]) for k in FEATURES)
        x = (side, side * trend, side * r5, side * r20, volume, volatility, spread)
        net = float(row["net_return_after_all_cost_pct"])
        pnl = float(row["realized_net_pnl_components"]["reported_realized_net_pnl_usdt"])
    except (KeyError, ValueError, TypeError):
        return None, "missing_input_or_net_return"
    if not all(map(math.isfinite, (*x, net, pnl))):
        return None, "nonfinite_input_or_net_return"
    return {
        "x": x, "net": net, "pnl": pnl, "decision": decision, "key": row["lifecycle_key"],
        "side": row["side"], "at": opened.isoformat(), "closed": closed.isoformat(),
        "known": max(closed, receipt).isoformat(), "stop_fill": bool(row.get("verified_stop_loss_fills")),
        "holding_minutes": (closed - opened).total_seconds() / 60,
    }, None


async def load(start, end):
    from sqlalchemy import select

    from db.session import get_read_session_ctx
    from models.trade import OkxPositionHistory as H
    from services.authoritative_trade_outcome import load_authoritative_trade_outcomes

    db = partial(get_read_session_ctx, statement_timeout_ms=20_000,
                 transaction_isolation="REPEATABLE READ READ ONLY")
    cursor = start
    result, reasons, seen = [], Counter(), set()
    while cursor < end:
        upper = min(cursor + timedelta(hours=12), end)
        page = await load_authoritative_trade_outcomes(
            mode="paper", since=cursor, until=upper, limit=161, compact=True,
            include_training_features=True, session_factory=db,
        )
        if len(page) >= 161:
            raise ValueError("settlement page budget exceeded; refusing incomplete research")
        async with db() as session:
            clocks = dict((await session.execute(select(H.row_identity, H.synced_at).where(
                H.mode == "paper", H.row_identity.in_([r["lifecycle_key"] for r in page]),
            ))).all())
        for row in page:
            key = row["lifecycle_key"]
            if key in seen:
                raise ValueError("overlapping lifecycle pages")
            seen.add(key)
            item, reason = sample(row, clocks.get(key))
            if reason:
                reasons[reason] += 1
            elif datetime.fromisoformat(item["at"]) < start or not (
                cursor <= datetime.fromisoformat(item["closed"]) < upper
            ):
                reasons["outside_open_or_settlement_window"] += 1
            else:
                result.append(item)
        cursor = upper
    counts = Counter(r["decision"] for r in result)
    reasons["repeated_entry_decision"] += sum(n for n in counts.values() if n > 1)
    return sorted((r for r in result if counts[r["decision"]] == 1),
                  key=lambda r: (r["at"], r["key"])), dict(reasons)


def fit(rows, start, split):
    # The exchange close time alone does not prove the mirror had the current facts then.
    train = [r for r in rows if start <= datetime.fromisoformat(r["at"])
             and datetime.fromisoformat(r["known"]) < split]
    if len(train) < 32 or {r["side"] for r in train} != {"long", "short"}:
        raise ValueError(f"insufficient two-sided settled training support: {len(train)}")
    x = np.asarray([r["x"] for r in train], dtype=float)
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.
    # Equal weight per observed entry; penalties/thresholds are fixed before evaluation.
    reg = Ridge(alpha=float(len(train))).fit((x - center) / scale, [r["net"] for r in train])
    return {
        "version": VERSION, "features": INPUTS, "start": start.isoformat(), "split": split.isoformat(),
        "frozen_at": datetime.now(UTC).isoformat(), "training_sha256": digest(train),
        "train_decisions": sorted(r["decision"] for r in train),
        "available_training_count": len(train), "excluded_after_cutoff": len(rows) - len(train),
        "center": center.tolist(), "scale": scale.tolist(), "weights": reg.coef_.tolist(),
        "intercept": float(reg.intercept_),
    }


def keep(model, x):
    score = model["intercept"] + math.fsum(
        w * (v - c) / s for v, c, s, w in zip(
            x, model["center"], model["scale"], model["weights"], strict=True,
        )
    )
    return math.isfinite(score) and score > 0


def evaluate(model, rows):
    from services.strategy_learning import _return_metrics

    if set(model["train_decisions"]).intersection(r["decision"] for r in rows):
        raise ValueError("training and evaluation decisions overlap")

    def stats(group):
        return _return_metrics([{
            "timestamp": r["at"], "net_return_after_all_cost_pct": r["net"],
            "net_pnl_after_all_costs_usdt": r["pnl"],
        } for r in sorted(group, key=lambda v: (v["closed"], v["key"]))])

    kept, skipped = [], []
    for r in rows:
        (kept if keep(model, r["x"]) else skipped).append(r)
    return {
        "all_observed_entries": stats(rows), "kept": stats(kept), "not_selected": stats(skipped),
        "kept_by_side": {s: stats([r for r in kept if r["side"] == s]) for s in ("long", "short")},
        "observed_exit_groups": {name: {
            "count": len(group), "net_usdt": math.fsum(r["pnl"] for r in group),
            "mean_holding_minutes": math.fsum(r["holding_minutes"] for r in group) / len(group)
            if group else None,
        } for name, group in (
            ("verified_stop_fill", [r for r in rows if r["stop_fill"]]),
            ("no_verified_stop_fill", [r for r in rows if not r["stop_fill"]]),
        )},
        "no_trade_net_usdt": 0, "evaluation_sha256": digest(rows), "promotable": False,
        "scope": "same_side_observed_fills_only_no_counterfactual_or_portfolio_replay",
    }


async def run(args):
    start, split, end = args.start, args.split, args.end
    if not (start and split and end and start < split < end <= datetime.now(UTC)
            and end - start <= timedelta(days=31)):
        raise ValueError("require aware ordered historical bounds within 31 days")
    if args.artifact.exists():
        model = json.loads(args.artifact.read_text(encoding="utf-8"))
        if (model["version"] != VERSION or tuple(model["features"]) != INPUTS
                or model["start"] != start.isoformat() or model["split"] != split.isoformat()):
            raise ValueError("frozen artifact does not match research contract")
    elif args.forward:
        raise ValueError("forward evaluation requires an existing frozen model")
    else:
        train, exclusions = await load(start, split)
        model = {**fit(train, start, split), "training_exclusions": exclusions}
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        with args.artifact.open("x", encoding="utf-8") as f:
            json.dump(model, f, allow_nan=False)
    evaluation_start = max(split, datetime.fromisoformat(model["frozen_at"])) if args.forward else split
    if end <= evaluation_start:
        raise ValueError("empty evaluation time window")
    rows, exclusions = await load(evaluation_start, end)
    known_rows = [row for row in rows if datetime.fromisoformat(row["known"]) < end]
    return {
        "artifact_sha256": digest(model), "frozen_at": model["frozen_at"],
        "evaluation_from": evaluation_start.isoformat(), "evaluation_until": end.isoformat(),
        "prospective": args.forward, "trained": model["available_training_count"],
        "training_exclusions": model["training_exclusions"],
        "training_mirror_after_cutoff": model["excluded_after_cutoff"],
        "evaluation_exclusions": {
            **exclusions, "mirror_at_or_after_cutoff": len(rows) - len(known_rows),
        },
        **evaluate(model, known_rows),
    }


if __name__ == "__main__":
    from scripts.runtime_env_bootstrap import (
        drop_privileges_to_runtime_user_if_needed,
        load_runtime_env_files,
    )
    load_runtime_env_files(project_root=ROOT)
    drop_privileges_to_runtime_user_if_needed(project_root=ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("start", "split", "end"):
        parser.add_argument("--" + option, required=True, type=availability_timestamp)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--forward", action="store_true")
    print(json.dumps(asyncio.run(run(parser.parse_args())), allow_nan=False))
