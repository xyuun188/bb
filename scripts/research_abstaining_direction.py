"""Fit a frozen, research-only continuation/reversal/hold candidate on paper observations."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

FEATURES = ("change_24h_pct", "returns_5", "returns_20", "volume_ratio", "volatility_20", "spread_pct")
SIDES = ("long", "short")
HORIZON = 240


def observation(row):
    if row.get("training_eligible") is not True:
        return None
    try:
        values = tuple(float(row["feature_snapshot"][k]) for k in FEATURES)
        returns = tuple(float(row[f"{s}_net_return_after_all_cost_pct"]) for s in SIDES)
    except (KeyError, TypeError, ValueError):
        return None
    if not all(map(math.isfinite, (*values, *returns))) or values[0] == 0:
        return None
    # Express both targets relative to the trend observable at the decision.
    trend = int(values[0] < 0)
    return {
        "x": values, "trend": trend, "net": (returns[trend], returns[1 - trend]),
        "id": row["source_id"], "decision": row["decision_id"], "symbol": row["symbol"],
        "at": row["decision_timestamp"], "label": row["label_timestamp"],
    }


def fit(rows):
    x = np.asarray([r["x"] for r in rows])
    net = np.asarray([r["net"] for r in rows])
    labels = np.where((net.max(axis=1) > 0) & (net[:, 0] != net[:, 1]), net.argmax(axis=1), 2)
    if set(labels) != {0, 1, 2}:
        raise ValueError("incomplete continuation/reversal/hold training classes")
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=32, random_state=42).fit(x, labels)
    leaves = tree.apply(x)
    actions = {}
    for leaf in np.unique(leaves):
        means = net[leaves == leaf].mean(axis=0)
        # A frequent small winner need not have positive expectancy.
        vote = int(np.bincount(labels[leaves == leaf], minlength=3).argmax())
        actions[int(leaf)] = vote if vote != 2 and means[vote] > 0 and means[vote] > means[1 - vote] else 2
    t = tree.tree_
    return {
        "features": FEATURES, "left": t.children_left.tolist(), "right": t.children_right.tolist(),
        "feature": t.feature.tolist(), "threshold": t.threshold.tolist(), "actions": actions,
        "train_count": len(rows), "classes": np.bincount(labels, minlength=3).tolist(),
        "training_sha256": digest(rows),
    }


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def predict(model, row):
    node = 0
    while model["left"][node] != -1:
        value = np.float32(row["x"][model["feature"][node]])
        node = model["left"][node] if float(value) <= model["threshold"][node] else model["right"][node]
    action = model["actions"][str(node)]
    return "hold" if action == 2 else SIDES[row["trend"] ^ action]


async def load(start, end):
    from sqlalchemy import select

    from db.session import get_read_session_ctx
    from models.learning import ShadowBacktest as S
    from services.ml_training_dataset import (
        crypto_shadow_candidate_filters,
        shadow_training_columns,
    )
    from services.strategy_learning import _build_historical_replay_observations
    from services.training_epoch import load_training_data_start

    result, excluded, seen = [], Counter(), set()
    cursor = 0
    epoch = load_training_data_start()
    async with get_read_session_ctx(statement_timeout_ms=30000, transaction_isolation="REPEATABLE READ READ ONLY") as db:
        while True:
            page = (await db.execute(select(*shadow_training_columns()).where(
                S.created_at >= start, S.created_at < end, S.due_at < end, S.id > cursor,
                S.id % 11 == 0, S.horizon_minutes == HORIZON, S.execution_mode == "paper",
                S.status == "completed", *crypto_shadow_candidate_filters(),
            ).order_by(S.id).limit(100))).mappings().all()
            if not page:
                break
            cursor = page[-1]["id"]
            items, reasons = _build_historical_replay_observations(
                [SimpleNamespace(**p) for p in page], epoch_start=epoch,
            )
            excluded.update(reasons)
            for item in items:
                row = observation(item)
                if row is None:
                    excluded["quality_or_features_incomplete"] += 1
                elif datetime.fromisoformat(row["label"]) >= end:
                    excluded["late_label"] += 1
                elif not row["decision"] or row["decision"] in seen:
                    excluded["missing_or_duplicate_decision"] += 1
                else:
                    seen.add(row["decision"])
                    result.append(row)
    return sorted(result, key=lambda r: (r["at"], r["id"])), dict(excluded)


async def evaluate(model, rows):
    from services.authoritative_trade_outcome import load_authoritative_trade_outcomes
    from services.ml_signal_service import ML_SIGNAL_ARTIFACT_REGISTRY, MLSignalService
    from services.strategy_learning import _return_metrics

    metrics, actions, lifecycle_pnl = {}, {}, {}
    # Evaluate the incumbent on full point-in-time features, never on the candidate's reduced vector.
    from sqlalchemy import select

    from db.session import get_read_session_ctx
    from models.learning import ShadowBacktest as S
    artifact = ML_SIGNAL_ARTIFACT_REGISTRY.resolve_current(load_bundle=False)
    service = MLSignalService(artifact_registry=ML_SIGNAL_ARTIFACT_REGISTRY, resolved_artifact=artifact)
    calls = {}
    async with get_read_session_ctx(statement_timeout_ms=30000) as db:
        for offset in range(0, len(rows), 64):
            group = rows[offset:offset + 64]
            raw = dict((await db.execute(select(S.id, S.training_feature_snapshot).where(
                S.id.in_([r["id"] for r in group]),
            ))).all())
            preds = service.predict_strategy_replay_batch(
                [{**raw[r["id"]], "symbol": r["symbol"]} for r in group], horizon_minutes=HORIZON,
            )
            for r, pred in zip(group, preds, strict=True):
                p = pred["predictions"][0]
                side = p["best_side"]
                calls[r["id"]] = side
    outcomes = await load_authoritative_trade_outcomes(
        mode="paper", since=datetime.fromisoformat(rows[0]["at"]), limit=500, compact=True,
        include_decision_evidence=True,
    ) if rows else []
    for name in ("candidate", "incumbent_direction", "no_trade"):
        samples, decisions, counts = [], {}, Counter()
        for r in rows:
            side = predict(model, r) if name == "candidate" else calls[r["id"]] if name == "incumbent_direction" else "hold"
            counts[side] += 1
            if side != "hold":
                decisions[r["decision"]] = (r["symbol"], side)
                samples.append({
                    "net_return_after_all_cost_pct": r["net"][r["trend"] ^ SIDES.index(side)],
                    "timestamp": r["at"],
                })
        metrics[name] = {k: v for k, v in _return_metrics(samples).items()
                         if k in {"average_net_return_pct", "profit_factor", "return_lcb_pct", "sample_count"}}
        actions[name] = dict(counts)
        matched = {
            o["outcome_id"]: o["realized_pnl"] for o in outcomes
            if o.get("trade_fact_trusted") is True and o.get("outcome_complete") is True
            and decisions.get(o.get("decision_id")) == (o.get("symbol"), o.get("side"))
        }
        lifecycle_pnl[name] = {"matching_settlements": len(matched), "observed_net_usdt": sum(matched.values()) if matched else None}
    return {"rows": len(rows), "input_sha256": digest(rows), "actions": actions, "horizon_labels": metrics,
            "matching_actual_paths_only": lifecycle_pnl, "incumbent": artifact.version,
            "full_execution_replay_available": False, "promotable": False}


async def run(args):
    start, split, end = [datetime.fromisoformat(t).astimezone(UTC) for t in (args.start, args.split, args.end)]
    if not start < split < end <= datetime.now(UTC):
        raise ValueError("invalid chronological bounds")
    path = args.artifact
    if path.exists():
        model = json.loads(path.read_text())
        if model["start"] != start.isoformat() or model["split"] != split.isoformat():
            raise ValueError("existing research artifact has different training dates")
    else:
        training, excluded = await load(start, split)
        if len(training) < 96:
            raise ValueError(f"insufficient qualified training decisions: {len(training)}; {excluded}")
        model = {**fit(training), "start": start.isoformat(), "split": split.isoformat(),
                 "frozen_at": datetime.now(UTC).isoformat(), "excluded": excluded}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as f:
            json.dump(model, f)
        model = json.loads(path.read_text())
    eval_start = max(split, datetime.fromisoformat(model["frozen_at"])) if args.forward else split
    rows, excluded = await load(eval_start, end)
    return {"artifact_sha256": digest(model), "trained": model["train_count"], "classes": model["classes"],
            "frozen_at": model["frozen_at"], "evaluation_start": eval_start.isoformat(), "end": end.isoformat(),
            "prospective": args.forward, "excluded": excluded, **await evaluate(model, rows)}


if __name__ == "__main__":
    from scripts.runtime_env_bootstrap import (
        drop_privileges_to_runtime_user_if_needed,
        load_runtime_env_files,
    )
    load_runtime_env_files(project_root=ROOT)
    drop_privileges_to_runtime_user_if_needed(project_root=ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("start", "split", "end"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--forward", action="store_true")
    print(json.dumps(asyncio.run(run(parser.parse_args())), allow_nan=False))
