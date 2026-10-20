"""Read-only, same-window diagnostics for immutable registered ML artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select

from db.session import get_read_session_ctx
from models.learning import ShadowBacktest
from services.ml_training_contract import PRIMARY_PREDICTION_HORIZON_MINUTES
from services.ml_training_dataset import shadow_training_columns
from services.strategy_learning import _build_historical_replay_observations, _return_metrics
from services.training_epoch import load_training_data_start


def _timestamp(value: Any) -> datetime:
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def common_forward_observations(
    observations: list[dict[str, Any]],
    *,
    cutoff: datetime,
    evaluated_at: datetime,
    horizon_minutes: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Freeze membership using source identity, quality and decision/label time."""

    excluded: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    identities: set[int] = set()
    for row in sorted(
        observations, key=lambda item: (str(item.get("decision_timestamp")), item["source_id"])
    ):
        try:
            decision = _timestamp(row.get("decision_timestamp"))
            label = _timestamp(row.get("label_timestamp"))
        except (TypeError, ValueError):
            excluded["invalid_timestamp"] += 1
            continue
        reason = (
            "not_forward"
            if decision <= cutoff
            else "label_not_mature"
            if label > evaluated_at or label <= decision
            else "horizon_mismatch"
            if row.get("horizon_minutes") != horizon_minutes
            else "quality_ineligible"
            if row.get("training_eligible") is not True
            else "duplicate_source"
            if row["source_id"] in identities
            else None
        )
        if reason is not None:
            excluded[reason] += 1
            continue
        identities.add(row["source_id"])
        selected.append(row)
    return selected, dict(excluded)


def summarize_forward_predictions(
    observations: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    expected_version: str,
    horizon_minutes: int,
) -> dict[str, Any]:
    """Evaluate frozen model decisions; never tune selectors on these outcomes."""

    if len(predictions) != len(observations):
        raise ValueError("common-window prediction count mismatch")
    samples: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    prediction_digest = hashlib.sha256()
    for row, prediction in zip(observations, predictions, strict=True):
        if (
            prediction.get("available") is not True
            or prediction.get("model_version") != expected_version
        ):
            raise ValueError("common-window prediction unavailable or model version mismatch")
        primary = next(
            (
                item
                for item in prediction.get("predictions", [])
                if item.get("horizon_minutes") == horizon_minutes
            ),
            {},
        )
        side = primary.get("best_side")
        if side not in {"long", "short"}:
            raise ValueError("common-window model side missing")
        contract = (primary.get("return_distribution_contract") or {}).get(side) or {}
        expected = _finite(contract.get("objective_expected_return_pct"))
        lower = _finite(contract.get("lower_quantile_return_pct"))
        cost = _finite(row.get("execution_cost_pct"))
        funding = _finite(row.get(f"{side}_funding_return_pct"))
        realized = _finite(row.get(f"{side}_net_return_after_all_cost_pct"))
        if None in (expected, lower, cost, funding, realized):
            raise ValueError("common-window fee-after evidence incomplete")
        net_expected = float(expected) - float(cost) + float(funding)
        net_lower = float(lower) - float(cost) + float(funding)
        cost_ready = (
            (primary.get("counterfactual_execution_cost_distribution") or {}).get(side) or {}
        ).get("distribution_ready") is True
        gate_passed = (
            contract.get("production_eligible") is True
            and cost_ready
            and net_expected > 0
            and net_lower > 0
        )
        if not gate_passed:
            failures["fee_after_distribution_gate_failed"] += 1
        sample = {
            "source_id": row["source_id"],
            "decision_id": row.get("decision_id"),
            "symbol": row["symbol"],
            "side": side,
            "timestamp": row["decision_timestamp"],
            "net_return_after_all_cost_pct": realized,
            "model_expected_net_return_pct": net_expected,
            "return_gate_passed": gate_passed,
        }
        samples.append(sample)
        prediction_digest.update(json.dumps(sample, sort_keys=True, allow_nan=False).encode())
    selected = [row for row in samples if row["return_gate_passed"]]
    return {
        "model_version": expected_version,
        "prediction_sha256": prediction_digest.hexdigest(),
        "prediction_count": len(samples),
        "decision_group_count": len(
            {row["decision_id"] or f"shadow:{row['source_id']}" for row in samples}
        ),
        "all_model_selected_directions": _return_metrics(samples),
        "fee_after_return_gate": _return_metrics(selected),
        "gate_pass_count": len(selected),
        "gate_pass_share": len(selected) / len(samples) if samples else 0.0,
        "gate_failures": dict(failures),
        "sides": {
            side: _return_metrics([row for row in samples if row["side"] == side])
            for side in ("long", "short")
        },
        "observed_prediction_bias_pct": (
            sum(
                row["model_expected_net_return_pct"] - row["net_return_after_all_cost_pct"]
                for row in samples
            )
            / len(samples)
            if samples
            else None
        ),
    }


async def run_model_forward_comparison(*, limit: int = 500) -> dict[str, Any]:
    from types import SimpleNamespace

    from services.ml_signal_service import ML_SIGNAL_ARTIFACT_REGISTRY, MLSignalService

    registry = ML_SIGNAL_ARTIFACT_REGISTRY
    artifacts = {
        "current": registry.resolve_current(load_bundle=False),
        "challenger": registry.resolve_challenger(load_bundle=False),
    }
    if any(artifact is None for artifact in artifacts.values()):
        raise ValueError("current and challenger must both be registered")
    cutoff = max(
        _timestamp(artifact.manifest[field])
        for artifact in artifacts.values()
        for field in ("trained_at", "created_at")
    )
    evaluated_at = datetime.now(UTC)
    horizon = PRIMARY_PREDICTION_HORIZON_MINUTES
    async with get_read_session_ctx() as session:
        inventory = await session.execute(
            select(
                ShadowBacktest.execution_mode,
                ShadowBacktest.status,
                ShadowBacktest.horizon_minutes,
                func.count(ShadowBacktest.id).label("sample_count"),
                func.max(ShadowBacktest.created_at).label("latest_decision_at"),
                func.max(ShadowBacktest.due_at).label("latest_label_at"),
            )
            .where(ShadowBacktest.created_at > cutoff.replace(tzinfo=None))
            .group_by(
                ShadowBacktest.execution_mode,
                ShadowBacktest.status,
                ShadowBacktest.horizon_minutes,
            )
        )
        inventory_rows = [
            {
                key: value.isoformat() if isinstance(value, datetime) else value
                for key, value in row.items()
            }
            for row in inventory.mappings()
        ]
        result = await session.execute(
            select(*shadow_training_columns())
            .where(
                ShadowBacktest.execution_mode == "paper",
                ShadowBacktest.status == "completed",
                ShadowBacktest.created_at > cutoff.replace(tzinfo=None),
                ShadowBacktest.due_at <= evaluated_at.replace(tzinfo=None),
                ShadowBacktest.horizon_minutes == horizon,
                ShadowBacktest.long_return_pct.is_not(None),
                ShadowBacktest.short_return_pct.is_not(None),
            )
            .order_by(ShadowBacktest.created_at.asc(), ShadowBacktest.id.asc())
            .limit(min(max(limit, 1), 2000))
        )
        rows = [SimpleNamespace(**dict(row)) for row in result.mappings()]
    observations, cost_exclusions = _build_historical_replay_observations(
        rows, epoch_start=load_training_data_start()
    )
    observations, exclusions = common_forward_observations(
        observations, cutoff=cutoff, evaluated_at=evaluated_at, horizon_minutes=horizon
    )
    sealed = json.dumps(
        observations, sort_keys=True, ensure_ascii=True, allow_nan=False, default=str
    ).encode()
    models = {}
    for role, artifact in artifacts.items():
        service = MLSignalService(artifact_registry=registry, resolved_artifact=artifact)
        features = [{**row["feature_snapshot"], "symbol": row["symbol"]} for row in observations]
        predictions = service.predict_strategy_replay_batch(features, horizon_minutes=horizon)
        models[role] = {
            "artifact_sha256": artifact.sha256,
            "trained_at": artifact.manifest["trained_at"],
            **summarize_forward_predictions(
                observations,
                predictions,
                expected_version=artifact.version,
                horizon_minutes=horizon,
            ),
        }
    return {
        "status": "complete" if observations else "insufficient_forward_evidence",
        "generated_at": evaluated_at.isoformat(),
        "comparison_scope": "same_cost_complete_post_training_decisions",
        "diagnostic_only": True,
        "promotion_authorized": False,
        "submits_orders": False,
        "forward_cutoff": cutoff.isoformat(),
        "horizon_minutes": horizon,
        "queried_row_count": len(rows),
        "post_training_inventory": inventory_rows,
        "sample_count": len(observations),
        "window_start": observations[0]["decision_timestamp"] if observations else None,
        "window_end": observations[-1]["decision_timestamp"] if observations else None,
        "source_ids": [row["source_id"] for row in observations],
        "input_sha256": hashlib.sha256(sealed).hexdigest(),
        "cost_exclusions": cost_exclusions,
        "exclusions": exclusions,
        "no_trade_baseline_return_pct": 0.0,
        "metric_limitations": "shadow opportunities, not executed portfolio PnL; block lower bounds are diagnostics",
        "models": models,
    }
