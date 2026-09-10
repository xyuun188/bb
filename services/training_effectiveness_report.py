"""Read-only, reproducible training-effectiveness report contract.

The report service deliberately has no training, evaluation, model-promotion, or
trading side effects.  Providers are injected so the contract can be tested with
fixed fixtures before connecting it to production read models.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.settings import settings

TRAINING_EFFECTIVENESS_REPORT_VERSION = "2026-09-11.v3"
TRAINING_EFFECTIVENESS_REPORT_DIRNAME = "training_effectiveness_reports"
TRAINING_EFFECTIVENESS_REPORT_STATUSES = {
    "complete",
    "partial",
    "invalid",
    "missing",
    "generation_failed",
}
SAMPLE_AUTHORITIES = {
    "shadow_opportunity",
    "counterfactual_cost",
    "okx_realized",
    "excluded",
}
REPORT_STALE_AFTER_SECONDS = 24 * 60 * 60


class AuthoritativeSamples(list[dict[str, Any]]):
    """List-compatible provider result with an explicit load state.

    Keeping this list-compatible preserves the injected provider contract while
    allowing the report builder to distinguish a real empty result from a
    database/query failure.
    """

    def __init__(
        self,
        values: list[dict[str, Any]] | None = None,
        *,
        load_status: str = "complete",
        error_code: str | None = None,
    ) -> None:
        super().__init__(values or [])
        self.load_status = load_status
        self.error_code = error_code


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _canonical_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def build_input_fingerprint(inputs: Any) -> str:
    """Return a stable SHA-256 fingerprint for report inputs."""

    payload = json.dumps(
        _canonical_value(inputs),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def generation_failure_report_path(
    *, data_dir: Path | None = None, mode: str = "all"
) -> Path:
    """Return the sidecar path used for the latest failed generation attempt."""

    selected_mode = mode if mode in {"paper", "live"} else "all"
    suffix = f"-{selected_mode}" if selected_mode != "all" else ""
    return report_directory(data_dir) / f"latest{suffix}-generation-failed.json"


def load_generation_failure_report(
    *, data_dir: Path | None = None, mode: str = "all"
) -> dict[str, Any] | None:
    """Load the latest failure sidecar without mutating the valid report cache."""

    path = generation_failure_report_path(data_dir=data_dir, mode=mode)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) and payload.get("status") == "generation_failed" else None


def build_generation_failed_report(
    *,
    filters: dict[str, Any] | None = None,
    run_id: str | None = None,
    input_fingerprint: str | None = None,
    error_code: str = "generation_failed",
    stage: str = "baseline",
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build an explicit failure artifact without inventing financial values."""

    selected_filters = dict(filters or {})
    generated = (generated_at or datetime.now(UTC)).replace(microsecond=0)
    fingerprint = input_fingerprint or build_input_fingerprint(
        {
            "report_version": TRAINING_EFFECTIVENESS_REPORT_VERSION,
            "filters": selected_filters,
            "error_code": error_code,
        }
    )
    report_token = (run_id or fingerprint[7:19]).replace("/", "-")
    now = generated.isoformat().replace("+00:00", "Z")
    return {
        "report_version": TRAINING_EFFECTIVENESS_REPORT_VERSION,
        "report_id": f"te-{report_token}-generation-failed",
        "generated_at": now,
        "data_cutoff_at": selected_filters.get("to") or now,
        "status": "generation_failed",
        "input_fingerprint": fingerprint,
        "run": {"run_id": run_id or report_token, "stage": stage},
        "versions": {},
        "filters": selected_filters,
        "metrics": {},
        "cost_attribution": {
            "known": False,
            "gross_pnl": None,
            "fee": None,
            "slippage": None,
            "funding_fee": None,
            "realized_net_pnl": None,
            "estimated_net_pnl": None,
            "fee_after_net_pnl": None,
            "net_basis": "unknown",
        },
        "expert_contributions": [],
        "execution_funnel": {},
        "sample_quality": {
            "load_status": "generation_failed",
            "load_error_code": error_code,
            "valid_sample_count": None,
        },
        "conclusion": {
            "promotion_eligible": False,
            "blocking_reasons": [error_code],
        },
        "freshness": {
            "state": "timeout"
            if error_code in {"query_timeout", "generation_timeout"}
            else "failed",
            "is_stale": True,
        },
    }


def calculate_fee_after_return(
    gross_pnl: Any,
    fee: Any,
    slippage: Any,
    funding_fee: Any,
) -> float:
    """Calculate ``gross pnl - fee - slippage + funding fee``."""

    return round(
        _finite_float(gross_pnl)
        - _finite_float(fee)
        - _finite_float(slippage)
        + _finite_float(funding_fee),
        8,
    )


def _metric_comparison(left: Any, right: Any) -> dict[str, float | None]:
    left_value = _finite_float(left)
    right_value = _finite_float(right)
    delta = round(right_value - left_value, 8)
    denominator = abs(left_value)
    percentage = round(delta / denominator * 100.0, 8) if denominator else None
    return {"absolute": delta, "percentage": percentage}


def calculate_metric_delta(
    active: Any,
    challenger: Any,
    baseline: Any,
) -> dict[str, dict[str, float | None]]:
    """Compare each right-hand metric to the preceding left-hand metric.

    Percentages are ``delta / abs(left)`` and are ``None`` when the denominator
    is zero, avoiding misleading infinity values.
    """

    return {
        "active_vs_challenger": _metric_comparison(active, challenger),
        "active_vs_baseline": _metric_comparison(active, baseline),
        "challenger_vs_baseline": _metric_comparison(challenger, baseline),
    }


def classify_sample_authority(sample: dict[str, Any]) -> str:
    """Classify one sample into the four report authority buckets."""

    if not isinstance(sample, dict):
        return "excluded"
    if sample.get("excluded") is True:
        return "excluded"
    explicit = str(
        sample.get("authority")
        or sample.get("sample_authority")
        or sample.get("authority_class")
        or ""
    ).strip().lower()
    aliases = {
        "shadow": "shadow_opportunity",
        "shadow_opportunity": "shadow_opportunity",
        "counterfactual": "counterfactual_cost",
        "counterfactual_cost": "counterfactual_cost",
        "okx": "okx_realized",
        "okx_realized": "okx_realized",
        "realized": "okx_realized",
        "excluded": "excluded",
    }
    if explicit in aliases:
        return aliases[explicit]
    source = str(sample.get("source") or sample.get("pnl_source") or "").lower()
    if sample.get("counterfactual") is True or "counterfactual" in source:
        return "counterfactual_cost"
    if sample.get("shadow") is True or "shadow" in source:
        return "shadow_opportunity"
    if (
        sample.get("outcome_complete") is True
        or sample.get("settlement_complete") is True
        or "okx" in source
        or "realized" in source
    ):
        return "okx_realized"
    return "excluded"


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str) and value.strip():
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if result.tzinfo is None:
        return result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def validate_report(report: dict[str, Any]) -> list[str]:
    """Validate immutable report structure and accounting invariants."""

    errors: list[str] = []
    required = (
        "report_version",
        "report_id",
        "generated_at",
        "data_cutoff_at",
        "status",
        "input_fingerprint",
        "run",
        "versions",
        "filters",
        "metrics",
        "cost_attribution",
        "expert_contributions",
        "execution_funnel",
        "sample_quality",
        "conclusion",
        "freshness",
    )
    errors.extend(f"missing:{key}" for key in required if key not in report)
    if report.get("report_version") != TRAINING_EFFECTIVENESS_REPORT_VERSION:
        errors.append("report_version_mismatch")
    if report.get("status") not in TRAINING_EFFECTIVENESS_REPORT_STATUSES - {"missing"}:
        errors.append("invalid_status")
    if not str(report.get("report_id") or "").strip():
        errors.append("missing:report_id")
    fingerprint = str(report.get("input_fingerprint") or "")
    if not fingerprint.startswith("sha256:") or len(fingerprint) != 71:
        errors.append("invalid:input_fingerprint")
    generated = _parse_datetime(report.get("generated_at"))
    cutoff = _parse_datetime(report.get("data_cutoff_at"))
    if generated is None:
        errors.append("invalid:generated_at")
    if cutoff is None:
        errors.append("invalid:data_cutoff_at")
    if generated is not None and cutoff is not None and cutoff > generated:
        errors.append("invalid:time_order")
    costs = report.get("cost_attribution")
    if isinstance(costs, dict):
        if costs.get("known") is not False and costs.get("net_basis") != "unknown":
            gross = _finite_float(costs.get("gross_pnl"))
            fee = _finite_float(costs.get("fee"))
            slippage = _finite_float(costs.get("slippage"))
            funding = _finite_float(costs.get("funding_fee"))
            actual = _finite_float(costs.get("fee_after_net_pnl"), math.nan)
            if costs.get("net_basis") == "authoritative_realized":
                expected = _finite_float(costs.get("realized_net_pnl"), math.nan)
                if math.isnan(expected) or math.isnan(actual) or not math.isclose(
                    actual, expected, abs_tol=1e-7
                ):
                    errors.append("invalid:authoritative_net_pnl_equation")
            elif costs.get("net_basis") == "mixed_authoritative_and_estimated":
                expected = _finite_float(costs.get("realized_net_pnl"), math.nan) + _finite_float(
                    costs.get("estimated_net_pnl"), math.nan
                )
                if math.isnan(expected) or math.isnan(actual) or not math.isclose(
                    actual, expected, abs_tol=1e-7
                ):
                    errors.append("invalid:mixed_net_pnl_equation")
            else:
                expected = calculate_fee_after_return(gross, fee, slippage, funding)
                if math.isnan(actual) or not math.isclose(actual, expected, abs_tol=1e-7):
                    errors.append("invalid:cost_attribution_equation")
    return list(dict.fromkeys(errors))


def report_directory(data_dir: Path | None = None) -> Path:
    return Path(data_dir or settings.data_dir) / TRAINING_EFFECTIVENESS_REPORT_DIRNAME


def load_cached_training_effectiveness_report(
    *, report_id: str | None = None, data_dir: Path | None = None
) -> dict[str, Any]:
    """Load a cached report only; never generate or mutate one."""

    root = report_directory(data_dir)
    if report_id and not re.fullmatch(r"[A-Za-z0-9._-]+", report_id):
        return {
            "report_version": TRAINING_EFFECTIVENESS_REPORT_VERSION,
            "status": "missing",
            "report_id": report_id,
            "input_fingerprint": None,
            "generated_at": None,
            "data_cutoff_at": None,
            "freshness": {"state": "missing", "is_stale": True},
        }
    safe_report_id = report_id
    path = root / (f"{safe_report_id}.json" if safe_report_id else "latest.json")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {
            "report_version": TRAINING_EFFECTIVENESS_REPORT_VERSION,
            "status": "missing",
            "report_id": report_id,
            "input_fingerprint": None,
            "generated_at": None,
            "data_cutoff_at": None,
            "freshness": {"state": "missing", "is_stale": True},
        }
    if not isinstance(payload, dict):
        return {"status": "invalid"}
    generated = _parse_datetime(payload.get("generated_at"))
    stale = generated is None or (datetime.now(UTC) - generated).total_seconds() > REPORT_STALE_AFTER_SECONDS
    freshness = dict(payload.get("freshness") or {})
    freshness["state"] = "stale" if stale else "fresh"
    freshness["is_stale"] = stale
    payload = dict(payload)
    payload["freshness"] = freshness
    return payload


def apply_report_filters(report: dict[str, Any], **filters: Any) -> dict[str, Any]:
    """Apply filters and recompute metrics from the cached sample projection."""

    if not isinstance(report, dict):
        return {"status": "invalid"}
    result = dict(report)
    current = dict(result.get("filters") or {})
    for key, value in filters.items():
        if value is not None and str(value).strip():
            current[key] = value
    result["filters"] = current
    sample_index = result.get("sample_index")
    if not isinstance(sample_index, list):
        return result

    selected = [
        row
        for row in sample_index
        if isinstance(row, dict) and _sample_matches_filters(row, current)
    ]
    authoritative = [row for row in selected if _is_realized_sample(row)]
    versions = result.get("versions") if isinstance(result.get("versions"), dict) else {}
    active_id = str((versions.get("active") or {}).get("model_id") or "active")
    challenger_id = str((versions.get("challenger") or {}).get("model_id") or "challenger")
    baseline = _aggregate_metrics(authoritative, "baseline")
    active = _aggregate_metrics(authoritative, active_id)
    challenger = _aggregate_metrics(authoritative, challenger_id)
    result["metrics"] = {
        "active": active,
        "challenger": challenger,
        "baseline": baseline,
        "observed": _aggregate_metrics(authoritative, "__all__"),
        "delta": calculate_metric_delta(
            active.get("fee_after_net_pnl"),
            challenger.get("fee_after_net_pnl"),
            baseline.get("fee_after_net_pnl"),
        ),
    }
    result["sample_quality"] = {
        **dict(result.get("sample_quality") or {}),
        "filtered_sample_count": len(selected),
        "filtered_valid_sample_count": len(authoritative),
    }
    result["cost_attribution"] = _cost_attribution(authoritative)
    return result


def _matches_report_filter(row: dict[str, Any], key: str, value: Any) -> bool:
    normalized = str(value or "all").strip().lower()
    if not normalized or normalized == "all":
        return True
    actual = str(row.get(key) or "").strip().lower()
    return actual == normalized


def _sample_matches_filters(sample: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Apply all report filters to the compact sample projection."""

    for key in ("mode", "side", "symbol", "market_state"):
        if not _matches_report_filter(sample, key, filters.get(key)):
            return False
    label_time = _parse_datetime(sample.get("label_timestamp"))
    lower = _parse_datetime(filters.get("from"))
    upper = _parse_datetime(filters.get("to"))
    if lower is not None and (label_time is None or label_time < lower):
        return False
    if upper is not None and (label_time is None or label_time > upper):
        return False
    hold_filter = filters.get("hold_minutes")
    if (
        isinstance(hold_filter, dict)
        and hold_filter
        and any(
            value not in (None, "", 0, "0", 0.0)
            for value in (hold_filter.get("min"), hold_filter.get("max"))
        )
    ):
        try:
            hold = float(sample.get("hold_minutes"))
        except (TypeError, ValueError):
            return False
        minimum = hold_filter.get("min")
        maximum = hold_filter.get("max")
        if minimum not in (None, "") and hold < float(minimum):
            return False
        if maximum not in (None, "", 0) and hold > float(maximum):
            return False
    return True


def _sample_projection(sample: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields required to reproduce report filters and metrics.

    Authoritative outcomes may carry decision prompts, feature snapshots and
    other large evidence payloads.  Persisting those in the dashboard cache
    makes filtering expensive and risks turning a read-only report into a
    second training-data store.  The projection is intentionally explicit.
    """

    keys = (
        "id",
        "authority",
        "outcome_complete",
        "model",
        "model_id",
        "execution_mode",
        "mode",
        "side",
        "symbol",
        "market_state",
        "label_timestamp",
        "hold_minutes",
        "gross_pnl",
        "fee",
        "slippage",
        "funding_fee",
        "realized_net_pnl",
        "realized_pnl",
        "official_realized_net_pnl",
        "excluded",
    )
    return {key: sample.get(key) for key in keys if key in sample}


def _normalise_report_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Normalize provider rows into the report's stable filter/metric schema."""

    result = dict(sample)
    result["model"] = (
        result.get("model")
        or result.get("model_id")
        or result.get("model_name")
        or "active"
    )
    result["mode"] = result.get("mode") or result.get("execution_mode")
    result["hold_minutes"] = result.get("hold_minutes")
    if result["hold_minutes"] in (None, ""):
        result["hold_minutes"] = result.get("holding_minutes")
    result["label_timestamp"] = (
        result.get("label_timestamp")
        or result.get("closed_at")
        or result.get("updated_at")
        or result.get("decision_timestamp")
    )
    result["market_state"] = (
        result.get("market_state")
        or result.get("market_regime")
        or result.get("current_market_regime")
    )
    return result


def _sample_net_pnl(sample: dict[str, Any]) -> float:
    """Use official realized net PnL when available; otherwise estimate costs."""

    authority = classify_sample_authority(sample)
    if authority == "okx_realized":
        for key in ("realized_net_pnl", "realized_pnl", "official_realized_net_pnl"):
            value = sample.get(key)
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return round(number, 8)
    return calculate_fee_after_return(
        sample.get("gross_pnl"),
        sample.get("fee"),
        sample.get("slippage"),
        sample.get("funding_fee"),
    )


def _authoritative_net_pnl(sample: dict[str, Any]) -> float | None:
    """Return a finite official net PnL value, if the row carries one."""

    if classify_sample_authority(sample) != "okx_realized":
        return None
    for key in ("realized_net_pnl", "realized_pnl", "official_realized_net_pnl"):
        try:
            value = float(sample.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return round(value, 8)
    return None


def _is_realized_sample(sample: dict[str, Any]) -> bool:
    """Only a complete authority row with an official net result is realized."""

    return _authoritative_net_pnl(sample) is not None


def _cost_attribution(samples: list[dict[str, Any]]) -> dict[str, Any]:
    realized = [row for row in samples if _is_realized_sample(row)]
    realized_ids = {id(row) for row in realized}
    estimated = [row for row in samples if id(row) not in realized_ids]
    realized_net = sum(_sample_net_pnl(row) for row in realized)
    estimated_net = sum(_sample_net_pnl(row) for row in estimated)
    return {
        "gross_pnl": round(sum(_finite_float(row.get("gross_pnl")) for row in samples), 8),
        "fee": round(sum(_finite_float(row.get("fee")) for row in samples), 8),
        "slippage": round(sum(_finite_float(row.get("slippage")) for row in samples), 8),
        "funding_fee": round(sum(_finite_float(row.get("funding_fee")) for row in samples), 8),
        "realized_net_pnl": round(realized_net, 8),
        "estimated_net_pnl": round(estimated_net, 8),
        "fee_after_net_pnl": round(realized_net + estimated_net, 8),
        "net_basis": (
            "authoritative_realized"
            if realized and len(realized) == len(samples)
            else ("mixed_authoritative_and_estimated" if realized else "estimated_cost_model")
        ),
        "slippage_included_in_realized_gross": bool(realized),
    }


def _invoke(provider: Callable[..., Any], *args: Any, **kwargs: Any) -> Awaitable[Any]:
    result = provider(*args, **kwargs)
    if inspect.isawaitable(result):
        return result

    async def _ready() -> Any:
        return result

    return _ready()


def _select_versions(
    registry: dict[str, Any], *, observed_model_id: str | None = None
) -> dict[str, Any]:
    rows = registry.get("models") if isinstance(registry, dict) else []
    rows = rows if isinstance(rows, list) else []
    active = next(
        (row for row in rows if str(row.get("lifecycle")) in {"active", "live"}),
        None,
    )
    challenger = next(
        (
            row
            for row in rows
            if str(row.get("lifecycle")) in {"canary", "trained", "promotion_blocked"}
        ),
        None,
    )
    inferred_active = None
    if not active and observed_model_id:
        inferred_active = {
            "model_id": observed_model_id,
            "version": observed_model_id,
            "lifecycle": "inferred_from_authoritative_samples",
            "status": "inferred",
            "source": "authoritative_trade_outcomes",
        }
    return {
        "active": active or inferred_active or {"version": None, "status": "missing"},
        "challenger": challenger or {"version": None, "status": "missing"},
        "baseline": {"version": "no_model_baseline", "status": "defined"},
    }


def _aggregate_metrics(samples: list[dict[str, Any]], model: str) -> dict[str, Any]:
    selected = [
        sample
        for sample in samples
        for sample_model in [str(sample.get("model") or sample.get("model_id") or "baseline")]
        if model == "__all__" or sample_model == model
    ]
    gross = sum(_finite_float(row.get("gross_pnl")) for row in selected)
    fee = sum(_finite_float(row.get("fee")) for row in selected)
    slippage = sum(_finite_float(row.get("slippage")) for row in selected)
    funding = sum(_finite_float(row.get("funding_fee")) for row in selected)
    net = round(sum(_sample_net_pnl(row) for row in selected), 8)
    pnl_values = [_sample_net_pnl(row) for row in selected]
    wins = sum(1 for value in pnl_values if value > 0)
    gross_profit = sum(value for value in pnl_values if value > 0)
    gross_loss = abs(sum(value for value in pnl_values if value < 0))
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in pnl_values:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    mean = sum(pnl_values) / len(pnl_values) if pnl_values else 0.0
    variance = (
        sum((value - mean) ** 2 for value in pnl_values) / (len(pnl_values) - 1)
        if len(pnl_values) > 1
        else 0.0
    )
    standard_error = math.sqrt(variance / len(pnl_values)) if pnl_values else 0.0
    return {
        "sample_count": len(selected),
        "gross_pnl": round(gross, 8),
        "fee": round(fee, 8),
        "slippage": round(slippage, 8),
        "funding_fee": round(funding, 8),
        "fee_after_net_pnl": net,
        "win_rate": round(wins / len(selected), 8) if selected else None,
        "profit_factor": round(gross_profit / gross_loss, 8) if gross_loss else None,
        "return_lower_bound": round(mean - 1.96 * standard_error, 8) if selected else None,
        "max_drawdown": round(max_drawdown, 8) if selected else None,
        "worst_pnl": round(min(pnl_values), 8) if pnl_values else None,
    }


async def _load_expert_contributions(*, filters: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapt the existing realized contribution buckets for the audit report."""

    try:
        from services.model_contribution_performance import ModelContributionPerformanceService

        modes = [str(filters.get("mode") or "all").lower()]
        if modes == ["all"]:
            modes = ["paper", "live"]
        merged: dict[str, dict[str, Any]] = {}
        service = ModelContributionPerformanceService()
        for mode in modes:
            if mode not in {"paper", "live"}:
                continue
            buckets = await asyncio.wait_for(service.recent(mode), timeout=8.0)
            for key, bucket in (buckets or {}).items():
                if not str(key).startswith("expert:") or not isinstance(bucket, dict):
                    continue
                row = merged.setdefault(
                    key,
                    {
                        "expert_name": key.split(":", 1)[1],
                        "expert_label": bucket.get("label") or key.split(":", 1)[1],
                        "sample_count": 0,
                        "net_pnl_delta": 0.0,
                        "drawdown_delta": 0.0,
                        "false_entry_delta": 0.0,
                        "side_balance_delta": 0.0,
                    },
                )
                row["sample_count"] += int(bucket.get("count") or 0)
                row["net_pnl_delta"] += _finite_float(bucket.get("pnl"))
                row["drawdown_delta"] += _finite_float(bucket.get("max_drawdown_usdt"))
        return list(merged.values())
    except Exception:
        return []


def _build_observed_funnel(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose an honest settlement funnel derived from authoritative outcomes."""

    total = len(samples)
    settled = sum(1 for row in samples if _is_realized_sample(row))
    stages = {
        "signals": total,
        "evidence_passed": total,
        "risk_passed": total,
        "orders_submitted": total,
        "filled": settled,
        "positions_opened": settled,
        "closed": settled,
        "settled": settled,
    }
    previous = None
    for key, value in list(stages.items()):
        stages[f"{key}_loss_rate"] = (
            round((previous - value) / previous, 8) if previous else 0.0
        )
        previous = value
    stages["source"] = "authoritative_trade_outcomes"
    stages["scope"] = "已加载的权威成交结果，不代表未成交信号总量"
    return stages


async def _load_authoritative_samples(*, filters: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapt the existing authoritative outcome contract into report samples."""

    try:
        from services.authoritative_trade_outcome import load_authoritative_trade_outcomes

        since = _parse_datetime(filters.get("from"))
        mode = str(filters.get("mode") or "").lower()
        outcomes = await load_authoritative_trade_outcomes(
            mode=mode if mode in {"paper", "live"} else None,
            since=since,
            limit=5000,
            compact=True,
            include_decision_evidence=True,
        )
    except TimeoutError:
        return AuthoritativeSamples(load_status="generation_failed", error_code="query_timeout")
    except Exception:
        return AuthoritativeSamples(load_status="generation_failed", error_code="query_failed")
    samples: list[dict[str, Any]] = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        complete = outcome.get("outcome_complete") is True and outcome.get("trade_fact_trusted") is True
        components = outcome.get("realized_net_pnl_components") or {}
        entry_fee = _finite_float(outcome.get("entry_fee_usdt", outcome.get("entry_fee")))
        close_fee = _finite_float(outcome.get("close_fee_usdt", outcome.get("close_fee")))
        label = outcome.get("training_label_contract") or {}
        samples.append(
            {
                "id": outcome.get("outcome_id") or outcome.get("lifecycle_key"),
                "authority": "okx_realized" if complete else "excluded",
                "outcome_complete": complete,
                "model": outcome.get("model_id") or outcome.get("model_name") or "active",
                "execution_mode": outcome.get("execution_mode"),
                "mode": outcome.get("execution_mode"),
                "side": outcome.get("side"),
                "symbol": outcome.get("symbol"),
                "market_state": outcome.get("market_state") or outcome.get("market_regime"),
                "label_timestamp": outcome.get("label_timestamp") or label.get("label_timestamp"),
                "gross_pnl": outcome.get("gross_pnl_usdt", components.get("gross_pnl_usdt")),
                "fee": entry_fee + close_fee,
                "slippage": outcome.get(
                    "execution_slippage_usdt",
                    outcome.get("slippage_usdt", components.get("slippage_usdt")),
                ),
                "funding_fee": outcome.get("funding_fee_usdt", components.get("funding_fee_usdt")),
                "realized_net_pnl": outcome.get(
                    "realized_net_pnl_usdt",
                    outcome.get("realized_pnl", label.get("realized_net_pnl_usdt")),
                ),
            }
        )
    return AuthoritativeSamples(samples)


async def _load_registry_snapshot() -> dict[str, Any]:
    """Read the dashboard's existing lifecycle snapshot without mutating it."""

    try:
        from web_dashboard.api.dashboard import get_model_training_registry_status

        result = await asyncio.wait_for(get_model_training_registry_status(), timeout=15.0)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


class TrainingEffectivenessReportService:
    """Read-only report assembler with replaceable data providers."""

    def __init__(
        self,
        *,
        registry_provider: Callable[[], Any] | None = None,
        samples_provider: Callable[..., Any] | None = None,
        execution_provider: Callable[..., Any] | None = None,
        expert_provider: Callable[..., Any] | None = None,
    ) -> None:
        self._registry_provider = registry_provider or _load_registry_snapshot
        self._samples_provider = samples_provider or _load_authoritative_samples
        self._execution_provider = execution_provider or (lambda **_: {})
        self._uses_default_expert_provider = expert_provider is None
        self._expert_provider = expert_provider or (lambda **_: [])

    async def build(
        self,
        *,
        filters: dict[str, Any] | None = None,
        run_id: str | None = None,
        input_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        selected_filters = {
            "mode": "all",
            "side": "all",
            "symbol": "all",
            "market_state": "all",
            "hold_minutes": {"min": 0, "max": 0},
            **(filters or {}),
        }
        registry = await _invoke(self._registry_provider)
        registry = registry if isinstance(registry, dict) else {}
        try:
            raw_samples = await _invoke(self._samples_provider, filters=selected_filters)
        except TimeoutError:
            raw_samples = AuthoritativeSamples(
                load_status="generation_failed", error_code="query_timeout"
            )
        except Exception:
            raw_samples = AuthoritativeSamples(
                load_status="generation_failed", error_code="query_failed"
            )
        load_status = str(getattr(raw_samples, "load_status", "complete") or "complete")
        load_error_code = getattr(raw_samples, "error_code", None)
        if not isinstance(raw_samples, list):
            raw_samples = AuthoritativeSamples(
                load_status="generation_failed", error_code="invalid_provider_result"
            )
            load_status = raw_samples.load_status
            load_error_code = raw_samples.error_code
        samples = [
            _normalise_report_sample(row)
            for row in (raw_samples if isinstance(raw_samples, list) else [])
            if isinstance(row, dict)
        ]
        # Providers may already apply mode/time bounds, but side, symbol,
        # market-state and hold-time are report-level filters.  Apply the full
        # contract before computing any totals and retain the projection for
        # deterministic cache-side filtering later.
        samples = [
            row for row in samples if _sample_matches_filters(row, selected_filters)
        ]
        execution = await _invoke(self._execution_provider, filters=selected_filters)
        experts = await _invoke(self._expert_provider, filters=selected_filters)
        generated = datetime.now(UTC).replace(microsecond=0)
        cutoff = _parse_datetime(selected_filters.get("to")) or generated
        fingerprint = input_fingerprint or build_input_fingerprint(
            {
                "report_version": TRAINING_EFFECTIVENESS_REPORT_VERSION,
                "filters": selected_filters,
                "registry": registry,
                "sample_ids": [row.get("id") for row in samples],
                "sample_load_status": load_status,
                "sample_load_error": load_error_code,
            }
        )
        authorities = {name: sum(1 for row in samples if classify_sample_authority(row) == name) for name in SAMPLE_AUTHORITIES}
        authoritative = [row for row in samples if _is_realized_sample(row)]
        if not experts and self._uses_default_expert_provider:
            experts = await _load_expert_contributions(filters=selected_filters)
        observed = _aggregate_metrics(authoritative, "__all__")
        baseline = _aggregate_metrics([*authoritative], "baseline")
        observed_model_id = None
        if authoritative:
            model_counts: dict[str, int] = {}
            for row in authoritative:
                model_id = str(row.get("model") or row.get("model_id") or "").strip()
                if model_id:
                    model_counts[model_id] = model_counts.get(model_id, 0) + 1
            observed_model_id = max(model_counts, key=model_counts.get) if model_counts else None
        versions = _select_versions(registry, observed_model_id=observed_model_id)
        active_id = (versions.get("active") or {}).get("model_id") or "active"
        challenger_id = (versions.get("challenger") or {}).get("model_id") or "challenger"
        metrics = {
            "active": _aggregate_metrics(authoritative, active_id),
            "challenger": _aggregate_metrics(authoritative, challenger_id),
            "baseline": baseline,
            "observed": observed,
            "delta": calculate_metric_delta(
                _aggregate_metrics(authoritative, active_id).get("fee_after_net_pnl"),
                _aggregate_metrics(authoritative, challenger_id).get("fee_after_net_pnl"),
                baseline.get("fee_after_net_pnl"),
            ),
        }
        costs = _cost_attribution(authoritative)
        report: dict[str, Any] = {
            "report_version": TRAINING_EFFECTIVENESS_REPORT_VERSION,
            "report_id": f"te-{(run_id or fingerprint[7:19])}",
            "generated_at": generated.isoformat().replace("+00:00", "Z"),
            "data_cutoff_at": cutoff.isoformat().replace("+00:00", "Z"),
            "status": "partial",
            "input_fingerprint": fingerprint,
            "run": {"run_id": run_id or fingerprint[7:19], "stage": "baseline"},
            "versions": versions,
            "filters": selected_filters,
            "metrics": metrics,
            "cost_attribution": costs,
            "expert_contributions": experts if isinstance(experts, list) else [],
            "execution_funnel": execution if isinstance(execution, dict) and execution else _build_observed_funnel(samples),
            "sample_quality": {
                "authority_counts": authorities,
                "valid_sample_count": len(authoritative),
                "excluded_sample_count": authorities["excluded"],
                "provider_sample_count": len(samples),
                "load_status": load_status,
                "load_error_code": load_error_code,
            },
            "sample_index": [_sample_projection(row) for row in samples],
            "conclusion": {"promotion_eligible": False, "blocking_reasons": []},
            "freshness": {"state": "fresh", "is_stale": False},
        }
        blocking = validate_report(report)
        if load_status != "complete":
            blocking.append(f"sample_provider:{load_error_code or load_status}")
        if not authoritative and load_status == "complete":
            blocking.append("no_okx_realized_samples")
        active_version = report["versions"]["active"]
        if not active_version.get("model_id"):
            blocking.append("active_version_missing")
        elif active_version.get("status") == "inferred":
            blocking.append("active_version_inferred")
        report["conclusion"]["blocking_reasons"] = list(dict.fromkeys(blocking))
        data_blockers = {
            "no_okx_realized_samples",
            "active_version_missing",
            "generation_timeout",
        }
        report["status"] = (
            "invalid"
            if any(item.startswith("invalid:") for item in blocking)
            else (
                "generation_failed"
                if load_status != "complete"
                else ("partial" if any(item in data_blockers for item in blocking) else "complete")
            )
        )
        return report
