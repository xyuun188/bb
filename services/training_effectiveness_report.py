"""Read-only, reproducible training-effectiveness report contract.

The report service deliberately has no training, evaluation, model-promotion, or
trading side effects.  Providers are injected so the contract can be tested with
fixed fixtures before connecting it to production read models.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from config.settings import settings
from services.model_training_registry import build_model_training_registry

TRAINING_EFFECTIVENESS_REPORT_VERSION = "2026-08-25.v1"
TRAINING_EFFECTIVENESS_REPORT_DIRNAME = "training_effectiveness_reports"
TRAINING_EFFECTIVENESS_REPORT_STATUSES = {"complete", "partial", "invalid", "missing"}
SAMPLE_AUTHORITIES = {
    "shadow_opportunity",
    "counterfactual_cost",
    "okx_realized",
    "excluded",
}
REPORT_STALE_AFTER_SECONDS = 24 * 60 * 60


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
        gross = _finite_float(costs.get("gross_pnl"))
        fee = _finite_float(costs.get("fee"))
        slippage = _finite_float(costs.get("slippage"))
        funding = _finite_float(costs.get("funding_fee"))
        expected = calculate_fee_after_return(gross, fee, slippage, funding)
        actual = _finite_float(costs.get("fee_after_net_pnl"), math.nan)
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
    freshness.setdefault("state", "stale" if stale else "fresh")
    freshness["is_stale"] = stale
    payload = dict(payload)
    payload["freshness"] = freshness
    return payload


def apply_report_filters(report: dict[str, Any], **filters: Any) -> dict[str, Any]:
    """Apply display filters without changing the cached input fingerprint."""

    if not isinstance(report, dict):
        return {"status": "invalid"}
    result = dict(report)
    current = dict(result.get("filters") or {})
    for key, value in filters.items():
        if value is not None and str(value).strip():
            current[key] = value
    result["filters"] = current
    return result


def _invoke(provider: Callable[..., Any], *args: Any, **kwargs: Any) -> Awaitable[Any]:
    result = provider(*args, **kwargs)
    if inspect.isawaitable(result):
        return result

    async def _ready() -> Any:
        return result

    return _ready()


def _select_versions(registry: dict[str, Any]) -> dict[str, Any]:
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
    return {
        "active": active or {"version": None, "status": "missing"},
        "challenger": challenger or {"version": None, "status": "missing"},
        "baseline": {"version": "no_model_baseline", "status": "defined"},
    }


def _aggregate_metrics(samples: list[dict[str, Any]], model: str) -> dict[str, Any]:
    selected = [
        sample
        for sample in samples
        if str(sample.get("model") or sample.get("model_id") or "baseline") == model
    ]
    gross = sum(_finite_float(row.get("gross_pnl")) for row in selected)
    fee = sum(_finite_float(row.get("fee")) for row in selected)
    slippage = sum(_finite_float(row.get("slippage")) for row in selected)
    funding = sum(_finite_float(row.get("funding_fee")) for row in selected)
    net = calculate_fee_after_return(gross, fee, slippage, funding)
    wins = sum(1 for row in selected if _finite_float(row.get("realized_net_pnl", row.get("net_pnl"))) > 0)
    return {
        "sample_count": len(selected),
        "gross_pnl": round(gross, 8),
        "fee": round(fee, 8),
        "slippage": round(slippage, 8),
        "funding_fee": round(funding, 8),
        "fee_after_net_pnl": net,
        "win_rate": round(wins / len(selected), 8) if selected else None,
    }


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
        )
    except Exception:
        return []
    samples: list[dict[str, Any]] = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        complete = outcome.get("outcome_complete") is True and outcome.get("trade_fact_trusted") is True
        components = outcome.get("realized_net_pnl_components") or {}
        entry_fee = _finite_float(outcome.get("entry_fee_usdt", outcome.get("entry_fee")))
        close_fee = _finite_float(outcome.get("close_fee_usdt", outcome.get("close_fee")))
        samples.append(
            {
                "id": outcome.get("outcome_id") or outcome.get("lifecycle_key"),
                "authority": "okx_realized" if complete else "excluded",
                "outcome_complete": complete,
                "model": outcome.get("model_id") or outcome.get("model_name") or "active",
                "gross_pnl": outcome.get("gross_pnl_usdt", components.get("gross_pnl_usdt")),
                "fee": entry_fee + close_fee,
                "slippage": outcome.get("execution_slippage_usdt", components.get("slippage_usdt")),
                "funding_fee": outcome.get("funding_fee_usdt", components.get("funding_fee_usdt")),
                "realized_net_pnl": outcome.get("realized_net_pnl_usdt"),
            }
        )
    return samples


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
        self._registry_provider = registry_provider or build_model_training_registry
        self._samples_provider = samples_provider or _load_authoritative_samples
        self._execution_provider = execution_provider or (lambda **_: {})
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
        samples = await _invoke(self._samples_provider, filters=selected_filters)
        samples = [row for row in (samples if isinstance(samples, list) else []) if isinstance(row, dict)]
        execution = await _invoke(self._execution_provider, filters=selected_filters)
        experts = await _invoke(self._expert_provider, filters=selected_filters)
        generated = datetime.now(UTC).replace(microsecond=0)
        cutoff = _parse_datetime(selected_filters.get("to")) or generated
        fingerprint = input_fingerprint or build_input_fingerprint(
            {"filters": selected_filters, "registry": registry, "sample_ids": [row.get("id") for row in samples]}
        )
        authorities = {name: sum(1 for row in samples if classify_sample_authority(row) == name) for name in SAMPLE_AUTHORITIES}
        authoritative = [row for row in samples if classify_sample_authority(row) == "okx_realized"]
        baseline = _aggregate_metrics([*authoritative], "baseline")
        active_id = (_select_versions(registry).get("active") or {}).get("model_id") or "active"
        challenger_id = (_select_versions(registry).get("challenger") or {}).get("model_id") or "challenger"
        metrics = {
            "active": _aggregate_metrics(authoritative, active_id),
            "challenger": _aggregate_metrics(authoritative, challenger_id),
            "baseline": baseline,
            "delta": calculate_metric_delta(
                _aggregate_metrics(authoritative, active_id).get("fee_after_net_pnl"),
                _aggregate_metrics(authoritative, challenger_id).get("fee_after_net_pnl"),
                baseline.get("fee_after_net_pnl"),
            ),
        }
        costs = {
            "gross_pnl": sum(_finite_float(row.get("gross_pnl")) for row in authoritative),
            "fee": sum(_finite_float(row.get("fee")) for row in authoritative),
            "slippage": sum(_finite_float(row.get("slippage")) for row in authoritative),
            "funding_fee": sum(_finite_float(row.get("funding_fee")) for row in authoritative),
        }
        costs["fee_after_net_pnl"] = calculate_fee_after_return(**costs)
        report: dict[str, Any] = {
            "report_version": TRAINING_EFFECTIVENESS_REPORT_VERSION,
            "report_id": f"te-{(run_id or fingerprint[7:19])}",
            "generated_at": generated.isoformat().replace("+00:00", "Z"),
            "data_cutoff_at": cutoff.isoformat().replace("+00:00", "Z"),
            "status": "partial",
            "input_fingerprint": fingerprint,
            "run": {"run_id": run_id or fingerprint[7:19], "stage": "baseline"},
            "versions": _select_versions(registry),
            "filters": selected_filters,
            "metrics": metrics,
            "cost_attribution": {key: round(value, 8) for key, value in costs.items()},
            "expert_contributions": experts if isinstance(experts, list) else [],
            "execution_funnel": execution if isinstance(execution, dict) else {},
            "sample_quality": {"authority_counts": authorities, "valid_sample_count": len(authoritative), "excluded_sample_count": authorities["excluded"]},
            "conclusion": {"promotion_eligible": False, "blocking_reasons": []},
            "freshness": {"state": "fresh", "is_stale": False},
        }
        blocking = validate_report(report)
        if not authoritative:
            blocking.append("no_okx_realized_samples")
        if not report["versions"]["active"].get("model_id"):
            blocking.append("active_version_missing")
        report["conclusion"]["blocking_reasons"] = list(dict.fromkeys(blocking))
        report["status"] = "invalid" if any(item.startswith("invalid:") for item in blocking) else ("complete" if not blocking else "partial")
        return report
