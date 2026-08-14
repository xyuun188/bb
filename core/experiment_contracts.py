"""Immutable experiment identities for reproducible BB research runs."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from core.secret_utils import is_sensitive_key

EXPERIMENT_CONTRACT_VERSION = "bb.experiment-spec.v1"
EXPERIMENT_RESULT_VERSION = "bb.experiment-result.v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{7,40}$")


class ExperimentContractError(ValueError):
    """Raised when experiment evidence is incomplete or internally inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON evidence with stable ordering and no non-finite numbers."""

    normalized = _normalize_json(value)
    try:
        text = json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError(f"value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def build_strategy_identity(
    *,
    strategy_id: str,
    strategy_version: str,
    implementation: str,
    git_commit: str,
    source_sha256: str,
    model_versions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity = {
        "strategy_id": _required_text(strategy_id, "strategy_id", max_length=120),
        "strategy_version": _required_text(
            strategy_version,
            "strategy_version",
            max_length=120,
        ),
        "implementation": _required_text(
            implementation,
            "implementation",
            max_length=240,
        ),
        "git_commit": _required_git_commit(git_commit),
        "source_sha256": _required_sha256(source_sha256, "source_sha256"),
        "model_versions": _normalize_json(dict(model_versions or {})),
    }
    identity["strategy_fingerprint"] = content_sha256(identity)
    return identity


def build_parameter_set(
    values: Mapping[str, Any],
    *,
    version: str | None = None,
) -> dict[str, Any]:
    normalized = _normalize_json(dict(values or {}), reject_secrets=True)
    digest = content_sha256(normalized)
    return {
        "parameter_set_id": f"params_{digest[:24]}",
        "parameter_version": (
            _required_text(version, "parameter_version", max_length=120)
            if version
            else f"sha256:{digest[:16]}"
        ),
        "sha256": digest,
        "values": normalized,
    }


def build_dataset_manifest(
    *,
    source: str,
    symbols: Sequence[str],
    timeframe: str,
    started_at: datetime | str,
    ended_at: datetime | str,
    timezone: str,
    row_count: int,
    data_sha256: str,
    columns: Sequence[str],
    quality: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    start = _utc_datetime_text(started_at, "started_at")
    end = _utc_datetime_text(ended_at, "ended_at")
    if datetime.fromisoformat(end) < datetime.fromisoformat(start):
        raise ExperimentContractError("dataset ended_at must not precede started_at")
    safe_symbols = sorted(
        {
            _required_text(symbol, "symbol", max_length=40)
            for symbol in symbols
            if str(symbol or "").strip()
        }
    )
    if not safe_symbols:
        raise ExperimentContractError("dataset symbols must not be empty")
    safe_columns = [
        _required_text(column, "dataset column", max_length=80) for column in columns
    ]
    if len(set(safe_columns)) != len(safe_columns):
        raise ExperimentContractError("dataset columns must be unique")
    rows = int(row_count)
    if rows <= 0:
        raise ExperimentContractError("dataset row_count must be positive")
    manifest_content = {
        "source": _required_text(source, "dataset source", max_length=160),
        "symbols": safe_symbols,
        "timeframe": _required_text(timeframe, "timeframe", max_length=40),
        "started_at": start,
        "ended_at": end,
        "timezone": _required_text(timezone, "timezone", max_length=80),
        "row_count": rows,
        "columns": safe_columns,
        "data_sha256": _required_sha256(data_sha256, "data_sha256"),
        "quality": _normalize_json(dict(quality or {})),
    }
    manifest_sha256 = content_sha256(manifest_content)
    return {
        "dataset_id": f"dataset_{manifest_sha256[:24]}",
        "manifest_sha256": manifest_sha256,
        **manifest_content,
    }


def build_experiment_spec(
    *,
    experiment_type: str,
    strategy: Mapping[str, Any],
    parameters: Mapping[str, Any],
    dataset: Mapping[str, Any],
    execution_assumptions: Mapping[str, Any],
    portfolio_assumptions: Mapping[str, Any],
    validation_windows: Sequence[Mapping[str, Any]],
    runner: Mapping[str, Any],
    environment: Mapping[str, Any],
    random_seed: int,
    authority_contract: Mapping[str, Any],
) -> dict[str, Any]:
    content = {
        "contract_version": EXPERIMENT_CONTRACT_VERSION,
        "immutable": True,
        "experiment_type": _required_text(
            experiment_type,
            "experiment_type",
            max_length=60,
        ),
        "strategy": _normalize_json(dict(strategy)),
        "parameters": _normalize_json(dict(parameters), reject_secrets=True),
        "dataset": _normalize_json(dict(dataset)),
        "execution_assumptions": _normalize_json(dict(execution_assumptions)),
        "portfolio_assumptions": _normalize_json(dict(portfolio_assumptions)),
        "validation_windows": _normalize_windows(validation_windows),
        "runner": _normalize_json(dict(runner)),
        "environment": _normalize_json(dict(environment)),
        "random_seed": int(random_seed),
        "authority_contract": _normalize_json(dict(authority_contract)),
    }
    _validate_spec_content(content)
    digest = content_sha256(content)
    spec = {
        **content,
        "experiment_id": f"exp_{digest[:24]}",
        "spec_sha256": digest,
    }
    verify_experiment_spec(spec)
    return spec


def build_experiment_result(
    spec: Mapping[str, Any],
    *,
    status: str,
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    verify_experiment_spec(spec)
    normalized_status = _required_text(status, "result status", max_length=30).lower()
    if normalized_status not in {"complete", "failed"}:
        raise ExperimentContractError("result status must be complete or failed")
    content = {
        "result_version": EXPERIMENT_RESULT_VERSION,
        "experiment_id": spec["experiment_id"],
        "spec_sha256": spec["spec_sha256"],
        "status": normalized_status,
        "metrics": _normalize_json(dict(metrics or {})),
        "artifacts": _normalize_json(dict(artifacts or {})),
        "diagnostics": _normalize_json(dict(diagnostics or {})),
    }
    return {**content, "result_sha256": content_sha256(content)}


def verify_experiment_spec(spec: Mapping[str, Any]) -> None:
    payload = dict(spec or {})
    experiment_id = _required_text(payload.pop("experiment_id", ""), "experiment_id")
    recorded_hash = _required_sha256(payload.pop("spec_sha256", ""), "spec_sha256")
    _validate_spec_content(payload)
    actual_hash = content_sha256(payload)
    if recorded_hash != actual_hash:
        raise ExperimentContractError("experiment spec SHA-256 mismatch")
    if experiment_id != f"exp_{actual_hash[:24]}":
        raise ExperimentContractError("experiment_id does not match spec SHA-256")


def verify_experiment_result(result: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    verify_experiment_spec(spec)
    payload = dict(result or {})
    recorded_hash = _required_sha256(payload.pop("result_sha256", ""), "result_sha256")
    if payload.get("result_version") != EXPERIMENT_RESULT_VERSION:
        raise ExperimentContractError("unsupported experiment result version")
    if payload.get("experiment_id") != spec.get("experiment_id"):
        raise ExperimentContractError("result experiment_id mismatch")
    if payload.get("spec_sha256") != spec.get("spec_sha256"):
        raise ExperimentContractError("result spec_sha256 mismatch")
    if content_sha256(payload) != recorded_hash:
        raise ExperimentContractError("experiment result SHA-256 mismatch")


def _validate_spec_content(content: Mapping[str, Any]) -> None:
    if content.get("contract_version") != EXPERIMENT_CONTRACT_VERSION:
        raise ExperimentContractError("unsupported experiment contract version")
    if content.get("immutable") is not True:
        raise ExperimentContractError("experiment spec must be immutable")
    strategy = _mapping(content.get("strategy"), "strategy")
    parameters = _mapping(content.get("parameters"), "parameters")
    dataset = _mapping(content.get("dataset"), "dataset")
    execution = _mapping(content.get("execution_assumptions"), "execution_assumptions")
    portfolio = _mapping(content.get("portfolio_assumptions"), "portfolio_assumptions")
    runner = _mapping(content.get("runner"), "runner")
    authority = _mapping(content.get("authority_contract"), "authority_contract")
    _required_text(strategy.get("strategy_id"), "strategy.strategy_id")
    _required_text(strategy.get("strategy_version"), "strategy.strategy_version")
    _required_git_commit(strategy.get("git_commit"))
    _required_sha256(strategy.get("source_sha256"), "strategy.source_sha256")
    recorded_strategy_fingerprint = _required_sha256(
        strategy.get("strategy_fingerprint"),
        "strategy.strategy_fingerprint",
    )
    strategy_content = dict(strategy)
    strategy_content.pop("strategy_fingerprint", None)
    if content_sha256(strategy_content) != recorded_strategy_fingerprint:
        raise ExperimentContractError("strategy fingerprint mismatch")
    _required_sha256(parameters.get("sha256"), "parameters.sha256")
    if content_sha256(parameters.get("values")) != parameters.get("sha256"):
        raise ExperimentContractError("parameter set SHA-256 mismatch")
    expected_parameter_id = f"params_{str(parameters['sha256'])[:24]}"
    if parameters.get("parameter_set_id") != expected_parameter_id:
        raise ExperimentContractError("parameter_set_id does not match parameter SHA-256")
    _required_sha256(dataset.get("data_sha256"), "dataset.data_sha256")
    manifest_sha256 = _required_sha256(
        dataset.get("manifest_sha256"),
        "dataset.manifest_sha256",
    )
    dataset_content = dict(dataset)
    dataset_content.pop("dataset_id", None)
    dataset_content.pop("manifest_sha256", None)
    if content_sha256(dataset_content) != manifest_sha256:
        raise ExperimentContractError("dataset manifest SHA-256 mismatch")
    if dataset.get("dataset_id") != f"dataset_{manifest_sha256[:24]}":
        raise ExperimentContractError("dataset_id does not match manifest SHA-256")
    for field in ("commission_rate", "slippage_rate"):
        value = _finite_float(execution.get(field), f"execution_assumptions.{field}")
        if value < 0:
            raise ExperimentContractError(f"execution_assumptions.{field} must be non-negative")
    if _finite_float(portfolio.get("initial_cash"), "portfolio.initial_cash") <= 0:
        raise ExperimentContractError("portfolio.initial_cash must be positive")
    _required_text(runner.get("runner_id"), "runner.runner_id")
    _required_text(runner.get("runner_version"), "runner.runner_version")
    _required_text(authority.get("orders"), "authority_contract.orders")
    _required_text(authority.get("fills"), "authority_contract.fills")
    _required_text(authority.get("fees"), "authority_contract.fees")
    _required_text(authority.get("settlement"), "authority_contract.settlement")
    windows = content.get("validation_windows")
    if not isinstance(windows, list) or not windows:
        raise ExperimentContractError("validation_windows must not be empty")
    _normalize_windows(windows)


def _normalize_windows(
    windows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    previous_end: datetime | None = None
    for index, raw in enumerate(windows):
        row = dict(raw or {})
        role = _required_text(row.get("role"), f"validation_windows[{index}].role")
        start_text = _utc_datetime_text(
            row.get("started_at"),
            f"validation_windows[{index}].started_at",
        )
        end_text = _utc_datetime_text(
            row.get("ended_at"),
            f"validation_windows[{index}].ended_at",
        )
        start = datetime.fromisoformat(start_text)
        end = datetime.fromisoformat(end_text)
        if end < start:
            raise ExperimentContractError(f"validation window {index} ends before it starts")
        if previous_end is not None and start < previous_end:
            raise ExperimentContractError("validation windows must be chronological and disjoint")
        result.append(
            {
                "role": role,
                "started_at": start_text,
                "ended_at": end_text,
            }
        )
        previous_end = end
    return result


def _normalize_json(value: Any, *, reject_secrets: bool = False) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExperimentContractError("non-finite numeric values are not allowed")
        return value
    if isinstance(value, datetime):
        return _utc_datetime_text(value, "datetime")
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key in sorted(value, key=lambda item: str(item)):
            key = str(raw_key)
            if not key:
                raise ExperimentContractError("JSON object keys must not be empty")
            if reject_secrets and is_sensitive_key(key):
                raise ExperimentContractError(
                    f"sensitive key {key!r} is forbidden in experiment evidence"
                )
            normalized[key] = _normalize_json(value[raw_key], reject_secrets=reject_secrets)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_json(item, reject_secrets=reject_secrets) for item in value]
    raise ExperimentContractError(f"unsupported JSON value type: {type(value).__name__}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentContractError(f"{name} must be an object")
    return value


def _required_text(value: Any, name: str, *, max_length: int = 500) -> str:
    text = str(value or "").strip()
    if not text:
        raise ExperimentContractError(f"{name} must not be empty")
    if len(text) > max_length:
        raise ExperimentContractError(f"{name} exceeds {max_length} characters")
    return text


def _required_sha256(value: Any, name: str) -> str:
    text = str(value or "").strip().lower()
    if not SHA256_PATTERN.fullmatch(text):
        raise ExperimentContractError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _required_git_commit(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not GIT_COMMIT_PATTERN.fullmatch(text):
        raise ExperimentContractError("git_commit must be a 7-40 character hexadecimal SHA")
    return text


def _finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ExperimentContractError(f"{name} must be finite")
    return number


def _utc_datetime_text(value: Any, name: str) -> str:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ExperimentContractError(f"{name} is not a valid datetime") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ExperimentContractError(f"{name} must be a datetime")
    if parsed.tzinfo is None:
        raise ExperimentContractError(f"{name} must include timezone evidence")
    return parsed.astimezone(UTC).isoformat()
