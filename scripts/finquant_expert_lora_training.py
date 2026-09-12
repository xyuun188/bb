#!/usr/bin/env python3
"""Export authoritative BB-FinQuant data and train the Qwen3.8-27B adapter."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
import posixpath
import subprocess
import sys
import textwrap
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from core import finquant_training_contract as training_contract  # noqa: E402
from core import model_host_deployment  # noqa: E402
from core.finquant_adapter_registry import REMOTE_REGISTRY_TOOL_CODE  # noqa: E402
from core.finquant_remote_trainer import REMOTE_TRAINER_CODE  # noqa: E402
from core.model_candidate_manifest import ModelCandidateManifest  # noqa: E402
from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402
from core.model_training_backend import ModelTrainingBackendManifest  # noqa: E402
from core.model_training_service_lease import (  # noqa: E402
    capture_service_states_command,
    parse_service_states,
    restore_services_command,
    service_states_match,
    stop_services_command,
    wrap_command_with_restore,
)
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_error_text, safe_print  # noqa: E402
from core.training_contracts import is_authoritative_expert_memory_extra  # noqa: E402
from db.session import get_session_ctx  # noqa: E402
from models.learning import ExpertMemory  # noqa: E402
from scripts.train_local_ai_tools_models import (  # noqa: E402
    _load_shadow_samples,
    _load_trade_samples,
)
from services.training_data_quality import annotate_training_payload  # noqa: E402
from services.training_epoch import load_training_epoch_start  # noqa: E402

REMOTE_ROOT = "/data/BB"
REMOTE_TRAINING_DIR = f"{REMOTE_ROOT}/training/finquant_expert"
REMOTE_SERVICE_DIR = f"{REMOTE_ROOT}/services/finquant_expert_training"
REMOTE_DATASET_VERSIONS_DIR = f"{REMOTE_TRAINING_DIR}/versions"
REMOTE_DATASET_CURRENT = f"{REMOTE_TRAINING_DIR}/current.json"
REMOTE_TRAINER = f"{REMOTE_SERVICE_DIR}/train_finquant_lora.py"
REMOTE_REGISTRY_TOOL = f"{REMOTE_SERVICE_DIR}/finquant_registry.py"
REMOTE_ADAPTER_ROOT = f"{REMOTE_ROOT}/models/finquant_target_27b"
REMOTE_ADAPTER_VERSIONS_DIR = f"{REMOTE_ADAPTER_ROOT}/versions"
REMOTE_ADAPTER_CURRENT = f"{REMOTE_ADAPTER_ROOT}/current.json"
REMOTE_ADAPTER_ROLLBACK = f"{REMOTE_ADAPTER_ROOT}/rollback.json"
TARGET_MODEL_REPO = training_contract.TARGET_MODEL_REPO
TARGET_MODEL_TYPE = training_contract.TARGET_MODEL_TYPE
TARGET_MODEL_ARCHITECTURE = training_contract.TARGET_MODEL_ARCHITECTURE
TARGET_MODEL_ID = training_contract.TARGET_MODEL_ID
REMOTE_INFERENCE_BASE_MODEL = os.environ.get(
    "BB_TARGET_INFERENCE_MODEL_PATH", "/home/linux/trade_models/qwen3.8-27b"
).strip()
REMOTE_TRAIN_BASE_REPO = TARGET_MODEL_REPO
REMOTE_TRAIN_BASE_MODEL = os.environ.get(
    "BB_TARGET_TRAINING_MODEL_PATH", REMOTE_INFERENCE_BASE_MODEL
).strip()
REMOTE_QWEN_START_SCRIPT = "/data/BB/scripts/start_target_single_model.sh"
REMOTE_TARGET_MODEL_SERVICE = "bb-phase3-llm-target.service"
REMOTE_TRAINING_CONFLICT_SERVICES = (
    REMOTE_TARGET_MODEL_SERVICE,
    "bb-phase3-quant-api.service",
)
REMOTE_TRAIN_LOG_DIR = f"{REMOTE_TRAINING_DIR}/logs"
REMOTE_DOWNLOAD_MANIFEST = f"{REMOTE_ROOT}/manifests/phase3_model_download_manifest.json"
REMOTE_VALIDATION_MANIFEST = f"{REMOTE_ROOT}/manifests/phase3_model_validation.json"
REMOTE_TARGET_MODEL_MANIFEST = f"{REMOTE_ROOT}/manifests/target_model_candidate.json"
REMOTE_PLATFORM_APP_DIR = "/data/bb/app"
REMOTE_PLATFORM_SCRIPT = f"{REMOTE_PLATFORM_APP_DIR}/scripts/finquant_expert_lora_training.py"
REMOTE_PLATFORM_EXPORT_DIR = f"{REMOTE_PLATFORM_APP_DIR}/data/finquant_expert_training"
REMOTE_PLATFORM_EXPORT_WRAPPER = f"{REMOTE_PLATFORM_EXPORT_DIR}/export_wrapper.py"
MODEL_NAME = TARGET_MODEL_ID
BASE_MODEL_NAME = os.environ.get("BB_TARGET_BASE_MODEL_NAME", "qwen3.8-27b-base").strip()
DATASET_SCHEMA_VERSION = training_contract.DATASET_SCHEMA_VERSION
DATASET_POLICY = training_contract.DATASET_POLICY
TRAINING_INHERITANCE_CONTRACT = {
    "authoritative_samples": "inherited",
    "preference_pairs": "inherited",
    "task_and_label_contracts": "inherited",
    "evaluation_workflow": "inherited",
    "legacy_adapter_weights": "forbidden",
    "legacy_optimizer_state": "forbidden",
}
RETURN_OBJECTIVE_NAME = training_contract.RETURN_OBJECTIVE_NAME
RETURN_OBJECTIVE_VERSION = training_contract.RETURN_OBJECTIVE_VERSION
PREFERENCE_CONTRACT_VERSION = training_contract.PREFERENCE_CONTRACT_VERSION
ADAPTER_REGISTRY_VERSION = "bb_finquant_target_27b.v1"
DATASET_VERSION_PATTERN = training_contract.DATASET_VERSION_PATTERN
ADAPTER_VERSION_PATTERN = training_contract.ADAPTER_VERSION_PATTERN
REQUIRED_TRAINING_TABLES = (
    "trade_reflections",
    "positions",
    "orders",
    "shadow_backtests",
    "expert_memories",
)
TARGET_TRAINING_BACKEND_MANIFEST_ENV = "BB_TARGET_TRAINING_BACKEND_MANIFEST"
LIVE_SWITCH_GATE_ENV = "BB_FINQUANT_LIVE_SWITCH_GATE"
REMOTE_TRAINING_TIMEOUT_SECONDS = 6900
REMOTE_TRAINING_SSH_TIMEOUT_SECONDS = 7200


def _verified_training_candidate() -> ModelCandidateManifest:
    """Load the exact candidate used for a new training run.

    Historical dataset exports retain lineage, but a new adapter must never be
    trained against an old base implicitly. The
    caller provides a model-host-generated manifest through
    ``BB_TARGET_MODEL_MANIFEST``; without it training is refused closed.
    """

    manifest_path = str(os.environ.get("BB_TARGET_MODEL_MANIFEST") or "").strip()
    if not manifest_path:
        raise RuntimeError(
            "target model training requires BB_TARGET_MODEL_MANIFEST pointing to a verified candidate"
        )
    candidate = ModelCandidateManifest.load(manifest_path)
    evidence_errors = candidate.validate_evidence()
    if evidence_errors:
        raise RuntimeError("target model candidate evidence invalid: " + ", ".join(evidence_errors))
    return candidate


def _verified_training_backend(candidate: ModelCandidateManifest) -> ModelTrainingBackendManifest:
    """Require independent Qwen3.5 training evidence before any remote mutation."""

    manifest_path = str(os.environ.get(TARGET_TRAINING_BACKEND_MANIFEST_ENV) or "").strip()
    if not manifest_path:
        raise RuntimeError(
            "Qwen3.5 target training backend is not independently verified; "
            f"set {TARGET_TRAINING_BACKEND_MANIFEST_ENV} to a verified capability manifest"
        )
    backend = ModelTrainingBackendManifest.load(manifest_path)
    trainer_sha256 = _sha256_bytes(REMOTE_TRAINER_CODE)
    errors = backend.validate_for_candidate(candidate, trainer_sha256=trainer_sha256)
    if errors:
        raise RuntimeError(
            "Qwen3.5 target training backend evidence invalid: " + ", ".join(errors)
        )
    return backend


def _verified_training_context() -> tuple[ModelCandidateManifest, ModelTrainingBackendManifest]:
    candidate = _verified_training_candidate()
    return candidate, _verified_training_backend(candidate)


def sh(value: str | Path) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        number = float(value)
        return number if number == number else default
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sha256_bytes(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_code_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _finalize_dataset_contract(
    dataset_jsonl: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    return training_contract.finalize_dataset_contract(
        dataset_jsonl,
        manifest,
        source_code_version=_source_code_version(),
        source_script_sha256=_sha256_file(Path(__file__)),
        inheritance_contract=TRAINING_INHERITANCE_CONTRACT,
    )


def _validate_dataset_contract(dataset_jsonl: str, manifest: dict[str, Any]) -> None:
    training_contract.validate_dataset_contract(
        dataset_jsonl,
        manifest,
        inheritance_contract=TRAINING_INHERITANCE_CONTRACT,
    )


def _new_adapter_version(dataset_manifest: dict[str, Any], *, now: datetime | None = None) -> str:
    return training_contract.new_adapter_version(dataset_manifest, now=now)


def _remote_dataset_paths(dataset_version: str) -> tuple[str, str]:
    if not DATASET_VERSION_PATTERN.fullmatch(dataset_version):
        raise ValueError("invalid remote dataset version")
    root = f"{REMOTE_DATASET_VERSIONS_DIR}/{dataset_version}"
    return f"{root}/dataset.jsonl", f"{root}/manifest.json"


def _remote_adapter_paths(adapter_version: str) -> tuple[str, str, str]:
    if not ADAPTER_VERSION_PATTERN.fullmatch(adapter_version):
        raise ValueError("invalid remote adapter version")
    root = f"{REMOTE_ADAPTER_VERSIONS_DIR}/{adapter_version}"
    return (
        root,
        f"{root}/specialization_manifest.json",
        f"{REMOTE_TRAIN_LOG_DIR}/{adapter_version}.log",
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _bounded_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        return {
            "truncated": True,
            "sha256": _sha256_bytes(serialized),
            "type": type(value).__name__,
        }
    if isinstance(value, dict):
        items = list(value.items())
        bounded = {str(key): _bounded_json_value(item, depth=depth + 1) for key, item in items[:24]}
        if len(items) > 24:
            bounded["_truncated_keys"] = len(items) - 24
        return bounded
    if isinstance(value, (list, tuple)):
        bounded_list = [_bounded_json_value(item, depth=depth + 1) for item in value[:8]]
        if len(value) > 8:
            bounded_list.append({"_truncated_items": len(value) - 8})
        return bounded_list
    if isinstance(value, str) and len(value) > 240:
        return {
            "truncated": True,
            "sha256": _sha256_bytes(value),
            "preview": value[:160],
            "original_length": len(value),
        }
    return value


def _json_compact(value: Any, limit: int = 2200) -> str:
    original = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(original) <= limit:
        return original
    bounded = _bounded_json_value(value)
    compacted = json.dumps(bounded, ensure_ascii=False, sort_keys=True, default=str)
    if len(compacted) <= limit:
        return compacted
    fallback = {
        "truncated": True,
        "sha256": _sha256_bytes(original),
        "top_level_keys": sorted(str(key) for key in value) if isinstance(value, dict) else [],
        "preview": compacted[: min(800, max(limit - 300, 0))],
    }
    return json.dumps(fallback, ensure_ascii=False, sort_keys=True)


def _json_object_from_remote_output(raw: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    parsed: dict[str, Any] | None = None
    parsed_span = -1
    text = str(raw or "")
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and end > parsed_span:
            parsed = value
            parsed_span = end
    if parsed is None:
        raise ValueError(f"remote output did not contain a JSON object: {text[:1000]}")
    return parsed


def _trade_response(sample: dict[str, Any]) -> dict[str, Any]:
    realized = _safe_float(sample.get("realized_pnl"))
    entry_fee = _safe_float(sample.get("entry_fee"))
    close_fee = _safe_float(sample.get("close_fee"))
    funding_signed = _safe_float(sample.get("funding_fee"))
    liquidation_signed = _safe_float(sample.get("liquidation_penalty"))
    cost_drag = entry_fee + close_fee - min(funding_signed, 0.0) - min(
        liquidation_signed,
        0.0,
    )
    side = str(sample.get("side") or "").lower()
    return {
        "verdict": "good_trade" if realized > 0 else "bad_trade" if realized < 0 else "flat_trade",
        "side": side,
        "net_pnl_after_all_costs_usdt": round(realized, 6),
        "gross_pnl_usdt": round(_safe_float(sample.get("gross_pnl")), 6),
        "entry_fee_usdt": round(entry_fee, 6),
        "close_fee_usdt": round(close_fee, 6),
        "funding_fee_usdt_signed": round(funding_signed, 6),
        "liquidation_penalty_usdt_signed": round(liquidation_signed, 6),
        "total_cost_drag_usdt": round(cost_drag, 6),
        "lesson": sample.get("improvement_summary")
        or sample.get("mistake_summary")
        or (
            "Prefer similar setups only when expected net profit, liquidity, and exit discipline are stronger."
            if realized <= 0
            else "This setup had positive after-fee outcome; reuse only with comparable evidence and risk control."
        ),
        "risk_guidance": {
            "increase_size": bool(
                realized > 0 and _safe_float(sample.get("holding_minutes")) > 0
            ),
            "avoid_tiny_fee_drag": bool(cost_drag > abs(realized) and realized <= 0),
            "requires_after_fee_positive_expectancy": True,
        },
    }


def _required_shadow_fee_after_returns(sample: dict[str, Any]) -> tuple[float, float]:
    values: list[float] = []
    for key in (
        "long_net_return_after_all_cost_pct",
        "short_net_return_after_all_cost_pct",
    ):
        try:
            value = float(sample.get(key))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"shadow training sample is missing {key}") from exc
        if not math.isfinite(value):
            raise ValueError(f"shadow training sample has invalid {key}")
        values.append(value)
    return values[0], values[1]


def _shadow_response(sample: dict[str, Any]) -> dict[str, Any]:
    long_return, short_return = _required_shadow_fee_after_returns(sample)
    best_side = "long" if long_return >= short_return else "short"
    return {
        "verdict": (
            "missed_opportunity" if sample.get("missed_opportunity") else "shadow_observation"
        ),
        "best_side": best_side,
        "long_net_return_after_all_cost_pct": round(long_return, 6),
        "short_net_return_after_all_cost_pct": round(short_return, 6),
        "lesson": "Rank future candidates by after-cost payoff and avoid suppressing high-quality opportunities without explicit counter-evidence.",
        "risk_guidance": {
            "do_not_trade_without_confirmation": True,
            "use_as_shadow_supervision": True,
        },
    }


def _messages(
    kind: str,
    payload: dict[str, Any],
    response: dict[str, Any],
    *,
    rejected_response: dict[str, Any] | None = None,
    preference_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    system = (
        "You are BB-FinQuant-Qwen3.8-27B, a cryptocurrency futures expert. "
        "Learn from audited after-fee outcomes. Reply only as compact JSON with "
        "verdict, side or best_side, lesson, and risk_guidance."
    )
    user = {
        "task": "learn_finquant_trade_policy",
        "sample_kind": kind,
        "payload": payload,
    }
    row = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": _json_compact(user)},
            {
                "role": "assistant",
                "content": json.dumps(response, ensure_ascii=False, sort_keys=True),
            },
        ],
        "metadata": {
            "kind": kind,
            "symbol": payload.get("symbol"),
            "side": payload.get("side") or payload.get("decision_action"),
            "source_id": payload.get("id"),
        },
    }
    if rejected_response is not None:
        prompt = "\n".join(
            (
                f"<|system|>\n{system}",
                f"<|user|>\n{_json_compact(user)}",
                "<|assistant|>\n",
            )
        )
        row["preference"] = {
            "contract_version": PREFERENCE_CONTRACT_VERSION,
            "objective": RETURN_OBJECTIVE_NAME,
            "prompt": prompt,
            "chosen": json.dumps(response, ensure_ascii=False, sort_keys=True),
            "rejected": json.dumps(rejected_response, ensure_ascii=False, sort_keys=True),
            "metrics": dict(preference_metrics or {}),
        }
    return row


def _trade_rejected_response(response: dict[str, Any]) -> dict[str, Any]:
    rejected = dict(response)
    realized = _safe_float(response.get("net_pnl_after_all_costs_usdt"))
    rejected["verdict"] = "good_trade" if realized <= 0 else "bad_trade"
    rejected["lesson"] = (
        "Repeat this setup and increase risk because outcome frequency matters more than payoff."
        if realized <= 0
        else "Avoid this setup even though its audited fee-after outcome was positive."
    )
    rejected["risk_guidance"] = {
        "increase_size": bool(realized <= 0),
        "avoid_tiny_fee_drag": False,
        "requires_after_fee_positive_expectancy": False,
    }
    return rejected


def _shadow_rejected_response(
    sample: dict[str, Any],
    response: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    long_return, short_return = _required_shadow_fee_after_returns(sample)
    chosen_side = str(response.get("best_side") or "long")
    rejected_side = "short" if chosen_side == "long" else "long"
    rejected = dict(response)
    rejected["best_side"] = rejected_side
    rejected["lesson"] = "Choose the lower fee-after return side because direction hit rate is enough."
    chosen_return = long_return if chosen_side == "long" else short_return
    rejected_return = short_return if chosen_side == "long" else long_return
    return rejected, {
        "chosen_net_return_after_all_cost_pct": round(chosen_return, 8),
        "rejected_net_return_after_all_cost_pct": round(rejected_return, 8),
        "return_uplift_pct": round(chosen_return - rejected_return, 8),
    }


def _return_objective_counterexample() -> dict[str, Any]:
    payload = {
        "scenario": "low_win_high_payoff_vs_high_win_negative_expectancy",
        "candidate_a": {
            "win_rate": 0.35,
            "avg_win_pct": 4.0,
            "avg_loss_pct": -1.0,
            "expected_net_return_after_all_cost_pct": 0.75,
        },
        "candidate_b": {
            "win_rate": 0.80,
            "avg_win_pct": 0.10,
            "avg_loss_pct": -2.0,
            "expected_net_return_after_all_cost_pct": -0.32,
        },
    }
    chosen = {
        "verdict": "prefer_candidate_a",
        "best_side": "candidate_a",
        "lesson": "Lower win rate is acceptable when fee-after expectancy and payoff are superior.",
        "risk_guidance": {"requires_after_fee_positive_expectancy": True},
    }
    rejected = {
        "verdict": "prefer_candidate_b",
        "best_side": "candidate_b",
        "lesson": "Prefer the higher win rate despite negative fee-after expectancy.",
        "risk_guidance": {"requires_after_fee_positive_expectancy": False},
    }
    return _messages(
        "return_preference_counterexample",
        payload,
        chosen,
        rejected_response=rejected,
        preference_metrics={
            "chosen_net_return_after_all_cost_pct": 0.75,
            "rejected_net_return_after_all_cost_pct": -0.32,
            "return_uplift_pct": 1.07,
        },
    )


async def _load_expert_memory_examples() -> list[dict[str, Any]]:
    epoch_start = load_training_epoch_start()
    async with get_session_ctx() as session:
        result = await session.execute(
            select(ExpertMemory)
            .where(
                ExpertMemory.is_active.is_(True),
                ExpertMemory.created_at >= epoch_start,
            )
            .order_by(ExpertMemory.confidence_score.desc(), ExpertMemory.id.desc())
        )
        rows = list(result.scalars().all())
    examples: list[dict[str, Any]] = []
    for row in rows:
        extra = row.extra if isinstance(row.extra, dict) else {}
        if not is_authoritative_expert_memory_extra(extra):
            continue
        payload = {
            "id": int(row.id or 0),
            "expert_name": row.expert_name,
            "symbol": row.symbol,
            "side": row.side,
            "market_pattern": row.market_pattern,
            "confidence_score": _safe_float(row.confidence_score),
            "success_count": int(row.success_count or 0),
            "failure_count": int(row.failure_count or 0),
            "net_return_after_all_cost_pct": _safe_float(
                extra.get("net_return_after_all_cost_pct")
            ),
            "objective": extra.get("objective"),
            "objective_version": extra.get("objective_version"),
        }
        response = {
            "verdict": "expert_memory_lesson",
            "side": row.side,
            "lesson": row.lesson,
            "observation_policy": "fee_after_outcome_only_no_live_risk_authority",
        }
        examples.append(_messages("expert_memory", payload, response))
    return examples


def _db_kind() -> str:
    url = str(settings.database_url or "").lower()
    if "postgresql" in url:
        return "postgresql"
    if "sqlite" in url:
        return "sqlite"
    return "unknown"


async def _missing_training_tables() -> list[str]:
    expected = set(REQUIRED_TRAINING_TABLES)
    async with get_session_ctx() as session:
        if _db_kind() == "sqlite":
            result = await session.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            present = {str(row[0]) for row in result.fetchall()} & expected
        elif _db_kind() == "postgresql":
            result = await session.execute(text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = current_schema()
                    """))
            present = {str(row[0]) for row in result.fetchall()} & expected
        else:
            present = set()
    return sorted(expected - present)


async def _training_data_source_ready() -> tuple[bool, list[str]]:
    try:
        missing = await _missing_training_tables()
    except SQLAlchemyError:
        raise
    return (not missing, missing)


async def _assert_training_data_source_ready() -> None:
    ready, missing = await _training_data_source_ready()
    if ready:
        return
    raise RuntimeError(
        "BB-FinQuant-Qwen3.8-27B training data source is not ready: "
        f"missing tables {missing}. Use --source platform or run this script on "
        "the online platform server where the trading PostgreSQL database is configured."
    )


async def build_dataset() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    await _assert_training_data_source_ready()
    trade_samples = await _load_trade_samples()
    shadow_samples = await _load_shadow_samples()
    quality = annotate_training_payload(
        shadow_samples=shadow_samples,
        trade_samples=trade_samples,
        sequence_samples=[],
        text_sentiment_samples=[],
    )
    quality_report = _safe_dict(quality.get("quality_report"))
    consistency = _safe_dict(quality_report.get("training_label_consistency"))
    if consistency.get("status") == "blocked" or consistency.get("promotion_blocked"):
        raise RuntimeError(
            "BB-FinQuant dataset label consistency is blocked; refusing to export training data: "
            f"{_json_compact(consistency, limit=2400)}"
        )
    examples: list[dict[str, Any]] = []
    for sample in quality["trade_samples"]:
        if str(sample.get("trade_fact_trust_reason") or "").strip():
            continue
        payload = {
            key: sample.get(key)
            for key in (
                "id",
                "position_id",
                "symbol",
                "side",
                "entry_order_id",
                "close_order_id",
                "entry_price",
                "close_price",
                "quantity",
                "notional",
                "notional_source",
                "realized_pnl",
                "gross_pnl",
                "entry_fee",
                "close_fee",
                "entry_fee_source",
                "close_fee_source",
                "funding_fee",
                "funding_fee_source",
                "liquidation_penalty",
                "settlement_components_total",
                "slippage",
                "slippage_source",
                "holding_minutes",
                "net_return_after_all_cost_pct",
                "leverage",
                "outcome",
                "raw_llm_response",
                "settlement_status",
                "settlement_source",
            )
            if key in sample
        }
        response = _trade_response(sample)
        examples.append(
            _messages(
                "closed_trade",
                payload,
                response,
                rejected_response=_trade_rejected_response(response),
                preference_metrics={
                    "chosen_realized_net_pnl_usdt": response.get(
                        "net_pnl_after_all_costs_usdt"
                    ),
                    "chosen_matches_audited_outcome": True,
                    "rejected_matches_audited_outcome": False,
                },
            )
        )
    for sample in quality["shadow_samples"]:
        payload = {
            key: sample.get(key)
            for key in (
                "id",
                "symbol",
                "analysis_type",
                "decision_action",
                "decision_confidence",
                "horizon_minutes",
                "features",
                "gross_long_return_pct",
                "gross_short_return_pct",
                "long_net_return_after_all_cost_pct",
                "short_net_return_after_all_cost_pct",
                "best_action_after_all_cost",
                "missed_opportunity",
            )
            if key in sample
        }
        response = _shadow_response(sample)
        rejected, preference_metrics = _shadow_rejected_response(sample, response)
        examples.append(
            _messages(
                "shadow_backtest",
                payload,
                response,
                rejected_response=rejected,
                preference_metrics=preference_metrics,
            )
        )
    examples.extend(await _load_expert_memory_examples())
    examples.append(_return_objective_counterexample())
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "example_count": len(examples),
        "example_counts_by_kind": {
            kind: sum(
                1 for example in examples if _safe_dict(example.get("metadata")).get("kind") == kind
            )
            for kind in ("closed_trade", "shadow_backtest", "expert_memory")
        },
        "quality_report": quality_report,
        "objective_name": RETURN_OBJECTIVE_NAME,
        "objective_version": RETURN_OBJECTIVE_VERSION,
        "preference_contract_version": PREFERENCE_CONTRACT_VERSION,
        "preference_example_count": sum(
            1 for example in examples if isinstance(example.get("preference"), dict)
        ),
        "source": "platform_db_clean_training_view",
    }
    return examples, manifest


def _upload_text(ssh, remote_path: str, content: str, *, mode: int = 0o644) -> None:
    run_remote_text(ssh, f"mkdir -p {sh(posixpath.dirname(remote_path))}", timeout=30)
    sftp = ssh.open_sftp()
    try:
        with sftp.file(remote_path, "w") as remote:
            remote.write(content)
        sftp.chmod(remote_path, mode)
    finally:
        sftp.close()


def _upload_text_atomic(ssh, remote_path: str, content: str, *, mode: int = 0o644) -> None:
    temporary_path = f"{remote_path}.tmp.{time.time_ns()}"
    _upload_text(ssh, temporary_path, content, mode=mode)
    run_remote_text(
        ssh,
        f"mv -f {sh(temporary_path)} {sh(remote_path)}",
        timeout=30,
        check=True,
    )


def _remote_immutable_file_exists(ssh, remote_path: str, expected_sha256: str) -> bool:
    raw = run_remote_text(
        ssh,
        f"if [ -e {sh(remote_path)} ]; then sha256sum {sh(remote_path)} | cut -d' ' -f1; "
        "else echo missing; fi",
        timeout=30,
        check=True,
    ).strip()
    if raw == "missing":
        return False
    if raw != expected_sha256:
        raise RuntimeError(f"refusing to overwrite immutable remote artifact: {remote_path}")
    return True


def _download_text(ssh, remote_path: str) -> str:
    sftp = ssh.open_sftp()
    try:
        with sftp.file(remote_path, "r") as remote:
            data = remote.read()
    finally:
        sftp.close()
    if isinstance(data, bytes):
        return data.decode("utf-8")
    return str(data)


def export_dataset_from_platform() -> tuple[str, str, dict[str, Any]]:
    ssh = connect_remote_ssh(ROOT, timeout=20)
    try:
        _upload_text(
            ssh,
            REMOTE_PLATFORM_SCRIPT,
            Path(__file__).read_text(encoding="utf-8"),
            mode=0o755,
        )
        wrapper = f"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path({REMOTE_PLATFORM_APP_DIR!r})

def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip(chr(34)).strip(chr(39))
        if key:
            os.environ[key] = value

os.chdir(ROOT)
load_env(ROOT / ".env")
load_env(Path("/etc/bb/bb-runtime.env"))
sys.argv = [
    "scripts/finquant_expert_lora_training.py",
    "--source",
    "local",
    "--export-only",
]
runpy.run_path("scripts/finquant_expert_lora_training.py", run_name="__main__")
"""
        _upload_text(ssh, REMOTE_PLATFORM_EXPORT_WRAPPER, wrapper, mode=0o755)
        run_remote_text(
            ssh,
            f"chown -R bb:bb {sh(REMOTE_PLATFORM_EXPORT_DIR)} 2>/dev/null || true",
            timeout=30,
            check=False,
        )
        inner_command = (
            f"cd {sh(REMOTE_PLATFORM_APP_DIR)} && "
            "PYBIN=python3; "
            "if [ -x .venv/bin/python ]; then PYBIN=.venv/bin/python; "
            "elif [ -x venv/bin/python ]; then PYBIN=venv/bin/python; fi; "
            f"$PYBIN {sh(REMOTE_PLATFORM_EXPORT_WRAPPER)}"
        )
        command = (
            "if id -u bb >/dev/null 2>&1; then "
            f"sudo -u bb -H bash -lc {sh(inner_command)}; "
            "else "
            f"{inner_command}; "
            "fi"
        )
        raw = run_remote_text(ssh, command, timeout=600, check=True)
        remote_result: dict[str, Any] = {}
        try:
            remote_result = json.loads(raw)
        except json.JSONDecodeError:
            remote_result = {"raw": raw[:1200]}
        dataset_path = str(remote_result.get("local_dataset") or "")
        manifest_path = str(remote_result.get("local_manifest") or "")
        allowed_prefix = REMOTE_PLATFORM_EXPORT_DIR.rstrip("/") + "/"
        if not dataset_path.startswith(allowed_prefix) or not manifest_path.startswith(
            allowed_prefix
        ):
            raise RuntimeError(
                "online platform export did not return versioned FinQuant artifact paths"
            )
        dataset_jsonl = _download_text(ssh, dataset_path)
        manifest_json = _download_text(ssh, manifest_path)
        return dataset_jsonl, manifest_json, remote_result
    finally:
        ssh.close()


def _adapter_deployment_payload(
    *,
    adapter_path: str,
    candidate: ModelCandidateManifest,
) -> dict[str, Any]:
    if not str(adapter_path or "").startswith(f"{REMOTE_ADAPTER_ROOT}/"):
        raise ValueError("target model service cannot start without a verified FinQuant 27B adapter")
    candidate.to_topology()
    qwen_script = model_host_deployment.target_start_script(
        candidate.to_dict(),
        adapter_path=adapter_path,
        base_model_name=BASE_MODEL_NAME,
    )
    return {
        "candidate": candidate.to_dict(),
        "adapter_path": adapter_path,
        "target_service": REMOTE_TARGET_MODEL_SERVICE,
        "conflicting_services": list(REMOTE_TRAINING_CONFLICT_SERVICES[1:]),
        "start_script": qwen_script,
        "unit": (
            "[Unit]\n"
            "Description=BB target Qwen3.8-27B model service\n"
            "After=network-online.target\n"
            "Wants=network-online.target\n\n"
            "[Service]\n"
            "Type=simple\n"
            "User=linux\n"
            "WorkingDirectory=/data/BB\n"
            "ExecStart=/data/BB/scripts/start_target_single_model.sh\n"
            "Restart=always\n"
            "RestartSec=5\n"
            "LimitNOFILE=65535\n\n"
            "[Install]\n"
            "WantedBy=multi-user.target\n"
        ),
        "service_manifest": {
            "manifest_version": "bb.phase3.model-service.v2",
            "topology_profile": "target_single_model",
            "services": [
                {
                    "name": REMOTE_TARGET_MODEL_SERVICE,
                    "role": "target_single_model",
                    "model_id": candidate.model_id,
                    "port": 8000,
                    "enabled": True,
                },
                {
                    "name": "bb-phase3-quant-api.service",
                    "role": "quant_api",
                    "port": 8101,
                    "enabled": True,
                },
            ],
        },
    }


def _remote_adapter_deploy_command(payload: dict[str, Any]) -> str:
    code = inspect.getsource(model_host_deployment)
    return (
        "python3 - <<'BB_TARGET_ADAPTER_DEPLOY_PY'\n"
        + code
        + "\n"
        + f"print(json.dumps(deploy_target_adapter(json.loads({json.dumps(payload)!r}))))\n"
        + "BB_TARGET_ADAPTER_DEPLOY_PY\n"
    )


def _remote_adapter_rollback_command(backup_path: str) -> str:
    code = inspect.getsource(model_host_deployment)
    return (
        "python3 - <<'BB_TARGET_ADAPTER_ROLLBACK_PY'\n"
        + code
        + "\n"
        + f"print(json.dumps(rollback_target_adapter({backup_path!r})))\n"
        + "BB_TARGET_ADAPTER_ROLLBACK_PY\n"
    )


def _attempt_adapter_deployment_rollback(ssh, backup_path: str) -> dict[str, Any]:
    try:
        raw = run_remote_text(
            ssh,
            _remote_adapter_rollback_command(backup_path),
            timeout=300,
            check=False,
        )
        state = _json_object_from_remote_output(raw)
    except Exception as exc:
        return {
            "status": "rollback_failed",
            "errors": [safe_error_text(exc, limit=800)],
            "backup": backup_path,
        }
    errors = state.get("errors")
    if state.get("status") != "rolled_back" or not isinstance(errors, list) or errors:
        return {
            "status": "rollback_failed",
            "errors": errors if isinstance(errors, list) and errors else ["invalid rollback status"],
            "backup": backup_path,
            "remote_status": state.get("status"),
        }
    return state


def _switch_verified_adapter(ssh, *, rollback_service: bool) -> dict[str, Any]:
    _upload_text(ssh, REMOTE_REGISTRY_TOOL, REMOTE_REGISTRY_TOOL_CODE, mode=0o755)
    result: dict[str, Any] = {}
    if rollback_service:
        raw = run_remote_text(
            ssh,
            f"/data/BB/envs/phase3-quant/bin/python {sh(REMOTE_REGISTRY_TOOL)} rollback",
            timeout=120,
            check=True,
        )
        result["rollback"] = _json_object_from_remote_output(raw)
    verified = run_remote_text(
        ssh,
        f"/data/BB/envs/phase3-quant/bin/python {sh(REMOTE_REGISTRY_TOOL)} verify",
        timeout=180,
        check=True,
    )
    verified_state = _json_object_from_remote_output(verified)
    adapter = str(verified_state.get("adapter_path") or "")
    if not adapter:
        raise RuntimeError("verified FinQuant adapter state did not include an adapter path")
    try:
        candidate_payload = json.loads(_download_text(ssh, REMOTE_TARGET_MODEL_MANIFEST))
    except json.JSONDecodeError as exc:
        raise RuntimeError("target model candidate manifest is not valid JSON") from exc
    candidate = ModelCandidateManifest.from_dict(candidate_payload)
    candidate_errors = candidate.validate_evidence()
    if candidate_errors:
        raise RuntimeError(
            "target model candidate evidence is invalid: " + ", ".join(candidate_errors)
        )
    if verified_state.get("model_name") != candidate.model_id or candidate.model_id != MODEL_NAME:
        raise RuntimeError("FinQuant adapter and target candidate model identities differ")
    deployment_payload = _adapter_deployment_payload(
        adapter_path=adapter,
        candidate=candidate,
    )
    deployed = run_remote_text(
        ssh,
        _remote_adapter_deploy_command(deployment_payload),
        timeout=720,
        check=True,
    )
    deployment_state = _json_object_from_remote_output(deployed)
    backup_path = str(deployment_state.get("backup") or "")
    if not backup_path:
        raise RuntimeError("target adapter deployment did not return rollback evidence")
    if (
        deployment_state.get("status") != "shadow"
        or deployment_state.get("live_routing_enabled") is not False
        or deployment_state.get("model_id") != candidate.model_id
        or deployment_state.get("adapter_path") != adapter
    ):
        rollback_state = _attempt_adapter_deployment_rollback(ssh, backup_path)
        raise RuntimeError(
            "target adapter deployment returned inconsistent evidence; "
            f"rollback: {_json_compact(rollback_state, limit=1200)}"
        )
    try:
        synced = run_remote_text(
            ssh,
            f"/data/BB/envs/phase3-quant/bin/python {sh(REMOTE_REGISTRY_TOOL)} sync-evidence",
            timeout=180,
            check=True,
        )
    except Exception as exc:
        rollback_state = _attempt_adapter_deployment_rollback(ssh, backup_path)
        rolled_back = rollback_state.get("status") == "rolled_back" and not rollback_state.get(
            "errors"
        )
        rollback_message = (
            "deployment was rolled back"
            if rolled_back
            else "deployment rollback failed; manual intervention required"
        )
        raise RuntimeError(
            "target adapter runtime passed but registry evidence sync failed; "
            f"{rollback_message}. Error: {safe_error_text(exc, limit=800)}; "
            f"rollback: {_json_compact(rollback_state, limit=1200)}"
        ) from None
    result.update(
        {
            "service_switched": True,
            "adapter_state": verified_state,
            "deployment": deployment_state,
            "specialization_evidence": _json_object_from_remote_output(synced),
        }
    )
    return result


def rollback_and_switch_service() -> dict[str, Any]:
    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    try:
        return _switch_verified_adapter(ssh, rollback_service=True)
    finally:
        ssh.close()


def remote_registry_status() -> dict[str, Any]:
    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    try:
        _upload_text(ssh, REMOTE_REGISTRY_TOOL, REMOTE_REGISTRY_TOOL_CODE, mode=0o755)
        raw = run_remote_text(
            ssh,
            f"/data/BB/envs/phase3-quant/bin/python {sh(REMOTE_REGISTRY_TOOL)} status",
            timeout=180,
            check=True,
        )
        return _json_object_from_remote_output(raw)
    finally:
        ssh.close()


def sync_remote_registry_evidence() -> dict[str, Any]:
    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    try:
        _upload_text(ssh, REMOTE_REGISTRY_TOOL, REMOTE_REGISTRY_TOOL_CODE, mode=0o755)
        raw = run_remote_text(
            ssh,
            f"/data/BB/envs/phase3-quant/bin/python {sh(REMOTE_REGISTRY_TOOL)} sync-evidence",
            timeout=180,
            check=True,
        )
        return _json_object_from_remote_output(raw)
    finally:
        ssh.close()


def _capture_remote_training_service_states(ssh):
    services = tuple(dict.fromkeys(REMOTE_TRAINING_CONFLICT_SERVICES))
    raw = run_remote_text(
        ssh,
        capture_service_states_command(services),
        timeout=60,
        check=True,
    )
    return services, parse_service_states(raw, services)


def deploy_and_optionally_train(
    *,
    dataset_jsonl: str,
    manifest_json: str,
    train: bool,
    switch_service: bool,
    stop_inference_for_training: bool,
    max_steps: int,
    adapter_version: str | None = None,
) -> dict[str, Any]:
    try:
        dataset_manifest = json.loads(manifest_json)
    except json.JSONDecodeError as exc:
        raise ValueError("BB-FinQuant dataset manifest is not valid JSON") from exc
    if not isinstance(dataset_manifest, dict):
        raise ValueError("BB-FinQuant dataset manifest must be an object")
    _validate_dataset_contract(dataset_jsonl, dataset_manifest)
    dataset_version = str(dataset_manifest["dataset_version"])
    remote_dataset, remote_dataset_manifest = _remote_dataset_paths(dataset_version)
    if train and not stop_inference_for_training:
        raise ValueError(
            "LoRA training requires an explicit stop-inference-for-training acknowledgement"
        )
    backend = None
    candidate = None
    if train:
        # This is deliberately before platform credentials, SSH, uploads, or service mutation.
        candidate, backend = _verified_training_context()
    if train and switch_service:
        raise ValueError("training produces a shadow artifact only; service switching is a separate gated operation")
    if switch_service and os.environ.get(LIVE_SWITCH_GATE_ENV) != "1":
        raise ValueError(f"service switching requires {LIVE_SWITCH_GATE_ENV}=1")
    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    try:
        if not _remote_immutable_file_exists(
            ssh,
            remote_dataset,
            str(dataset_manifest["dataset_sha256"]),
        ):
            _upload_text_atomic(ssh, remote_dataset, dataset_jsonl)
        if not _remote_immutable_file_exists(
            ssh,
            remote_dataset_manifest,
            _sha256_bytes(manifest_json),
        ):
            _upload_text_atomic(ssh, remote_dataset_manifest, manifest_json)
        _upload_text_atomic(
            ssh,
            REMOTE_DATASET_CURRENT,
            json.dumps(
                {
                    "dataset_schema_version": DATASET_SCHEMA_VERSION,
                    "dataset_version": dataset_version,
                    "dataset_sha256": dataset_manifest["dataset_sha256"],
                    "dataset_path": remote_dataset,
                    "manifest_path": remote_dataset_manifest,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        _upload_text(ssh, REMOTE_TRAINER, REMOTE_TRAINER_CODE, mode=0o755)
        _upload_text(ssh, REMOTE_REGISTRY_TOOL, REMOTE_REGISTRY_TOOL_CODE, mode=0o755)
        result: dict[str, Any] = {
            "uploaded": {
                "dataset": remote_dataset,
                "dataset_manifest": remote_dataset_manifest,
                "dataset_pointer": REMOTE_DATASET_CURRENT,
                "trainer": REMOTE_TRAINER,
                "registry_tool": REMOTE_REGISTRY_TOOL,
            },
            "trained": False,
            "service_switched": False,
        }
        if train:
            assert candidate is not None
            assert backend is not None
            selected_version = adapter_version or _new_adapter_version(dataset_manifest)
            adapter_dir, specialization_manifest, train_log = _remote_adapter_paths(
                selected_version
            )
            training_services, service_states = _capture_remote_training_service_states(ssh)
            restore_services = restore_services_command(service_states)
            stop_services = stop_services_command(service_states)
            train_body = (
                f"mkdir -p {sh(REMOTE_TRAINING_DIR)} {sh(REMOTE_ADAPTER_VERSIONS_DIR)} "
                f"{sh(REMOTE_TRAIN_LOG_DIR)}; "
                f"{_remote_train_base_prepare_command(repo_id=candidate.repo_id, model_path=candidate.model_path)}; "
                "/data/BB/envs/phase3-quant/bin/python -c "
                "'import datasets, trl; print(trl.__version__)'; "
                f"{stop_services}; "
                f"timeout --signal=TERM --kill-after=60s {REMOTE_TRAINING_TIMEOUT_SECONDS}s "
                f"/data/BB/envs/phase3-quant/bin/python {sh(REMOTE_TRAINER)} "
                f"--dataset {sh(remote_dataset)} "
                f"--dataset-manifest {sh(remote_dataset_manifest)} "
                f"--output-dir {sh(adapter_dir)} "
                f"--base-model {sh(candidate.model_path)} "
                f"--base-model-repo {sh(candidate.repo_id)} "
                f"--model-name {sh(candidate.model_id)} "
                f"--inference-base-model {sh(candidate.model_path)} "
                f"--manifest {sh(specialization_manifest)} "
                f"--version-id {sh(selected_version)} "
                f"--max-steps {int(max_steps)} "
                f"> {sh(train_log)} 2>&1; "
                f"cat {sh(specialization_manifest)}"
            )
            train_cmd = wrap_command_with_restore(train_body, restore_services)
            try:
                raw = run_remote_text(
                    ssh,
                    train_cmd,
                    timeout=REMOTE_TRAINING_SSH_TIMEOUT_SECONDS,
                    check=True,
                )
            except Exception as exc:
                tail = run_remote_text(
                    ssh,
                    f"tail -n 160 {sh(train_log)} 2>/dev/null || true",
                    timeout=120,
                    check=False,
                )
                restore_evidence = "unavailable"
                try:
                    _actual_services, actual_states = _capture_remote_training_service_states(ssh)
                    restore_evidence = (
                        "verified" if service_states_match(service_states, actual_states)
                        else "mismatch"
                    )
                except Exception as state_exc:
                    restore_evidence = safe_error_text(state_exc, limit=300)
                raise RuntimeError(
                    "remote LoRA training failed. "
                    f"Command error: {safe_error_text(exc, limit=1400)}\n"
                    f"Service restore: {restore_evidence}\n"
                    f"Train log tail:\n{tail}"
                ) from None
            _actual_services, actual_states = _capture_remote_training_service_states(ssh)
            if not service_states_match(service_states, actual_states):
                raise RuntimeError(
                    "remote LoRA training completed but the pre-training service state was not restored"
                )
            result["trained"] = True
            result["adapter_version"] = selected_version
            result["specialization_manifest"] = _json_object_from_remote_output(raw)
            shadowed = run_remote_text(
                ssh,
                f"BB_TARGET_MODEL_ID={sh(candidate.model_id)} "
                f"BB_TARGET_INFERENCE_MODEL_PATH={sh(candidate.model_path)} "
                f"/data/BB/envs/phase3-quant/bin/python {sh(REMOTE_REGISTRY_TOOL)} "
                f"register-shadow --manifest {sh(specialization_manifest)}",
                timeout=180,
                check=True,
            )
            result["artifact_registry"] = _json_object_from_remote_output(shadowed)
            result["training_backend"] = backend.to_dict()
            result["service_restore"] = "verified"
            result["live_routing_enabled"] = False
        if switch_service:
            result.update(_switch_verified_adapter(ssh, rollback_service=False))
        return result
    finally:
        ssh.close()


def _remote_train_base_prepare_command(*, repo_id: str, model_path: str) -> str:
    return textwrap.dedent(f"""
        if [ -s {sh(model_path + "/config.json")} ]; then
          echo train-base-ready
        else
          mkdir -p {sh(model_path)}
          /data/BB/envs/phase3-quant/bin/python - <<'PY'
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id={repo_id!r},
    local_dir={model_path!r},
    resume_download=True,
    local_dir_use_symlinks=False,
    max_workers=4,
    ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.ot"],
)
print("train-base-downloaded")
PY
        fi
        """).strip()


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("auto", "local", "platform"),
        default="auto",
        help=(
            "Training data source. auto uses the local DB when it has the required "
            "training tables, otherwise exports from the online platform server."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument(
        "--dataset-version",
        default="",
        help="Reuse one already-exported immutable local dataset version.",
    )
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--switch-service", action="store_true")
    parser.add_argument("--rollback-service", action="store_true")
    parser.add_argument("--registry-status", action="store_true")
    parser.add_argument("--sync-registry-evidence", action="store_true")
    parser.add_argument("--stop-inference-for-training", action="store_true")
    args = parser.parse_args()

    # Training contract validation is intentionally the first operation for a
    # training request.  It must fail before DB access, platform export, SSH,
    # uploads, downloads, or any systemd mutation can occur.
    if args.train:
        _verified_training_context()

    if args.registry_status:
        if (
            args.train
            or args.switch_service
            or args.rollback_service
            or args.export_only
            or args.sync_registry_evidence
        ):
            raise SystemExit("--registry-status cannot be combined with mutating operations")
        result = remote_registry_status()
        safe_print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.sync_registry_evidence:
        if args.train or args.switch_service or args.rollback_service or args.export_only:
            raise SystemExit("--sync-registry-evidence cannot be combined with other operations")
        result = sync_remote_registry_evidence()
        safe_print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.rollback_service:
        if not args.switch_service:
            raise SystemExit("--rollback-service requires --switch-service")
        if args.train or args.export_only:
            raise SystemExit("rollback cannot be combined with training or dataset export")
        result = rollback_and_switch_service()
        result["source"] = "existing_verified_remote_artifact"
        safe_print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    requested_dataset_version = str(args.dataset_version or "").strip()
    if requested_dataset_version:
        if not DATASET_VERSION_PATTERN.fullmatch(requested_dataset_version):
            raise SystemExit("invalid --dataset-version")
        source = "immutable_local_version"
        local_dir = ROOT / "data" / "finquant_expert_training"
        version_dir = local_dir / "versions" / requested_dataset_version
        local_dataset = version_dir / "dataset.jsonl"
        local_manifest = version_dir / "manifest.json"
        if not local_dataset.is_file() or not local_manifest.is_file():
            raise SystemExit(f"dataset version is incomplete or missing: {version_dir}")
        dataset_jsonl = local_dataset.read_bytes().decode("utf-8")
        try:
            manifest = json.loads(local_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"dataset manifest is invalid: {local_manifest}") from exc
        _validate_dataset_contract(dataset_jsonl, manifest)
        manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2)
        example_count = int(manifest["example_count"])
    else:
        source = args.source
        if source == "auto":
            try:
                local_ready, _missing = await _training_data_source_ready()
            except SQLAlchemyError:
                local_ready = False
            source = "local" if local_ready else "platform"

    if not requested_dataset_version and source == "platform":
        dataset_jsonl, manifest_json, platform_export = export_dataset_from_platform()
        try:
            manifest = json.loads(manifest_json)
        except json.JSONDecodeError:
            manifest = {}
        example_count = int(manifest.get("example_count") or 0)
        if example_count <= 0:
            example_count = sum(1 for line in dataset_jsonl.splitlines() if line.strip())
        manifest["source_transport"] = "online_platform_export"
        manifest["platform_export"] = platform_export
    elif not requested_dataset_version:
        examples, manifest = await build_dataset()
        if not examples:
            raise SystemExit("No BB-FinQuant SFT examples were generated.")
        dataset_jsonl = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples)
        manifest["source_transport"] = "local_database"
        example_count = len(examples)

    if not dataset_jsonl.strip():
        raise SystemExit("No BB-FinQuant SFT examples were generated.")
    if not requested_dataset_version:
        manifest = _finalize_dataset_contract(dataset_jsonl, manifest)
    _validate_dataset_contract(dataset_jsonl, manifest)
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2)
    example_count = int(manifest["example_count"])
    local_dir = ROOT / "data" / "finquant_expert_training"
    version_dir = local_dir / "versions" / str(manifest["dataset_version"])
    local_dataset = version_dir / "dataset.jsonl"
    local_manifest = version_dir / "manifest.json"
    if version_dir.exists():
        if not local_dataset.is_file() or not local_manifest.is_file():
            raise RuntimeError(f"incomplete immutable dataset version: {version_dir}")
        if (
            _sha256_file(local_dataset) != manifest["dataset_sha256"]
            or local_manifest.read_text(encoding="utf-8") != manifest_json
        ):
            raise RuntimeError(f"refusing to overwrite immutable dataset version: {version_dir}")
    else:
        version_dir.mkdir(parents=True, exist_ok=False)
        local_dataset.write_bytes(dataset_jsonl.encode("utf-8"))
        local_manifest.write_text(manifest_json, encoding="utf-8")
    _write_json_atomic(
        local_dir / "current.json",
        {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "dataset_version": manifest["dataset_version"],
            "dataset_sha256": manifest["dataset_sha256"],
            "dataset_path": str(local_dataset.relative_to(local_dir)),
            "manifest_path": str(local_manifest.relative_to(local_dir)),
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    if args.export_only:
        safe_print(
            json.dumps(
                {
                    "exported": True,
                    "source": source,
                    "dataset_version": manifest["dataset_version"],
                    "dataset_sha256": manifest["dataset_sha256"],
                    "local_dataset": str(local_dataset),
                    "local_manifest": str(local_manifest),
                    "example_count": example_count,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    result = deploy_and_optionally_train(
        dataset_jsonl=dataset_jsonl,
        manifest_json=manifest_json,
        train=args.train,
        switch_service=args.switch_service,
        stop_inference_for_training=args.stop_inference_for_training,
        max_steps=args.max_steps,
    )
    result["source"] = source
    result["example_count"] = example_count
    safe_print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
