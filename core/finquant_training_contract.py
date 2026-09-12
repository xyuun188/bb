"""Pure contracts for BB-FinQuant dataset and adapter lineage.

This module deliberately has no database, SSH, subprocess, or service-control
side effects.  The CLI/orchestrator can therefore validate an immutable
dataset before it opens credentials or mutates the model host.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any

DATASET_SCHEMA_VERSION = "bb_finquant_expert_sft.v3"
DATASET_POLICY = "bb_finquant_target_agnostic.v1"
RETURN_OBJECTIVE_NAME = "maximize_expected_realized_net_return_after_cost"
RETURN_OBJECTIVE_VERSION = "2026-07-12.v1"
PREFERENCE_CONTRACT_VERSION = "bb_finquant_return_preference.v1"
TARGET_MODEL_REPO = "Qwen/Qwen3.8-27B"
TARGET_MODEL_ID = "qwen3.8-27b"
TARGET_MODEL_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
TARGET_MODEL_TYPE = "qwen3_5"
DATASET_VERSION_PATTERN = re.compile(r"^bb-finquant-sft-v3-[0-9a-f]{12}-[0-9a-f]{8}$")
ADAPTER_VERSION_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}$")


def sha256_bytes(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def finalize_dataset_contract(
    dataset_jsonl: str,
    manifest: dict[str, Any],
    *,
    source_code_version: str,
    source_script_sha256: str,
    inheritance_contract: dict[str, str],
) -> dict[str, Any]:
    dataset_sha256 = sha256_bytes(dataset_jsonl)
    lineage_payload = json.dumps(
        {
            "created_at": manifest.get("created_at"),
            "source": manifest.get("source"),
            "source_transport": manifest.get("source_transport"),
            "source_code_version": source_code_version,
            "source_script_sha256": source_script_sha256,
            "training_target_repo": TARGET_MODEL_REPO,
            "training_target_architecture": TARGET_MODEL_ARCHITECTURE,
            "training_target_model_type": TARGET_MODEL_TYPE,
            "training_inheritance_contract": inheritance_contract,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    lineage_sha256 = sha256_bytes(lineage_payload)
    finalized = dict(manifest)
    finalized.update(
        {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "dataset_policy": DATASET_POLICY,
            "dataset_version": f"bb-finquant-sft-v3-{dataset_sha256[:12]}-{lineage_sha256[:8]}",
            "dataset_sha256": dataset_sha256,
            "dataset_lineage_sha256": lineage_sha256,
            "source_code_version": source_code_version,
            "source_script_sha256": source_script_sha256,
            "training_target_contract": {
                "repo_id": TARGET_MODEL_REPO,
                "architecture": TARGET_MODEL_ARCHITECTURE,
                "model_type": TARGET_MODEL_TYPE,
                "candidate_manifest_required": True,
                "training_backend_manifest_required": True,
            },
            "training_inheritance_contract": dict(inheritance_contract),
            "example_count": sum(1 for line in dataset_jsonl.splitlines() if line.strip()),
        }
    )
    return finalized


def validate_dataset_contract(
    dataset_jsonl: str,
    manifest: dict[str, Any],
    *,
    inheritance_contract: dict[str, str],
) -> None:
    if manifest.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported BB-FinQuant dataset schema")
    version = str(manifest.get("dataset_version") or "")
    if not DATASET_VERSION_PATTERN.fullmatch(version):
        raise ValueError("invalid BB-FinQuant dataset version")
    actual_hash = sha256_bytes(dataset_jsonl)
    if manifest.get("dataset_sha256") != actual_hash:
        raise ValueError("BB-FinQuant dataset SHA-256 mismatch")
    lineage_hash = str(manifest.get("dataset_lineage_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", lineage_hash):
        raise ValueError("BB-FinQuant dataset lineage hash is invalid")
    expected_version = f"bb-finquant-sft-v3-{actual_hash[:12]}-{lineage_hash[:8]}"
    if version != expected_version:
        raise ValueError("BB-FinQuant dataset version does not match its content and lineage")
    expected_count = sum(1 for line in dataset_jsonl.splitlines() if line.strip())
    if int(manifest.get("example_count") or 0) != expected_count:
        raise ValueError("BB-FinQuant dataset example count mismatch")
    if manifest.get("objective_name") != RETURN_OBJECTIVE_NAME:
        raise ValueError("BB-FinQuant dataset return objective mismatch")
    if manifest.get("objective_version") != RETURN_OBJECTIVE_VERSION:
        raise ValueError("BB-FinQuant dataset return objective version mismatch")

    preference_count = 0
    for line_number, line in enumerate(dataset_jsonl.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"BB-FinQuant dataset row {line_number} is not valid JSON") from exc
        messages = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(messages, list) or len(messages) < 3:
            raise ValueError(f"BB-FinQuant dataset row {line_number} has invalid messages")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(f"BB-FinQuant dataset row {line_number} has invalid message")
            if message.get("role") not in {"user", "assistant"}:
                continue
            try:
                content = json.loads(str(message.get("content") or ""))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"BB-FinQuant dataset row {line_number} has invalid JSON message content"
                ) from exc
            if not isinstance(content, dict):
                raise ValueError(
                    f"BB-FinQuant dataset row {line_number} JSON message must be an object"
                )
        preference = row.get("preference") if isinstance(row, dict) else None
        if preference is None:
            continue
        if not isinstance(preference, dict):
            raise ValueError(f"BB-FinQuant dataset row {line_number} has invalid preference")
        if preference.get("contract_version") != PREFERENCE_CONTRACT_VERSION:
            raise ValueError(f"BB-FinQuant dataset row {line_number} preference version mismatch")
        if preference.get("objective") != RETURN_OBJECTIVE_NAME:
            raise ValueError(f"BB-FinQuant dataset row {line_number} preference objective mismatch")
        for key in ("prompt", "chosen", "rejected"):
            if not str(preference.get(key) or "").strip():
                raise ValueError(f"BB-FinQuant dataset row {line_number} preference missing {key}")
        preference_count += 1

    if preference_count <= 0 or int(manifest.get("preference_example_count") or 0) != preference_count:
        raise ValueError("BB-FinQuant dataset has no valid return-preference examples")
    target = manifest.get("training_target_contract")
    target = target if isinstance(target, dict) else {}
    if (
        target.get("repo_id") != TARGET_MODEL_REPO
        or target.get("architecture") != TARGET_MODEL_ARCHITECTURE
        or target.get("model_type") != TARGET_MODEL_TYPE
        or target.get("candidate_manifest_required") is not True
        or target.get("training_backend_manifest_required") is not True
    ):
        raise ValueError("BB-FinQuant target-model training contract mismatch")
    if (manifest.get("training_inheritance_contract") or {}) != inheritance_contract:
        raise ValueError("BB-FinQuant training inheritance contract mismatch")
    if not str(manifest.get("source_code_version") or "").strip():
        raise ValueError("BB-FinQuant dataset has no source code version")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("source_script_sha256") or "")):
        raise ValueError("BB-FinQuant dataset has no valid source script hash")


def new_adapter_version(dataset_manifest: dict[str, Any], *, now: datetime | None = None) -> str:
    dataset_hash = str(dataset_manifest.get("dataset_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", dataset_hash):
        raise ValueError("cannot version adapter without a valid dataset hash")
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{dataset_hash[:12]}"
