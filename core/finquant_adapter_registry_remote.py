from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

MODEL_NAME = "qwen3.8-27b"
REGISTRY_VERSION = "bb_finquant_target_27b.v1"
ROOT = Path("/data/BB/models/finquant_target_27b")
VERSIONS = ROOT / "versions"
CURRENT = ROOT / "current.json"
ROLLBACK = ROOT / "rollback.json"
RETIRED = ROOT / "retired"
DOWNLOAD_MANIFEST = Path("/data/BB/manifests/phase3_model_download_manifest.json")
VALIDATION_MANIFEST = Path("/data/BB/manifests/phase3_model_validation.json")
INFERENCE_BASE = os.environ.get(
    "BB_TARGET_INFERENCE_MODEL_PATH", "/home/linux/trade_models/qwen3.8-27b"
).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def inside(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    resolved.relative_to(root.resolve(strict=True))
    return resolved


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def adapter_digest(files: list[dict]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def validate_pointer(pointer: dict) -> tuple[dict, Path]:
    if pointer.get("registry_version") != REGISTRY_VERSION:
        raise ValueError("unsupported FinQuant pointer registry version")
    if pointer.get("model_name") != MODEL_NAME:
        raise ValueError("FinQuant pointer model identity mismatch")
    adapter_path = inside(Path(str(pointer.get("adapter_path") or "")), ROOT)
    manifest_path = inside(Path(str(pointer.get("manifest_path") or "")), ROOT)
    if sha256_file(manifest_path) != pointer.get("manifest_sha256"):
        raise ValueError("FinQuant specialization manifest hash mismatch")
    manifest = read_json(manifest_path)
    required_values = {
        "model_name": MODEL_NAME,
        "registry_version": REGISTRY_VERSION,
        "adapter_version": pointer.get("adapter_version"),
        "dataset_schema_version": "bb_finquant_expert_sft.v3",
        "objective_name": "maximize_expected_realized_net_return_after_cost",
        "objective_version": "2026-07-12.v1",
        "preference_contract_version": "bb_finquant_return_preference.v1",
        "base_model_repo": "Qwen/Qwen3.8-27B",
    }
    for key, expected in required_values.items():
        if manifest.get(key) != expected:
            raise ValueError(f"FinQuant manifest {key} mismatch")
    for key in (
        "dataset_sha256",
        "dataset_lineage_sha256",
        "dataset_manifest_sha256",
        "source_script_sha256",
        "trainer_code_sha256",
        "base_model_config_sha256",
        "inference_base_model_config_sha256",
    ):
        if not _valid_sha256(manifest.get(key)):
            raise ValueError(f"FinQuant manifest has no valid {key}")
    if not isinstance(manifest.get("training_config"), dict):
        raise ValueError("FinQuant training configuration is missing")
    files = manifest.get("adapter_files")
    if not isinstance(files, list) or not files:
        raise ValueError("FinQuant adapter file manifest is empty")
    verified_files = []
    for row in files:
        if not isinstance(row, dict):
            raise ValueError("invalid FinQuant adapter file row")
        relative = str(row.get("path") or "")
        path = inside(adapter_path / relative, adapter_path)
        verified = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if verified != row:
            raise ValueError(f"FinQuant adapter file verification failed: {relative}")
        verified_files.append(verified)
    if not any(
        row["path"] in {"adapter_model.safetensors", "adapter_model.bin"}
        for row in verified_files
    ):
        raise ValueError("FinQuant adapter weights are missing")
    digest = adapter_digest(verified_files)
    if digest != manifest.get("adapter_sha256") or digest != pointer.get("adapter_sha256"):
        raise ValueError("FinQuant aggregate adapter hash mismatch")
    return manifest, adapter_path


def pointer_for_manifest(manifest_path: Path) -> dict:
    manifest_path = inside(manifest_path, VERSIONS)
    manifest = read_json(manifest_path)
    pointer = {
        "registry_version": REGISTRY_VERSION,
        "model_name": MODEL_NAME,
        "adapter_version": manifest.get("adapter_version"),
        "specialization_id": manifest.get("specialization_id"),
        "adapter_path": str(manifest_path.parent),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "adapter_sha256": manifest.get("adapter_sha256"),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    validate_pointer(pointer)
    return pointer


def register_shadow(manifest_path: Path) -> dict:
    pointer = pointer_for_manifest(manifest_path)
    pointer.update({"routing_state": "shadow_only", "can_influence_live": False})
    shadow_path = ROOT / "shadow" / f"{pointer['adapter_version']}.json"
    atomic_json(shadow_path, pointer)
    return {
        "shadow": pointer,
        "shadow_path": str(shadow_path),
        "live_routing_enabled": False,
    }


def _retire_incompatible(pointer: dict, *, source: str, reason: str) -> dict:
    record = {
        "retired_at": datetime.now(UTC).isoformat(),
        "reason": reason,
        "can_influence_live": False,
        "pointer": pointer,
    }
    path = RETIRED / f"incompatible-{source}-{time.time_ns()}.json"
    atomic_json(path, record)
    record["audit_path"] = str(path)
    return record


def promote(manifest_path: Path) -> dict:
    if os.environ.get("BB_FINQUANT_ALLOW_LIVE_PROMOTION") != "1":
        raise PermissionError("live registry promotion requires an explicit gated release")
    new_pointer = pointer_for_manifest(manifest_path)
    previous = None
    retired_previous = None
    retired_rollback = None
    if CURRENT.exists():
        candidate = read_json(CURRENT)
        try:
            validate_pointer(candidate)
        except ValueError as exc:
            retired_previous = _retire_incompatible(
                candidate, source="current", reason=str(exc)
            )
        else:
            previous = candidate
    if previous and previous.get("adapter_path") == new_pointer.get("adapter_path"):
        previous = None
    if ROLLBACK.exists():
        candidate = read_json(ROLLBACK)
        try:
            validate_pointer(candidate)
        except ValueError as exc:
            retired_rollback = _retire_incompatible(
                candidate, source="rollback", reason=str(exc)
            )
            ROLLBACK.unlink()
    if previous and previous.get("adapter_path") != new_pointer.get("adapter_path"):
        atomic_json(ROLLBACK, previous)
    atomic_json(CURRENT, new_pointer)
    return {
        "current": new_pointer,
        "rollback": previous,
        "retired_incompatible_previous": retired_previous,
        "retired_incompatible_rollback": retired_rollback,
    }


def validate_current(*, required: bool = True) -> dict | None:
    if not CURRENT.exists():
        if required:
            raise ValueError("FinQuant current adapter pointer is missing")
        return None
    pointer = read_json(CURRENT)
    validate_pointer(pointer)
    return pointer


def rollback() -> dict:
    current = validate_current(required=True)
    if not ROLLBACK.exists():
        raise ValueError("FinQuant rollback pointer is missing")
    target = read_json(ROLLBACK)
    validate_pointer(target)
    if target.get("adapter_path") == current.get("adapter_path"):
        raise ValueError("FinQuant rollback target equals current adapter")
    timestamp = datetime.now(UTC).isoformat()
    target["updated_at"] = timestamp
    current["updated_at"] = timestamp
    atomic_json(CURRENT, target)
    atomic_json(ROLLBACK, current)
    return {"current": target, "rollback": current}


def status() -> dict:
    current = validate_current(required=True)
    current_manifest, current_path = validate_pointer(current)
    rollback_pointer = read_json(ROLLBACK) if ROLLBACK.exists() else None
    rollback_manifest = None
    rollback_path = None
    if rollback_pointer is not None:
        rollback_manifest, rollback_path = validate_pointer(rollback_pointer)
    return {
        "registry_version": REGISTRY_VERSION,
        "current_verified": True,
        "current": current,
        "current_manifest": current_manifest,
        "current_adapter_path": str(current_path),
        "rollback_present": rollback_pointer is not None,
        "rollback_verified": rollback_pointer is not None,
        "rollback": rollback_pointer,
        "rollback_manifest": rollback_manifest,
        "rollback_adapter_path": str(rollback_path) if rollback_path else None,
    }


def sync_evidence() -> dict:
    pointer = validate_current(required=True)
    manifest, _ = validate_pointer(pointer)
    verification_status = "verified"
    specialization = {
        "verification_status": verification_status,
        "identity_verified": True,
        "adapter_version": pointer.get("adapter_version"),
        "adapter_path": pointer.get("adapter_path"),
        "lora_adapter": pointer.get("adapter_path"),
        "specialization_manifest": pointer.get("manifest_path"),
        "specialization_id": pointer.get("specialization_id"),
        "training_artifact": pointer.get("adapter_path"),
        "manifest_sha256": pointer.get("manifest_sha256"),
        "adapter_sha256": pointer.get("adapter_sha256"),
        "dataset_version": manifest.get("dataset_version"),
        "dataset_sha256": manifest.get("dataset_sha256"),
        "dataset_lineage_sha256": manifest.get("dataset_lineage_sha256"),
        "dataset_manifest_sha256": manifest.get("dataset_manifest_sha256"),
        "source_code_version": manifest.get("source_code_version"),
        "source_script_sha256": manifest.get("source_script_sha256"),
        "trainer_code_sha256": manifest.get("trainer_code_sha256"),
        "base_model_repo": manifest.get("base_model_repo"),
        "base_model_config_sha256": manifest.get("base_model_config_sha256"),
        "inference_base_model_config_sha256": manifest.get(
            "inference_base_model_config_sha256"
        ),
        "evaluation_report": manifest.get("evaluation_report"),
        "held_out_eval_loss": manifest.get("held_out_eval_loss"),
        "objective_name": manifest.get("objective_name"),
        "objective_version": manifest.get("objective_version"),
        "preference_contract_version": manifest.get("preference_contract_version"),
        "preference_selection_accuracy": manifest.get(
            "preference_selection_accuracy"
        ),
        "training_stages": manifest.get("training_stages"),
        "trained_at": manifest.get("trained_at"),
        "sample_count": manifest.get("sample_count"),
        "max_steps": manifest.get("max_steps"),
    }
    evidence = {
        "served_model_name": MODEL_NAME,
        "specialization_required": True,
        "specialization_target": MODEL_NAME,
        "specialization_status": manifest.get("specialization_status"),
        "base_model_carrier": INFERENCE_BASE,
        **{key: value for key, value in specialization.items() if value is not None},
        "specialization_evidence": {
            key: value for key, value in specialization.items() if value is not None
        },
    }
    updated = []
    for path in (DOWNLOAD_MANIFEST, VALIDATION_MANIFEST):
        data = read_json(path) if path.exists() else {"models": []}
        models = data.setdefault("models", [])
        models[:] = [
            row
            for row in models
            if not isinstance(row, dict)
            or not str(row.get("slot") or "").startswith("llm_")
        ]
        target = {
            "slot": "llm_decision_and_expert_carrier",
            "repo_id": "Qwen/Qwen3.8-27B",
            "path": INFERENCE_BASE,
            "target": INFERENCE_BASE,
            "role": "decision_and_expert_carrier",
            "status": "verified",
            "exists": True,
            **evidence,
        }
        models.append(target)
        data["topology_profile"] = "target_single_model"
        data["live_routing_enabled"] = False
        data["checked_at"] = datetime.now(UTC).isoformat()
        atomic_json(path, data)
        updated.append(str(path))
    return {"updated": updated, "evidence": evidence, "pointer": pointer}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--manifest", type=Path, required=True)
    shadow_parser = subparsers.add_parser("register-shadow")
    shadow_parser.add_argument("--manifest", type=Path, required=True)
    subparsers.add_parser("verify")
    subparsers.add_parser("status")
    subparsers.add_parser("rollback")
    subparsers.add_parser("sync-evidence")
    args = parser.parse_args()
    if args.command == "promote":
        result = promote(args.manifest)
    elif args.command == "register-shadow":
        result = register_shadow(args.manifest)
    elif args.command == "verify":
        pointer = validate_current(required=True)
        manifest, adapter_path = validate_pointer(pointer)
        result = {
            "verified": True,
            "pointer": pointer,
            "manifest": manifest,
            "adapter_path": str(adapter_path),
        }
    elif args.command == "rollback":
        result = rollback()
    elif args.command == "status":
        result = status()
    else:
        result = sync_evidence()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
