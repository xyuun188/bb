#!/usr/bin/env python3
"""Build a training-backend manifest from independently captured probe facts.

This command does not run training and does not infer capability from installed
packages.  It accepts only a completed forward/backward, DPO, save/reload and
resource probe bound to the same immutable model candidate and trainer bytes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_candidate_manifest import (  # noqa: E402
    MAX_MANIFEST_AGE,
    MAX_MANIFEST_FUTURE_SKEW,
    ModelCandidateManifest,
    sha256_file,
)
from core.model_training_backend import (  # noqa: E402
    TRAINING_BACKEND_ID,
    TRAINING_BACKEND_MANIFEST_VERSION,
    TRAINING_COMPUTE_DTYPE,
    TRAINING_LOADER_CLASS,
    TRAINING_QUANTIZATION,
    ModelTrainingBackendManifest,
)

TRAINING_PROBE_VERSION = "bb.model-training-capability-probe.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--trainer-file", required=True, type=Path)
    parser.add_argument("--probe-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validator-version", default="bb-training-backend-validator.v1")
    return parser


def _json_object(path: Path, *, label: str) -> dict:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"{label} does not exist: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {source}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _observed_at(value: object) -> str:
    try:
        observed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("training probe observed_at is invalid") from exc
    if observed.tzinfo is None:
        raise ValueError("training probe observed_at must include a timezone")
    observed = observed.astimezone(UTC)
    now = datetime.now(UTC)
    if observed > now + MAX_MANIFEST_FUTURE_SKEW:
        raise ValueError("training probe observed_at is in the future")
    if now - observed > MAX_MANIFEST_AGE:
        raise ValueError("training probe evidence is stale")
    return observed.isoformat()


def _require_probe_contract(probe: dict, candidate: ModelCandidateManifest) -> None:
    if probe.get("probe_version") != TRAINING_PROBE_VERSION:
        raise ValueError("training capability probe version is unsupported")
    if probe.get("status") != "verified":
        raise ValueError("training capability probe status must be verified")
    expected = {
        "backend_id": TRAINING_BACKEND_ID,
        "model_id": candidate.model_id,
        "repo_id": candidate.repo_id,
        "revision": candidate.revision,
        "architecture": candidate.architecture,
        "model_type": candidate.model_type,
        "training_model_path": candidate.model_path,
        "model_config_sha256": candidate.config_sha256,
        "tokenizer_sha256": candidate.tokenizer_sha256,
        "loader_class": TRAINING_LOADER_CLASS,
        "quantization": TRAINING_QUANTIZATION,
        "compute_dtype": TRAINING_COMPUTE_DTYPE,
    }
    mismatches = [field for field, expected_value in expected.items() if probe.get(field) != expected_value]
    for field in (
        "text_forward_backward_verified",
        "dpo_step_verified",
        "adapter_save_reload_verified",
    ):
        if probe.get(field) is not True:
            mismatches.append(field)
    if mismatches:
        raise ValueError(
            "training capability probe does not match candidate: "
            + ", ".join(sorted(set(mismatches)))
        )
    _observed_at(probe.get("observed_at"))


def build_manifest(args: argparse.Namespace) -> dict:
    candidate = ModelCandidateManifest.load(args.candidate_manifest)
    evidence_errors = candidate.validate_evidence()
    if evidence_errors:
        raise ValueError("candidate evidence is invalid: " + ", ".join(evidence_errors))
    trainer_path = args.trainer_file.expanduser().resolve()
    if not trainer_path.is_file():
        raise ValueError(f"trainer file does not exist: {trainer_path}")
    probe_path = args.probe_file.expanduser().resolve()
    probe = _json_object(probe_path, label="training capability probe")
    _require_probe_contract(probe, candidate)
    payload = {
        "manifest_version": TRAINING_BACKEND_MANIFEST_VERSION,
        "status": "verified",
        "backend_id": TRAINING_BACKEND_ID,
        "model_id": candidate.model_id,
        "repo_id": candidate.repo_id,
        "revision": candidate.revision,
        "architecture": candidate.architecture,
        "model_type": candidate.model_type,
        "training_model_path": candidate.model_path,
        "model_config_sha256": candidate.config_sha256,
        "tokenizer_sha256": candidate.tokenizer_sha256,
        "loader_class": TRAINING_LOADER_CLASS,
        "quantization": TRAINING_QUANTIZATION,
        "compute_dtype": TRAINING_COMPUTE_DTYPE,
        "torch_version": probe.get("torch_version"),
        "transformers_version": probe.get("transformers_version"),
        "peft_version": probe.get("peft_version"),
        "trl_version": probe.get("trl_version"),
        "bitsandbytes_version": probe.get("bitsandbytes_version"),
        "trainer_sha256": sha256_file(trainer_path),
        "runtime_probe_sha256": sha256_file(probe_path),
        "text_forward_backward_verified": True,
        "dpo_step_verified": True,
        "adapter_save_reload_verified": True,
        "gpu_memory_peak_gib": probe.get("gpu_memory_peak_gib"),
        "storage_available_gib": probe.get("storage_available_gib"),
        "storage_required_free_gib": probe.get("storage_required_free_gib"),
        "validated_at": _observed_at(probe.get("observed_at")),
        "validator_version": args.validator_version,
    }
    backend = ModelTrainingBackendManifest.from_dict(payload)
    errors = backend.validate_for_candidate(
        candidate,
        trainer_sha256=sha256_file(trainer_path),
    )
    if errors:
        raise ValueError("training backend evidence is invalid: " + ", ".join(errors))
    return backend.to_dict()


def write_atomic(path: Path, payload: dict) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        manifest = build_manifest(args)
        write_atomic(args.output, manifest)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"training backend validation failed: {exc}") from None
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
