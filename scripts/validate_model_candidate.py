#!/usr/bin/env python3
"""Generate a verified single-model candidate manifest on the model host.

This command is deliberately explicit: it never downloads weights, guesses a
Hugging Face revision, or labels a model verified without measured runtime
evidence supplied by the operator/benchmark harness.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_candidate_manifest import (  # noqa: E402
    MANIFEST_VERSION,
    MAX_MANIFEST_AGE,
    MAX_MANIFEST_FUTURE_SKEW,
    ModelCandidateManifest,
    sha256_file,
)

WEIGHT_SUFFIXES = {".safetensors", ".bin", ".gguf", ".pt", ".pth"}
TOKENIZER_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
RUNTIME_PROBE_VERSION = "bb.model-runtime-probe.v1"
MIN_RUNTIME_PROBE_REQUESTS = 20


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--license", required=True)
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--context-length", required=True, type=int)
    parser.add_argument("--gpu-memory-peak-gib", required=True, type=float)
    parser.add_argument("--inference-p95-ms", required=True, type=float)
    parser.add_argument("--max-concurrency", required=True, type=int)
    parser.add_argument("--storage-required-free-gib", required=True, type=float)
    parser.add_argument(
        "--text-inference-verified",
        action="store_true",
        help="Set only after the isolated runtime completed a text-only inference probe.",
    )
    parser.add_argument(
        "--runtime-engine",
        required=True,
        choices=("transformers", "vllm", "sglang"),
    )
    parser.add_argument("--runtime-engine-version", required=True)
    parser.add_argument("--transformers-version", required=True)
    parser.add_argument(
        "--runtime-probe-file",
        required=True,
        type=Path,
        help="Saved JSON evidence from the successful isolated runtime probe.",
    )
    parser.add_argument("--validator-version", default="bb-model-validator.v1")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist or is not a file: {resolved}")
    return resolved


def _weight_files(model_path: Path) -> tuple[dict, ...]:
    files = sorted(
        path for path in model_path.rglob("*") if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES
    )
    if not files:
        raise ValueError(f"no model weight files found under {model_path}")
    return tuple(
        {
            "path": path.relative_to(model_path).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    )


def _hash_tokenizer(tokenizer_path: Path) -> str:
    if not tokenizer_path.is_dir():
        raise ValueError(f"tokenizer path does not exist or is not a directory: {tokenizer_path}")
    existing = [_require_file(tokenizer_path / name, f"tokenizer file {name}") for name in TOKENIZER_NAMES if (tokenizer_path / name).is_file()]
    if not existing:
        raise ValueError(f"no supported tokenizer files found under {tokenizer_path}")
    payload = "\n".join(f"{item.name}:{sha256_file(item)}" for item in existing).encode("utf-8")
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _available_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def _read_runtime_probe(path: Path) -> dict:
    source = _require_file(path, "runtime probe")
    try:
        probe = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"runtime probe is not valid JSON: {source}") from exc
    if not isinstance(probe, dict):
        raise ValueError("runtime probe must be a JSON object")
    if probe.get("probe_version") != RUNTIME_PROBE_VERSION:
        raise ValueError("runtime probe version is unsupported")
    if probe.get("status") != "verified":
        raise ValueError("runtime probe status must be verified")
    observed_at = probe.get("observed_at")
    try:
        observed = datetime.fromisoformat(str(observed_at or "").replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("runtime probe observed_at is invalid") from exc
    if observed.tzinfo is None:
        raise ValueError("runtime probe observed_at must include a timezone")
    now = datetime.now(UTC)
    observed = observed.astimezone(UTC)
    if observed > now + MAX_MANIFEST_FUTURE_SKEW:
        raise ValueError("runtime probe observed_at is in the future")
    if now - observed > MAX_MANIFEST_AGE:
        raise ValueError("runtime probe evidence is stale")
    request_count = probe.get("request_count")
    successful_count = probe.get("successful_request_count")
    if (
        isinstance(request_count, bool)
        or not isinstance(request_count, int)
        or request_count < MIN_RUNTIME_PROBE_REQUESTS
        or successful_count != request_count
    ):
        raise ValueError("runtime probe does not contain a complete pressure-test sample")
    for field in ("oom_count", "timeout_count", "json_contract_failure_count"):
        if probe.get(field) != 0:
            raise ValueError(f"runtime probe has non-zero {field}")
    return probe


def _same_number(actual: object, expected: float | int) -> bool:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-6)


def _validate_runtime_probe(
    probe: dict,
    args: argparse.Namespace,
    *,
    model_path: Path,
    tokenizer_path: Path,
    architecture: str,
    model_type: str,
) -> None:
    expected = {
        "model_id": args.model_id,
        "repo_id": args.repo_id,
        "revision": args.revision,
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "quantization": args.quantization,
        "architecture": architecture,
        "model_type": model_type,
        "runtime_engine": args.runtime_engine,
        "runtime_engine_version": args.runtime_engine_version,
        "transformers_version": args.transformers_version,
    }
    mismatches = [field for field, value in expected.items() if probe.get(field) != value]
    for field, value in (
        ("context_length", args.context_length),
        ("max_concurrency", args.max_concurrency),
        ("gpu_memory_peak_gib", args.gpu_memory_peak_gib),
        ("inference_p95_ms", args.inference_p95_ms),
    ):
        if not _same_number(probe.get(field), value):
            mismatches.append(field)
    if probe.get("text_inference_verified") is not True or not args.text_inference_verified:
        mismatches.append("text_inference_verified")
    if mismatches:
        raise ValueError(
            "runtime probe does not match candidate inputs: " + ", ".join(sorted(set(mismatches)))
        )


def _architecture_identity(config_path: Path) -> tuple[str, str, bool]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"model config is not valid JSON: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError("model config must be an object")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ValueError("model config must declare exactly one architecture")
    architecture = architectures[0]
    model_type = config.get("model_type")
    if not isinstance(architecture, str) or not isinstance(model_type, str):
        raise ValueError("model config is missing architecture identity")
    # Qwen3.8 is a multimodal model and its official config does not carry a
    # project-specific ``language_model_only`` key.  Absence therefore means
    # false; text-only request support is proved separately by the runtime
    # probe and ``--text-inference-verified`` acknowledgement.
    language_model_only = config.get("language_model_only", False)
    if not isinstance(language_model_only, bool):
        raise ValueError("model config language_model_only identity must be boolean")
    return architecture, model_type, language_model_only


def build_manifest(args: argparse.Namespace) -> dict:
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise ValueError(f"model path does not exist or is not a directory: {model_path}")
    tokenizer_path = (args.tokenizer_path or model_path).expanduser().resolve()
    config_path = _require_file(model_path / "config.json", "model config")
    architecture, model_type, language_model_only = _architecture_identity(config_path)
    runtime_probe = _require_file(args.runtime_probe_file, "runtime probe")
    runtime_probe_payload = _read_runtime_probe(runtime_probe)
    _validate_runtime_probe(
        runtime_probe_payload,
        args,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        architecture=architecture,
        model_type=model_type,
    )
    weight_files = _weight_files(model_path)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "status": "verified",
        "model_id": args.model_id,
        "repo_id": args.repo_id,
        "revision": args.revision,
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "license": args.license,
        "quantization": args.quantization,
        "architecture": architecture,
        "model_type": model_type,
        "language_model_only": language_model_only,
        "text_inference_verified": args.text_inference_verified,
        "runtime": {
            "engine": args.runtime_engine,
            "engine_version": args.runtime_engine_version,
            "transformers_version": args.transformers_version,
            "probe_sha256": sha256_file(runtime_probe),
        },
        "context_length": args.context_length,
        "config_sha256": sha256_file(config_path),
        "tokenizer_sha256": _hash_tokenizer(tokenizer_path),
        "weight_files": list(weight_files),
        "gpu_memory_peak_gib": args.gpu_memory_peak_gib,
        "inference_p95_ms": args.inference_p95_ms,
        "max_concurrency": args.max_concurrency,
        "storage_available_gib": _available_gib(model_path),
        "storage_required_free_gib": args.storage_required_free_gib,
        "validated_at": datetime.now(UTC).isoformat(),
        "validator_version": args.validator_version,
    }
    parsed = ModelCandidateManifest.from_dict(manifest)
    parsed.to_topology()
    return parsed.to_dict()


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        manifest = build_manifest(args)
        write_atomic(args.output.expanduser().resolve(), manifest)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"candidate validation failed: {exc}") from None
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
