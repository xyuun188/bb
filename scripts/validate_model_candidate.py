#!/usr/bin/env python3
"""Generate a verified single-model candidate manifest on the model host.

This command is deliberately explicit: it never downloads weights, guesses a
Hugging Face revision, or labels a model verified without measured runtime
evidence supplied by the operator/benchmark harness.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from core.model_candidate_manifest import MANIFEST_VERSION, ModelCandidateManifest, sha256_file

WEIGHT_SUFFIXES = {".safetensors", ".bin", ".gguf", ".pt", ".pth"}
TOKENIZER_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


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


def build_manifest(args: argparse.Namespace) -> dict:
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise ValueError(f"model path does not exist or is not a directory: {model_path}")
    tokenizer_path = (args.tokenizer_path or model_path).expanduser().resolve()
    config_path = _require_file(model_path / "config.json", "model config")
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
        "context_length": args.context_length,
        "config_sha256": sha256_file(config_path),
        "tokenizer_sha256": _hash_tokenizer(tokenizer_path),
        "weight_files": list(weight_files),
        "gpu_memory_peak_gib": args.gpu_memory_peak_gib,
        "inference_p95_ms": args.inference_p95_ms,
        "max_concurrency": args.max_concurrency,
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
