#!/usr/bin/env python3
"""Refresh the verified target-model evidence before it expires.

The refresh is fail-closed: the existing candidate manifest remains untouched
unless the running endpoint passes the complete pressure probe and the model
artifacts pass the candidate validator again.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_candidate_manifest import ModelCandidateManifest  # noqa: E402
from scripts.validate_model_candidate import write_atomic  # noqa: E402

DEFAULT_CANDIDATE_MANIFEST = Path("/data/BB/manifests/target_model_candidate.json")
DEFAULT_PROBE_OUTPUT = Path("/data/BB/manifests/qwen3.8-27b-runtime-probe.json")
DEFAULT_LOCK_PATH = Path("/data/BB/runtime/model-candidate-evidence-refresh.lock")
DEFAULT_ENDPOINT = "http://127.0.0.1:8000"
DEFAULT_REFRESH_AGE_HOURS = 120.0


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("candidate validated_at must include a timezone")
    return parsed.astimezone(UTC)


def candidate_age_hours(candidate: ModelCandidateManifest, *, now: datetime | None = None) -> float:
    checked_at = (now or datetime.now(UTC)).astimezone(UTC)
    age = checked_at - _parse_timestamp(candidate.validated_at)
    return max(age.total_seconds() / 3600.0, 0.0)


def refresh_required(
    candidate: ModelCandidateManifest,
    *,
    max_age_hours: float,
    force: bool = False,
    now: datetime | None = None,
) -> bool:
    return bool(force or candidate_age_hours(candidate, now=now) >= max(float(max_age_hours), 0.0))


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with path.open("r+b") as handle:
        if os.name != "nt":
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name != "nt":
                fcntl.flock(handle, fcntl.LOCK_UN)


def _probe_args(candidate: ModelCandidateManifest, args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_id=candidate.model_id,
        repo_id=candidate.repo_id,
        revision=candidate.revision,
        model_path=Path(candidate.model_path),
        tokenizer_path=Path(candidate.tokenizer_path),
        quantization=candidate.quantization,
        context_length=candidate.context_length,
        max_concurrency=candidate.max_concurrency,
        runtime_engine=candidate.runtime.engine,
        endpoint=args.endpoint,
        request_count=args.request_count,
        request_timeout_seconds=args.request_timeout_seconds,
        long_prompt_tokens=args.long_prompt_tokens,
    )


def _validator_args(
    candidate: ModelCandidateManifest,
    probe: dict[str, Any],
    probe_path: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_id=candidate.model_id,
        repo_id=candidate.repo_id,
        revision=candidate.revision,
        model_path=Path(candidate.model_path),
        tokenizer_path=Path(candidate.tokenizer_path),
        license=candidate.license,
        quantization=candidate.quantization,
        context_length=candidate.context_length,
        gpu_memory_peak_gib=probe["gpu_memory_peak_gib"],
        inference_p95_ms=probe["inference_p95_ms"],
        max_concurrency=candidate.max_concurrency,
        storage_required_free_gib=candidate.storage_required_free_gib,
        text_inference_verified=True,
        runtime_engine=candidate.runtime.engine,
        runtime_engine_version=str(probe["runtime_engine_version"]),
        transformers_version=str(probe["transformers_version"]),
        runtime_probe_file=probe_path,
        validator_version=candidate.validator_version,
    )


def refresh_candidate_evidence(
    args: argparse.Namespace,
    *,
    probe_runner: Callable[[Any], dict[str, Any]] | None = None,
    manifest_builder: Callable[[Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidate_path = Path(args.candidate_manifest).expanduser().resolve()
    candidate = ModelCandidateManifest.load(candidate_path)
    age_hours = candidate_age_hours(candidate)
    if not refresh_required(
        candidate,
        max_age_hours=args.max_age_hours,
        force=bool(args.force),
    ):
        return {
            "status": "fresh",
            "refreshed": False,
            "model_id": candidate.model_id,
            "validated_at": candidate.validated_at,
            "age_hours": round(age_hours, 3),
        }

    if probe_runner is None:
        from scripts.probe_model_candidate_runtime import run_probe

        probe_runner = run_probe
    if manifest_builder is None:
        from scripts.validate_model_candidate import build_manifest

        manifest_builder = build_manifest

    probe = probe_runner(_probe_args(candidate, args))
    if probe.get("status") != "verified":
        raise RuntimeError("target-model runtime pressure probe did not verify")

    probe_output = Path(args.probe_output).expanduser().resolve()
    probe_output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{probe_output.name}.", suffix=".tmp", dir=probe_output.parent
    )
    os.close(fd)
    temporary_probe = Path(temporary_name)
    try:
        temporary_probe.write_text(
            json.dumps(probe, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        refreshed = manifest_builder(_validator_args(candidate, probe, temporary_probe))
        ModelCandidateManifest.from_dict(refreshed).to_topology(stage="paper")
        write_atomic(candidate_path, refreshed)
        os.replace(temporary_probe, probe_output)
    finally:
        temporary_probe.unlink(missing_ok=True)

    return {
        "status": "refreshed",
        "refreshed": True,
        "model_id": refreshed["model_id"],
        "validated_at": refreshed["validated_at"],
        "request_count": probe.get("request_count"),
        "inference_p95_ms": probe.get("inference_p95_ms"),
        "gpu_memory_peak_gib": probe.get("gpu_memory_peak_gib"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--probe-output", type=Path, default=DEFAULT_PROBE_OUTPUT)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--max-age-hours", type=float, default=DEFAULT_REFRESH_AGE_HOURS)
    parser.add_argument("--request-count", type=int, default=20)
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--long-prompt-tokens", type=int, default=4096)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        with _exclusive_lock(Path(args.lock_path)):
            result = refresh_candidate_evidence(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)[:500]}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
