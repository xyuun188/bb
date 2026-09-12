#!/usr/bin/env python3
"""Measure a running Qwen3.8-27B endpoint and emit candidate probe evidence.

Run this inside the exact target inference environment on the model host.  The
probe is read-only: it does not download weights, change services, or write the
candidate manifest.  A separate validator fingerprints this evidence and all
model artifacts before deployment.
"""

# The direct-execution path bootstraps the repository root before importing the
# validator module; suppress the intentional post-bootstrap import warning.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_model_candidate import (
    MIN_RUNTIME_PROBE_REQUESTS,
    RUNTIME_PROBE_VERSION,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--context-length", required=True, type=int)
    parser.add_argument("--max-concurrency", default=1, type=int)
    parser.add_argument(
        "--runtime-engine",
        default="transformers",
        choices=("transformers", "vllm", "sglang"),
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--request-count", default=MIN_RUNTIME_PROBE_REQUESTS, type=int)
    parser.add_argument("--request-timeout-seconds", default=180.0, type=float)
    parser.add_argument("--long-prompt-tokens", default=4096, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _json_request(url: str, *, payload: dict | None, timeout: float) -> dict:
    parsed_url = urllib.parse.urlsplit(url)
    if parsed_url.scheme != "http" or parsed_url.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("runtime probe only permits a loopback HTTP endpoint")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - URL is restricted to loopback HTTP above.
        url,
        data=data,
        headers={"content-type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("runtime endpoint response must be a JSON object")
    return value


def _config_identity(model_path: Path) -> tuple[str, str]:
    config_path = model_path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read model config: {config_path}") from exc
    architectures = config.get("architectures") if isinstance(config, dict) else None
    model_type = config.get("model_type") if isinstance(config, dict) else None
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ValueError("model config must contain exactly one architecture")
    if not isinstance(architectures[0], str) or not isinstance(model_type, str):
        raise ValueError("model config identity is incomplete")
    return architectures[0], model_type


def _long_prompt(tokenizer_path: Path, requested_tokens: int, context_length: int) -> tuple[str, int]:
    target = min(max(int(requested_tokens), 256), max(int(context_length) - 128, 256))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    seed = (
        "BTC USDT perpetual market facts funding fee spread volatility liquidity "
        "risk position expected net return after all costs. "
    )
    seed_ids = tokenizer(seed, add_special_tokens=False).get("input_ids") or []
    if not seed_ids:
        raise ValueError("tokenizer could not encode the runtime probe prompt")
    repeated = (seed_ids * math.ceil(target / len(seed_ids)))[:target]
    return tokenizer.decode(repeated), len(repeated)


def _assistant_content(response: dict) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    return str(message.get("content") or "").strip()


def _valid_contract(content: str) -> bool:
    value = content.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
        if value.lower().startswith("json"):
            value = value[4:].strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return False
    return parsed == {"status": "ok"}


def _gpu_memory_used_gib() -> float:
    result = subprocess.run(  # noqa: S603
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    values = [float(line.strip()) / 1024.0 for line in result.stdout.splitlines() if line.strip()]
    if not values:
        raise RuntimeError("nvidia-smi returned no GPU memory measurements")
    return max(values)


def _percentile_95(values: list[float]) -> float:
    if not values:
        raise ValueError("latency sample is empty")
    ordered = sorted(values)
    return ordered[max(math.ceil(len(ordered) * 0.95) - 1, 0)]


def run_probe(args: argparse.Namespace) -> dict:
    model_path = args.model_path.expanduser().resolve()
    tokenizer_path = (args.tokenizer_path or model_path).expanduser().resolve()
    architecture, model_type = _config_identity(model_path)
    if int(args.max_concurrency) != 1:
        raise ValueError("single-A100 runtime probe requires max_concurrency=1")
    request_count = int(args.request_count)
    if request_count < MIN_RUNTIME_PROBE_REQUESTS:
        raise ValueError(f"runtime probe requires at least {MIN_RUNTIME_PROBE_REQUESTS} requests")
    model_listing = _json_request(
        args.endpoint.rstrip("/") + "/v1/models",
        payload=None,
        timeout=min(float(args.request_timeout_seconds), 30.0),
    )
    model_ids = {
        str(row.get("id") or "")
        for row in model_listing.get("data", [])
        if isinstance(row, dict)
    }
    if args.model_id not in model_ids:
        raise RuntimeError("runtime endpoint does not expose the requested model identity")

    long_prompt, tested_prompt_tokens = _long_prompt(
        tokenizer_path,
        int(args.long_prompt_tokens),
        int(args.context_length),
    )
    compact_prompt = '/no_think\nReturn exactly this JSON object and nothing else: {"status":"ok"}'
    latencies_ms: list[float] = []
    successful = 0
    oom_count = 0
    timeout_count = 0
    json_failures = 0
    peak_gpu_gib = _gpu_memory_used_gib()
    for index in range(request_count):
        prompt = compact_prompt if index else long_prompt + "\n" + compact_prompt
        payload = {
            "model": args.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.perf_counter()
        try:
            response = _json_request(
                args.endpoint.rstrip("/") + "/v1/chat/completions",
                payload=payload,
                timeout=float(args.request_timeout_seconds),
            )
        except (TimeoutError, urllib.error.URLError) as exc:
            message = str(exc).lower()
            if "out of memory" in message or "cuda oom" in message:
                oom_count += 1
            else:
                timeout_count += 1
            continue
        except (OSError, ValueError) as exc:
            message = str(exc).lower()
            if "out of memory" in message or "cuda oom" in message:
                oom_count += 1
            else:
                json_failures += 1
            continue
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        if _valid_contract(_assistant_content(response)):
            successful += 1
        else:
            json_failures += 1
        peak_gpu_gib = max(peak_gpu_gib, _gpu_memory_used_gib())

    transformers_version = metadata.version("transformers")
    engine_version = (
        transformers_version
        if args.runtime_engine == "transformers"
        else metadata.version(args.runtime_engine)
    )
    verified = (
        successful == request_count
        and oom_count == 0
        and timeout_count == 0
        and json_failures == 0
    )
    return {
        "probe_version": RUNTIME_PROBE_VERSION,
        "status": "verified" if verified else "failed",
        "observed_at": datetime.now(UTC).isoformat(),
        "model_id": args.model_id,
        "repo_id": args.repo_id,
        "revision": args.revision,
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "quantization": args.quantization,
        "architecture": architecture,
        "model_type": model_type,
        "runtime_engine": args.runtime_engine,
        "runtime_engine_version": engine_version,
        "transformers_version": transformers_version,
        "context_length": int(args.context_length),
        "max_concurrency": 1,
        "tested_prompt_tokens": tested_prompt_tokens,
        "gpu_memory_peak_gib": round(peak_gpu_gib, 6),
        "inference_p95_ms": round(_percentile_95(latencies_ms), 3) if latencies_ms else None,
        "text_inference_verified": verified,
        "request_count": request_count,
        "successful_request_count": successful,
        "oom_count": oom_count,
        "timeout_count": timeout_count,
        "json_contract_failure_count": json_failures,
    }


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
    result = run_probe(args)
    write_atomic(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") != "verified":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
