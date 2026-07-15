#!/usr/bin/env python3
"""Probe Phase 3 shadow LLM endpoints through platform loopback tunnels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import parse, request

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_runtime import (  # noqa: E402
    HIGH_RISK_REVIEW_TOKEN_CAP,
    apply_non_thinking_request_controls,
)


@dataclass(frozen=True, slots=True)
class ProbeSpec:
    name: str
    api_base: str
    model: str
    role: str
    max_tokens: int = 48
    allow_reasoning_prefix: bool = False


DEFAULT_PROBES = (
    ProbeSpec(
        name="decision_maker",
        api_base="http://127.0.0.1:18000/v1",
        model="qwen3-32b-trade",
        role="decision",
    ),
    ProbeSpec(
        name="high_risk_review",
        api_base="http://127.0.0.1:18002/v1",
        model="deepseek-r1-14b-risk",
        role="risk",
        max_tokens=HIGH_RISK_REVIEW_TOKEN_CAP,
        allow_reasoning_prefix=True,
    ),
    ProbeSpec(
        name="expert_pool",
        api_base="http://127.0.0.1:18003/v1",
        model="BB-FinQuant-Expert-14B",
        role="expert",
    ),
)


def _request_body(spec: ProbeSpec) -> dict[str, Any]:
    body = {
        "model": spec.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a Phase 3 shadow probe. Return exactly one compact JSON "
                    "object and no markdown."
                ),
            },
            {
                "role": "user",
                "content": f'Return exactly {{"ok":true,"role":"{spec.role}"}}.',
            },
        ],
        "max_tokens": int(spec.max_tokens),
        "temperature": 0,
    }
    return apply_non_thinking_request_controls(spec.model, body)


def _content_from_response(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    return str(message.get("content") or first.get("text") or "").strip()


def _parsed_json_object(content: str) -> dict[str, Any] | None:
    text = str(content or "").strip()
    if not text:
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _json_object_available(content: str) -> bool:
    parsed = _parsed_json_object(content)
    return isinstance(parsed, dict) and parsed.get("ok") is True


def _probe_url(api_base: str) -> str:
    parsed = parse.urlsplit(str(api_base or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("shadow probe URL must use http or https")
    if parsed.username or parsed.password:
        raise ValueError("shadow probe URL must not contain credentials")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("shadow probe URL must target a loopback tunnel")
    if parsed.query or parsed.fragment:
        raise ValueError("shadow probe URL must not contain query or fragment data")
    path = f"{parsed.path.rstrip('/')}/chat/completions"
    return parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def probe_one(spec: ProbeSpec, *, timeout_seconds: float) -> dict[str, Any]:
    body = _request_body(spec)
    url = _probe_url(spec.api_base)
    started = time.monotonic()
    req = request.Request(  # noqa: S310 - _probe_url permits loopback HTTP(S) only.
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(  # noqa: S310 - req uses the validated loopback URL above.
            req, timeout=timeout_seconds
        ) as response:
            payload = json.loads(response.read(256_000).decode("utf-8", "replace"))
        latency_ms = round((time.monotonic() - started) * 1000, 1)
        content = _content_from_response(payload)
        parsed = _parsed_json_object(content)
        raw_has_think_tag = "<think>" in content.lower() or "</think>" in content.lower()
        contract_json_ok = isinstance(parsed, dict) and parsed.get("ok") is True
        reasoning_prefix_allowed = bool(spec.allow_reasoning_prefix and contract_json_ok)
        return {
            "name": spec.name,
            "api_base": spec.api_base,
            "model": spec.model,
            "ok": bool(contract_json_ok and (not raw_has_think_tag or reasoning_prefix_allowed)),
            "endpoint_ok": True,
            "json_object_available": contract_json_ok,
            "raw_has_think_tag": raw_has_think_tag,
            "reasoning_prefix_allowed": reasoning_prefix_allowed,
            "max_tokens": int(spec.max_tokens),
            "latency_ms": latency_ms,
            "content_preview": content[:240],
        }
    except Exception as exc:
        return {
            "name": spec.name,
            "api_base": spec.api_base,
            "model": spec.model,
            "ok": False,
            "endpoint_ok": False,
            "json_object_available": False,
            "raw_has_think_tag": False,
            "reasoning_prefix_allowed": False,
            "max_tokens": int(spec.max_tokens),
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "error": str(exc)[:240],
        }


def collect_report(*, timeout_seconds: float) -> dict[str, Any]:
    rows = [probe_one(spec, timeout_seconds=timeout_seconds) for spec in DEFAULT_PROBES]
    return {
        "status": "ready" if all(row.get("ok") for row in rows) else "blocked",
        "shadow_only": True,
        "live_routing_enabled": False,
        "probe_count": len(rows),
        "ready_count": sum(1 for row in rows if row.get("ok")),
        "probes": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = collect_report(timeout_seconds=max(float(args.timeout_seconds or 1), 1.0))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.fail_on_blocked and report["status"] != "ready":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
