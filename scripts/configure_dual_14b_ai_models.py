"""Generate or apply the approved dual-14B fixed expert model config.

Default behavior prints the AI_MODELS JSON that assigns:
- Qwen3-14B-AWQ vLLM to trend/momentum/final decision.
- DeepSeek-R1-Distill-Qwen-14B-AWQ vLLM to sentiment/position/risk.

The payload intentionally leaves api_key empty. At runtime LLMAgent falls back to
AI_API_KEY when a per-slot key is not set, so this script can configure model
routing without reading or exposing secrets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib import parse, request

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import FIXED_AI_MODEL_SLOTS  # noqa: E402
from core.remote_ai_service_spec import (  # noqa: E402
    DEEPSEEK_R1_14B_RISK_SERVICE,
    QWEN3_14B_TRADE_SERVICE,
)
from core.safe_output import safe_print  # noqa: E402

QWEN_EXPERT_NAMES = {"trend_expert", "momentum_expert", "decision_maker"}
R1_EXPERT_NAMES = {"sentiment_expert", "position_expert", "risk_expert"}


def _api_base(host: str, port: int) -> str:
    normalized = str(host or "127.0.0.1").strip().rstrip("/")
    if normalized.startswith(("http://", "https://")):
        return f"{normalized}:{port}/v1"
    return f"http://{normalized}:{port}/v1"


def build_dual_14b_ai_models(*, host: str = "127.0.0.1") -> list[dict[str, Any]]:
    """Return fixed-slot AI_MODELS entries for the dual-14B deployment."""
    qwen_base = _api_base(host, QWEN3_14B_TRADE_SERVICE.port)
    r1_base = _api_base(host, DEEPSEEK_R1_14B_RISK_SERVICE.port)
    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for slot in FIXED_AI_MODEL_SLOTS:
        name = str(slot["name"])
        if name in QWEN_EXPERT_NAMES:
            api_base = qwen_base
            model = QWEN3_14B_TRADE_SERVICE.served_model_name
        elif name in R1_EXPERT_NAMES:
            api_base = r1_base
            model = DEEPSEEK_R1_14B_RISK_SERVICE.served_model_name
        else:
            raise ValueError(f"No dual-14B provider assignment for fixed slot: {name}")
        seen.add(name)
        models.append(
            {
                "name": name,
                "role": slot["role"],
                "label": slot["label"],
                "weight": slot["weight"],
                "api_base": api_base,
                "api_key": "",
                "model": model,
                "enabled": True,
            }
        )
    missing = (QWEN_EXPERT_NAMES | R1_EXPERT_NAMES) - seen
    if missing:
        raise ValueError(f"Missing fixed AI model slots: {sorted(missing)}")
    return models


def _json_payload(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _validate_local_dashboard_url(value: str) -> str:
    url = str(value or "").strip().rstrip("/")
    parsed = parse.urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("--apply-dashboard only supports local http Dashboard URLs.")
    return url


def _apply_via_dashboard(models: list[dict[str, Any]], *, dashboard_url: str) -> None:
    base = _validate_local_dashboard_url(dashboard_url)
    for model in models:
        name = str(model["name"])
        payload = json.dumps(
            {
                "name": name,
                "api_base": model["api_base"],
                "api_key": "",
                "model": model["model"],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        req = request.Request(  # noqa: S310 - validated local Dashboard URL only.
            f"{base}/api/settings/ai-models/{name}",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        with request.urlopen(req, timeout=20) as response:  # noqa: S310
            body = response.read().decode("utf-8", errors="replace")
        safe_print(f"updated {name}: {body[:240]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host/IP used by this backend to reach the vLLM services.",
    )
    parser.add_argument(
        "--apply-dashboard",
        action="store_true",
        help="Apply slot updates through the running local Dashboard settings API.",
    )
    parser.add_argument(
        "--dashboard-url",
        default="http://127.0.0.1:8002",
        help="Dashboard base URL used with --apply-dashboard.",
    )
    args = parser.parse_args(argv)

    models = build_dual_14b_ai_models(host=args.host)
    if args.apply_dashboard:
        _apply_via_dashboard(models, dashboard_url=args.dashboard_url)
        return

    safe_print("AI_MODELS=" + _json_payload(models))


if __name__ == "__main__":
    main()
