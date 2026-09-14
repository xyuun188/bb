"""Measure the live Qwen3.8-27B request path without changing trading state.

The probe is deliberately read-only.  It uses the configured model-server
credentials, sends a tiny deterministic JSON request, and reports timing
percentiles plus service/GPU pressure.  It never touches orders, positions,
training state, or routing flags.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402


def _remote_command(runs: int, timeout: int, prompt_chars: int) -> str:
    # Keep the script stdlib-only so it works in the target inference env.
    payload = {
        "model": "qwen3.8-27b",
        "messages": [
            {
                "role": "user",
                "content": "/no_think Return exactly {\"status\":\"ok\"}"
                + ("\\ncontext=" + ("x" * max(int(prompt_chars) - 54, 0))),
            }
        ],
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    encoded = json.dumps(payload, ensure_ascii=False)
    remote = f"""
set -u
python3 - {shlex.quote(encoded)} {int(runs)} <<'PY'
import json, sys, time, urllib.request

payload = json.loads(sys.argv[1])
runs = max(int(sys.argv[2]), 1)
# Keep probes inside the target API contract (max_tokens <= 256). A 512-token
# request is a client-side validation error, not a model-latency measurement.
for tokens in (32, 64, 128, 256):
    values = []
    for index in range(runs):
        body = dict(payload)
        body["max_tokens"] = tokens
        request = urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={{"content-type": "application/json"}},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout={int(timeout)}) as response:
                raw = response.read()
                status = response.status
            elapsed = time.perf_counter() - started
            values.append(elapsed)
            print("sample tokens=%d run=%d status=%d seconds=%.3f bytes=%d" %
                  (tokens, index + 1, status, elapsed, len(raw)), flush=True)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print("sample tokens=%d run=%d error=%s detail=%s seconds=%.3f" %
                  (tokens, index + 1, type(exc).__name__, str(exc)[:180], elapsed), flush=True)
    if values:
        ordered = sorted(values)
        p50 = ordered[len(ordered) // 2]
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        print("summary tokens=%d count=%d p50=%.3f p95=%.3f max=%.3f" %
              (tokens, len(values), p50, p95, max(values)), flush=True)
PY
echo ---service-metrics---
systemctl show bb-phase3-llm-target.service \\
  -p ActiveState -p MainPID -p CPUUsageNSec -p MemoryCurrent -p TasksCurrent -p NRestarts || true
echo ---health-ready---
curl -sS --max-time 8 http://127.0.0.1:8000/health/ready || true
echo ---start-script---
sed -n '1,180p' /data/BB/scripts/start_target_single_model.sh 2>/dev/null || true
echo ---runtime-packages---
/data/BB/envs/target-inference/bin/python - <<'PY' 2>/dev/null || true
import importlib.util
for name in ("vllm", "torch", "transformers", "flash_attn", "fla", "causal_conv1d"):
    print("%s=%s" % (name, bool(importlib.util.find_spec(name))))
PY
echo ---gpu---
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
echo ---journal-tail---
journalctl -u bb-phase3-llm-target.service -n 80 --no-pager 2>/dev/null || true
""".strip()
    return remote


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--prompt-chars", type=int, default=64)
    args = parser.parse_args()
    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    try:
        output = run_remote_text(
            ssh,
            _remote_command(
                max(int(args.runs), 1),
                max(int(args.timeout), 5),
                max(int(args.prompt_chars), 64),
            ),
            timeout=240,
            max_output_chars=20_000,
            check=False,
        )
    finally:
        ssh.close()
    print(output, end="" if output.endswith("\n") else "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
