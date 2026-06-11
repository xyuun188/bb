"""Remote model-server monitor probe generation."""

from __future__ import annotations

import json
import textwrap

SERVER_MONITOR_REMOTE_COMMAND_TIMEOUT_SECONDS = 45


def display_provider_model_name(model_id: str | None) -> str:
    """Return a short label for the configured primary provider model."""
    value = str(model_id or "").strip()
    lowered = value.lower()
    if "qwen3" in lowered and "32b" in lowered:
        return "Qwen3-32B-AWQ" if "awq" in lowered else "Qwen3-32B"
    if "qwen3" in lowered and "14b" in lowered:
        return "Qwen3-14B-Instruct"
    if "qwen2.5" in lowered and "32b" in lowered:
        return "Qwen2.5-32B-Instruct"
    if "deepseek" in lowered:
        return value or "DeepSeek"
    return value or "Local Model"


def render_python_here_doc(script: str) -> str:
    """Wrap a Python script as a quoted remote shell heredoc command."""
    clean_script = script.strip()
    if "\nPY\n" in f"\n{clean_script}\n":
        raise ValueError("Remote monitor script cannot contain a bare PY heredoc delimiter.")
    return f"python3 - <<'PY'\n{clean_script}\nPY"


def render_server_monitor_probe(primary_model_id: str, primary_model_label: str) -> str:
    """Render the Python probe executed on the remote model server."""
    return textwrap.dedent(
        f"""
        import json
        import os
        import re
        import shutil
        import subprocess
        import time
        import urllib.error
        import urllib.request

        PRIMARY_MODEL_ID = {json.dumps(primary_model_id, ensure_ascii=False)}
        PRIMARY_MODEL_LABEL = {json.dumps(primary_model_label, ensure_ascii=False)}
        ERROR_TEXT_LIMIT = 240
        HTTP_BODY_READ_LIMIT = 512 * 1024
        MAX_MODEL_ROWS = 24
        SECRET_TEXT_RE = re.compile(
            r"(Authorization\\s*:\\s*Bearer\\s+)[^\\s,;\\\"']+"
            r"|((?:['\\\"]?\\b(?:api[_-]?key|api[_-]?secret|secret|password|"
            r"passphrase|token|authorization|access[_-]?key|access[_-]?token|webhook)"
            r"\\b['\\\"]?\\s*[:=]\\s*['\\\"]?))[^'\\\"\\s,;]+",
            re.IGNORECASE,
        )


        def safe_error(value, limit=ERROR_TEXT_LIMIT):
            text = str(value or "").strip()
            if not text:
                return ""

            def repl(match):
                auth_prefix = match.group(1)
                key_prefix = match.group(2)
                if auth_prefix:
                    return auth_prefix + "***"
                if key_prefix:
                    return key_prefix + "***"
                return "***"

            redacted = SECRET_TEXT_RE.sub(repl, text)
            if limit and len(redacted) > limit:
                return redacted[:limit] + "..."
            return redacted


        def elapsed_ms(started):
            return round((time.monotonic() - started) * 1000, 1)


        def read_response_text(response):
            raw = response.read(HTTP_BODY_READ_LIMIT + 1)
            truncated = len(raw) > HTTP_BODY_READ_LIMIT
            return raw[:HTTP_BODY_READ_LIMIT].decode("utf-8", "replace"), truncated


        def safe_string_list(values, limit=MAX_MODEL_ROWS):
            rows = []
            for value in values or []:
                text = safe_error(value, limit=160)
                if not text:
                    continue
                rows.append(text)
                if len(rows) >= limit:
                    break
            return rows


        def safe_model_map(value):
            if not isinstance(value, dict):
                return {{}}
            result = {{}}
            for key in list(value.keys())[:MAX_MODEL_ROWS]:
                result[safe_error(key, limit=80)] = safe_error(value.get(key), limit=160)
            return result


        def run_argv(args, timeout=4):
            try:
                p = subprocess.run(
                    args,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                )
                return p.returncode, p.stdout.strip(), p.stderr.strip()
            except Exception as exc:
                return 124, "", safe_error(exc)


        def to_float(value, default=0.0):
            try:
                return float(value)
            except Exception:
                return default


        def read_cpu():
            with open("/proc/stat", "r", encoding="utf-8") as f:
                parts = f.readline().split()[1:]
            nums = [int(x) for x in parts]
            idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
            total = sum(nums)
            return idle, total


        def cpu_percent():
            idle1, total1 = read_cpu()
            time.sleep(0.25)
            idle2, total2 = read_cpu()
            total_delta = max(total2 - total1, 1)
            idle_delta = max(idle2 - idle1, 0)
            return round((1 - idle_delta / total_delta) * 100, 1)


        def meminfo():
            data = {{}}
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    key, value = line.split(":", 1)
                    data[key] = int(value.strip().split()[0])
            total = data.get("MemTotal", 0) / 1024
            available = data.get("MemAvailable", 0) / 1024
            used = max(total - available, 0)
            return {{
                "total_mb": round(total, 1),
                "used_mb": round(used, 1),
                "available_mb": round(available, 1),
                "used_pct": round((used / total * 100) if total else 0, 1),
            }}


        def disk_usage(path):
            if not os.path.exists(path):
                return None
            usage = shutil.disk_usage(path)
            total = usage.total / 1024 / 1024 / 1024
            used = usage.used / 1024 / 1024 / 1024
            free = usage.free / 1024 / 1024 / 1024
            return {{
                "path": path,
                "total_gb": round(total, 1),
                "used_gb": round(used, 1),
                "free_gb": round(free, 1),
                "used_pct": round((used / total * 100) if total else 0, 1),
            }}


        def gpu_status():
            args = [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,"
                "temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ]
            code, out, err = run_argv(args, timeout=5)
            if code != 0:
                return {{
                    "available": False,
                    "error": err or out or "nvidia-smi unavailable",
                    "gpus": [],
                }}
            rows = []
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    continue
                used = to_float(parts[1])
                total = to_float(parts[2])
                rows.append(
                    {{
                        "name": parts[0],
                        "memory_used_mb": used,
                        "memory_total_mb": total,
                        "memory_used_pct": round((used / total * 100) if total else 0, 1),
                        "utilization_pct": to_float(parts[3]),
                        "temperature_c": to_float(parts[4]),
                        "power_w": to_float(parts[5]),
                    }}
                )
            return {{"available": bool(rows), "gpus": rows}}


        def gpu_processes():
            args = [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
            code, out, _err = run_argv(args, timeout=5)
            if code != 0:
                return []
            rows = []
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                rows.append(
                    {{
                        "pid": parts[0],
                        "process_name": parts[1],
                        "used_memory_mb": to_float(parts[2]),
                    }}
                )
            return rows


        def service_status(name):
            code, out, err = run_argv(["systemctl", "is-active", name], timeout=3)
            _pid_code, pid_out, _pid_err = run_argv(
                ["systemctl", "show", name, "-p", "MainPID", "--value"],
                timeout=3,
            )
            _since_code, since_out, _since_err = run_argv(
                ["systemctl", "show", name, "-p", "ActiveEnterTimestamp", "--value"],
                timeout=3,
            )
            pid = pid_out.strip()
            elapsed_out = ""
            if pid and pid != "0":
                _elapsed_code, elapsed_out, _elapsed_err = run_argv(
                    ["ps", "-p", pid, "-o", "etime="],
                    timeout=3,
                )
            return {{
                "name": name,
                "active": out.strip() == "active",
                "status": out.strip() or err.strip() or "unknown",
                "pid": pid if pid and pid != "0" else "",
                "active_since": since_out.strip(),
                "elapsed": elapsed_out.strip(),
            }}


        def vllm_model_service_status(models, endpoint):
            wanted = (PRIMARY_MODEL_ID or "").lower()
            model_ids = [str(m or "") for m in (models or []) if str(m or "")]
            active = bool(model_ids) and (
                not wanted
                or any(
                    wanted == m.lower() or wanted in m.lower() or m.lower() in wanted
                    for m in model_ids
                )
            )
            return {{
                "name": PRIMARY_MODEL_LABEL or PRIMARY_MODEL_ID or "Local LLM",
                "service_name": "vllm-openai-api",
                "provider_model": PRIMARY_MODEL_ID,
                "active": active,
                "status": "active" if active else "model_not_available",
                "pid": "",
                "active_since": "",
                "elapsed": "",
                "endpoint": endpoint,
                "models": model_ids,
            }}


        def http_json(url, timeout=3):
            started = time.monotonic()
            try:
                req = urllib.request.Request(url, headers={{"Accept": "application/json"}})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    text, truncated = read_response_text(resp)
                    return {{
                        "ok": 200 <= resp.status < 300,
                        "status_code": resp.status,
                        "latency_ms": elapsed_ms(started),
                        "truncated": truncated,
                        "data": json.loads(text),
                    }}
            except urllib.error.HTTPError as exc:
                try:
                    text, truncated = read_response_text(exc)
                except Exception:
                    text, truncated = "", False
                return {{
                    "ok": False,
                    "status_code": int(getattr(exc, "code", 0) or 0),
                    "latency_ms": elapsed_ms(started),
                    "truncated": truncated,
                    "error": safe_error(text or exc),
                    "data": None,
                }}
            except Exception as exc:
                return {{
                    "ok": False,
                    "status_code": 0,
                    "latency_ms": elapsed_ms(started),
                    "truncated": False,
                    "error": safe_error(exc),
                    "data": None,
                }}


        def endpoint_health(result):
            return {{
                "ok": bool(result.get("ok")),
                "status_code": int(result.get("status_code") or 0),
                "latency_ms": result.get("latency_ms"),
                "truncated": bool(result.get("truncated")),
                "error": result.get("error", ""),
            }}


        def primary_model_available(model_ids):
            wanted = (PRIMARY_MODEL_ID or "").lower()
            rows = [str(m or "") for m in (model_ids or []) if str(m or "")]
            return bool(rows) and (
                not wanted
                or any(
                    wanted == m.lower() or wanted in m.lower() or m.lower() in wanted
                    for m in rows
                )
            )


        def model_runtime():
            vllm = http_json("http://127.0.0.1:8000/v1/models", timeout=4)
            local_status = http_json("http://127.0.0.1:8001/models/status", timeout=4)
            local_health = http_json("http://127.0.0.1:8001/health", timeout=3)
            vllm_models = []
            if isinstance(vllm.get("data"), dict):
                vllm_models = [
                    item.get("id") or item.get("root") or ""
                    for item in vllm["data"].get("data", [])
                    if isinstance(item, dict)
                ]
            vllm_models = safe_string_list(vllm_models)
            vllm_model_ok = primary_model_available(vllm_models)
            tools = local_status.get("data") if isinstance(local_status.get("data"), dict) else {{}}
            local_available = bool(local_status.get("ok") or local_health.get("ok"))
            return {{
                "vllm": {{
                    "available": bool(vllm.get("ok") and vllm_model_ok),
                    "endpoint_available": bool(vllm.get("ok")),
                    "model_available": bool(vllm_model_ok),
                    "label": PRIMARY_MODEL_LABEL or PRIMARY_MODEL_ID or "Local LLM",
                    "provider_model": PRIMARY_MODEL_ID,
                    "endpoint": "127.0.0.1:8000/v1",
                    "models": vllm_models,
                    "health": endpoint_health(vllm),
                    "status": (
                        "active"
                        if vllm.get("ok") and vllm_model_ok
                        else "model_not_available"
                        if vllm.get("ok")
                        else "endpoint_unavailable"
                    ),
                    "error": vllm.get("error", ""),
                }},
                "local_ai_tools": {{
                    "available": local_available,
                    "endpoint": "127.0.0.1:8001",
                    "status": "active" if local_available else "endpoint_unavailable",
                    "status_health": endpoint_health(local_status),
                    "health": endpoint_health(local_health),
                    "trained_at": tools.get("trained_at"),
                    "shadow_sample_count": tools.get("shadow_sample_count"),
                    "trade_sample_count": tools.get("trade_sample_count"),
                    "sequence_sample_count": tools.get("sequence_sample_count"),
                    "text_sentiment_sample_count": tools.get("text_sentiment_sample_count"),
                    "models": safe_model_map(tools.get("models")),
                    "error": local_status.get("error") or local_health.get("error") or "",
                }},
            }}


        load = os.getloadavg()
        runtime = model_runtime()
        payload = {{
            "hostname": os.uname().nodename,
            "cpu": {{
                "usage_pct": cpu_percent(),
                "load_1m": load[0],
                "load_5m": load[1],
                "load_15m": load[2],
                "cores": os.cpu_count() or 0,
            }},
            "memory": meminfo(),
            "disks": [d for d in [disk_usage("/"), disk_usage("/data")] if d],
            "gpu": gpu_status(),
            "gpu_processes": gpu_processes(),
            "services": [
                vllm_model_service_status(
                    runtime.get("vllm", {{}}).get("models") or [],
                    runtime.get("vllm", {{}}).get("endpoint") or "127.0.0.1:8000/v1",
                ),
                service_status("local-ai-tools.service"),
            ],
            "model_runtime": runtime,
        }}
        print(json.dumps(payload, ensure_ascii=False))
        """
    ).strip()
