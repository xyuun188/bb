#!/usr/bin/env python3
"""Download and validate the Phase 3 32B decision-maker model on the model server."""

from __future__ import annotations

import argparse
import posixpath
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

PHASE3_ROOT = "/data/BB"
PYTHON_BIN = f"{PHASE3_ROOT}/envs/phase3-quant/bin/python"
REMOTE_SCRIPT = f"{PHASE3_ROOT}/scripts/download_phase3_decision_32b.py"
PID_PATH = f"{PHASE3_ROOT}/runtime/download_phase3_decision_32b.pid"
LOG_PATH = f"{PHASE3_ROOT}/logs/downloads/download_phase3_decision_32b.log"
MODEL_REPO = "Qwen/Qwen3-32B-AWQ"
TARGET_DIR = f"{PHASE3_ROOT}/models/llm_decision_maker/Qwen--Qwen3-32B-AWQ"
DOWNLOAD_MANIFEST = f"{PHASE3_ROOT}/manifests/phase3_model_download_manifest.json"
VALIDATION_MANIFEST = f"{PHASE3_ROOT}/manifests/phase3_model_validation.json"
REPORT_DOWNLOAD_MANIFEST = (
    f"{PHASE3_ROOT}/reports/inventory/phase3_model_download_manifest_latest.json"
)
REPORT_VALIDATION_MANIFEST = (
    f"{PHASE3_ROOT}/reports/inventory/phase3_model_validation_latest.json"
)


def sh(value: str | int | float) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def render_remote_downloader() -> str:
    """Return the Python downloader written to the model server."""

    return textwrap.dedent(
        f"""
        from __future__ import annotations

        import json
        import os
        import time
        from datetime import datetime, timezone
        from pathlib import Path

        PHASE3_ROOT = {PHASE3_ROOT!r}
        MODEL_REPO = {MODEL_REPO!r}
        TARGET_DIR = Path({TARGET_DIR!r})
        DOWNLOAD_MANIFEST = Path({DOWNLOAD_MANIFEST!r})
        VALIDATION_MANIFEST = Path({VALIDATION_MANIFEST!r})
        REPORT_DOWNLOAD_MANIFEST = Path({REPORT_DOWNLOAD_MANIFEST!r})
        REPORT_VALIDATION_MANIFEST = Path({REPORT_VALIDATION_MANIFEST!r})
        EXPECTED_LLM_CANDIDATES = {{
            "decision_maker": MODEL_REPO,
            "expert_pool": "Qwen/Qwen3-14B-AWQ",
            "high_risk_review": "casperhansen/deepseek-r1-distill-qwen-14b-awq",
        }}


        def now_iso() -> str:
            return datetime.now(timezone.utc).isoformat()


        def read_json(path: Path) -> dict:
            if not path.exists():
                return {{}}
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {{}}
            return data if isinstance(data, dict) else {{}}


        def write_json(path: Path, data: dict) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
            tmp.replace(path)


        def artifact_stats(path: Path) -> dict:
            files = [item for item in path.rglob("*") if item.is_file()]
            return {{
                "file_count": len(files),
                "size_bytes": sum(item.stat().st_size for item in files),
            }}


        def validate_artifact(path: Path) -> dict:
            required_any = [
                path / "tokenizer.json",
                path / "tokenizer.model",
                path / "tokenizer_config.json",
            ]
            missing = []
            if not (path / "config.json").exists():
                missing.append("config.json")
            if not any(item.exists() and item.stat().st_size > 0 for item in required_any):
                missing.append("tokenizer")
            shards = list(path.glob("*.safetensors"))
            index_path = path / "model.safetensors.index.json"
            if index_path.exists():
                try:
                    index = json.loads(index_path.read_text(encoding="utf-8"))
                    required_shards = sorted(set((index.get("weight_map") or {{}}).values()))
                except Exception:
                    required_shards = []
                shard_missing = [
                    name
                    for name in required_shards
                    if not (path / name).exists() or (path / name).stat().st_size <= 0
                ]
                missing.extend(shard_missing)
            elif not shards:
                missing.append("*.safetensors")
            else:
                missing.extend(str(item.name) for item in shards if item.stat().st_size <= 0)
            stats = artifact_stats(path)
            return {{
                **stats,
                "exists": path.exists(),
                "required_missing": missing,
                "required_any_ok": not missing,
                "status": "ok" if not missing else "incomplete",
            }}


        def update_model_row(data: dict, *, validation: bool) -> dict:
            rows = data.get("models")
            if not isinstance(rows, list):
                rows = []
            next_rows = []
            replaced = False
            stats = validate_artifact(TARGET_DIR)
            for item in rows:
                row = dict(item) if isinstance(item, dict) else {{}}
                if row.get("slot") != "llm_decision_maker":
                    next_rows.append(row)
                    continue
                row.update(
                    {{
                        "slot": "llm_decision_maker",
                        "repo_id": MODEL_REPO,
                        "target": str(TARGET_DIR),
                        "path": str(TARGET_DIR),
                        "role": "decision_maker",
                        "stage": "shadow_candidate_not_live",
                        "live_routing_enabled": False,
                        "checked_at": now_iso(),
                        "validation_note": "Phase 3 corrected 32B decision-maker artifact.",
                        **stats,
                    }}
                )
                next_rows.append(row)
                replaced = True
            if not replaced:
                next_rows.append(
                    {{
                        "slot": "llm_decision_maker",
                        "repo_id": MODEL_REPO,
                        "target": str(TARGET_DIR),
                        "path": str(TARGET_DIR),
                        "role": "decision_maker",
                        "stage": "shadow_candidate_not_live",
                        "live_routing_enabled": False,
                        "checked_at": now_iso(),
                        **stats,
                    }}
                )
            data["models"] = next_rows
            data["checked_at" if validation else "updated_at"] = now_iso()
            data.setdefault("storage_root", PHASE3_ROOT)
            return data


        def update_policy(data: dict) -> dict:
            policy = data.get("policy")
            if not isinstance(policy, dict):
                policy = {{}}
            llm_candidates = policy.get("llm_candidates")
            if not isinstance(llm_candidates, dict):
                llm_candidates = {{}}
            llm_candidates.update(EXPECTED_LLM_CANDIDATES)
            policy["llm_candidates"] = llm_candidates
            policy["llm_candidates_not_activated"] = True
            policy["llm_live_routing_enabled"] = False
            policy["quant_server_only"] = True
            data["policy"] = policy
            return data


        def update_manifests() -> None:
            download = read_json(DOWNLOAD_MANIFEST)
            download = update_policy(update_model_row(download, validation=False))
            nested = download.get("validation")
            if isinstance(nested, dict):
                download["validation"] = update_model_row(nested, validation=True)
            write_json(DOWNLOAD_MANIFEST, download)
            write_json(REPORT_DOWNLOAD_MANIFEST, download)

            validation = read_json(VALIDATION_MANIFEST)
            validation = update_model_row(validation, validation=True)
            write_json(VALIDATION_MANIFEST, validation)
            write_json(REPORT_VALIDATION_MANIFEST, validation)


        def main() -> None:
            os.environ.setdefault("HF_HOME", f"{{PHASE3_ROOT}}/runtime/huggingface")
            os.environ.setdefault("HF_HUB_CACHE", f"{{PHASE3_ROOT}}/runtime/huggingface/hub")
            os.environ.setdefault("TRANSFORMERS_CACHE", f"{{PHASE3_ROOT}}/runtime/huggingface/transformers")
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
            print(json.dumps({{"event": "download_start", "repo": MODEL_REPO, "target": str(TARGET_DIR), "at": now_iso()}}, ensure_ascii=False), flush=True)
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=MODEL_REPO,
                local_dir=str(TARGET_DIR),
                max_workers=8,
                resume_download=True,
            )
            stats = validate_artifact(TARGET_DIR)
            print(json.dumps({{"event": "validation", **stats, "at": now_iso()}}, ensure_ascii=False), flush=True)
            if stats["required_missing"]:
                raise SystemExit("incomplete Phase 3 decision-maker artifact: " + ",".join(stats["required_missing"]))
            update_manifests()
            print(json.dumps({{"event": "manifest_updated", "status": "ok", "at": now_iso()}}, ensure_ascii=False), flush=True)


        if __name__ == "__main__":
            started = time.monotonic()
            try:
                main()
            finally:
                print(json.dumps({{"event": "finished", "duration_seconds": round(time.monotonic() - started, 3)}}, ensure_ascii=False), flush=True)
        """
    ).strip()


def _upload_command() -> str:
    script = render_remote_downloader()
    if "\nPY\n" in f"\n{script}\n":
        raise ValueError("remote downloader cannot contain a bare PY delimiter")
    return (
        f"mkdir -p {sh(posixpath.dirname(REMOTE_SCRIPT))} {sh(posixpath.dirname(LOG_PATH))} "
        f"{sh(posixpath.dirname(PID_PATH))} {sh(posixpath.dirname(TARGET_DIR))} && "
        f"cat > {sh(REMOTE_SCRIPT)} <<'PY'\n{script}\nPY\n"
    )


def _start_command() -> str:
    return (
        f"test -x {sh(PYTHON_BIN)} && "
        f"(nohup {sh(PYTHON_BIN)} -u {sh(REMOTE_SCRIPT)} >> {sh(LOG_PATH)} 2>&1 & "
        f"echo $! > {sh(PID_PATH)}) && "
        f"echo started phase3 decision 32b download pid=$(cat {sh(PID_PATH)}) log={sh(LOG_PATH)}"
    )


def _status_command() -> str:
    return "\n".join(
        [
            f"echo '--- pid ---'; cat {sh(PID_PATH)} 2>/dev/null || true",
            "echo '--- process ---'; "
            f"pid=$(cat {sh(PID_PATH)} 2>/dev/null || true); "
            "if [ -n \"$pid\" ] && kill -0 \"$pid\" 2>/dev/null; then ps -p \"$pid\" -o pid,etime,cmd; else echo not-running; fi",
            f"echo '--- artifact ---'; test -f {sh(TARGET_DIR + '/config.json')} && echo config=yes || echo config=no; "
            f"find {sh(TARGET_DIR)} -maxdepth 1 -type f -name '*.safetensors' -printf '%f %s\\n' 2>/dev/null | head -20 || true",
            f"echo '--- tail ---'; tail -n 60 {sh(LOG_PATH)} 2>/dev/null || true",
        ]
    )


def _run_remote(command: str, *, timeout: int) -> str:
    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    try:
        return run_remote_text(ssh, command, timeout=timeout, check=False, max_output_chars=60_000)
    finally:
        ssh.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true", help="Upload and start the downloader.")
    parser.add_argument("--status", action="store_true", help="Show remote download status.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.start:
        safe_print(_run_remote(_upload_command() + _start_command(), timeout=120))
    if args.status or not args.start:
        safe_print(_run_remote(_status_command(), timeout=60))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
