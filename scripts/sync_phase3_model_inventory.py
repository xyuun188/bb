#!/usr/bin/env python3
"""Synchronize the verified single Qwen3.8-27B inventory to the model host."""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402
from core.phase3_model_contract import PHASE3_TARGET_MODEL_REPO_ID  # noqa: E402
from core.remote_server_info import parse_remote_server_info  # noqa: E402
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

PHASE3_ROOT = "/data/BB"
DOWNLOAD_MANIFEST = f"{PHASE3_ROOT}/manifests/phase3_model_download_manifest.json"
VALIDATION_MANIFEST = f"{PHASE3_ROOT}/manifests/phase3_model_validation.json"
TARGET_CANDIDATE_MANIFEST = f"{PHASE3_ROOT}/manifests/target_model_candidate.json"
REPORT_DOWNLOAD_MANIFEST = f"{PHASE3_ROOT}/reports/inventory/phase3_model_download_manifest_latest.json"
REPORT_VALIDATION_MANIFEST = f"{PHASE3_ROOT}/reports/inventory/phase3_model_validation_latest.json"


def render_target_inventory_sync() -> str:
    """Render a fail-closed remote inventory update script."""

    return textwrap.dedent(
        f"""
        from __future__ import annotations

        import json
        import re
        import urllib.request
        from datetime import datetime, timezone
        from pathlib import Path

        DOWNLOAD_MANIFEST = Path({DOWNLOAD_MANIFEST!r})
        VALIDATION_MANIFEST = Path({VALIDATION_MANIFEST!r})
        REPORT_DOWNLOAD_MANIFEST = Path({REPORT_DOWNLOAD_MANIFEST!r})
        REPORT_VALIDATION_MANIFEST = Path({REPORT_VALIDATION_MANIFEST!r})
        CANDIDATE_MANIFEST = Path({TARGET_CANDIDATE_MANIFEST!r})
        TARGET_REPO = {PHASE3_TARGET_MODEL_REPO_ID!r}
        HEALTH_RESPONSE_MAX_BYTES = 4 * 1024 * 1024


        def now_iso():
            return datetime.now(timezone.utc).isoformat()


        def read_json(path):
            if not path.is_file():
                raise FileNotFoundError(str(path))
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"manifest is not an object: {{path}}")
            return value


        def write_json(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\\n",
                encoding="utf-8",
            )
            temporary.replace(path)


        def candidate():
            value = read_json(CANDIDATE_MANIFEST)
            if value.get("status") != "verified":
                raise ValueError("target candidate manifest is not verified")
            if value.get("model_id") != "qwen3.8-27b":
                raise ValueError("target candidate model id mismatch")
            if value.get("repo_id") != TARGET_REPO:
                raise ValueError("target candidate repository mismatch")
            if not re.fullmatch(r"[0-9a-f]{{40}}", str(value.get("revision") or "")):
                raise ValueError("target candidate revision is not immutable")
            model_path = str(value.get("model_path") or "")
            if not model_path.startswith(("/data/", "/home/linux/")):
                raise ValueError("target candidate path is outside approved roots")
            return value


        def update(data, row):
            rows = data.get("models") if isinstance(data.get("models"), list) else []
            rows = [
                item for item in rows
                if isinstance(item, dict) and not str(item.get("slot") or "").startswith("llm_")
            ]
            rows.append(row)
            data["models"] = rows
            data["topology_profile"] = "target_single_model"
            data["live_routing_enabled"] = False
            data["updated_at"] = now_iso()
            return data


        def health():
            try:
                with urllib.request.urlopen("http://127.0.0.1:8101/health", timeout=8) as response:
                    raw = response.read(HEALTH_RESPONSE_MAX_BYTES + 1)
                    if len(raw) > HEALTH_RESPONSE_MAX_BYTES:
                        raise ValueError("health response exceeds 4 MiB")
                    value = json.loads(raw.decode("utf-8", "replace"))
            except Exception as exc:
                return {{"ok": False, "error": str(exc)[:240]}}
            return value if isinstance(value, dict) else {{"ok": False, "error": "invalid health payload"}}


        def main():
            item = candidate()
            model_id = item["model_id"]
            row = {{
                "slot": "llm_decision_and_expert_carrier",
                "role": "decision_and_expert_carrier",
                "repo_id": item["repo_id"],
                "served_model_name": model_id,
                "path": item["model_path"],
                "target": item["model_path"],
                "stage": "candidate_not_live",
                "live_routing_enabled": False,
                "status": "verified",
                "exists": True,
                "required_any_ok": True,
                "required_missing": [],
                "candidate_revision": item["revision"],
                "candidate_manifest": str(CANDIDATE_MANIFEST),
                "checked_at": now_iso(),
            }}
            download = update(read_json(DOWNLOAD_MANIFEST), row)
            validation = update(read_json(VALIDATION_MANIFEST), row)
            policy = download.get("policy") if isinstance(download.get("policy"), dict) else {{}}
            policy.update({{
                "topology_profile": "target_single_model",
                "llm_candidates": {{"target_single_model": model_id}},
                "llm_candidates_not_activated": True,
                "llm_live_routing_enabled": False,
                "quant_server_only": True,
            }})
            download["policy"] = policy
            for value in (download, validation):
                llm_rows = [
                    item for item in value.get("models", [])
                    if isinstance(item, dict) and str(item.get("slot") or "").startswith("llm_")
                ]
                if len(llm_rows) != 1 or llm_rows[0].get("served_model_name") != model_id:
                    raise RuntimeError("target inventory did not produce exactly one LLM row")
            for path, value in (
                (DOWNLOAD_MANIFEST, download),
                (VALIDATION_MANIFEST, validation),
                (REPORT_DOWNLOAD_MANIFEST, download),
                (REPORT_VALIDATION_MANIFEST, validation),
            ):
                write_json(path, value)
            probe = health()
            print(json.dumps({{
                "event": "phase3_target_inventory_synced",
                "status": "ok",
                "model_id": model_id,
                "revision": item["revision"],
                "health_ok": bool(probe.get("ok")),
                "checked_at": now_iso(),
            }}, ensure_ascii=False, indent=2, sort_keys=True))


        if __name__ == "__main__":
            main()
        """
    ).strip()


def _remote_command() -> str:
    script = render_target_inventory_sync()
    if "\nPY\n" in f"\n{script}\n":
        raise ValueError("inventory sync cannot contain a bare heredoc delimiter")
    return f"python3 - <<'PY'\n{script}\nPY"


def _load_info(info_file: Path | None):
    if info_file is None:
        return load_model_server_info_from_platform(ROOT)
    return parse_remote_server_info(
        info_file.read_text(encoding="utf-8", errors="replace"),
        source_path=info_file,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--info-file", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        safe_print(_remote_command())
        return 0
    info = _load_info(args.info_file)
    ssh = connect_remote_ssh(ROOT, timeout=20, banner_timeout=20, auth_timeout=20, info=info)
    try:
        safe_print(run_remote_text(ssh, _remote_command(), timeout=90, check=True, max_output_chars=80_000))
    finally:
        ssh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
