#!/usr/bin/env python3
"""Synchronize Phase 3 model inventory reports from canonical manifests."""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402
from core.remote_server_info import parse_remote_server_info  # noqa: E402
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

PHASE3_ROOT = "/data/BB"
DOWNLOAD_MANIFEST = f"{PHASE3_ROOT}/manifests/phase3_model_download_manifest.json"
VALIDATION_MANIFEST = f"{PHASE3_ROOT}/manifests/phase3_model_validation.json"
REPORT_DOWNLOAD_MANIFEST = (
    f"{PHASE3_ROOT}/reports/inventory/phase3_model_download_manifest_latest.json"
)
REPORT_VALIDATION_MANIFEST = (
    f"{PHASE3_ROOT}/reports/inventory/phase3_model_validation_latest.json"
)

EXPECTED_LLM_CANDIDATES = {
    "decision_maker": "Qwen/Qwen3-32B-AWQ",
    "expert_pool": "BB-FinQuant-Expert-14B",
    "high_risk_review": "casperhansen/deepseek-r1-distill-qwen-14b-awq",
}

EXPECTED_LLM_SLOTS = {
    "llm_decision_maker": {
        "repo_id": "Qwen/Qwen3-32B-AWQ",
        "served_model_name": "qwen3-32b-trade",
        "path": f"{PHASE3_ROOT}/models/llm_decision_maker/Qwen--Qwen3-32B-AWQ",
        "target": f"{PHASE3_ROOT}/models/llm_decision_maker/Qwen--Qwen3-32B-AWQ",
        "role": "decision_maker",
        "stage": "shadow_candidate_not_live",
    },
    "llm_expert_pool": {
        "repo_id": "Qwen/Qwen3-14B-AWQ",
        "served_model_name": "BB-FinQuant-Expert-14B",
        "path": f"{PHASE3_ROOT}/models/llm_expert_pool/Qwen--Qwen3-14B-AWQ",
        "target": f"{PHASE3_ROOT}/models/llm_expert_pool/Qwen--Qwen3-14B-AWQ",
        "role": "expert_pool",
        "stage": "shadow_candidate_not_live",
        "specialization_required": True,
        "specialization_target": "BB-FinQuant-Expert-14B",
        "specialization_status": "pending",
        "base_model_carrier": "Qwen/Qwen3-14B-AWQ",
    },
    "llm_high_risk_review": {
        "repo_id": "casperhansen/deepseek-r1-distill-qwen-14b-awq",
        "served_model_name": "deepseek-r1-14b-risk",
        "path": (
            f"{PHASE3_ROOT}/models/llm_high_risk_review/"
            "casperhansen--deepseek-r1-distill-qwen-14b-awq"
        ),
        "target": (
            f"{PHASE3_ROOT}/models/llm_high_risk_review/"
            "casperhansen--deepseek-r1-distill-qwen-14b-awq"
        ),
        "role": "high_risk_review",
        "stage": "shadow_candidate_not_live",
    },
}


def sh(value: str | int | float) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def render_remote_inventory_sync() -> str:
    """Return the remote Python script that fixes canonical and report manifests."""

    return textwrap.dedent(
        f"""
        from __future__ import annotations

        import json
        import urllib.request
        from datetime import datetime, timezone
        from pathlib import Path

        DOWNLOAD_MANIFEST = Path({DOWNLOAD_MANIFEST!r})
        VALIDATION_MANIFEST = Path({VALIDATION_MANIFEST!r})
        REPORT_DOWNLOAD_MANIFEST = Path({REPORT_DOWNLOAD_MANIFEST!r})
        REPORT_VALIDATION_MANIFEST = Path({REPORT_VALIDATION_MANIFEST!r})
        EXPECTED_LLM_CANDIDATES = {EXPECTED_LLM_CANDIDATES!r}
        EXPECTED_LLM_SLOTS = {EXPECTED_LLM_SLOTS!r}


        def now_iso() -> str:
            return datetime.now(timezone.utc).isoformat()


        def read_json(path: Path) -> dict:
            if not path.exists():
                raise FileNotFoundError(str(path))
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"manifest is not an object: {{path}}")
            return data


        def write_json(path: Path, data: dict) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\\n",
                encoding="utf-8",
            )
            tmp.replace(path)


        def artifact_stats(path: Path) -> dict:
            if not path.exists():
                return {{"exists": False, "file_count": 0, "size_bytes": 0}}
            files = [item for item in path.rglob("*") if item.is_file()]
            return {{
                "exists": True,
                "file_count": len(files),
                "size_bytes": sum(item.stat().st_size for item in files),
            }}


        def validate_artifact(path: Path) -> dict:
            missing = []
            if not (path / "config.json").exists():
                missing.append("config.json")
            tokenizer_candidates = (
                path / "tokenizer.json",
                path / "tokenizer.model",
                path / "tokenizer_config.json",
                path / "vocab.txt",
            )
            if not any(item.exists() and item.stat().st_size > 0 for item in tokenizer_candidates):
                missing.append("tokenizer")
            index_path = path / "model.safetensors.index.json"
            shards = list(path.glob("*.safetensors"))
            if index_path.exists():
                try:
                    index = json.loads(index_path.read_text(encoding="utf-8"))
                    required_shards = sorted(set((index.get("weight_map") or {{}}).values()))
                except Exception:
                    required_shards = []
                missing.extend(
                    name
                    for name in required_shards
                    if not (path / name).exists() or (path / name).stat().st_size <= 0
                )
            elif shards:
                missing.extend(item.name for item in shards if item.stat().st_size <= 0)
            else:
                missing.append("*.safetensors")
            return {{
                **artifact_stats(path),
                "required_missing": missing,
                "required_any_ok": not missing,
                "status": "ok" if not missing else "incomplete",
            }}


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


        def update_llm_rows(data: dict, *, validation: bool) -> dict:
            rows = data.get("models")
            if not isinstance(rows, list):
                rows = []
            by_slot = {{
                str(row.get("slot") or ""): dict(row)
                for row in rows
                if isinstance(row, dict) and str(row.get("slot") or "")
            }}
            for slot, expected in EXPECTED_LLM_SLOTS.items():
                row = by_slot.get(slot, {{}})
                model_path = Path(str(expected["path"]))
                row.update(expected)
                row["live_routing_enabled"] = False
                row["checked_at"] = now_iso()
                row["validation_note"] = "Phase 3 inventory synchronized from canonical /data/BB manifest."
                row.update(validate_artifact(model_path))
                by_slot[slot] = row
            next_rows = []
            emitted = set()
            for item in rows:
                row = dict(item) if isinstance(item, dict) else {{}}
                slot = str(row.get("slot") or "")
                if slot in by_slot:
                    next_rows.append(by_slot[slot])
                    emitted.add(slot)
                else:
                    next_rows.append(row)
            for slot, row in by_slot.items():
                if slot not in emitted:
                    next_rows.append(row)
            data["models"] = next_rows
            data["checked_at" if validation else "updated_at"] = now_iso()
            return data


        def health_probe() -> dict:
            try:
                with urllib.request.urlopen("http://127.0.0.1:8101/health", timeout=8) as response:
                    payload = json.loads(response.read(256_000).decode("utf-8", "replace"))
            except Exception as exc:
                return {{"ok": False, "error": str(exc)[:240]}}
            if not isinstance(payload, dict):
                return {{"ok": False, "error": "health payload is not an object"}}
            return payload


        def llm_rows(data: dict) -> list[dict]:
            result = []
            for row in data.get("models", []) if isinstance(data.get("models"), list) else []:
                if not isinstance(row, dict):
                    continue
                slot = str(row.get("slot") or "")
                if not slot.startswith("llm_"):
                    continue
                result.append({{
                    "slot": slot,
                    "repo_id": row.get("repo_id"),
                    "path": row.get("path") or row.get("target"),
                    "status": row.get("status"),
                    "required_missing": row.get("required_missing") or [],
                    "live_routing_enabled": row.get("live_routing_enabled"),
                }})
            return result


        def main() -> None:
            download = update_policy(update_llm_rows(read_json(DOWNLOAD_MANIFEST), validation=False))
            if isinstance(download.get("validation"), dict):
                download["validation"] = update_llm_rows(download["validation"], validation=True)
            validation = update_llm_rows(read_json(VALIDATION_MANIFEST), validation=True)

            write_json(DOWNLOAD_MANIFEST, download)
            write_json(VALIDATION_MANIFEST, validation)
            write_json(REPORT_DOWNLOAD_MANIFEST, download)
            write_json(REPORT_VALIDATION_MANIFEST, validation)

            health = health_probe()
            print(json.dumps({{
                "event": "phase3_inventory_synced",
                "status": "ok",
                "canonical_download_manifest": str(DOWNLOAD_MANIFEST),
                "canonical_validation_manifest": str(VALIDATION_MANIFEST),
                "report_download_manifest": str(REPORT_DOWNLOAD_MANIFEST),
                "report_validation_manifest": str(REPORT_VALIDATION_MANIFEST),
                "policy_llm_candidates": download.get("policy", {{}}).get("llm_candidates"),
                "llm_rows": llm_rows(validation),
                "health_decision_model": next((
                    item.get("repo_id")
                    for item in health.get("model_status", [])
                    if isinstance(item, dict) and item.get("slot") == "llm_decision_maker"
                ), ""),
                "health_ok": bool(health.get("ok")),
                "checked_at": now_iso(),
            }}, ensure_ascii=False, indent=2, sort_keys=True))


        if __name__ == "__main__":
            main()
        """
    ).strip()


def _remote_command() -> str:
    script = render_remote_inventory_sync()
    if "\nPY\n" in f"\n{script}\n":
        raise ValueError("Phase 3 inventory sync cannot contain a bare PY delimiter.")
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
    parser.add_argument(
        "--info-file",
        type=Path,
        default=None,
        help="Optional ignored model-server info file for direct operator-side sync.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the remote script only.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        safe_print(_remote_command())
        return 0
    info = _load_info(args.info_file)
    ssh = connect_remote_ssh(ROOT, timeout=20, banner_timeout=20, auth_timeout=20, info=info)
    try:
        safe_print(
            run_remote_text(
                ssh,
                _remote_command(),
                timeout=90,
                check=True,
                max_output_chars=80_000,
            )
        )
    finally:
        ssh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
