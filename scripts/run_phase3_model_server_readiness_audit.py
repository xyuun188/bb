"""Build a read-only Phase 3 quant model-server readiness report."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.runtime_env_bootstrap import (  # noqa: E402
    drop_privileges_to_runtime_user_if_needed,
    load_runtime_env_files,
)

load_runtime_env_files(project_root=ROOT)
drop_privileges_to_runtime_user_if_needed(project_root=ROOT)

from config.settings import settings
from core.remote_server_info import parse_remote_server_info
from core.safe_output import safe_error_text
from services.phase3_model_server_readiness import Phase3ModelServerReadinessAuditService

DEFAULT_REPORT_DIR = "phase3_model_server_readiness_reports"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_report_name(timestamp: str) -> str:
    return timestamp.replace(":", "").replace("-", "").replace("+", "Z").replace(".", "_")


def _report_output_dir(value: Path | None) -> Path:
    if value is not None:
        return value
    return settings.data_dir / DEFAULT_REPORT_DIR


def _report_has_config_environment_error(report: dict[str, Any]) -> bool:
    """Return True when the report failed before reaching the model server.

    This protects the operator-facing ``latest.json`` from being polluted by
    ad-hoc shell runs that forget to provide ``BB_SECURE_SETTINGS_KEY``.  The
    dated artifact is still written for audit, but the last verified latest
    report remains the dashboard source of truth.
    """

    if str(report.get("status") or "").lower() != "unverified":
        return False
    error_text = str(report.get("error") or "")
    blockers = report.get("blockers") if isinstance(report.get("blockers"), list) else []
    blocker_codes = {
        str(item.get("code") or "")
        for item in blockers
        if isinstance(item, dict)
    }
    if "model_server_config_error" in blocker_codes:
        return True
    return "BB_SECURE_SETTINGS_KEY" in error_text


def _latest_is_verified(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    status = str(payload.get("status") or "").lower()
    return bool(
        status == "ready"
        and payload.get("runtime_ready") is True
        and payload.get("artifact_ready") is True
        and payload.get("phase3_model_service_go_live_blocked") is False
    )


def write_report(
    report: dict[str, Any],
    output_dir: Path,
    *,
    indent: int | None,
    protect_verified_latest: bool = True,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = str(report.get("checked_at") or _now_iso())
    report_path = output_dir / f"phase3-model-server-readiness-{_safe_report_name(timestamp)}.json"
    latest_path = output_dir / "latest.json"
    preserve_latest = bool(
        protect_verified_latest
        and _report_has_config_environment_error(report)
        and latest_path.exists()
        and _latest_is_verified(latest_path)
    )
    artifacts = {
        "report_path": str(report_path),
        "latest_path": str(latest_path),
        "latest_preserved": preserve_latest,
    }
    report["report_artifacts"] = artifacts
    if preserve_latest:
        report["latest_preserved_reason"] = (
            "config_environment_error_did_not_overwrite_last_verified_latest"
        )
    text = json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True)
    report_path.write_text(text + "\n", encoding="utf-8")
    if not preserve_latest:
        latest_path.write_text(text + "\n", encoding="utf-8")
    return artifacts


def _info_loader_from_file(path: Path):
    def load_info(_project_root: Path):
        text = path.read_text(encoding="utf-8", errors="replace")
        return parse_remote_server_info(text, source_path=path)

    return load_info


async def collect_phase3_model_server_readiness(
    *,
    timeout_seconds: int,
    info_file: Path | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"timeout_seconds": timeout_seconds}
    if info_file is not None:
        kwargs["info_loader"] = _info_loader_from_file(info_file)
        kwargs["async_info_loader"] = None
    return await Phase3ModelServerReadinessAuditService(**kwargs).report()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument(
        "--info-file",
        type=Path,
        default=None,
        help="Optional ignored local model-server info file for operator-side verification.",
    )
    parser.add_argument("--json-indent", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for dated report files. Defaults to data/phase3_model_server_readiness_reports.",
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print the report without writing report artifacts.",
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Return exit code 2 when the model-server gate is blocked.",
    )
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    report = await collect_phase3_model_server_readiness(
        timeout_seconds=max(int(args.timeout_seconds or 1), 1),
        info_file=args.info_file,
    )
    indent = None if int(args.json_indent or 0) <= 0 else int(args.json_indent)
    if not args.stdout_only:
        try:
            write_report(report, _report_output_dir(args.output_dir), indent=indent)
        except Exception as exc:
            report["status"] = "blocked"
            report["report_artifact_error"] = {
                "code": "report_artifact_write_failed",
                "message": safe_error_text(exc, limit=240),
                "output_dir": str(_report_output_dir(args.output_dir)),
            }
    print(json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True))
    if args.fail_on_blocked and report.get("phase3_model_service_go_live_blocked"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
