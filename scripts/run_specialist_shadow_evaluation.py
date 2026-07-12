"""Generate a read-only Phase 3 specialist shadow evaluation report."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.runtime_env_bootstrap import (  # noqa: E402
    drop_privileges_to_runtime_user_if_needed,
    load_runtime_env_files,
)

load_runtime_env_files(project_root=ROOT)
drop_privileges_to_runtime_user_if_needed(project_root=ROOT)

from config.settings import settings  # noqa: E402
from core.safe_output import safe_print  # noqa: E402
from scripts.train_local_ai_tools_models import (  # noqa: E402
    _load_authoritative_trade_samples,
)
from services.specialist_shadow_evaluation import (  # noqa: E402
    DEFAULT_LIMIT,
    DEFAULT_WINDOW_HOURS,
    SpecialistShadowEvaluationService,
)

DEFAULT_REPORT_DIR = "phase3"


def _safe_report_name(timestamp: str) -> str:
    return timestamp.replace(":", "").replace("-", "").replace("+", "Z").replace(".", "_")


def _report_output_dir(value: Path | None) -> Path:
    if value is not None:
        return value
    return settings.data_dir / DEFAULT_REPORT_DIR


async def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a read-only specialist shadow challenger report from completed "
            "shadow backtests and authoritative OKX lifecycle facts. This never starts "
            "trading or promotes a model."
        )
    )
    parser.add_argument("--hours", type=int, default=DEFAULT_WINDOW_HOURS)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--authoritative-limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--json-indent", type=int, default=2)
    args = parser.parse_args()

    authoritative_trade_samples = await _load_authoritative_trade_samples(
        max(1, int(args.authoritative_limit or DEFAULT_LIMIT))
    )
    report = await SpecialistShadowEvaluationService().report(
        hours=args.hours,
        limit=args.limit,
        authoritative_trade_samples=authoritative_trade_samples,
    )
    generated_at = str(report.get("generated_at") or datetime.now(UTC).isoformat())
    output_dir = _report_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"specialist_shadow_evaluation_{_safe_report_name(generated_at)}.json"
    latest_path = output_dir / "specialist_shadow_evaluation_latest.json"
    indent = None if int(args.json_indent or 0) <= 0 else int(args.json_indent)
    payload = json.dumps(report, ensure_ascii=False, indent=indent)
    report_path.write_text(payload + "\n", encoding="utf-8")
    latest_path.write_text(payload + "\n", encoding="utf-8")

    summary = {
        "ok": True,
        "report_path": str(report_path),
        "latest_path": str(latest_path),
        "completed_count": report.get("completed_count"),
        "eligible_shadow_count": report.get("eligible_shadow_count"),
        "authoritative_eligible_count": report.get("authoritative_eligible_count"),
        "model_count": report.get("model_count"),
        "promotion_ready_count": (report.get("summary") or {}).get("promotion_ready_count"),
        "blocked_count": (report.get("summary") or {}).get("blocked_count"),
        "top_blocked_reasons": (report.get("summary") or {}).get("top_blocked_reasons") or [],
        "live_mutation": False,
    }
    safe_print(json.dumps(report if args.print_json else summary, ensure_ascii=False, indent=indent))


if __name__ == "__main__":
    asyncio.run(_main())
