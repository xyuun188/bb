"""Repair mojibake text in the strategy learning state cache.

The state file can contain LLM-generated candidate labels/descriptions that were
persisted after UTF-8 Chinese text was decoded as GBK/CP936.  This script keeps
the cleanup scoped to the strategy-learning JSON state and defaults to a dry run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402
from web_dashboard.api.text_sanitize import sanitize_payload  # noqa: E402

STATE_FILE_NAME = "strategy_learning_state.json"


def _state_path(raw_path: str | None) -> Path:
    if raw_path:
        return Path(raw_path)
    return settings.data_dir / STATE_FILE_NAME


def _changed(before: Any, after: Any) -> bool:
    return json.dumps(before, ensure_ascii=False, sort_keys=True) != json.dumps(
        after,
        ensure_ascii=False,
        sort_keys=True,
    )


def cleanup_state(path: Path, *, write: bool) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "changed": False, "candidate_count": 0}

    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise RuntimeError(f"{path} did not contain a JSON object")

    cleaned = sanitize_payload(state)
    changed = _changed(state, cleaned)
    if changed and write:
        path.write_text(
            json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    cache = cleaned.get("llm_candidate_cache") if isinstance(cleaned, dict) else {}
    candidates = cache.get("candidates") if isinstance(cache, dict) else []
    return {
        "path": str(path),
        "exists": True,
        "changed": changed,
        "written": bool(changed and write),
        "candidate_count": len(candidates) if isinstance(candidates, list) else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", help="Override strategy_learning_state.json path")
    parser.add_argument("--write", action="store_true", help="Write repaired JSON back to disk")
    args = parser.parse_args()

    result = cleanup_state(_state_path(args.path), write=args.write)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
