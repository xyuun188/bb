"""Print the BB daily briefing, locally or against the online server.

The briefing itself lives in ``services/daily_briefing.py`` so this operator
command and the online timer render exactly the same report.  By default the
report runs on the online host (where the live database and model artifact
live) and the text is streamed back here.  Use ``--local`` when this checkout
already points at the database you want to inspect.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402
from services import daily_briefing  # noqa: E402

REMOTE_APP_DIR = "/data/bb/app"


def _remote_report(window_hours: int, decision_limit: int, as_json: bool) -> str:
    inner = (
        f"cd {REMOTE_APP_DIR} && export PYTHONPATH={REMOTE_APP_DIR} && "
        "exec .venv/bin/python scripts/run_bb_daily_briefing.py "
        f"--hours {window_hours} --limit {decision_limit}"
        + (" --json" if as_json else "")
    )
    command = ["runuser", "-u", "bb", "--", "/bin/bash", "-lc", inner]
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        return run_remote_text(
            ssh,
            shlex.join(command),
            timeout=300,
            max_output_chars=200_000,
            check=True,
        ).rstrip()
    finally:
        ssh.close()


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=daily_briefing.DEFAULT_WINDOW_HOURS)
    parser.add_argument("--limit", type=int, default=daily_briefing.DEFAULT_DECISION_LIMIT)
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead.")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Query this checkout's database instead of the online server.",
    )
    args = parser.parse_args(argv)

    window_hours = max(1, min(args.hours, 720))
    decision_limit = max(1, args.limit)
    if args.local:
        exit_code = daily_briefing.main(
            [
                f"--hours={window_hours}",
                f"--limit={decision_limit}",
                *(["--json"] if args.json else []),
            ]
        )
        return int(exit_code or 0)
    else:
        safe_print(_remote_report(window_hours, decision_limit, args.json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
