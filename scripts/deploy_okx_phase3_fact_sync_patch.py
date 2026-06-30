#!/usr/bin/env python3
"""Deploy the Phase 3 OKX fact-sync/account-equity patch only.

This intentionally uploads a fixed allowlist instead of syncing the dirty
working tree.  This script restarts the dashboard only; paper trading is not
started or restarted here.
"""

from __future__ import annotations

import posixpath
import shlex
import stat
import sys
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

REMOTE_APP_DIR = "/data/bb/app"
REMOTE_OWNER = "bb:bb"
DASHBOARD_SERVICE = "bb-dashboard.service"
PAPER_SERVICE = "bb-paper-trading.service"

ALLOWLIST = (
    "models/trade.py",
    "db/session.py",
    "core/server_monitor_probe.py",
    "data_feed/feature_vector.py",
    "data_feed/external_event_scraper.py",
    "data_feed/sentiment_scraper.py",
    "services/external_event_service.py",
    "services/okx_native_facts.py",
    "services/okx_order_fact_sync.py",
    "services/phase3_boundary.py",
    "services/phase3_model_server_readiness.py",
    "services/okx_position_confirmation.py",
    "services/okx_position_ledger_view.py",
    "services/historical_trade_fact_audit.py",
    "services/training_data_quality.py",
    "services/ml_signal_service.py",
    "services/manual_close_marker.py",
    "services/trade_fact_trust.py",
    "services/okx_trade_fact_integrity.py",
    "services/equity_baseline.py",
    "services/execution_allocation_service.py",
    "services/crypto_feature_coverage.py",
    "services/server_monitor_status.py",
    "services/vector_memory/service.py",
    "services/vector_memory/store.py",
    "services/strategy_learning.py",
    "services/sync_service.py",
    "services/trading_service.py",
    "scripts/train_local_ai_tools_models.py",
    "scripts/repair_missing_position_links_from_okx_fills.py",
    "scripts/run_okx_daily_reconciliation_report.py",
    "scripts/run_phase3_okx_fact_sync.py",
    "scripts/install_okx_daily_reconciliation_timer.py",
    "web_dashboard/api/dashboard.py",
    "web_dashboard/api/data_collection.py",
    "web_dashboard/api/system_health.py",
    "web_dashboard/api/system_audit.py",
    "web_dashboard/api/trades.py",
    "web_dashboard/static/css/dashboard.css",
    "web_dashboard/static/index.html",
    "web_dashboard/static/js/dashboard.js",
    "docs/superpowers/plans/2026-06-26-phase-3-quant-master-control-plan.md",
)


def remote_quote(value: str) -> str:
    return shlex.quote(value)


def remote_path(rel: str) -> str:
    return str(PurePosixPath(REMOTE_APP_DIR) / PurePosixPath(rel))


def ensure_remote_dir(sftp, remote_dir: str) -> None:
    current = PurePosixPath(remote_dir)
    path = "/" if current.is_absolute() else "."
    start = 1 if current.is_absolute() else 0
    for part in current.parts[start:]:
        path = posixpath.join(path, part)
        try:
            sftp.stat(path)
        except OSError:
            sftp.mkdir(path)


def upload_allowlist(ssh) -> list[str]:
    stamp = time.strftime("%Y%m%d%H%M%S")
    backup_dir = f"/data/bb/backups/phase3_okx_fact_sync_{stamp}"
    run_remote_text(
        ssh,
        f"mkdir -p {remote_quote(backup_dir)}",
        timeout=30,
    )
    sftp = ssh.open_sftp()
    uploaded: list[str] = []
    try:
        for rel in ALLOWLIST:
            local = ROOT / rel
            if not local.exists():
                raise FileNotFoundError(local)
            dst = remote_path(rel)
            backup = f"{backup_dir}/{rel}"
            run_remote_text(
                ssh,
                (
                    f"mkdir -p {remote_quote(posixpath.dirname(backup))} && "
                    f"if [ -f {remote_quote(dst)} ]; then "
                    f"cp -a {remote_quote(dst)} {remote_quote(backup)}; fi"
                ),
                timeout=30,
            )
            ensure_remote_dir(sftp, posixpath.dirname(dst))
            sftp.put(str(local), dst)
            mode = stat.S_IMODE(local.stat().st_mode)
            sftp.chmod(dst, mode)
            mtime = local.stat().st_mtime
            sftp.utime(dst, (mtime, mtime))
            uploaded.append(dst)
            safe_print(f"uploaded {rel}")
    finally:
        sftp.close()
    run_remote_text(
        ssh,
        f"chown -R {remote_quote(REMOTE_OWNER)} {' '.join(remote_quote(path) for path in uploaded)}",
        timeout=120,
    )
    return uploaded


def main() -> None:
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        before = run_remote_text(
            ssh,
            (
                f"systemctl is-active {remote_quote(PAPER_SERVICE)} || true; "
                f"systemctl is-active {remote_quote(DASHBOARD_SERVICE)} || true"
            ),
            timeout=30,
            check=False,
        ).strip()
        safe_print("service-state-before:")
        safe_print(before)

        upload_allowlist(ssh)

        py_files = " ".join(
            remote_quote(rel)
            for rel in ALLOWLIST
            if rel.endswith(".py")
        )
        run_remote_text(
            ssh,
            (
                f"cd {remote_quote(REMOTE_APP_DIR)} && "
                "PYBIN=python3; "
                "if [ -x .venv/bin/python ]; then PYBIN=.venv/bin/python; "
                "elif [ -x venv/bin/python ]; then PYBIN=venv/bin/python; fi; "
                "$PYBIN -m py_compile "
                f"{py_files}"
            ),
            timeout=120,
        )

        inner_migrate_cmd = (
            f"cd {remote_quote(REMOTE_APP_DIR)} && "
            "PYBIN=python3; "
            "if [ -x .venv/bin/python ]; then PYBIN=.venv/bin/python; "
            "elif [ -x venv/bin/python ]; then PYBIN=venv/bin/python; fi; "
            "$PYBIN - <<'PY'\n"
            "import asyncio\n"
            "from db.session import init_db, close_db\n"
            "async def main():\n"
            "    await init_db()\n"
            "    await close_db()\n"
            "asyncio.run(main())\n"
            "print('db-init-ok')\n"
            "PY"
        )
        migrate_cmd = f"sudo -u bb -H bash -lc {remote_quote(inner_migrate_cmd)}"
        safe_print(run_remote_text(ssh, migrate_cmd, timeout=180))

        restart_parts = [f"systemctl restart {remote_quote(DASHBOARD_SERVICE)}"]
        restart_parts.extend(
            [
                f"systemctl is-active {remote_quote(DASHBOARD_SERVICE)}",
                f"echo {PAPER_SERVICE}-not-touched",
                (
                    "for i in $(seq 1 30); do "
                    "code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 "
                    "http://127.0.0.1:8002/ || true); "
                    "case \"$code\" in 200|302|401) echo dashboard-ok:$code; exit 0;; esac; "
                    "sleep 2; done; echo dashboard-timeout; exit 7"
                ),
            ]
        )
        safe_print(run_remote_text(ssh, " && ".join(restart_parts), timeout=160))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
