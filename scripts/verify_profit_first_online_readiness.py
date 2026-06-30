#!/usr/bin/env python3
"""Run a read-only online Profit-First v3 validation pass."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_error_text, safe_print  # noqa: E402

REMOTE_APP_DIR = "/data/bb/app"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _json_or_error(raw: str, *, source: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        return {
            "status": "unavailable",
            "source": source,
            "error": safe_error_text(exc, limit=200),
            "raw_excerpt": safe_error_text(raw, limit=500),
        }
    return parsed if isinstance(parsed, dict) else {"status": "invalid", "source": source}


def _remote_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _remote_validation_command(remote_app_dir: str) -> str:
    quoted_app_dir = _remote_quote(remote_app_dir)
    return textwrap.dedent(
        f"""\
        set -u
        cd {quoted_app_dir}
        python3 - <<'PY'
        import json
        import os
        import shlex
        import subprocess
        from datetime import datetime, timezone
        from pathlib import Path

        def run(cmd, timeout=180):
            completed = subprocess.run(
                cmd,
                shell=True,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            return {{
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }}

        def pybin():
            for candidate in (".venv/bin/python", "venv/bin/python", "python3"):
                if candidate == "python3" or Path(candidate).exists():
                    return candidate
            return "python3"

        def load_control_state():
            state_path = os.environ.get("BB_TRADING_CONTROL_STATE_PATH", "").strip()
            if not state_path:
                for env_path in (Path(".env"), Path("/etc/bb/bb-runtime.env")):
                    try:
                        for line in env_path.read_text(encoding="utf-8").splitlines():
                            if line.startswith("BB_TRADING_CONTROL_STATE_PATH="):
                                state_path = line.split("=", 1)[1].strip()
                    except FileNotFoundError:
                        continue
            if not state_path:
                state_path = "data/trading-control-state.json"
            path = Path(state_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return {{"path": str(path), "available": False, "paused": False}}
            except Exception as exc:
                return {{"path": str(path), "available": False, "error": str(exc), "paused": False}}
            return {{
                "path": str(path),
                "available": True,
                "mode": payload.get("mode"),
                "paused": bool(payload.get("paused")),
                "scan_mode": payload.get("scan_mode"),
                "mode_changed_at": payload.get("mode_changed_at"),
                "live_model_name": payload.get("live_model_name"),
            }}

        def latest_json(path):
            try:
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
            except FileNotFoundError:
                return {{"available": False, "path": path}}
            except Exception as exc:
                return {{"available": False, "path": path, "error": str(exc)}}
            return {{"available": True, "path": path, "payload": payload}}

        def historical_package_args(recovery_plan):
            args = ["--skip-current-blockers"]
            seen = set()

            def add(flag, value):
                text = str(value or "").strip()
                if not text:
                    return
                key = (flag, text)
                if key in seen:
                    return
                seen.add(key)
                args.extend([flag, text])

            for item in recovery_plan.get("blocking_actions") or []:
                if not isinstance(item, dict):
                    continue
                code = str(item.get("code") or "")
                target = item.get("target") if isinstance(item.get("target"), dict) else {{}}
                decision_ids = target.get("decision_ids") if isinstance(target.get("decision_ids"), list) else []
                if code in {{"missing_profit_first_trade_plan", "missing_profit_first_position_ladder"}}:
                    for decision_id in decision_ids:
                        add("--entry-decision-id", decision_id)
                elif code == "missing_profit_first_exit_plan_reference":
                    for decision_id in decision_ids:
                        add("--exit-decision-id", decision_id)
                order_ids = target.get("order_ids") if isinstance(target.get("order_ids"), list) else []
                for order_id in order_ids:
                    add("--order-id", order_id)
                exchange_order_ids = (
                    target.get("exchange_order_ids")
                    if isinstance(target.get("exchange_order_ids"), list)
                    else []
                )
                for exchange_order_id in exchange_order_ids:
                    add("--exchange-order-id", exchange_order_id)
            return args

        python = pybin()
        service_state = {{
            "paper": run("systemctl is-active bb-paper-trading.service || true", timeout=20)["stdout"],
            "dashboard": run("systemctl is-active bb-dashboard.service || true", timeout=20)["stdout"],
            "model_tunnels": run("systemctl is-active bb-model-tunnels.service || true", timeout=20)["stdout"],
        }}
        control_state = load_control_state()

        governance_run = run(
            f"{{python}} scripts/run_profit_first_governance_report.py --stdout-only --json-indent 0",
            timeout=240,
        )
        go_no_go_run = run(
            f"{{python}} scripts/run_phase3_go_no_go_report.py --stdout-only --json-indent 0",
            timeout=360,
        )
        recovery_plan_run = run(
            f"{{python}} scripts/plan_profit_first_recovery_repairs.py --stdout-only --json-indent 0",
            timeout=360,
        )
        governance = (
            json.loads(governance_run["stdout"]) if governance_run["returncode"] == 0 else {{}}
        )
        go_no_go = json.loads(go_no_go_run["stdout"]) if go_no_go_run["returncode"] == 0 else {{}}
        recovery_plan = (
            json.loads(recovery_plan_run["stdout"]) if recovery_plan_run["returncode"] == 0 else {{}}
        )
        historical_args = historical_package_args(recovery_plan)
        historical_cmd = (
            f"{{python}} scripts/plan_profit_first_historical_recovery_package.py "
            "--stdout-only --json-indent 0 "
            + " ".join(shlex.quote(item) for item in historical_args)
        )
        historical_package_run = run(historical_cmd, timeout=240)
        historical_package = (
            json.loads(historical_package_run["stdout"])
            if historical_package_run["returncode"] == 0
            else {{}}
        )

        latest_go_no_go = latest_json("data/phase3_go_no_go_reports/latest.json")
        latest_go_no_go_payload = latest_go_no_go.get("payload") if latest_go_no_go.get("available") else {{}}
        latest_go_no_go_details = (
            latest_go_no_go_payload.get("go_no_go")
            if isinstance(latest_go_no_go_payload.get("go_no_go"), dict)
            else {{}}
        )

        gate = go_no_go.get("go_no_go") if isinstance(go_no_go.get("go_no_go"), dict) else {{}}
        report = {{
            "report_type": "profit_first_online_readiness",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "read_only": True,
            "audit_only": True,
            "starts_trading_service": False,
            "submits_orders": False,
            "changes_model_routing": False,
            "changes_live_sizing": False,
            "live_mutation": False,
            "remote_app_dir": str(Path.cwd()),
            "service_state": service_state,
            "control_state": control_state,
            "governance": {{
                "command_returncode": governance_run["returncode"],
                "status": governance.get("status", "unavailable"),
                "read_only": governance.get("read_only"),
                "audit_only": governance.get("audit_only"),
                "can_submit_orders": governance.get("can_submit_orders"),
                "can_start_trading_service": governance.get("can_start_trading_service"),
                "missing_brain_outputs": governance.get("missing_brain_outputs"),
                "stderr_excerpt": governance_run["stderr"][:500],
            }},
            "go_no_go": {{
                "command_returncode": go_no_go_run["returncode"],
                "status": go_no_go.get("status", "unavailable"),
                "gate_status": gate.get("status"),
                "can_start_paper_with_operator_approval": gate.get("can_start_paper_with_operator_approval"),
                "critical_blocker_count": len(gate.get("critical_blockers") or gate.get("blockers") or []),
                "blocker_codes": [
                    item.get("code")
                    for item in (gate.get("critical_blockers") or gate.get("blockers") or [])
                    if isinstance(item, dict)
                ][:20],
                "stderr_excerpt": go_no_go_run["stderr"][:500],
            }},
            "recovery_repair_plan": {{
                "command_returncode": recovery_plan_run["returncode"],
                "status": recovery_plan.get("status", "unavailable"),
                "dry_run": recovery_plan.get("dry_run"),
                "read_only": recovery_plan.get("read_only"),
                "mutates_database": recovery_plan.get("mutates_database"),
                "starts_trading_service": recovery_plan.get("starts_trading_service"),
                "submits_orders": recovery_plan.get("submits_orders"),
                "changes_model_routing": recovery_plan.get("changes_model_routing"),
                "changes_live_sizing": recovery_plan.get("changes_live_sizing"),
                "live_mutation": recovery_plan.get("live_mutation"),
                "resume_allowed_by_this_plan": recovery_plan.get("resume_allowed_by_this_plan"),
                "summary": recovery_plan.get("summary") if isinstance(recovery_plan.get("summary"), dict) else {{}},
                "blocking_actions": [
                    {{
                        "category": item.get("category"),
                        "code": item.get("code"),
                        "action_type": item.get("action_type"),
                        "status": item.get("status"),
                        "resume_gate_effect": item.get("resume_gate_effect"),
                        "target": item.get("target") if isinstance(item.get("target"), dict) else {{}},
                    }}
                    for item in (recovery_plan.get("blocking_actions") or [])
                    if isinstance(item, dict)
                ][:20],
                "actions": [
                    {{
                        "category": item.get("category"),
                        "code": item.get("code"),
                        "action_type": item.get("action_type"),
                        "status": item.get("status"),
                        "resume_gate_effect": item.get("resume_gate_effect"),
                    }}
                    for item in (recovery_plan.get("actions") or [])
                    if isinstance(item, dict)
                ][:20],
                "stderr_excerpt": recovery_plan_run["stderr"][:500],
            }},
            "historical_recovery_package": {{
                "command_returncode": historical_package_run["returncode"],
                "status": historical_package.get("status", "unavailable"),
                "dry_run": historical_package.get("dry_run"),
                "read_only": historical_package.get("read_only"),
                "mutates_database": historical_package.get("mutates_database"),
                "starts_trading_service": historical_package.get("starts_trading_service"),
                "submits_orders": historical_package.get("submits_orders"),
                "changes_model_routing": historical_package.get("changes_model_routing"),
                "changes_live_sizing": historical_package.get("changes_live_sizing"),
                "live_mutation": historical_package.get("live_mutation"),
                "resume_allowed_by_this_package": historical_package.get("resume_allowed_by_this_package"),
                "summary": (
                    historical_package.get("summary")
                    if isinstance(historical_package.get("summary"), dict)
                    else {{}}
                ),
                "targets": (
                    historical_package.get("targets")
                    if isinstance(historical_package.get("targets"), dict)
                    else {{}}
                ),
                "items": [
                    {{
                        "item_type": item.get("item_type"),
                        "decision_id": item.get("decision_id"),
                        "order_id": item.get("order_id"),
                        "exchange_order_id": item.get("exchange_order_id"),
                        "recommended_resolution": item.get("recommended_resolution"),
                        "operator_approval_required": item.get("operator_approval_required"),
                    }}
                    for item in (historical_package.get("items") or [])
                    if isinstance(item, dict)
                ][:20],
                "stderr_excerpt": historical_package_run["stderr"][:500],
            }},
            "latest_persisted_go_no_go": {{
                "available": latest_go_no_go.get("available"),
                "status": latest_go_no_go_payload.get("status") or latest_go_no_go_details.get("status"),
                "checked_at": latest_go_no_go_payload.get("checked_at"),
                "can_start_paper_with_operator_approval": latest_go_no_go_details.get("can_start_paper_with_operator_approval"),
            }},
        }}
        blocker_clear = int(report["go_no_go"].get("critical_blocker_count") or 0) == 0
        ready_for_resume = (
            control_state.get("paused") is False
            and report["governance"]["status"] == "ready"
            and report["go_no_go"]["status"] in {"paper_resume_ready", "paper_observation_healthy"}
            and bool(report["go_no_go"]["can_start_paper_with_operator_approval"])
        )
        resumed_observing = (
            control_state.get("paused") is False
            and report["governance"]["status"] == "ready"
            and report["go_no_go"]["status"] in {"post_resume_observing", "paper_observation_healthy"}
            and blocker_clear
            and report["recovery_repair_plan"]["status"] in {"clear", "ready"}
        )
        report["resume_allowed_by_this_check"] = bool(ready_for_resume or resumed_observing)
        if ready_for_resume:
            report["validation_status"] = "resume_ready"
        elif resumed_observing:
            report["validation_status"] = "resumed_observing"
        else:
            report["validation_status"] = "paused_or_blocked"
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        PY
        """
    )


def collect_online_readiness(*, remote_app_dir: str = REMOTE_APP_DIR) -> dict[str, Any]:
    ssh = None
    try:
        ssh = connect_remote_ssh(ROOT, timeout=20)
        raw = run_remote_text(
            ssh,
            _remote_validation_command(remote_app_dir),
            timeout=480,
            check=True,
            max_output_chars=20000,
        )
    except Exception as exc:
        return {
            "report_type": "profit_first_online_readiness",
            "checked_at": _now_iso(),
            "read_only": True,
            "audit_only": True,
            "starts_trading_service": False,
            "submits_orders": False,
            "changes_model_routing": False,
            "changes_live_sizing": False,
            "live_mutation": False,
            "validation_status": "unavailable",
            "resume_allowed_by_this_check": False,
            "error": safe_error_text(exc, limit=800),
        }
    finally:
        if ssh is not None:
            ssh.close()
    return _json_or_error(raw, source="remote_profit_first_online_readiness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-app-dir", default=REMOTE_APP_DIR)
    parser.add_argument("--json-indent", type=int, default=2)
    parser.add_argument(
        "--fail-on-resume-not-ready",
        action="store_true",
        help="Exit 2 when the read-only validation says resume is not ready.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = collect_online_readiness(remote_app_dir=args.remote_app_dir)
    indent = None if int(args.json_indent or 0) <= 0 else int(args.json_indent)
    safe_print(json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True))
    if args.fail_on_resume_not_ready and not bool(report.get("resume_allowed_by_this_check")):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
