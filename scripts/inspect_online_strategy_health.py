from __future__ import annotations

import argparse
import json
import re
import secrets
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

REMOTE_SCRIPT_TEMPLATE = r'''
import asyncio
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

APP_ROOT = Path("/data/bb/app")
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def _inherit_dashboard_runtime_environment() -> None:
    pid_text = subprocess.check_output(
        [
            "systemctl",
            "show",
            "--property=MainPID",
            "--value",
            "bb-dashboard.service",
        ],
        text=True,
    ).strip()
    pid = int(pid_text or "0")
    if pid <= 0:
        raise RuntimeError("bb-dashboard.service has no active MainPID")
    for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        key, separator, value = item.partition(b"=")
        if separator and key:
            os.environ[key.decode("utf-8", errors="surrogateescape")] = value.decode(
                "utf-8",
                errors="surrogateescape",
            )


_inherit_dashboard_runtime_environment()

from services.ml_signal_service import MLSignalService
from services.trade_execution_contract import TradeExecutionContractService
from web_dashboard.api.dashboard import get_model_training_registry_status
from db.session import get_read_session_ctx
from models.decision import AIDecision
from models.trade import Order
from sqlalchemy import select, text

WINDOW_MINUTES = __WINDOW_MINUTES__
SUMMARY_ONLY = __SUMMARY_ONLY__
MARKET_SYMBOL_ONLY = __MARKET_SYMBOL_ONLY__
ENTRY_ONLY = __ENTRY_ONLY__
DECISION_ID = __DECISION_ID__


async def main():
    since = datetime.now(UTC) - timedelta(minutes=WINDOW_MINUTES)
    contract = await TradeExecutionContractService().report(since=since, limit=5000)
    try:
        ml_status = MLSignalService().status()
    except Exception as exc:
        ml_status = {
            "available": False,
            "readiness_state": "unavailable",
            "allow_live_position_influence": False,
            "error": str(exc)[:180],
        }
    try:
        model_registry = await get_model_training_registry_status()
    except Exception as exc:
        model_registry = {
            "summary": {"status": "unavailable"},
            "models": [],
            "error": str(exc)[:180],
        }
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "window_minutes": WINDOW_MINUTES,
        "audit_only": True,
        "live_mutation": False,
        "optimization_target": "realized_fee_after_return",
        "trade_execution_contract": contract,
        "local_ml_readiness": ml_status,
        "model_training_registry": model_registry,
    }
    async with get_read_session_ctx() as session:
        schema_rows = (
            await session.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = current_schema() "
                    "AND table_name = 'expert_memories' ORDER BY column_name"
                )
            )
        ).fetchall()
    expert_memory_columns = [str(row[0]) for row in schema_rows]
    removed_memory_policy_columns = sorted(
        {
            "confidence_adjustment",
            "position_size_multiplier",
        }.intersection(expert_memory_columns)
    )
    payload["expert_memory_schema"] = {
        "column_count": len(expert_memory_columns),
        "removed_policy_columns_present": removed_memory_policy_columns,
        "migration_complete": not removed_memory_policy_columns,
    }
    if DECISION_ID > 0:
        async with get_read_session_ctx() as session:
            decision = await session.get(AIDecision, DECISION_ID)
            order_rows = list(
                (
                    await session.execute(
                        select(Order).where(Order.decision_id == DECISION_ID).order_by(Order.id)
                    )
                )
                .scalars()
                .all()
            )
        payload["selected_decision"] = (
            {
                "id": decision.id,
                "model_name": decision.model_name,
                "symbol": decision.symbol,
                "action": decision.action,
                "was_executed": decision.was_executed,
                "execution_reason": decision.execution_reason,
                "created_at": decision.created_at,
                "executed_at": decision.executed_at,
                "execution_price": decision.execution_price,
                "raw_llm_response": decision.raw_llm_response,
                "orders": [
                    {
                        "id": row.id,
                        "status": row.status,
                        "side": row.side,
                        "quantity": row.quantity,
                        "price": row.price,
                        "exchange_order_id": row.exchange_order_id,
                        "okx_inst_id": row.okx_inst_id,
                        "okx_state": row.okx_state,
                        "okx_sync_status": row.okx_sync_status,
                        "okx_raw_fills": row.okx_raw_fills,
                        "created_at": row.created_at,
                        "filled_at": row.filled_at,
                    }
                    for row in order_rows
                ],
            }
            if decision is not None
            else None
        )
    if SUMMARY_ONLY or MARKET_SYMBOL_ONLY or ENTRY_ONLY:
        payload = {
            **payload,
            "trade_execution_contract": {
                "summary": contract.get("summary", {}),
                "violation_reason_counts": contract.get("violation_reason_counts", {}),
                "policy": contract.get("policy", {}),
            },
        }
    print(json.dumps(payload, ensure_ascii=False, default=str))


asyncio.run(main())
'''


def _safe_token(token: str | None) -> str:
    value = token or secrets.token_hex(6)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise ValueError("token must contain only letters, digits, underscore or hyphen")
    return value


def _remote_result_path(minutes: int, token: str) -> str:
    safe_minutes = max(int(minutes or 480), 1)
    return f"/data/bb/app/tmp/codex-strategy-health/result_{safe_minutes}_{_safe_token(token)}.json"


def _build_remote_command(
    minutes: int,
    *,
    token: str | None = None,
    summary: bool = False,
    market_symbol_only: bool = False,
    entry_only: bool = False,
    decision_id: int = 0,
    output_path: str | None = None,
) -> str:
    safe_minutes = max(int(minutes or 480), 1)
    safe_token = _safe_token(token)
    tmp_dir = "/data/bb/app/tmp/codex-strategy-health"
    sample_path = f"{tmp_dir}/sample_{safe_minutes}_{safe_token}.py"
    result_path = output_path or ""
    if result_path and not result_path.startswith(f"{tmp_dir}/result_"):
        raise ValueError("output_path must stay inside the strategy-health temp directory")
    remote_script = (
        REMOTE_SCRIPT_TEMPLATE.replace("__WINDOW_MINUTES__", str(safe_minutes))
        .replace("__SUMMARY_ONLY__", "True" if summary else "False")
        .replace("__MARKET_SYMBOL_ONLY__", "True" if market_symbol_only else "False")
        .replace("__ENTRY_ONLY__", "True" if entry_only else "False")
        .replace("__DECISION_ID__", str(max(int(decision_id or 0), 0)))
    )
    quoted_sample = shlex.quote(sample_path)
    command = [
        "set -eo pipefail",
        "cd /data/bb/app",
        f"install -d -o bb -g bb -m 0750 {shlex.quote(tmp_dir)}",
        f"cat > {quoted_sample} <<'PY'",
        remote_script,
        "PY",
        f"chmod 0640 {quoted_sample}",
        f"chown bb:bb {quoted_sample}",
    ]
    python_command = (
        "systemd-run --quiet --wait --pipe --collect "
        "--property=WorkingDirectory=/data/bb/app "
        "--property=User=bb "
        "--property=Group=bb "
        "--property=EnvironmentFile=-/data/bb/app/.env "
        "--property=EnvironmentFile=/etc/bb/bb-runtime.env "
        f"/data/bb/app/.venv/bin/python {quoted_sample}"
    )
    if result_path:
        quoted_result = shlex.quote(result_path)
        command.extend(
            [
                f"{python_command} > {quoted_result}",
                f"chmod 0640 {quoted_result}",
                f"printf '%s\\n' {quoted_result}",
            ]
        )
    else:
        command.append(python_command)
    command.append(f"rm -f {quoted_sample}")
    return "\n".join(command)


def _summarize_report(report: dict) -> dict:
    contract = report.get("trade_execution_contract")
    contract = contract if isinstance(contract, dict) else {}
    ml_status = report.get("local_ml_readiness")
    ml_status = ml_status if isinstance(ml_status, dict) else {}
    registry = report.get("model_training_registry")
    registry = registry if isinstance(registry, dict) else {}
    models = registry.get("models")
    models = models if isinstance(models, list) else []
    return {
        "generated_at": report.get("generated_at"),
        "window_minutes": report.get("window_minutes"),
        "optimization_target": "realized_fee_after_return",
        "contract_summary": contract.get("summary") or {},
        "contract_violations": contract.get("violation_reason_counts") or {},
        "contract_policy": contract.get("policy") or {},
        "ml_readiness_state": ml_status.get("readiness_state") or ml_status.get("state"),
        "ml_live_influence": bool(ml_status.get("allow_live_position_influence")),
        "model_training_summary": registry.get("summary") or {},
        "trainable_models": [
            {
                key: row.get(key)
                for key in (
                    "model_id",
                    "lifecycle",
                    "runtime_available",
                    "artifact_available",
                    "sample_count",
                    "live_influence",
                    "quality_state",
                    "blocking_reasons",
                )
            }
            for row in models
            if isinstance(row, dict) and row.get("trainable") is True
        ],
        "selected_decision": report.get("selected_decision"),
        "expert_memory_schema": report.get("expert_memory_schema"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect online dynamic return contracts.")
    parser.add_argument("--minutes", type=int, default=480)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--market-symbol-only", action="store_true")
    parser.add_argument("--entry-only", action="store_true")
    parser.add_argument("--decision-id", type=int, default=0)
    args = parser.parse_args()
    minutes = max(int(args.minutes or 480), 1)
    token = secrets.token_hex(6)
    result_path = _remote_result_path(minutes, token)
    command = _build_remote_command(
        minutes,
        token=token,
        summary=args.summary,
        market_symbol_only=args.market_symbol_only,
        entry_only=args.entry_only,
        decision_id=args.decision_id,
        output_path=result_path,
    )
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        run_remote_text(ssh, command, timeout=220, max_output_chars=4000)
        sftp = ssh.open_sftp()
        try:
            with sftp.file(result_path, "r") as remote_file:
                output = remote_file.read().decode("utf-8", errors="replace")
            try:
                sftp.remove(result_path)
            except OSError:
                pass
        finally:
            sftp.close()
        payload = json.loads(output)
        safe_print(
            json.dumps(
                _summarize_report(payload) if args.summary else payload,
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
