"""Read-only OKX capacity probes and grouped recent execution failures."""

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

REMOTE_CODE = r'''
import asyncio
import json
import os
import subprocess
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

pid = int(subprocess.check_output(
    ["systemctl", "show", "--property=MainPID", "--value", "bb-paper-trading.service"],
    text=True,
).strip() or "0")
if pid <= 0:
    raise RuntimeError("bb-paper-trading.service has no active MainPID")
for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
    key, separator, value = item.partition(b"=")
    if separator and key:
        os.environ[key.decode()] = value.decode(errors="surrogateescape")

from config.settings import settings
from core.safe_output import safe_error_text
from core.symbols import okx_inst_id_from_symbol
from db.session import close_db, get_read_session_ctx
from executor.okx_executor import OKXExecutor
from models.decision import AIDecision
from models.trade import Order
from services.secure_runtime_config import load_secure_settings_into_runtime
from sqlalchemy import select

async def main():
    await load_secure_settings_into_runtime()
    executor = OKXExecutor(mode="paper", load_markets_on_initialize=False)
    try:
        await executor.initialize()
        exchange = executor._exchange
        since = (
            datetime.fromisoformat(__SINCE__.replace("Z", "+00:00"))
            if __SINCE__ else datetime.now(UTC) - timedelta(hours=__HOURS__)
        )
        counts = Counter()
        failures = []
        async with get_read_session_ctx(statement_timeout_ms=20000) as session:
            rows = (await session.execute(
                select(
                    Order,
                    AIDecision.execution_reason,
                    AIDecision.raw_llm_response["execution_result"],
                    AIDecision.action,
                )
                .outerjoin(AIDecision, AIDecision.id == Order.decision_id)
                .where(Order.execution_mode == "paper", Order.created_at >= since)
                .order_by(Order.id.desc()).limit(300)
            )).all()
            for order, reason, execution, action in rows:
                counts[f"{order.status}:{action or order.side}"] += 1
                if order.status not in {"rejected", "failed"}:
                    continue
                raw = (execution or {}).get("raw_response") or {}
                fact = order.okx_raw_fills or {}
                rules = raw.get("okx_order_rules") or fact.get("order_rules") or {}
                failures.append({
                    "id": order.id, "symbol": order.symbol, "action": action,
                    "created_at": str(order.created_at), "quantity": order.quantity,
                    "exchange_order_id": order.exchange_order_id,
                    "error_code": raw.get("okx_error_code") or fact.get("error_code"),
                    "reason": safe_error_text(reason or order.okx_last_error, limit=320),
                    "execution_status": (execution or {}).get("status"),
                    "execution_error": safe_error_text(raw.get("error"), limit=500),
                    "order_last_error": safe_error_text(order.okx_last_error, limit=500),
                    "sync_status": order.okx_sync_status,
                    "fact_error": safe_error_text(fact.get("error"), limit=500),
                    "blocker": raw.get("execution_blocker"),
                    "planned_contracts": raw.get("planned_order_contracts") or rules.get("final_contracts"),
                    "balance_recheck": rules.get("final_balance_recheck"),
                    "account_capacity": rules.get("account_order_capacity"),
                    "reconciliation_reasons": raw.get("reconciliation_reasons"),
                    "risk_contract_fields": {
                        key: rules.get(key) for key in (
                            "leverage", "planned_notional_usdt", "final_notional_usdt",
                            "authorized_fill_ceiling_usdt", "maximum_fill_notional_usdt",
                        )
                    },
                })
        grouped = Counter(
            (row["symbol"], str(row["error_code"]), str(row["blocker"]))
            for row in failures
        )
        probes = {}
        for symbol in ("CAP/USDT", "CRO/USDT", "BTC/USDT"):
            inst_id = okx_inst_id_from_symbol(symbol)
            try:
                response = await exchange._call_sdk(
                    lambda: exchange.account_api, "get_max_order_size",
                    instId=inst_id, tdMode="cross",
                )
                probe = {"native_max_size": response.get("data")}
                if hasattr(executor, "_fetch_account_order_capacity"):
                    buy = await executor._fetch_account_order_capacity(
                        exchange, symbol, side="buy", force=True,
                    )
                    cached_buy = await executor._fetch_account_order_capacity(
                        exchange, symbol, side="buy",
                    )
                    sell = await executor._fetch_account_order_capacity(
                        exchange, symbol, side="sell", force=True,
                    )
                    probe["buy_contract"] = buy
                    probe["buy_recheck"] = cached_buy
                    probe["sell_contract"] = sell
                    facts = await executor.pre_order_execution_facts(symbol, "long")
                    probe["pre_order_long"] = {
                        "production_eligible": facts.get("production_eligible"),
                        "reason": facts.get("reason"),
                        "reasons": facts.get("reasons"),
                        "account_order_capacity": facts.get("account_order_capacity"),
                    }
                probes[symbol] = probe
            except Exception as exc:
                probes[symbol] = {"error": safe_error_text(exc, limit=300)}
        snapshot = await executor.get_balance_snapshot()
        print(json.dumps({
            "checked_at": datetime.now(UTC).isoformat(),
            "since": since.isoformat(),
            "demo": settings.is_okx_demo("paper"),
            "order_counts": dict(counts),
            "failure_groups": [
                {"symbol": key[0], "code": key[1], "blocker": key[2], "count": value}
                for key, value in grouped.most_common()
            ],
            "failures": (
                failures[:3]
                + [row for row in failures[3:] if row["error_code"] != "51008"]
            ),
            "balance": {key: snapshot.get(key) for key in ("free", "equity", "used", "verified")},
            "probes": probes,
        }, ensure_ascii=True, default=str))
    finally:
        await executor.shutdown()
        await close_db()

asyncio.run(main())
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--since", default="")
    args = parser.parse_args()
    code = REMOTE_CODE.replace("__HOURS__", str(max(1, min(args.hours, 168)))).replace(
        "__SINCE__", repr(args.since)
    )
    command = "cd /data/bb/app && runuser -u bb -- .venv/bin/python -c " + shlex.quote(code)
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        safe_print(run_remote_text(
            ssh, command, timeout=180, check=True, max_output_chars=45000,
        ))
    finally:
        ssh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
