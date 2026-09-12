#!/usr/bin/env python3
"""Read-only evidence bundle for disputed OKX close orders.

The command deliberately does not repair or mutate the database.  It joins
local order/position lifecycle rows with OKX-native order history, fills, and
public contract specifications so a quantity mismatch can be classified
before any lineage repair is considered.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import String, cast, or_, select

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402
from db.session import get_read_session_ctx  # noqa: E402
from executor.okx_executor import OKXExecutor  # noqa: E402
from models.decision import AIDecision  # noqa: E402
from models.trade import OkxPositionHistory, Order, Position  # noqa: E402
from services.okx_native_facts import OkxNativeFactsClient  # noqa: E402

REMOTE_APP_DIR = "/data/bb/app"


def _json_safe(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (AttributeError, TypeError, ValueError):
            return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _order_snapshot(row: Order) -> dict[str, Any]:
    return {
        "id": row.id,
        "model_name": row.model_name,
        "execution_mode": row.execution_mode,
        "symbol": row.symbol,
        "side": row.side,
        "order_type": row.order_type,
        "quantity": row.quantity,
        "price": row.price,
        "status": row.status,
        "decision_id": row.decision_id,
        "exchange_order_id": row.exchange_order_id,
        "okx_inst_id": row.okx_inst_id,
        "okx_trade_ids": row.okx_trade_ids,
        "okx_fill_contracts": row.okx_fill_contracts,
        "okx_fill_pnl": row.okx_fill_pnl,
        "okx_state": row.okx_state,
        "okx_sync_status": row.okx_sync_status,
        "okx_last_error": row.okx_last_error,
        "okx_raw_fills": _json_safe(row.okx_raw_fills),
        "created_at": _json_safe(row.created_at),
        "filled_at": _json_safe(row.filled_at),
    }


def _position_snapshot(row: Position) -> dict[str, Any]:
    return {
        "id": row.id,
        "model_name": row.model_name,
        "execution_mode": row.execution_mode,
        "symbol": row.symbol,
        "side": row.side,
        "quantity": row.quantity,
        "entry_price": row.entry_price,
        "entry_exchange_order_id": row.entry_exchange_order_id,
        "close_exchange_order_id": row.close_exchange_order_id,
        "okx_inst_id": row.okx_inst_id,
        "okx_pos_id": row.okx_pos_id,
        "is_open": row.is_open,
        "settlement_status": row.settlement_status,
        "settlement_source": row.settlement_source,
        "settlement_raw": _json_safe(row.settlement_raw),
        "realized_pnl": row.realized_pnl,
        "close_fill_pnl": row.close_fill_pnl,
        "entry_fee": row.entry_fee,
        "close_fee": row.close_fee,
        "funding_fee": row.funding_fee,
        "created_at": _json_safe(row.created_at),
        "closed_at": _json_safe(row.closed_at),
    }


def _history_snapshot(row: OkxPositionHistory) -> dict[str, Any]:
    return {
        "id": row.id,
        "mode": row.mode,
        "row_identity": row.row_identity,
        "inst_id": row.inst_id,
        "symbol": row.symbol,
        "pos_id": row.pos_id,
        "pos_side": row.pos_side,
        "side": row.side,
        "close_type": row.close_type,
        "close_status": row.close_status,
        "opened_at": _json_safe(row.opened_at),
        "updated_at_okx": _json_safe(row.updated_at_okx),
        "open_avg_px": row.open_avg_px,
        "close_avg_px": row.close_avg_px,
        "open_max_pos": row.open_max_pos,
        "close_total_pos": row.close_total_pos,
        "realized_pnl": row.realized_pnl,
        "pnl": row.pnl,
        "pnl_ratio": row.pnl_ratio,
        "funding_fee": row.funding_fee,
        "fee": row.fee,
        "entry_order_ids": row.entry_order_ids,
        "close_order_ids": row.close_order_ids,
        "linked_order_ids": row.linked_order_ids,
        "position_ids": row.position_ids,
        "match_status": row.match_status,
        "evidence_gaps": row.evidence_gaps,
        "sync_status": row.sync_status,
        "last_sync_error": row.last_sync_error,
        "raw_row": _json_safe(row.raw_row),
    }


def _fill_snapshot(group: Any) -> dict[str, Any]:
    return {
        **group.as_dict(),
        "rows": [_json_safe(row) for row in group.rows],
    }


async def _collect(order_ids: list[str]) -> dict[str, Any]:
    requested = sorted({str(value).strip() for value in order_ids if str(value).strip()})
    if not requested:
        raise ValueError("at least one --exchange-order-id is required")

    async with get_read_session_ctx(statement_timeout_ms=30_000) as session:
        order_rows = list(
            (
                await session.execute(
                    select(Order).where(Order.exchange_order_id.in_(requested)).order_by(Order.id)
                )
            )
            .scalars()
            .all()
        )
        symbols = sorted({str(row.symbol or "").strip() for row in order_rows if row.symbol})
        positions = []
        if requested:
            position_filters = [
                cast(Position.close_exchange_order_id, String).contains(order_id)
                for order_id in requested
            ]
            positions = list(
                (
                    await session.execute(
                        select(Position)
                        .where(
                            Position.execution_mode == "paper",
                            or_(*position_filters),
                        )
                        .order_by(Position.id)
                    )
                )
                .scalars()
                .all()
            )
        history_rows = []
        if symbols:
            candidates = list(
                (
                    await session.execute(
                        select(OkxPositionHistory)
                        .where(OkxPositionHistory.symbol.in_(symbols), OkxPositionHistory.mode == "paper")
                        .order_by(OkxPositionHistory.updated_at_okx.desc())
                        .limit(300)
                    )
                )
                .scalars()
                .all()
            )
            requested_set = set(requested)
            for row in candidates:
                linked = {
                    str(value).strip()
                    for values in (row.entry_order_ids, row.close_order_ids, row.linked_order_ids)
                    if isinstance(values, list)
                    for value in values
                    if str(value).strip()
                }
                if linked.intersection(requested_set):
                    history_rows.append(row)
        decisions = []
        decision_ids = sorted({int(row.decision_id) for row in order_rows if row.decision_id})
        for decision_id in decision_ids:
            decision = await session.get(AIDecision, decision_id)
            if decision is not None:
                raw = decision.raw_llm_response if isinstance(decision.raw_llm_response, dict) else {}
                decisions.append(
                    {
                        "id": decision.id,
                        "model_name": decision.model_name,
                        "symbol": decision.symbol,
                        "action": decision.action,
                        "was_executed": decision.was_executed,
                        "execution_reason": decision.execution_reason,
                        "executed_at": _json_safe(decision.executed_at),
                        "exit_exchange_order_id": raw.get("exit_exchange_order_id"),
                        "exit_exchange_order_ids": raw.get("exit_exchange_order_ids"),
                    }
                )

    executor = OKXExecutor(mode="paper", load_markets_on_initialize=False)
    try:
        await executor.initialize()
        native = OkxNativeFactsClient(executor)
        inst_ids = sorted(
            {
                str(row.okx_inst_id).strip().upper()
                for row in order_rows
                if str(row.okx_inst_id or "").strip()
            }
            | {
                str(row.okx_inst_id).strip().upper()
                for row in positions
                if str(row.okx_inst_id or "").strip()
            }
        )
        fills = await native.fetch_fill_groups(
            order_ids=requested,
            inst_ids=inst_ids,
            target_orders_first=True,
            target_orders_only=True,
            target_order_query_limit=len(requested),
            include_historical=True,
            strict=False,
        )
        order_history = await native.fetch_order_history_rows(
            order_ids=requested,
            inst_ids=inst_ids,
            limit=100,
            max_pages=2,
            strict=False,
        )
        pos_ids = sorted(
            {
                str(row.okx_pos_id).strip()
                for row in positions
                if str(row.okx_pos_id or "").strip()
            }
        )
        position_history = await native.fetch_position_history_rows(
            inst_ids=inst_ids,
            pos_ids=pos_ids,
            limit=100,
            max_pages=2,
            strict=False,
        )
        specs = await native.fetch_contract_specs(inst_ids=inst_ids)
    finally:
        await executor.shutdown()

    return {
        "requested_order_ids": requested,
        "local": {
            "orders": [_order_snapshot(row) for row in order_rows],
            "positions": [_position_snapshot(row) for row in positions],
            "position_history": [_history_snapshot(row) for row in history_rows],
            "decisions": decisions,
        },
        "okx": {
            "fills": [_fill_snapshot(group) for group in fills if group.order_id in requested],
            "order_history": [
                _json_safe(row)
                for row in order_history
                if str(row.get("ordId") or row.get("clOrdId") or "").strip() in requested
            ],
            "position_history": [_json_safe(row) for row in position_history],
            "contract_specs": _json_safe(specs),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange-order-id", action="append", required=True)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser


def _online(order_ids: list[str]) -> str:
    remote_script = "\n".join(
        (
            "import asyncio, json, sys",
            f"sys.path.insert(0, {REMOTE_APP_DIR!r})",
            "from scripts.runtime_env_bootstrap import load_runtime_env_files, drop_privileges_to_runtime_user_if_needed",
            f"from pathlib import Path; root=Path({REMOTE_APP_DIR!r}); load_runtime_env_files(project_root=root); drop_privileges_to_runtime_user_if_needed(project_root=root)",
            "from scripts.inspect_okx_order_evidence import _collect",
            f"print(json.dumps(asyncio.run(_collect({order_ids!r})), ensure_ascii=False, default=str))",
        )
    )
    command = (
        f"cd {shlex.quote(REMOTE_APP_DIR)} && "
        "runuser -u bb -- /bin/bash -lc "
        + shlex.quote(".venv/bin/python - <<'PY'\n" + remote_script + "\nPY")
    )
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        return run_remote_text(ssh, command, timeout=240, max_output_chars=200_000)
    finally:
        ssh.close()


def _summarize(payload: dict[str, Any]) -> dict[str, Any]:
    local = payload.get("local") if isinstance(payload.get("local"), dict) else {}
    okx = payload.get("okx") if isinstance(payload.get("okx"), dict) else {}
    orders = local.get("orders") if isinstance(local.get("orders"), list) else []
    positions = local.get("positions") if isinstance(local.get("positions"), list) else []
    fills = okx.get("fills") if isinstance(okx.get("fills"), list) else []
    histories = okx.get("order_history") if isinstance(okx.get("order_history"), list) else []
    position_histories = (
        okx.get("position_history")
        if isinstance(okx.get("position_history"), list)
        else []
    )
    specs = okx.get("contract_specs") if isinstance(okx.get("contract_specs"), dict) else {}
    return {
        "requested_order_ids": payload.get("requested_order_ids", []),
        "orders": [
            {
                key: row.get(key)
                for key in (
                    "id",
                    "exchange_order_id",
                    "symbol",
                    "side",
                    "quantity",
                    "price",
                    "decision_id",
                    "okx_inst_id",
                    "okx_fill_contracts",
                    "okx_sync_status",
                )
            }
            for row in orders
            if isinstance(row, dict)
        ],
        "positions": [
            {
                key: row.get(key)
                for key in (
                    "id",
                    "symbol",
                    "side",
                    "quantity",
                    "entry_price",
                    "entry_exchange_order_id",
                    "close_exchange_order_id",
                    "okx_inst_id",
                    "okx_pos_id",
                    "settlement_status",
                )
            }
            for row in positions
            if isinstance(row, dict)
        ],
        "fills": [
            {
                key: row.get(key)
                for key in (
                    "order_id",
                    "inst_id",
                    "side",
                    "pos_side",
                    "contracts",
                    "avg_price",
                    "fee_abs",
                    "fill_pnl",
                    "trade_ids",
                )
            }
            for row in fills
            if isinstance(row, dict)
        ],
        "order_history": [
            {
                key: row.get(key)
                for key in (
                    "ordId",
                    "instId",
                    "side",
                    "posSide",
                    "sz",
                    "accFillSz",
                    "avgPx",
                    "state",
                    "reduceOnly",
                    "cTime",
                    "uTime",
                )
            }
            for row in histories
            if isinstance(row, dict)
        ],
        "position_history": [
            {
                key: row.get(key)
                for key in (
                    "posId",
                    "instId",
                    "posSide",
                    "direction",
                    "openMaxPos",
                    "closeTotalPos",
                    "closeAvgPx",
                    "realizedPnl",
                    "fee",
                    "fundingFee",
                    "cTime",
                    "uTime",
                    "type",
                )
            }
            for row in position_histories
            if isinstance(row, dict)
        ],
        "contract_specs": {
            inst_id: {
                key: spec.get(key)
                for key in ("instId", "ctVal", "ctMult", "ctValCcy", "lotSz", "minSz", "ctType")
            }
            for inst_id, spec in specs.items()
            if isinstance(spec, dict)
        },
        "decisions": local.get("decisions", []),
    }


def main() -> None:
    args = _parser().parse_args()
    if args.online:
        raw = _online(args.exchange_order_id)
        try:
            payload = json.loads(raw.splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            output = raw
        else:
            output = json.dumps(
                _summarize(payload) if args.summary else payload,
                ensure_ascii=False,
                default=str,
            )
    else:
        payload = asyncio.run(_collect(args.exchange_order_id))
        output = json.dumps(
            _summarize(payload) if args.summary else payload,
            ensure_ascii=False,
            default=str,
        )
    safe_print(output)


if __name__ == "__main__":
    main()
