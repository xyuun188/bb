from __future__ import annotations

import argparse
import json
import re
import secrets
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

INCIDENT_DECISION_IDS = (79318, 79568, 79757, 80207, 80213, 80325, 80426, 80855)
REMOTE_TMP_DIR = "/data/bb/app/tmp/codex-profit-integrity-fixtures"

REMOTE_SCRIPT_TEMPLATE = r'''
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

APP_ROOT = Path("/data/bb/app")
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def _inherit_dashboard_runtime_environment() -> None:
    pid = int(
        subprocess.check_output(
            ["systemctl", "show", "--property=MainPID", "--value", "bb-dashboard.service"],
            text=True,
        ).strip()
        or "0"
    )
    if pid <= 0:
        raise RuntimeError("bb-dashboard.service has no active MainPID")
    for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        key, separator, value = item.partition(b"=")
        if separator and key:
            os.environ[key.decode("utf-8", errors="surrogateescape")] = value.decode(
                "utf-8", errors="surrogateescape"
            )


_inherit_dashboard_runtime_environment()

from db.session import get_read_session_ctx
from executor.okx_executor import OKXExecutor
from core.symbols import normalize_trading_symbol
from models.decision import AIDecision
from models.learning import ExpertMemory, ShadowBacktest, StrategyLearningEvent, TradeReflection
from models.trade import Order, Position
from services.ml_signal_service import MLSignalService

DECISION_IDS = __DECISION_IDS__


def _safe(value, *, depth=0):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:2000]
    if depth >= 7:
        return "<depth-limited>"
    if isinstance(value, dict):
        return {
            str(key): _safe(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).lower() not in {"apikey", "api_key", "secret", "password", "passphrase"}
        }
    if isinstance(value, (list, tuple)):
        return [_safe(item, depth=depth + 1) for item in value[:50]]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)[:2000]


def _pick(payload, keys):
    source = payload if isinstance(payload, dict) else {}
    return {key: _safe(source.get(key)) for key in keys if key in source}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decision_payload(row):
    raw = row.raw_llm_response if isinstance(row.raw_llm_response, dict) else {}
    features = row.feature_snapshot if isinstance(row.feature_snapshot, dict) else {}
    return {
        "id": row.id,
        "created_at": _safe(row.created_at),
        "symbol": row.symbol,
        "action": row.action,
        "confidence": row.confidence,
        "position_size_pct": row.position_size_pct,
        "suggested_leverage": row.suggested_leverage,
        "stop_loss_pct": row.stop_loss_pct,
        "take_profit_pct": row.take_profit_pct,
        "was_executed": row.was_executed,
        "execution_reason": row.execution_reason,
        "execution_price": row.execution_price,
        "market_fact": _pick(
            features,
            (
                "symbol", "timestamp", "price_source", "current_price", "indicator_close_price",
                "bid", "ask", "spread_pct", "volume_24h", "volume_24h_source",
                "notional_24h_usdt", "orderbook_bid_depth", "orderbook_ask_depth",
                "price_reconciliation_warning", "indicator_price_gap_pct",
            ),
        ),
        "production_evidence": _pick(
            raw,
            (
                "ml_signal", "authoritative_return_candidate", "direction_competition",
                "opportunity_score", "live_ml_profit_contract",
                "profit_risk_sizing", "dynamic_leverage_decision", "execution_leverage",
                "execution_parameters", "pre_execution_price_check", "execution_result",
                "strategy_learning_context", "blocker", "policy_blocker", "skip_kind",
            ),
        ),
    }


def _shadow_payload(row):
    features = row.feature_snapshot if isinstance(row.feature_snapshot, dict) else {}
    return {
        "id": row.id,
        "decision_id": row.decision_id,
        "created_at": _safe(row.created_at),
        "due_at": _safe(row.due_at),
        "symbol": row.symbol,
        "horizon_minutes": row.horizon_minutes,
        "status": row.status,
        "entry_price": row.entry_price,
        "actual_price": row.actual_price,
        "long_return_pct": row.long_return_pct,
        "short_return_pct": row.short_return_pct,
        "best_action": row.best_action,
        "note": row.note,
        "market_fact": _pick(
            features,
            (
                "symbol", "timestamp", "price_source", "current_price", "indicator_close_price",
                "bid", "ask", "spread_pct", "volume_24h", "volume_24h_source",
                "notional_24h_usdt", "orderbook_bid_depth", "orderbook_ask_depth",
                "price_reconciliation_warning", "indicator_price_gap_pct",
            ),
        ),
    }


def _order_payload(row):
    return {
        "id": row.id,
        "created_at": _safe(row.created_at),
        "decision_id": row.decision_id,
        "symbol": row.symbol,
        "side": row.side,
        "quantity": row.quantity,
        "price": row.price,
        "status": row.status,
        "fee": row.fee,
        "exchange_order_id": row.exchange_order_id,
        "okx_inst_id": row.okx_inst_id,
        "okx_fill_contracts": row.okx_fill_contracts,
        "okx_fill_pnl": row.okx_fill_pnl,
        "okx_state": row.okx_state,
        "okx_sync_status": row.okx_sync_status,
    }


def _position_payload(row):
    return {
        "id": row.id,
        "created_at": _safe(row.created_at),
        "closed_at": _safe(row.closed_at),
        "symbol": row.symbol,
        "side": row.side,
        "quantity": row.quantity,
        "entry_price": row.entry_price,
        "leverage": row.leverage,
        "realized_pnl": row.realized_pnl,
        "entry_fee": row.entry_fee,
        "close_fee": row.close_fee,
        "funding_fee": row.funding_fee,
        "stop_loss_price": row.stop_loss_price,
        "take_profit_price": row.take_profit_price,
        "is_open": row.is_open,
        "okx_inst_id": row.okx_inst_id,
        "okx_pos_id": row.okx_pos_id,
        "entry_exchange_order_id": row.entry_exchange_order_id,
        "close_exchange_order_id": row.close_exchange_order_id,
        "settlement_status": row.settlement_status,
        "settlement_source": row.settlement_source,
    }


def _artifact_payload(status):
    registry = status.get("artifact_registry") if isinstance(status, dict) else {}
    registry = registry if isinstance(registry, dict) else {}
    manifest = registry.get("manifest") if isinstance(registry.get("manifest"), dict) else {}
    return {
        "available": bool(status.get("available")),
        "readiness_state": status.get("readiness_state") or status.get("status"),
        "live_ml_ready": bool(status.get("live_ml_ready")),
        "trained_at": status.get("trained_at"),
        "model_stage": status.get("model_stage"),
        "registry": _pick(registry, ("model_id", "registry_version", "version", "sha256")),
        "manifest": _pick(
            manifest,
            (
                "artifact_policy_id", "artifact_registry_version", "artifact_model_id",
                "artifact_version", "artifact_sha256", "created_at", "trained_at",
                "sample_count", "test_count", "training_data_version", "code_version",
                "objective_name", "objective_version", "label_version", "cost_model_version",
                "quality_report", "governance_report", "metrics", "training_window_composition",
                "score_bucket_diagnostics", "tail_loss_policy", "tail_loss_scale_pct",
            ),
        ),
    }


def _protection_inventory(positions, orders):
    active = set()
    position_rows = []
    for row in positions:
        info = row.get("info") if isinstance(row, dict) and isinstance(row.get("info"), dict) else {}
        symbol = str(row.get("symbol") or info.get("instId") or "")
        side = str(row.get("side") or info.get("posSide") or "").lower()
        contracts = abs(float(row.get("contracts") or info.get("pos") or 0.0))
        if contracts <= 0 or side not in {"long", "short"}:
            continue
        key = (normalize_trading_symbol(symbol), side)
        active.add(key)
        position_rows.append({"symbol": key[0], "side": side, "contracts": contracts})

    order_rows = []
    key_counts = Counter()
    for row in orders:
        symbol = normalize_trading_symbol(row.get("symbol"))
        side = str(row.get("position_side") or "").lower()
        key = (symbol, side)
        key_counts[key] += 1
        order_rows.append(
            _pick(
                row,
                (
                    "symbol", "position_side", "close_side", "order_type", "take_profit_price",
                    "stop_loss_price", "trigger_price", "algo_id", "updated_at_ms",
                ),
            )
        )
    return {
        "positions": position_rows,
        "protection_orders": order_rows,
        "orphan_keys": [list(key) for key in sorted(set(key_counts) - active)],
        "duplicate_keys": [list(key) for key, count in sorted(key_counts.items()) if count > 1],
        "missing_protection_keys": [list(key) for key in sorted(active - set(key_counts))],
    }


async def main():
    decisions = []
    shadows = []
    orders = []
    positions = []
    strategy_events = []
    async with get_read_session_ctx() as session:
        decision_rows = list(
            (await session.execute(select(AIDecision).where(AIDecision.id.in_(DECISION_IDS))
             .order_by(AIDecision.id))).scalars().all()
        )
        decisions = [_decision_payload(row) for row in decision_rows]
        shadow_rows = list(
            (await session.execute(select(ShadowBacktest).where(
                ShadowBacktest.decision_id.in_(DECISION_IDS)
            ).order_by(ShadowBacktest.decision_id, ShadowBacktest.horizon_minutes))).scalars().all()
        )
        shadows = [_shadow_payload(row) for row in shadow_rows]
        order_rows = list(
            (await session.execute(select(Order).where(Order.decision_id.in_(DECISION_IDS))
             .order_by(Order.id))).scalars().all()
        )
        orders = [_order_payload(row) for row in order_rows]
        linked_position_ids = {4879}
        position_rows = list(
            (await session.execute(select(Position).where(Position.id.in_(linked_position_ids))
             .order_by(Position.id))).scalars().all()
        )
        positions = [_position_payload(row) for row in position_rows]
        event_rows = list(
            (await session.execute(select(StrategyLearningEvent).where(
                StrategyLearningEvent.decision_id.in_(DECISION_IDS)
            ).order_by(StrategyLearningEvent.id))).scalars().all()
        )
        strategy_events = [
            {
                "id": row.id,
                "decision_id": row.decision_id,
                "order_id": row.order_id,
                "position_id": row.position_id,
                "event_type": row.event_type,
                "event_status": row.event_status,
                "profile_id": row.profile_id,
                "profile_version": row.profile_version,
                "strategy_snapshot": _safe(row.strategy_snapshot),
                "attribution": _safe(row.attribution),
            }
            for row in event_rows
        ]
        reflection_rows = list(
            (await session.execute(select(TradeReflection).where(TradeReflection.position_id == 4879)
             .order_by(TradeReflection.id))).scalars().all()
        )
        memory_rows = list(
            (await session.execute(select(ExpertMemory).where(ExpertMemory.source_position_id == 4879)
             .order_by(ExpertMemory.id))).scalars().all()
        )

    ml_status = MLSignalService().status()
    okx = OKXExecutor(mode="paper", load_markets_on_initialize=False)
    exchange_positions = await okx.get_positions_strict(None)
    protection_orders = await okx.get_position_protection_orders(None)
    source_paths = (
        "services/training_data_quality.py", "services/ml_readiness.py",
        "services/ml_signal_service.py", "services/entry_profit_risk_sizing.py",
        "services/entry_opportunity_scoring.py", "services/strategy_learning.py",
        "executor/okx_executor.py", "services/sync_service.py",
    )
    payload = {
        "fixture_version": "2026-07-14.profit-integrity-baseline.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "audit_only": True,
        "live_mutation": False,
        "decision_ids": DECISION_IDS,
        "decisions": decisions,
        "shadow_backtests": shadows,
        "orders": orders,
        "positions": positions,
        "strategy_learning_events": strategy_events,
        "position_4879_learning": {
            "reflections": [
                {
                    "id": row.id,
                    "position_id": row.position_id,
                    "symbol": row.symbol,
                    "side": row.side,
                    "entry_price": row.entry_price,
                    "exit_price": row.exit_price,
                    "quantity": row.quantity,
                    "realized_pnl": row.realized_pnl,
                    "fee_estimate": row.fee_estimate,
                    "hold_minutes": row.hold_minutes,
                    "closed_at": _safe(row.closed_at),
                    "outcome": row.outcome,
                    "mistake_summary": row.mistake_summary,
                    "improvement_summary": row.improvement_summary,
                    "expert_lessons": _safe(row.expert_lessons),
                    "source": row.source,
                }
                for row in reflection_rows
            ],
            "expert_memories": [
                {
                    "id": row.id,
                    "source_position_id": row.source_position_id,
                    "expert_name": row.expert_name,
                    "symbol": row.symbol,
                    "side": row.side,
                    "memory_type": row.memory_type,
                    "market_pattern": row.market_pattern,
                    "lesson": row.lesson,
                    "recommended_action": row.recommended_action,
                    "memory_key": row.memory_key,
                    "is_active": row.is_active,
                    "extra": _safe(row.extra),
                }
                for row in memory_rows
            ],
        },
        "local_ml_artifact": _artifact_payload(ml_status),
        "source_sha256": {path: _sha256(APP_ROOT / path) for path in source_paths},
        "git_head": "__LOCAL_GIT_HEAD__",
        "services": {
            name: subprocess.check_output(
                ["systemctl", "is-active", name], text=True
            ).strip()
            for name in ("bb-paper-trading.service", "bb-dashboard.service")
        },
        "okx_protection_inventory": _protection_inventory(
            exchange_positions, protection_orders
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, default=str))


asyncio.run(main())
'''


def _safe_token(token: str | None) -> str:
    value = token or secrets.token_hex(6)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise ValueError("token must contain only letters, digits, underscore or hyphen")
    return value


def _remote_result_path(token: str) -> str:
    return f"{REMOTE_TMP_DIR}/result_{_safe_token(token)}.json"


def _build_remote_command(
    decision_ids: tuple[int, ...] = INCIDENT_DECISION_IDS,
    *,
    token: str | None = None,
    output_path: str | None = None,
    git_head: str = "0" * 40,
) -> str:
    safe_ids = tuple(sorted({int(value) for value in decision_ids if int(value) > 0}))
    if not safe_ids:
        raise ValueError("at least one positive decision ID is required")
    safe_git_head = str(git_head or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", safe_git_head):
        raise ValueError("git_head must be a 40-character hexadecimal commit SHA")
    safe_token = _safe_token(token)
    sample_path = f"{REMOTE_TMP_DIR}/sample_{safe_token}.py"
    result_path = output_path or _remote_result_path(safe_token)
    if not result_path.startswith(f"{REMOTE_TMP_DIR}/result_"):
        raise ValueError("output_path must stay inside the profit-integrity temp directory")
    remote_script = REMOTE_SCRIPT_TEMPLATE.replace(
        "__DECISION_IDS__", repr(list(safe_ids))
    ).replace("__LOCAL_GIT_HEAD__", safe_git_head)
    quoted_sample = shlex.quote(sample_path)
    quoted_result = shlex.quote(result_path)
    command = [
        "set -eo pipefail",
        "cd /data/bb/app",
        f"install -d -o bb -g bb -m 0750 {shlex.quote(REMOTE_TMP_DIR)}",
        f"cat > {quoted_sample} <<'PY'",
        remote_script,
        "PY",
        f"chmod 0640 {quoted_sample}",
        f"chown bb:bb {quoted_sample}",
        (
            "systemd-run --quiet --wait --pipe --collect "
            "--property=WorkingDirectory=/data/bb/app --property=User=bb --property=Group=bb "
            "--property=EnvironmentFile=-/data/bb/app/.env "
            "--property=EnvironmentFile=/etc/bb/bb-runtime.env "
            f"/data/bb/app/.venv/bin/python {quoted_sample} > {quoted_result}"
        ),
        f"chmod 0640 {quoted_result}",
        f"rm -f {quoted_sample}",
    ]
    return "\n".join(command)


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    artifact = payload.get("local_ml_artifact")
    artifact = artifact if isinstance(artifact, dict) else {}
    inventory = payload.get("okx_protection_inventory")
    inventory = inventory if isinstance(inventory, dict) else {}
    return {
        "output_fixture_version": payload.get("fixture_version"),
        "decision_count": len(payload.get("decisions") or []),
        "shadow_count": len(payload.get("shadow_backtests") or []),
        "order_count": len(payload.get("orders") or []),
        "position_count": len(payload.get("positions") or []),
        "artifact_version": (artifact.get("registry") or {}).get("version"),
        "artifact_live_influence": bool(artifact.get("live_ml_ready")),
        "orphan_protection_count": len(inventory.get("orphan_keys") or []),
        "duplicate_protection_count": len(inventory.get("duplicate_keys") or []),
        "missing_protection_count": len(inventory.get("missing_protection_keys") or []),
    }


def _parse_remote_payload(raw: bytes) -> dict[str, Any]:
    text = raw.decode("utf-8", errors="replace")
    marker = '{"fixture_version":'
    start = text.rfind(marker)
    if start < 0:
        raise ValueError("remote fixture payload marker is missing")
    payload = json.loads(text[start:])
    if not isinstance(payload, dict):
        raise ValueError("remote fixture payload must be a JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the profit-integrity incident fixture")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    token = secrets.token_hex(6)
    result_path = _remote_result_path(token)
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        run_remote_text(
            ssh,
            _build_remote_command(
                token=token,
                output_path=result_path,
                git_head=git_head,
            ),
            timeout=240,
            max_output_chars=2000,
        )
        sftp = ssh.open_sftp()
        try:
            with sftp.file(result_path, "r") as remote_file:
                payload = _parse_remote_payload(remote_file.read())
            try:
                sftp.remove(result_path)
            except OSError:
                pass
        finally:
            sftp.close()
    finally:
        ssh.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    safe_print(json.dumps(_summary(payload), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
