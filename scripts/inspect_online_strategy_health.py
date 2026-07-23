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
from services.profit_training_contract import PROFIT_TRAINING_TARGET  # noqa: E402

REMOTE_SCRIPT_TEMPLATE = r'''
import asyncio
import json
import os
import subprocess
import sys
import time
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
from services.okx_training_gate import okx_training_refresh_gate
from services.protection_order_integrity import audit_protection_order_integrity
from services.trade_execution_contract import TradeExecutionContractService
from executor.okx_executor import OKXExecutor
from web_dashboard.api.dashboard import (
    get_expert_memories,
    get_model_training_registry_status,
    get_shadow_backtests,
    get_strategy_learning,
)
from web_dashboard.api.data_collection import get_data_collection_status
from db.session import get_read_session_ctx
from models.decision import AIDecision
from models.trade import Order
from sqlalchemy import select, text

WINDOW_MINUTES = __WINDOW_MINUTES__
SUMMARY_ONLY = __SUMMARY_ONLY__
MARKET_SYMBOL_ONLY = __MARKET_SYMBOL_ONLY__
ENTRY_ONLY = __ENTRY_ONLY__
DECISION_ID = __DECISION_ID__
REPLAY_ONLY = __REPLAY_ONLY__
PROFIT_TRAINING_TARGET = "net_return_after_all_cost_pct"


async def _read_stage(name, awaitable):
    try:
        return await awaitable
    except Exception as exc:
        return {
            "available": False,
            "stage": name,
            "error": f"{type(exc).__name__}: {str(exc)[:180]}",
        }


async def _read_positions_and_protection():
    executor = OKXExecutor(mode="paper", load_markets_on_initialize=False)
    try:
        await executor.initialize()
        positions = await executor.get_positions_strict()
        protection_orders = await executor.get_position_protection_orders()
        pending_orders = await executor.get_open_orders_strict()
        protection = audit_protection_order_integrity(
            positions,
            protection_orders,
            pending_orders,
            {},
            pending_snapshot_complete=True,
        )
        protection["available"] = True
        protection["invalid_order_count"] = len(protection.get("invalid_orders") or [])
        protection["inventory_fingerprint"] = protection.get("input_fingerprint")
        protection["blockers"] = protection.get("repair_blockers") or []
        return {
            "available": True,
            "count": len(positions),
            "total": len(positions),
            "protection_inventory": protection,
        }
    finally:
        await executor.shutdown()


async def main():
    if REPLAY_ONLY:
        started = time.perf_counter()
        strategy = await get_strategy_learning(mode="paper", detail="summary")
        schedule = strategy.get("schedule") if isinstance(strategy, dict) else {}
        schedule = schedule if isinstance(schedule, dict) else {}
        feedback = strategy.get("feedback") if isinstance(strategy, dict) else {}
        feedback = feedback if isinstance(feedback, dict) else {}
        print(json.dumps({
            "generated_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "candidate_count": schedule.get("candidate_count"),
            "governed_candidate_count": schedule.get("governed_candidate_count"),
            "rejected_candidate_count": schedule.get("rejected_candidate_count"),
            "scheduler_mode": schedule.get("scheduler_mode"),
            "historical_model_replay": schedule.get("historical_model_replay") or {},
            "paper_strategy_champion": strategy.get("paper_strategy_champion") or {},
            "shadow_feedback": feedback.get("shadow_feedback") or {},
        }, ensure_ascii=False, default=str))
        return
    since = datetime.now(UTC) - timedelta(minutes=WINDOW_MINUTES)
    contract = await TradeExecutionContractService().report(since=since, limit=5000)
    try:
        ml_status = MLSignalService().status()
    except Exception as exc:
        ml_status = {
            "available": False,
            "readiness_state": "unavailable",
            "live_ml_ready": False,
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
        "optimization_target": PROFIT_TRAINING_TARGET,
        "trade_execution_contract": contract,
        "local_ml_readiness": ml_status,
        "model_training_registry": model_registry,
        "okx_training_refresh_gate": okx_training_refresh_gate(),
    }
    payload["data_collection"] = await _read_stage(
        "data_collection",
        get_data_collection_status(include_feature_coverage=False),
    )
    payload["shadow_maturity"] = await _read_stage(
        "shadow_maturity",
        get_shadow_backtests(page_size=1, page=1),
    )
    payload["strategy_learning"] = await _read_stage(
        "strategy_learning",
        get_strategy_learning(mode="paper", detail="full"),
    )
    payload["expert_learning"] = await _read_stage(
        "expert_learning",
        get_expert_memories(page_size=20, mode="paper"),
    )
    payload["open_positions"] = await _read_stage(
        "open_positions",
        _read_positions_and_protection(),
    )
    reconciliation_path = APP_ROOT / "data" / "okx_daily_reconciliation_reports" / "latest.json"
    try:
        payload["latest_okx_reconciliation"] = json.loads(
            reconciliation_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        payload["latest_okx_reconciliation"] = {
            "available": False,
            "path": str(reconciliation_path),
            "error": f"{type(exc).__name__}: {str(exc)[:180]}",
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
    replay_only: bool = False,
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
        .replace("__REPLAY_ONLY__", "True" if replay_only else "False")
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
    strategy_blueprint = ml_status.get("strategy_blueprint")
    strategy_blueprint = (
        strategy_blueprint if isinstance(strategy_blueprint, dict) else {}
    )
    return {
        "generated_at": report.get("generated_at"),
        "window_minutes": report.get("window_minutes"),
        "optimization_target": PROFIT_TRAINING_TARGET,
        "contract_summary": contract.get("summary") or {},
        "contract_violations": contract.get("violation_reason_counts") or {},
        "contract_policy": contract.get("policy") or {},
        "ml_readiness_state": ml_status.get("readiness_state") or ml_status.get("state"),
        "ml_live_influence": bool(ml_status.get("live_ml_ready")),
        "model_strategy_blueprint": {
            key: strategy_blueprint.get(key)
            for key in (
                "strategy_id",
                "model_version",
                "artifact_stage",
                "execution_scope",
                "eligible_sides",
                "paper_execution_eligible",
                "live_execution_permission",
                "blocking_reasons",
                "model_quality",
                "entry_policy",
                "exit_policy",
                "risk_policy",
                "training_evidence",
                "historical_replay_policy",
            )
            if key in strategy_blueprint
        },
        "model_training_summary": registry.get("summary") or {},
        "training_scheduler_state": _summarize_training_scheduler_state(
            registry.get("scheduler_state")
        ),
        "okx_training_refresh_gate": report.get("okx_training_refresh_gate") or {},
        "profit_closed_loop": _summarize_profit_closed_loop(report),
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


def _summarize_profit_closed_loop(report: dict) -> dict:
    collection = report.get("data_collection")
    collection = collection if isinstance(collection, dict) else {}
    collection_training = collection.get("training")
    collection_training = collection_training if isinstance(collection_training, dict) else {}
    local_ai_tools = collection_training.get("local_ai_tools")
    local_ai_tools = local_ai_tools if isinstance(local_ai_tools, dict) else {}
    promotion = local_ai_tools.get("promotion_recommendation")
    promotion = promotion if isinstance(promotion, dict) else {}
    governance = collection_training.get("governance")
    governance = governance if isinstance(governance, dict) else {}
    artifact_governance = local_ai_tools.get("governance_report")
    artifact_governance = (
        artifact_governance if isinstance(artifact_governance, dict) else {}
    )
    artifact_quality = local_ai_tools.get("quality_report")
    artifact_quality = artifact_quality if isinstance(artifact_quality, dict) else {}
    quality_totals = artifact_quality.get("totals")
    quality_totals = quality_totals if isinstance(quality_totals, dict) else {}
    shadow = report.get("shadow_maturity")
    shadow = shadow if isinstance(shadow, dict) else {}
    strategy = report.get("strategy_learning")
    strategy = strategy if isinstance(strategy, dict) else {}
    schedule = strategy.get("schedule")
    schedule = schedule if isinstance(schedule, dict) else {}
    runtime = schedule.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    feedback = strategy.get("feedback")
    feedback = feedback if isinstance(feedback, dict) else {}
    expert = report.get("expert_learning")
    expert = expert if isinstance(expert, dict) else {}
    positions = report.get("open_positions")
    positions = positions if isinstance(positions, dict) else {}
    reconciliation = report.get("latest_okx_reconciliation")
    reconciliation = reconciliation if isinstance(reconciliation, dict) else {}
    source_rows = collection.get("sources")
    source_rows = source_rows if isinstance(source_rows, list) else []
    reflection_rows = expert.get("reflections")
    reflection_rows = reflection_rows if isinstance(reflection_rows, list) else []
    production_strategy = schedule.get("current_production_strategy")
    production_strategy = production_strategy if isinstance(production_strategy, dict) else {}
    paper_champion = strategy.get("paper_strategy_champion")
    paper_champion = paper_champion if isinstance(paper_champion, dict) else {}
    historical_replay = schedule.get("historical_model_replay")
    historical_replay = (
        historical_replay if isinstance(historical_replay, dict) else {}
    )
    protection = positions.get("protection_inventory")
    protection = protection if isinstance(protection, dict) else {}
    return {
        "collection": {
            "available": collection.get("available", True),
            "checked_at": collection.get("checked_at"),
            "sources": [
                {
                    key: row.get(key)
                    for key in ("key", "group", "enabled", "status")
                    if key in row
                }
                for row in source_rows
                if isinstance(row, dict)
            ],
            "training": {
                "local_ai_tools": {
                    key: local_ai_tools.get(key)
                    for key in (
                        "available",
                        "status",
                        "trained_at",
                        "training_mode",
                        "model_stage",
                        "model_bundle_available",
                        "completed_shadow_sample_count",
                        "completed_trade_sample_count",
                        "shadow_sample_count",
                        "trade_sample_count",
                        "sequence_sample_count",
                        "text_sentiment_sample_count",
                        "objective_name",
                        "objective_version",
                        "label_version",
                        "profit_supervision_version",
                        "live_influence",
                    )
                    if key in local_ai_tools
                },
                "promotion": {
                    key: promotion.get(key)
                    for key in (
                        "optimization_target",
                        "recommended_stage",
                        "canary_ready",
                        "live_ml_ready",
                        "canary_blocking_reasons",
                        "live_blocking_reasons",
                    )
                    if key in promotion
                },
                "artifact_governance": {
                    key: artifact_governance.get(key)
                    for key in (
                        "status",
                        "data_quality_version",
                        "training_policy",
                        "raw_records_preserved",
                        "cleanup_mode",
                        "quarantine_applied",
                        "downweight_applied",
                        "trainable_sample_count",
                        "excluded_sample_count",
                        "downweighted_sample_count",
                        "effective_weight_ratio",
                        "excluded_ratio",
                        "blocked_reason_ratio",
                        "contamination_risk",
                        "blocked_reason_count",
                        "requires_artifact_refresh",
                        "quality_fingerprint",
                        "artifact_quality_fingerprint",
                        "artifact_matches_quality",
                    )
                    if key in artifact_governance
                },
                "quality_totals": {
                    key: quality_totals.get(key)
                    for key in (
                        "total",
                        "included",
                        "downweighted",
                        "excluded",
                        "effective_weight",
                        "effective_weight_ratio",
                    )
                    if key in quality_totals
                },
                "quality_reasons": list(artifact_quality.get("top_reasons") or [])[:20],
                "governance": {
                    key: governance.get(key)
                    for key in (
                        "status",
                        "cleanup_effective",
                        "trainable_sample_count",
                        "contamination_risk",
                        "quality_state",
                        "training_data_version",
                        "data_fingerprint",
                        "blocking_reasons",
                    )
                    if key in governance
                },
            },
            "error": collection.get("error"),
        },
        "shadow_maturity": {
            "available": shadow.get("available", True),
            "total": shadow.get("count"),
            "pending_total": shadow.get("pending_count"),
            "completed_total": shadow.get("completed_count"),
            "error": shadow.get("error"),
        },
        "strategy_scheduler": {
            "available": strategy.get("available", True),
            "optimization_target": strategy.get("optimization_target"),
            "scheduler_mode": schedule.get("scheduler_mode"),
            "candidate_count": schedule.get("candidate_count"),
            "governed_candidate_count": schedule.get("governed_candidate_count"),
            "rejected_candidate_count": schedule.get("rejected_candidate_count"),
            "current_production_strategy": {
                key: production_strategy.get(key)
                for key in (
                    "id",
                    "version",
                    "name",
                    "objective",
                    "owner",
                    "enabled",
                    "status",
                    "scope",
                    "entry_permission_owner",
                    "historical_prior_role",
                    "historical_prior_can_authorize_entry",
                    "historical_prior_can_change_size_or_leverage",
                    "execution_owners",
                    "data_sources",
                    "historical_prior_matching_enabled",
                )
                if key in production_strategy
            },
            "paper_strategy_champion": {
                key: paper_champion.get(key)
                for key in (
                    "active",
                    "profile_id",
                    "profile_version",
                    "status",
                    "execution_scope",
                    "paper_execution_permission",
                    "live_execution_permission",
                    "model_strategy_id",
                    "model_version",
                    "selector",
                    "metrics",
                    "transition",
                    "reason",
                    "rollback_reason",
                    "model_rollback_required",
                    "model_rollback_target_version",
                )
                if key in paper_champion
            },
            "historical_model_replay": historical_replay,
            "production_influence_enabled": bool(
                runtime.get("production_influence_enabled")
            ),
            "can_authorize_entry": runtime.get("can_authorize_entry"),
            "policy_provenance": runtime.get("policy_provenance") or {},
            "feedback_generated_at": feedback.get("generated_at"),
            "error": strategy.get("error"),
        },
        "authoritative_settlement": {
            "contract_summary": (
                (report.get("trade_execution_contract") or {}).get("summary") or {}
            ),
            "outcome_contract": expert.get("authoritative_outcome_contract") or {},
        },
        "reflection_and_memory": {
            "available": expert.get("available", True),
            "memory_count": expert.get("count"),
            "reflection_count": expert.get("reflection_count"),
            "recent_reflections": [
                {
                    "id": row.get("id"),
                    "position_id": row.get("position_id"),
                    "symbol": row.get("symbol"),
                    "side": row.get("side"),
                    "closed_at": row.get("closed_at"),
                    "evidence_precedence": row.get("evidence_precedence"),
                    "authoritative_outcome": {
                        key: (row.get("authoritative_outcome") or {}).get(key)
                        for key in (
                            "outcome_id",
                            "outcome_version",
                            "complete",
                            "evidence_gaps",
                            "decision_id",
                            "realized_pnl",
                            "net_return_after_all_cost_pct",
                            "counterfactual_production_weight",
                        )
                    }
                    if isinstance(row.get("authoritative_outcome"), dict)
                    else None,
                }
                for row in reflection_rows
                if isinstance(row, dict)
            ],
            "error": expert.get("error"),
        },
        "positions_and_protection": {
            "available": positions.get("available", True),
            "total": positions.get("total"),
            "open_total": positions.get("count"),
            "protection_inventory": {
                key: protection.get(key)
                for key in (
                    "available",
                    "contract_version",
                    "position_count",
                    "protection_order_count",
                    "missing_keys",
                    "orphan_keys",
                    "split_coverage_keys",
                    "coverage_mismatches",
                    "invalid_order_count",
                    "repair_blockers",
                    "inventory_fingerprint",
                    "blockers",
                )
                if key in protection
            },
            "error": positions.get("error"),
        },
        "okx_reconciliation": {
            "available": reconciliation.get("available", True),
            "generated_at": reconciliation.get("generated_at"),
            "status": reconciliation.get("status"),
            "can_open_new_entries": reconciliation.get("can_open_new_entries"),
            "can_refresh_training": reconciliation.get("can_refresh_training"),
            "requires_attention": reconciliation.get("requires_attention"),
            "issue_summary": (reconciliation.get("issue_ledger") or {}).get("summary"),
            "error": reconciliation.get("error"),
        },
    }


def _summarize_training_scheduler_state(value: object) -> dict:
    payload = value if isinstance(value, dict) else {}
    raw_schedulers = payload.get("schedulers")
    raw_schedulers = raw_schedulers if isinstance(raw_schedulers, dict) else {}
    raw_models = payload.get("models")
    raw_models = raw_models if isinstance(raw_models, dict) else {}
    scheduler_fields = (
        "heartbeat_at",
        "heartbeat_age_seconds",
        "heartbeat_stale_after_seconds",
        "heartbeat_stale",
        "interval_seconds",
        "model_ids",
    )
    model_fields = (
        "state",
        "last_check_at",
        "last_started_at",
        "last_finished_at",
        "last_error",
        "last_result",
        "next_check_at",
        "retry_count",
        "sample_cursor",
        "training_timeout_exceeded",
    )
    return {
        "status": payload.get("status"),
        "updated_at": payload.get("updated_at"),
        "heartbeat_stale": bool(payload.get("heartbeat_stale")),
        "stale_scheduler_ids": payload.get("stale_scheduler_ids") or [],
        "training_timeout_exceeded": bool(payload.get("training_timeout_exceeded")),
        "timed_out_model_ids": payload.get("timed_out_model_ids") or [],
        "schedulers": {
            scheduler_id: {
                field: row.get(field)
                for field in scheduler_fields
                if field in row
            }
            for scheduler_id, row in raw_schedulers.items()
            if isinstance(row, dict)
        },
        "models": {
            model_id: {
                **{
                    field: row.get(field)
                    for field in model_fields
                    if field in row
                },
                "recent_history": list(row.get("history") or [])[-5:],
            }
            for model_id, row in raw_models.items()
            if isinstance(row, dict)
        },
    }


def _decode_remote_json(output: str) -> dict:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as original_error:
        for line in reversed(output.splitlines()):
            candidate = line.strip()
            if not candidate.startswith("{"):
                continue
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise original_error
    if not isinstance(payload, dict):
        raise ValueError("online strategy health output must be a JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect online dynamic return contracts.")
    parser.add_argument("--minutes", type=int, default=480)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--market-symbol-only", action="store_true")
    parser.add_argument("--entry-only", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
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
        replay_only=args.replay_only,
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
        payload = _decode_remote_json(output)
        safe_print(
            json.dumps(
                _summarize_report(payload)
                if args.summary and not args.replay_only
                else payload,
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
