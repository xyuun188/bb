"""Read bounded shadow queue, database contention and settlement evidence online."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh  # noqa: E402
from core.safe_output import safe_print  # noqa: E402
from scripts import audit_online_market_hold_economics as transport  # noqa: E402


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def verify_shadow_evidence_fingerprint(
    contract: dict[str, Any] | None,
    full_market_contract: dict[str, Any] | None,
    historical_evidence: Any = None,
) -> tuple[bool, str]:
    """Verify the lineage appropriate to a native or historical shadow row."""

    compact = contract if isinstance(contract, dict) else {}
    full = full_market_contract if isinstance(full_market_contract, dict) else {}
    source = str(compact.get("source") or "").strip()
    native = source == "shadow_entry_result_and_okx_native_1m_path" or (
        compact.get("executable_market_fact_verified") is True
        and compact.get("entry_fact_id")
        and compact.get("result_fact_id")
        and compact.get("path_fingerprint")
    )
    if native:
        entry = full.get("entry_fact")
        entry = entry if isinstance(entry, dict) else {}
        result = full.get("result_fact")
        result = result if isinstance(result, dict) else {}
        path = full.get("price_path")
        path = path if isinstance(path, dict) else {}
        entry_fact_id = str(compact.get("entry_fact_id") or entry.get("fact_id") or "").strip()
        result_fact_id = str(compact.get("result_fact_id") or result.get("fact_id") or "").strip()
        path_fingerprint = str(
            compact.get("path_fingerprint") or path.get("path_fingerprint") or ""
        ).strip()
        if not entry_fact_id or not result_fact_id or not path_fingerprint:
            return False, "native_lineage_field_missing"
        if full and (
            entry_fact_id != str(entry.get("fact_id") or "").strip()
            or result_fact_id != str(result.get("fact_id") or "").strip()
            or path_fingerprint != str(path.get("path_fingerprint") or "").strip()
        ):
            return False, "native_lineage_mismatch"
        expected = _fingerprint(
            {
                "entry_fact_id": entry_fact_id,
                "result_fact_id": result_fact_id,
                "path_fingerprint": path_fingerprint,
            }
        )
        if str(compact.get("data_fingerprint") or "").strip() != expected:
            return False, "native_data_fingerprint_mismatch"
        return True, "native_lineage_verified"

    expected = str(compact.get("data_fingerprint") or "").strip()
    if not isinstance(historical_evidence, dict) or not expected:
        return False, "historical_evidence_missing"
    verified = _fingerprint(historical_evidence) == expected
    return (
        verified,
        "historical_evidence_verified"
        if verified
        else "historical_data_fingerprint_mismatch",
    )


REMOTE_SCRIPT = r'''
import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic

from sqlalchemy import text
from db.session import get_read_session_ctx


async def main():
    report = {"checked_at": datetime.now(UTC).isoformat()}

    async def read(name, sql, params=None):
        started = monotonic()
        try:
            async with get_read_session_ctx(statement_timeout_ms=8000) as session:
                result = await session.execute(text(sql), params or {})
                rows = [dict(row) for row in result.mappings()]
            report[name] = {"seconds": round(monotonic() - started, 3), "rows": rows}
            return rows
        except Exception as exc:
            report[name] = {"seconds": round(monotonic() - started, 3),
                            "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            return []

    await read("database_activity", """
        SELECT state, wait_event_type, wait_event, count(*) AS count,
               max(extract(epoch from now()-query_start)) AS max_query_age_seconds
        FROM pg_stat_activity
        WHERE datname = current_database() AND pid <> pg_backend_pid()
        GROUP BY state, wait_event_type, wait_event
    """)
    await read("table_sizes", """
        SELECT relname, n_live_tup, n_dead_tup,
               pg_size_pretty(pg_total_relation_size(relid)) AS total_size
        FROM pg_stat_user_tables
        WHERE relname IN ('shadow_backtests', 'market_klines', 'positions')
    """)
    await read("indexes", """
        SELECT tablename, indexname, indexdef FROM pg_indexes
        WHERE tablename IN ('shadow_backtests', 'market_klines')
    """)
    await read("queue", """
        SELECT status, horizon_minutes, count(*) AS count,
               min(due_at) AS oldest_due_at, max(updated_at) AS latest_update
        FROM shadow_backtests
        WHERE execution_mode='paper' AND
              ((status='pending' AND due_at <= now()) OR
               (updated_at >= now()-interval '15 minutes'))
        GROUP BY status, horizon_minutes
    """)
    await read("settlement_throughput", """
        SELECT status, count(*) AS count,
               min(updated_at) AS first_update, max(updated_at) AS last_update
        FROM shadow_backtests
        WHERE execution_mode='paper' AND updated_at>=now()-interval '5 minutes'
              AND status<>'pending'
        GROUP BY status
    """)
    await read("native_fee_after_groups", """
        SELECT
            training_feature_snapshot->'training_market_fact_contract'->>'source'
                AS market_source,
            training_feature_snapshot->'training_label_contract'->>'fee_after_complete'
                AS fee_after_complete,
            count(*) AS rows,
            count(DISTINCT decision_id) AS decision_groups,
            count(DISTINCT horizon_minutes) AS horizons,
            min(updated_at) AS first_update,
            max(updated_at) AS last_update
        FROM shadow_backtests
        WHERE execution_mode='paper'
          AND status='completed'
          AND updated_at>=now()-interval '6 hours'
        GROUP BY
            training_feature_snapshot->'training_market_fact_contract'->>'source',
            training_feature_snapshot->'training_label_contract'->>'fee_after_complete'
        ORDER BY rows DESC
    """)
    await read("entry_exit_counts", """
        SELECT action, was_executed, count(*) AS count,
               min(created_at) AS first_decision, max(created_at) AS last_decision
        FROM ai_decisions
        WHERE is_paper AND created_at>=now()-interval '6 hours'
              AND action<>'hold'
        GROUP BY action, was_executed
    """)
    await read("recovery_coverage", """
        WITH recent AS (
            SELECT id FROM ai_decisions
            WHERE analysis_type='market' AND is_paper
                  AND created_at>=now()-interval '3 hours'
        ), coverage AS (
            SELECT recent.id, count(shadow.id) AS horizons
            FROM recent LEFT JOIN shadow_backtests shadow ON shadow.decision_id=recent.id
                AND shadow.label_version=:label_version
            GROUP BY recent.id
        )
        SELECT count(*) AS decisions,
               count(*) FILTER (WHERE horizons<4) AS incomplete_decisions,
               min(horizons) AS minimum_horizons
        FROM coverage
    """, {"label_version": "2026-07-27.market-opportunity-multitask.v2"})
    evidence_rows = await read("recent_historical_evidence", """
        SELECT id, symbol, horizon_minutes, status, updated_at,
               training_feature_snapshot->'training_market_fact_contract' AS contract,
               feature_snapshot->'market_fact_contract' AS full_market_contract,
               feature_snapshot->'market_fact_contract'->'price_path'
                   ->'ohlcv_evidence' AS evidence,
               json_array_length(
                   feature_snapshot->'market_fact_contract'->'price_path'
                   ->'ohlcv_evidence'->'bars'
               ) AS preserved_bar_count
        FROM shadow_backtests
        WHERE execution_mode='paper' AND status='completed'
              AND updated_at>=now()-interval '15 minutes'
        ORDER BY updated_at DESC LIMIT 4
    """)
    for row in evidence_rows:
        evidence = row.pop("evidence", None)
        full_market_contract = row.pop("full_market_contract", None)
        contract = row.get("contract") or {}
        source = str(contract.get("source") or "").strip()
        native = source == "shadow_entry_result_and_okx_native_1m_path" or (
            contract.get("executable_market_fact_verified") is True
            and contract.get("entry_fact_id")
            and contract.get("result_fact_id")
            and contract.get("path_fingerprint")
        )
        if native:
            entry = (
                full_market_contract.get("entry_fact")
                if isinstance(full_market_contract, dict)
                else {}
            )
            result = (
                full_market_contract.get("result_fact")
                if isinstance(full_market_contract, dict)
                else {}
            )
            path = (
                full_market_contract.get("price_path")
                if isinstance(full_market_contract, dict)
                else {}
            )
            entry_fact_id = str(
                contract.get("entry_fact_id")
                or (entry or {}).get("fact_id")
                or ""
            ).strip()
            result_fact_id = str(
                contract.get("result_fact_id")
                or (result or {}).get("fact_id")
                or ""
            ).strip()
            path_fingerprint = str(
                contract.get("path_fingerprint")
                or (path or {}).get("path_fingerprint")
                or ""
            ).strip()
            expected = hashlib.sha256(json.dumps(
                {
                    "entry_fact_id": entry_fact_id,
                    "result_fact_id": result_fact_id,
                    "path_fingerprint": path_fingerprint,
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")).hexdigest()
            row["evidence_fingerprint_verified"] = bool(
                entry_fact_id
                and result_fact_id
                and path_fingerprint
                and contract.get("data_fingerprint") == expected
                and (
                    not isinstance(full_market_contract, dict)
                    or (
                        entry_fact_id == str((entry or {}).get("fact_id") or "").strip()
                        and result_fact_id == str((result or {}).get("fact_id") or "").strip()
                        and path_fingerprint == str((path or {}).get("path_fingerprint") or "").strip()
                    )
                )
            )
            row["evidence_fingerprint_kind"] = "native_lineage"
        else:
            row["evidence_fingerprint_verified"] = bool(
                evidence and hashlib.sha256(json.dumps(
                    evidence, ensure_ascii=True, separators=(",", ":"), default=str,
                ).encode("utf-8")).hexdigest() == contract.get("data_fingerprint")
            )
            row["evidence_fingerprint_kind"] = "historical_ohlcv"
    native_rows = await read("native_fingerprint_rows", """
        SELECT id, symbol, decision_id, horizon_minutes, updated_at,
               training_feature_snapshot->'training_market_fact_contract' AS contract,
               feature_snapshot->'market_fact_contract' AS full_market_contract
        FROM shadow_backtests
        WHERE execution_mode='paper'
          AND status='completed'
          AND updated_at>=now()-interval '6 hours'
          AND training_feature_snapshot->'training_market_fact_contract'->>'source'
              = 'shadow_entry_result_and_okx_native_1m_path'
        ORDER BY updated_at DESC
        LIMIT 200
    """)
    report.pop("native_fingerprint_rows", None)
    native_fingerprint = {
        "checked_rows": len(native_rows),
        "verified_rows": 0,
        "failed_rows": 0,
        "distinct_decision_groups": len({
            int(row.get("decision_id") or 0)
            for row in native_rows
            if int(row.get("decision_id") or 0) > 0
        }),
        "failure_samples": [],
    }
    for row in native_rows:
        contract = row.get("contract") or {}
        full = row.get("full_market_contract")
        entry = full.get("entry_fact") if isinstance(full, dict) else {}
        result = full.get("result_fact") if isinstance(full, dict) else {}
        path = full.get("price_path") if isinstance(full, dict) else {}
        entry_id = str(contract.get("entry_fact_id") or (entry or {}).get("fact_id") or "").strip()
        result_id = str(contract.get("result_fact_id") or (result or {}).get("fact_id") or "").strip()
        path_id = str(contract.get("path_fingerprint") or (path or {}).get("path_fingerprint") or "").strip()
        expected = hashlib.sha256(json.dumps(
            {
                "entry_fact_id": entry_id,
                "result_fact_id": result_id,
                "path_fingerprint": path_id,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")).hexdigest()
        verified = bool(
            entry_id
            and result_id
            and path_id
            and contract.get("data_fingerprint") == expected
            and entry_id == str((entry or {}).get("fact_id") or "").strip()
            and result_id == str((result or {}).get("fact_id") or "").strip()
            and path_id == str((path or {}).get("path_fingerprint") or "").strip()
        )
        if verified:
            native_fingerprint["verified_rows"] += 1
        else:
            native_fingerprint["failed_rows"] += 1
            if len(native_fingerprint["failure_samples"]) < 6:
                native_fingerprint["failure_samples"].append({
                    "id": row.get("id"),
                    "symbol": row.get("symbol"),
                    "decision_id": row.get("decision_id"),
                    "horizon_minutes": row.get("horizon_minutes"),
                    "updated_at": row.get("updated_at"),
                    "has_entry_fact": bool(entry_id),
                    "has_result_fact": bool(result_id),
                    "has_path_fingerprint": bool(path_id),
                    "contract_data_fingerprint": contract.get("data_fingerprint"),
                })
    report["native_fingerprint_check"] = native_fingerprint
    await read("executed_entry_evidence", """
        SELECT id, symbol, created_at,
               raw_llm_response->'normal_paper_trade'->>'expected_net_return_pct'
                   AS expected_net_return_pct,
               raw_llm_response->'normal_paper_trade'->>'objective_net_return_pct'
                   AS return_lcb_pct,
               raw_llm_response->'normal_paper_trade'->>'loss_probability'
                   AS authorized_loss_probability,
               raw_llm_response->'opportunity_score'->>'server_profit_loss_probability'
                   AS execution_loss_probability,
               raw_llm_response->'profit_risk_sizing'->>'final_notional_usdt'
                   AS notional_usdt,
               raw_llm_response->'profit_risk_sizing'->>'expected_profit_usdt'
                   AS expected_profit_usdt
        FROM ai_decisions
        WHERE is_paper AND was_executed AND action IN ('long', 'short')
              AND created_at>=now()-interval '6 hours'
        ORDER BY id DESC LIMIT 12
    """)
    samples = await read("selected_windows", """
        WITH old AS (
            SELECT id FROM shadow_backtests
            WHERE status='pending' AND due_at<=now() ORDER BY due_at, id LIMIT 8
        ), fresh AS (
            SELECT id FROM shadow_backtests
            WHERE status='pending' AND due_at<=now()
            ORDER BY due_at DESC, id DESC LIMIT 4
        )
        SELECT id, symbol, horizon_minutes, due_at,
               feature_snapshot->'market_fact'->>'source_timestamp_ms' AS entry_ms,
               pg_column_size(feature_snapshot) AS feature_bytes,
               pg_column_size(raw_llm_response) AS response_bytes
        FROM shadow_backtests WHERE id IN (SELECT id FROM old UNION SELECT id FROM fresh)
        ORDER BY due_at, id
    """)
    for row in samples:
        end = row["due_at"].astimezone(UTC).replace(second=0, microsecond=0)
        start = (
            datetime.fromtimestamp(int(row["entry_ms"])/1000, tz=UTC)
            .replace(second=0, microsecond=0)
            if row["entry_ms"] else end-timedelta(minutes=row["horizon_minutes"])
        )
        row["expected_bars"] = max(int((end-start).total_seconds()//60)+1, 1)
        await read(f"window_{row['id']}", """
            SELECT count(*) AS bars, min(open_time) AS first_bar, max(open_time) AS last_bar
            FROM market_klines
            WHERE symbol=:symbol AND timeframe='1m' AND open_time BETWEEN :start AND :end
        """, {"symbol": row["symbol"], "start": start, "end": end})
    await read("fills_and_exits", """
        SELECT id, symbol, action, was_executed, created_at, execution_reason
        FROM ai_decisions WHERE is_paper
            AND created_at>=now()-interval '45 minutes' AND action<>'hold'
        ORDER BY id DESC LIMIT 15
    """)
    if __PROBE_HISTORY__:
        from services.okx_perpetual_sdk import OkxPerpetualSdkExchange
        exchange = OkxPerpetualSdkExchange("paper")
        started = monotonic()
        try:
            response = await exchange._call_sdk(
                lambda: exchange.market_api, "get_history_candlesticks",
                instId="BTC-USDT-SWAP", bar="1m", limit="243",
                after="1792484400000",
            )
            rows = response.get("data", [])
            report["history_endpoint_probe"] = {
                "seconds": round(monotonic()-started, 3),
                "code": response.get("code"), "count": len(rows),
                "first": rows[0] if rows else None,
                "last": rows[-1] if rows else None,
            }
        except Exception as exc:
            report["history_endpoint_probe"] = {
                "error": f"{type(exc).__name__}: {str(exc)[:300]}"
            }
        finally:
            await exchange.close()
    if __SETTLEMENT__:
        from services.okx_perpetual_sdk import OkxPerpetualSdkExchange
        exchange = OkxPerpetualSdkExchange("paper")
        try:
            response = await exchange.privateGetAccountPositionsHistory({"limit": "100"})
            since_ms = int((datetime.now(UTC)-timedelta(hours=6)).timestamp()*1000)
            rows = [
                {key: row.get(key) for key in (
                    "instId", "posId", "posSide", "openAvgPx", "closeAvgPx",
                    "pnl", "fee", "fundingFee", "realizedPnl", "cTime", "uTime",
                    "closeTotalPos", "type",
                )}
                for row in response.get("data", [])
                if int(row.get("uTime") or 0) >= since_ms
            ]
            report["okx_position_history"] = {
                "source": "OKX account positions-history, bounded latest 100 rows",
                "six_hour_rows": rows,
            }
        except Exception as exc:
            report["okx_position_history"] = {
                "error": f"{type(exc).__name__}: {str(exc)[:300]}"
            }
        finally:
            await exchange.close()
    status = json.loads(Path("data/trading_runtime_status.json").read_text())
    report["runtime"] = {
        key: status.get(key) for key in (
            "mode", "running", "paused", "heartbeat_at", "last_round_error",
            "market_last_error", "position_last_error",
            "last_position_price_refresh_diagnostics", "shadow_backtest_maintenance",
        )
    }
    print(json.dumps(report, ensure_ascii=False, default=str))

asyncio.run(main())
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-history", action="store_true")
    parser.add_argument("--settlement", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    transport.REMOTE_SCRIPT = REMOTE_SCRIPT.replace(
        "__PROBE_HISTORY__", repr(bool(args.probe_history)),
    ).replace("__SETTLEMENT__", repr(bool(args.settlement)))
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        report = transport._read_remote_report(ssh, 25, 45)
    finally:
        ssh.close()
    if args.compact:
        report = {
            key: value for key, value in report.items()
            if key not in {"indexes", "table_sizes", "selected_windows", "fills_and_exits"}
            and not key.startswith("window_")
        }
    safe_print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
