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

_COMPACT_REPORT_KEYS = frozenset(
    {
        "checked_at",
        "database_activity",
        "queue",
        "settlement_throughput",
        "native_fee_after_groups",
        "entry_exit_counts",
        "recovery_coverage",
        "native_fingerprint_check",
        "native_route_probe",
        "executed_entry_evidence",
        "runtime",
    }
)


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
from core.symbols import okx_inst_id_from_symbol


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
    if __PROBE_NATIVE__:
        probe_inputs = await read("native_route_probe_inputs", """
            SELECT id, symbol, horizon_minutes,
                   CASE
                       WHEN feature_snapshot->'market_fact_contract'->'entry_fact'
                           ->>'source_timestamp_ms' ~ '^[0-9]+$'
                       THEN (feature_snapshot->'market_fact_contract'->'entry_fact'
                           ->>'source_timestamp_ms')::BIGINT
                       WHEN feature_snapshot->'market_fact'->>'source_timestamp_ms'
                           ~ '^[0-9]+$'
                       THEN (feature_snapshot->'market_fact'
                           ->>'source_timestamp_ms')::BIGINT
                       ELSE NULL
                   END AS entry_timestamp_ms,
                   CASE
                       WHEN feature_snapshot->'market_fact_contract'->'result_fact'
                           ->>'source_timestamp_ms' ~ '^[0-9]+$'
                       THEN (feature_snapshot->'market_fact_contract'->'result_fact'
                           ->>'source_timestamp_ms')::BIGINT
                       ELSE NULL
                   END AS result_timestamp_ms,
                   (extract(epoch from due_at) * 1000)::BIGINT AS due_timestamp_ms,
                   training_feature_snapshot->'training_market_fact_contract'->>'status'
                       AS stored_contract_status,
                   training_feature_snapshot->'training_market_fact_contract'
                       ->>'violation_reason_codes' AS stored_reason_codes,
                   training_feature_snapshot->'training_market_fact_contract'
                       ->>'fallback_reason' AS stored_fallback_reason,
                   training_feature_snapshot->'training_market_fact_contract'
                       ->>'path_status' AS stored_path_status,
                   training_feature_snapshot->'training_market_fact_contract'
                       ->>'path_fingerprint' AS stored_path_fingerprint,
                   feature_snapshot->'market_fact_contract'->'entry_fact'
                       AS entry_fact,
                   feature_snapshot->'market_fact' AS market_fact,
                   feature_snapshot->'market_source_snapshots'
                       AS market_source_snapshots
            FROM shadow_backtests
            WHERE execution_mode='paper'
              AND status='quarantined'
              AND note ILIKE '%native_market_fact_invalid%'
              AND id > (
                  SELECT COALESCE(MAX(id), 0) - 4000
                  FROM shadow_backtests
              )
            ORDER BY id DESC
            LIMIT 8
        """)
        # The rows contain full market facts and source snapshots. Keep them
        # only for this audit's calculations, not in the emitted report.
        report.pop("native_route_probe_inputs", None)
        from services.okx_perpetual_sdk import OkxPerpetualSdkExchange

        exchange = OkxPerpetualSdkExchange("paper")
        probes = []
        probe_status_counts = {}
        price_check_counts = {
            "within_candle_range": 0,
            "outside_candle_range": 0,
            "candle_unavailable": 0,
        }

        def record_probe_status(status):
            probe_status_counts[status] = probe_status_counts.get(status, 0) + 1

        def append_probe(probe):
            if len(probes) < 1:
                probes.append(probe)

        try:
            for item in probe_inputs:
                def add_prices(
                    observations,
                    *,
                    source,
                    timestamp_ms,
                    prices,
                    fields,
                ):
                    try:
                        source_ms = int(float(timestamp_ms or 0))
                    except (TypeError, ValueError):
                        return
                    if source_ms <= 0 or not isinstance(prices, dict):
                        return
                    for field in fields:
                        try:
                            price = float(prices.get(field))
                        except (TypeError, ValueError):
                            continue
                        if price > 0:
                            observations.append({
                                "source": source,
                                "field": field,
                                "source_timestamp_ms": source_ms,
                                "price": price,
                            })

                entry_fact = item.get("entry_fact")
                entry_fact = entry_fact if isinstance(entry_fact, dict) else {}
                if not entry_fact:
                    entry_fact = item.get("market_fact")
                entry_fact = entry_fact if isinstance(entry_fact, dict) else {}
                identity = entry_fact.get("native_identity")
                identity = identity if isinstance(identity, dict) else {}
                saved_inst_id = str(identity.get("inst_id") or "").upper()
                try:
                    requested_inst_id = okx_inst_id_from_symbol(
                        str(item.get("symbol") or "")
                    ).upper()
                except Exception:
                    requested_inst_id = ""
                spec = identity.get("contract_spec")
                spec = spec if isinstance(spec, dict) else {}
                try:
                    tick_size = float(spec.get("tick_size") or 0)
                except (TypeError, ValueError):
                    tick_size = 0.0
                price_observations = []
                add_prices(
                    price_observations,
                    source=str(entry_fact.get("source_endpoint") or "market_fact"),
                    timestamp_ms=entry_fact.get("source_timestamp_ms"),
                    prices=entry_fact.get("prices"),
                    fields=("last", "bid", "ask"),
                )
                source_snapshots = item.get("market_source_snapshots")
                for snapshot in (
                    source_snapshots if isinstance(source_snapshots, list) else []
                ):
                    if not isinstance(snapshot, dict):
                        continue
                    source = str(
                        snapshot.get("source_endpoint")
                        or snapshot.get("source")
                        or "market_source_snapshot"
                    )
                    timestamp_ms = (
                        snapshot.get("source_timestamp_ms")
                        or snapshot.get("timestamp")
                    )
                    add_prices(
                        price_observations,
                        source=source,
                        timestamp_ms=timestamp_ms,
                        prices={
                            "last": snapshot.get("last_price") or snapshot.get("last"),
                            "bid": snapshot.get("bid"),
                            "ask": snapshot.get("ask"),
                        },
                        fields=("last", "bid", "ask"),
                    )
                    orderbook = snapshot.get("orderbook_fact")
                    if isinstance(orderbook, dict):
                        add_prices(
                            price_observations,
                            source=str(
                                orderbook.get("source_endpoint")
                                or f"{source}:orderbook"
                            ),
                            timestamp_ms=(
                                orderbook.get("source_timestamp_ms")
                                or timestamp_ms
                            ),
                            prices=orderbook,
                            fields=("bid", "ask"),
                        )
                consistency = entry_fact.get("source_consistency")
                consistency = consistency if isinstance(consistency, dict) else {}
                for interval in consistency.get("executable_intervals") or []:
                    if not isinstance(interval, dict):
                        continue
                    add_prices(
                        price_observations,
                        source=str(interval.get("source") or "executable_interval"),
                        timestamp_ms=interval.get("source_timestamp_ms"),
                        prices=interval,
                        fields=("bid", "ask"),
                    )
                consistency_orderbook = consistency.get("orderbook_fact")
                if isinstance(consistency_orderbook, dict):
                    add_prices(
                        price_observations,
                        source=str(
                            consistency_orderbook.get("source_endpoint")
                            or "orderbook_fact"
                        ),
                        timestamp_ms=consistency_orderbook.get("source_timestamp_ms"),
                        prices=consistency_orderbook,
                        fields=("bid", "ask"),
                    )
                deduplicated = {}
                for observation in price_observations:
                    key = (
                        observation["source"],
                        observation["field"],
                        observation["source_timestamp_ms"],
                        observation["price"],
                    )
                    deduplicated[key] = observation
                price_observations = list(deduplicated.values())[:24]
                entry_ms = int(item.get("entry_timestamp_ms") or 0)
                result_ms = int(item.get("result_timestamp_ms") or 0)
                due_ms = int(item.get("due_timestamp_ms") or 0)
                end_ms = result_ms if result_ms > 0 else due_ms
                if (
                    not saved_inst_id
                    or not requested_inst_id
                    or saved_inst_id != requested_inst_id
                ):
                    record_probe_status("stored_instrument_identity_mismatch")
                    append_probe({
                        "id": item.get("id"),
                        "symbol": item.get("symbol"),
                        "stored_inst_id": saved_inst_id or None,
                        "requested_inst_id": requested_inst_id or None,
                        "status": "stored_instrument_identity_mismatch",
                        "read_only": True,
                    })
                    continue
                if entry_ms <= 0 or end_ms <= 0 or end_ms < entry_ms:
                    record_probe_status("missing_or_invalid_path_timestamps")
                    append_probe({
                        "id": item.get("id"),
                        "symbol": item.get("symbol"),
                        "stored_inst_id": saved_inst_id,
                        "status": "missing_or_invalid_path_timestamps",
                        "has_entry_timestamp": entry_ms > 0,
                        "has_result_timestamp": result_ms > 0,
                        "has_due_timestamp": due_ms > 0,
                    })
                    continue
                start_open_ms = (entry_ms // 60_000) * 60_000
                end_open_ms = (end_ms // 60_000) * 60_000
                expected_opens = set(
                    range(start_open_ms, end_open_ms + 60_000, 60_000)
                )
                if len(expected_opens) > 300:
                    record_probe_status("window_exceeds_probe_limit")
                    append_probe({
                        "id": item.get("id"),
                        "symbol": item.get("symbol"),
                        "stored_inst_id": saved_inst_id,
                        "horizon_minutes": item.get("horizon_minutes"),
                        "status": "window_exceeds_probe_limit",
                        "expected_bar_count": len(expected_opens),
                        "path_end_source": (
                            "stored_result_fact" if result_ms > 0 else "due_at_fallback"
                        ),
                    })
                    continue
                started = monotonic()
                try:
                    rows = await asyncio.wait_for(
                        exchange.fetch_native_consistency_ohlcv(
                            str(item.get("symbol") or ""),
                            limit=min(max(len(expected_opens) + 2, 5), 300),
                            end_timestamp_ms=end_open_ms,
                        ),
                        timeout=12,
                    )
                    opens = []
                    for row in rows:
                        if isinstance(row, (list, tuple)) and row:
                            try:
                                opens.append(int(float(row[0])))
                            except (TypeError, ValueError):
                                continue
                    covered_count = len(expected_opens.intersection(opens))
                    bars_by_open = {}
                    for row in rows:
                        if not isinstance(row, (list, tuple)) or len(row) < 6:
                            continue
                        try:
                            open_ms = int(float(row[0]))
                            bars_by_open[open_ms - open_ms % 60_000] = {
                                "high": float(row[2]),
                                "low": float(row[3]),
                            }
                        except (TypeError, ValueError):
                            continue
                    for observation in price_observations:
                        source_ms = observation["source_timestamp_ms"]
                        bar = bars_by_open.get(source_ms - source_ms % 60_000)
                        observation["minute_open_ms"] = (
                            source_ms - source_ms % 60_000
                        )
                        observation["ohlc"] = bar
                        observation["within_candle_range"] = (
                            bool(
                                bar
                                and bar["low"] - tick_size
                                <= observation["price"]
                                <= bar["high"] + tick_size
                            )
                            if bar
                            else None
                        )
                        if observation["within_candle_range"] is True:
                            price_check_counts["within_candle_range"] += 1
                        elif observation["within_candle_range"] is False:
                            price_check_counts["outside_candle_range"] += 1
                        else:
                            price_check_counts["candle_unavailable"] += 1
                    status = (
                        "empty_response"
                        if not rows
                        else "complete_window"
                        if covered_count == len(expected_opens)
                        else "window_coverage_miss"
                    )
                    record_probe_status(status)
                    append_probe({
                        "id": item.get("id"),
                        "symbol": item.get("symbol"),
                        "stored_inst_id": saved_inst_id,
                        "horizon_minutes": item.get("horizon_minutes"),
                        "start_open_ms": start_open_ms,
                        "end_open_ms": end_open_ms,
                        "path_end_source": (
                            "stored_result_fact" if result_ms > 0 else "due_at_fallback"
                        ),
                        "stored_contract_status": item.get("stored_contract_status"),
                        "stored_reason_codes": item.get("stored_reason_codes"),
                        "stored_fallback_reason": item.get("stored_fallback_reason"),
                        "stored_path_status": item.get("stored_path_status"),
                        "has_stored_path_fingerprint": bool(
                            item.get("stored_path_fingerprint")
                        ),
                        "status": status,
                        "row_count": len(rows),
                        "covered_expected_bar_count": covered_count,
                        "expected_bar_count": len(expected_opens),
                        "missing_expected_bar_count": len(expected_opens.difference(opens)),
                        "tick_size": tick_size or None,
                        "ohlc_path_low": min(
                            (bar["low"] for bar in bars_by_open.values()),
                            default=None,
                        ),
                        "ohlc_path_high": max(
                            (bar["high"] for bar in bars_by_open.values()),
                            default=None,
                        ),
                        "saved_price_check_count": len(price_observations),
                        "saved_price_checks": price_observations[:2],
                        "elapsed_seconds": round(monotonic() - started, 3),
                    })
                except asyncio.TimeoutError:
                    record_probe_status("request_timeout")
                    append_probe({
                        "id": item.get("id"),
                        "symbol": item.get("symbol"),
                        "stored_inst_id": saved_inst_id,
                        "horizon_minutes": item.get("horizon_minutes"),
                        "start_open_ms": start_open_ms,
                        "end_open_ms": end_open_ms,
                        "path_end_source": (
                            "stored_result_fact" if result_ms > 0 else "due_at_fallback"
                        ),
                        "status": "request_timeout",
                        "elapsed_seconds": round(monotonic() - started, 3),
                    })
                except Exception as exc:
                    record_probe_status("request_error")
                    append_probe({
                        "id": item.get("id"),
                        "symbol": item.get("symbol"),
                        "stored_inst_id": saved_inst_id,
                        "horizon_minutes": item.get("horizon_minutes"),
                        "start_open_ms": start_open_ms,
                        "end_open_ms": end_open_ms,
                        "path_end_source": (
                            "stored_result_fact" if result_ms > 0 else "due_at_fallback"
                        ),
                        "status": "request_error",
                        "error_type": type(exc).__name__,
                        "elapsed_seconds": round(monotonic() - started, 3),
                    })
        finally:
            await exchange.close()
        report["native_route_probe"] = {
            "read_only": True,
            "database_mutated": False,
            "orders_submitted": False,
            "probe_count": len(probe_inputs),
            "status_counts": probe_status_counts,
            "saved_price_check_counts": price_check_counts,
            "probes": probes,
        }
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
            "market_current_stage", "market_round_active",
            "last_market_round_started_at", "last_market_round_finished_at",
            "market_stage_durations", "last_market_candidate_funnel",
            "last_market_round_summary", "ws_stats", "data_service_runtime",
            "last_position_price_refresh_diagnostics", "shadow_backtest_maintenance",
        )
    }
    print(json.dumps(report, ensure_ascii=False, default=str))

asyncio.run(main())
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-history", action="store_true")
    parser.add_argument("--probe-native", action="store_true")
    parser.add_argument("--settlement", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    transport.REMOTE_SCRIPT = REMOTE_SCRIPT.replace(
        "__PROBE_HISTORY__", repr(bool(args.probe_history)),
    ).replace("__PROBE_NATIVE__", repr(bool(args.probe_native))).replace(
        "__SETTLEMENT__", repr(bool(args.settlement))
    )
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        report = transport._read_remote_report(ssh, 25, 45)
    finally:
        ssh.close()
    if args.compact:
        runtime = report.get("runtime")
        compact_runtime = {
            key: runtime.get(key)
            for key in (
                "mode",
                "running",
                "paused",
                "heartbeat_at",
                "last_round_error",
                "market_last_error",
                "position_last_error",
                "market_current_stage",
                "market_round_active",
                "last_market_round_started_at",
                "last_market_round_finished_at",
            )
        } if isinstance(runtime, dict) else runtime
        report = {
            key: value
            for key, value in report.items()
            if key in _COMPACT_REPORT_KEYS
        }
        if compact_runtime is not None:
            report["runtime"] = compact_runtime
    safe_print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
