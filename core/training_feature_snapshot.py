"""Bounded training snapshots that retain sealed inputs without rewriting them."""

# DDL interpolates only module constants and escaped registry field names.
# ruff: noqa: S608

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from decimal import Decimal
from typing import Any

from core.feature_availability import FROZEN_FEATURE_FIELDS, availability_fingerprint
from core.market_facts import compact_market_fact

TRAINING_FEATURE_SNAPSHOT_VERSION = 3
FROZEN_TRAINING_FEATURE_MAX_BYTES = 96 * 1024
TRAINING_FEATURE_SNAPSHOT_MAX_BYTES = 128 * 1024
TRAINING_FEATURE_OBJECT_MAX_BYTES = 16 * 1024
TRAINING_FEATURE_MAX_STRING_CHARS = 512
TRAINING_FEATURE_MAX_DEPTH = 16
FEATURE_STORAGE_COMPACTION_VERSION = "2026-10-22.compact-feature-storage.v1"
TRAINING_FEATURE_METADATA_KEYS = (
    "feature_availability_contract",
    "feature_snapshot_frozen_at",
    "original_decision_timestamp",
    "feature_storage_compaction",
)
FROZEN_TRAINING_FEATURE_KEYS = tuple(
    dict.fromkeys((*FROZEN_FEATURE_FIELDS, *TRAINING_FEATURE_METADATA_KEYS))
)
TRAINING_FEATURE_RAW_TEXT_KEYS = (
    "recent_headlines",
    "headlines",
    "news_articles",
    "social_posts",
    "recent_news",
    "recent_social",
    "raw_llm_response",
    "raw_response",
    "transcript",
    "opinions",
    "experts",
)
_INVALID_SIZE = TRAINING_FEATURE_SNAPSHOT_MAX_BYTES + 1


def training_feature_storage_bytes(value: Any, *, _depth: int = 0) -> int:
    """Conservative byte cost, identical to the PostgreSQL snapshot function."""

    if _depth > TRAINING_FEATURE_MAX_DEPTH:
        return _INVALID_SIZE
    if value is None or isinstance(value, bool):
        return 32
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return _INVALID_SIZE
        magnitude = abs(value)
        if magnitude >= 1e64 or 0 < magnitude < 1e-64:
            return max(512, len(str(value)))
        return 128
    if isinstance(value, str):
        return 32 + 6 * len(value.encode("utf-8"))
    if isinstance(value, (dict, list)):
        size = 32
        values = value.items() if isinstance(value, dict) else enumerate(value)
        for key, child in values:
            if isinstance(value, dict):
                if not isinstance(key, str):
                    return _INVALID_SIZE
                size += 32 + 6 * len(key.encode("utf-8"))
            size += training_feature_storage_bytes(child, _depth=_depth + 1)
            if size > TRAINING_FEATURE_SNAPSHOT_MAX_BYTES:
                return _INVALID_SIZE
        return size
    return _INVALID_SIZE


def _jsonb_text(value: Any) -> str:
    if isinstance(value, dict):
        keys = sorted(value, key=lambda key: (len(str(key).encode("utf-8")), str(key)))
        return "{" + ", ".join(
            f"{json.dumps(str(key), ensure_ascii=False)}: {_jsonb_text(value[key])}"
            for key in keys
        ) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_jsonb_text(child) for child in value) + "]"
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite values cannot be fingerprinted")
        number = Decimal(str(value))
        if number.is_zero():
            return "0"
        number = number.normalize()
        return format(number, "f")
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _storage_fingerprint_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _storage_fingerprint_value(child)
            for key, child in value.items()
            if child is not None
        }
    if isinstance(value, list):
        return [_storage_fingerprint_value(child) for child in value]
    return value


def storage_feature_fingerprint(value: Any) -> str:
    """MD5 of the null-stripped, scale-normalized PostgreSQL JSONB projection."""

    return hashlib.md5(  # noqa: S324 - parity with PostgreSQL's built-in md5(jsonb::text)
        _jsonb_text(_storage_fingerprint_value(value)).encode("utf-8")
    ).hexdigest()


def frozen_feature_payload_size(features: dict[str, Any]) -> int:
    return training_feature_storage_bytes(_compact_frozen_features(features))


def _compact_frozen_value(key: str, value: Any) -> Any:
    if key == "market_fact" and isinstance(value, dict):
        return compact_market_fact(value)
    return deepcopy(value)


def _compact_frozen_features(features: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: _compact_frozen_value(key, features[key])
        for key in FROZEN_TRAINING_FEATURE_KEYS
        if key in features and key != "feature_storage_compaction"
    }
    original_contract = features.get("feature_availability_contract")
    if isinstance(original_contract, dict):
        compact["feature_storage_compaction"] = {
            "version": FEATURE_STORAGE_COMPACTION_VERSION,
            "full_input_fingerprint": original_contract.get("input_fingerprint"),
            "compact_input_fingerprint": storage_feature_fingerprint(
                {
                    key: compact[key]
                    for key in FROZEN_FEATURE_FIELDS
                    if key in compact
                }
            ),
        }
    elif "feature_storage_compaction" in features:
        compact["feature_storage_compaction"] = deepcopy(
            features["feature_storage_compaction"]
        )
    return compact


def frozen_training_feature_snapshot(features: dict[str, Any]) -> dict[str, Any]:
    """Preserve every present frozen value, including explicit JSON null."""

    frozen = _compact_frozen_features(features)
    if training_feature_storage_bytes(frozen) > FROZEN_TRAINING_FEATURE_MAX_BYTES:
        return {"training_feature_compaction_error": "frozen_input_budget_exceeded"}
    return deepcopy(frozen)


def compact_training_feature_snapshot(value: object) -> dict[str, Any]:
    """Drop raw text; never truncate, reseal, or repair frozen model inputs."""

    if not isinstance(value, dict):
        return {}
    original_contract = value.get("feature_availability_contract")
    storage_contract = value.get("feature_storage_compaction")
    if isinstance(original_contract, dict):
        if isinstance(storage_contract, dict):
            compact_input = {
                key: _compact_frozen_value(key, value[key])
                for key in FROZEN_FEATURE_FIELDS
                if key in value
            }
            current_fingerprint = storage_feature_fingerprint(
                compact_input
            )
            if (
                storage_contract.get("version") != FEATURE_STORAGE_COMPACTION_VERSION
                or storage_contract.get("full_input_fingerprint")
                != original_contract.get("input_fingerprint")
                or storage_contract.get("compact_input_fingerprint") != current_fingerprint
            ):
                return {"training_feature_compaction_error": "frozen_input_storage_mismatch"}
        else:
            original_fingerprint = availability_fingerprint(
                {key: value[key] for key in FROZEN_FEATURE_FIELDS if key in value}
            )
            if original_contract.get("input_fingerprint") != original_fingerprint:
                return {"training_feature_compaction_error": "frozen_input_storage_mismatch"}
    compact = frozen_training_feature_snapshot(value)
    if "training_feature_compaction_error" in compact:
        return compact
    # Optional diagnostics are useful only when they fit beside the sealed
    # training inputs.  Keep a deterministic prefix and drop the rest rather
    # than rejecting an otherwise valid frozen sample.
    for key in sorted(value, key=lambda item: str(item)):
        raw = value[key]
        if (
            not isinstance(key, str)
            or len(key) > 120
            or key in FROZEN_TRAINING_FEATURE_KEYS
            or key in TRAINING_FEATURE_RAW_TEXT_KEYS
        ):
            continue
        if raw is None or isinstance(raw, (bool, int, float)):
            keep = training_feature_storage_bytes(raw) <= TRAINING_FEATURE_OBJECT_MAX_BYTES
        elif isinstance(raw, str):
            keep = len(raw) <= TRAINING_FEATURE_MAX_STRING_CHARS
        else:
            keep = (
                isinstance(raw, dict)
                and training_feature_storage_bytes(raw) <= TRAINING_FEATURE_OBJECT_MAX_BYTES
            )
        if not keep:
            continue
        candidate = {**compact, key: deepcopy(raw)}
        if training_feature_storage_bytes(candidate) <= TRAINING_FEATURE_SNAPSHOT_MAX_BYTES:
            compact = candidate
    return compact


def postgres_training_feature_snapshot_functions() -> tuple[str, ...]:
    """Generate PostgreSQL compaction from the same fields and byte budgets."""

    frozen_keys = ",".join(
        "'" + key.replace("'", "''") + "'" for key in FROZEN_TRAINING_FEATURE_KEYS
    )
    input_keys = ",".join(
        "'" + key.replace("'", "''") + "'" for key in FROZEN_FEATURE_FIELDS
    )
    raw_keys = ",".join("'" + key.replace("'", "''") + "'" for key in TRAINING_FEATURE_RAW_TEXT_KEYS)
    empty_json = "{}"
    storage_sql = f"""
        CREATE OR REPLACE FUNCTION bb_training_feature_storage_bytes(
            input_value JSONB, input_depth INTEGER DEFAULT 0
        )
        RETURNS BIGINT
        LANGUAGE plpgsql IMMUTABLE
        AS $storage_bytes$
        DECLARE
            item RECORD;
            child JSONB;
            result_size BIGINT := 32;
            magnitude NUMERIC;
        BEGIN
            IF input_depth > {TRAINING_FEATURE_MAX_DEPTH} THEN
                RETURN {_INVALID_SIZE};
            END IF;
            CASE jsonb_typeof(input_value)
            WHEN 'null', 'boolean' THEN RETURN 32;
            WHEN 'number' THEN
                magnitude := abs((input_value #>> '{{}}')::NUMERIC);
                IF magnitude >= '1e64'::NUMERIC
                   OR (magnitude > 0 AND magnitude < '1e-64'::NUMERIC) THEN
                    RETURN greatest(512, length(input_value::TEXT));
                END IF;
                RETURN 128;
            WHEN 'string' THEN
                RETURN 32 + 6 * octet_length(convert_to(input_value #>> '{{}}', 'UTF8'));
            WHEN 'object' THEN
                FOR item IN SELECT key, value FROM jsonb_each(input_value) LOOP
                    result_size := result_size
                        + 32 + 6 * octet_length(convert_to(item.key, 'UTF8'))
                        + bb_training_feature_storage_bytes(item.value, input_depth + 1);
                    IF result_size > {TRAINING_FEATURE_SNAPSHOT_MAX_BYTES} THEN
                        RETURN {_INVALID_SIZE};
                    END IF;
                END LOOP;
                RETURN result_size;
            WHEN 'array' THEN
                FOR child IN SELECT value FROM jsonb_array_elements(input_value) LOOP
                    result_size := result_size
                        + bb_training_feature_storage_bytes(child, input_depth + 1);
                    IF result_size > {TRAINING_FEATURE_SNAPSHOT_MAX_BYTES} THEN
                        RETURN {_INVALID_SIZE};
                    END IF;
                END LOOP;
                RETURN result_size;
            ELSE RETURN {_INVALID_SIZE};
            END CASE;
        END;
        $storage_bytes$;
    """
    normalize_sql = f"""
        CREATE OR REPLACE FUNCTION bb_training_feature_normalize_jsonb(input_value JSONB)
        RETURNS JSONB
        LANGUAGE plpgsql IMMUTABLE
        AS $normalize_feature_json$
        DECLARE
            item RECORD;
            result JSONB;
        BEGIN
            CASE jsonb_typeof(input_value)
            WHEN 'number' THEN
                RETURN to_jsonb(trim_scale((input_value #>> '{{}}')::NUMERIC));
            WHEN 'object' THEN
                SELECT COALESCE(jsonb_object_agg(
                    entry.key,
                    bb_training_feature_normalize_jsonb(entry.value)
                ), '{empty_json}'::JSONB)
                    INTO result
                    FROM jsonb_each(input_value) AS entry(key, value);
                RETURN result;
            WHEN 'array' THEN
                SELECT COALESCE(jsonb_agg(
                    bb_training_feature_normalize_jsonb(entry.value)
                    ORDER BY entry.ordinality
                ), '[]'::JSONB)
                    INTO result
                    FROM jsonb_array_elements(input_value) WITH ORDINALITY
                        AS entry(value, ordinality);
                RETURN result;
            ELSE
                RETURN input_value;
            END CASE;
        END;
        $normalize_feature_json$;
    """

    market_fact_sql = f"""
        CREATE OR REPLACE FUNCTION bb_compact_market_fact(input_value JSONB)
        RETURNS JSONB
        LANGUAGE plpgsql IMMUTABLE
        AS $compact_market_fact$
        DECLARE
            source JSONB := COALESCE(input_value, '{empty_json}'::JSONB);
            storage JSONB := COALESCE(source->'storage_contract', '{empty_json}'::JSONB);
            identity JSONB := COALESCE(source->'native_identity', '{empty_json}'::JSONB);
            spec JSONB := COALESCE(identity->'contract_spec', '{empty_json}'::JSONB);
            consistency JSONB := COALESCE(source->'source_consistency', '{empty_json}'::JSONB);
            provenance JSONB := COALESCE(consistency->'provenance', '{empty_json}'::JSONB);
            consistency_path JSONB := COALESCE(consistency->'path', '{empty_json}'::JSONB);
            ticker_sources JSONB;
            executable_intervals JSONB;
            compact JSONB;
        BEGIN
            IF source->>'schema_version' IS DISTINCT FROM '2026-07-14.okx-native-market-fact.v1' THEN
                RETURN source;
            END IF;
            SELECT COALESCE(jsonb_agg(jsonb_strip_nulls(jsonb_build_object(
                'fact_id', item.value->'fact_id',
                'source_interface', item.value->'source_interface',
                'source_timestamp_ms', item.value->'source_timestamp_ms'
            ))), '[]'::JSONB)
                INTO ticker_sources
                FROM jsonb_array_elements(
                    COALESCE(consistency->'ticker_sources', '[]'::JSONB)
                ) AS item(value);
            SELECT COALESCE(jsonb_agg(jsonb_strip_nulls(jsonb_build_object(
                'source', item.value->'source',
                'bid', item.value->'bid',
                'ask', item.value->'ask',
                'source_timestamp_ms', item.value->'source_timestamp_ms'
            ))), '[]'::JSONB)
                INTO executable_intervals
                FROM jsonb_array_elements(
                    COALESCE(consistency->'executable_intervals', '[]'::JSONB)
                ) AS item(value);
            compact := jsonb_strip_nulls(jsonb_build_object(
                'schema_version', source->'schema_version',
                'symbol', source->'symbol',
                'source_interface', source->'source_interface',
                'source_endpoint', source->'source_endpoint',
                'source_channel', source->'source_channel',
                'source_timestamp_ms', source->'source_timestamp_ms',
                'source_sequence', source->'source_sequence',
                'received_at', source->'received_at',
                'stale', source->'stale',
                'source_quality_issue', source->'source_quality_issue',
                'fact_id', source->'fact_id',
                'native_identity', jsonb_strip_nulls(jsonb_build_object(
                    'inst_id', identity->'inst_id',
                    'inst_type', identity->'inst_type',
                    'uly', identity->'uly',
                    'contract_spec_version', identity->'contract_spec_version',
                    'contract_spec', jsonb_strip_nulls(jsonb_build_object(
                        'inst_id', spec->'inst_id',
                        'inst_type', spec->'inst_type',
                        'uly', spec->'uly',
                        'contract_type', spec->'contract_type',
                        'contract_value', spec->'contract_value',
                        'contract_multiplier', spec->'contract_multiplier',
                        'contract_value_currency', spec->'contract_value_currency',
                        'settle_currency', spec->'settle_currency',
                        'lot_size', spec->'lot_size',
                        'minimum_size', spec->'minimum_size',
                        'tick_size', spec->'tick_size',
                        'state', spec->'state',
                        'source', spec->'source',
                        'spec_version', spec->'spec_version'
                    ))
                )),
                'prices', source->'prices',
                'liquidity', source->'liquidity',
                'source_consistency', CASE
                    WHEN consistency <> '{empty_json}'::JSONB THEN jsonb_strip_nulls(jsonb_build_object(
                        'version', consistency->'version',
                        'status', consistency->'status',
                        'quotes_overlap', consistency->'quotes_overlap',
                        'quotes_temporally_reconciled', consistency->'quotes_temporally_reconciled',
                        'reasons', consistency->'reasons',
                        'reference_warnings', consistency->'reference_warnings',
                        'assertions', jsonb_strip_nulls(jsonb_build_object(
                            'native_identity_verified',
                                consistency->'assertions'->'native_identity_verified',
                            'executable_quotes_verified',
                                consistency->'assertions'->'executable_quotes_verified',
                            'tick_alignment_verified',
                                consistency->'assertions'->'tick_alignment_verified',
                            'reference_prices_verified',
                                consistency->'assertions'->'reference_prices_verified',
                            'one_minute_path_verified',
                                consistency->'assertions'->'one_minute_path_verified',
                            'market_continuity_verified',
                                consistency->'assertions'->'market_continuity_verified'
                        )),
                        'native_identity', jsonb_strip_nulls(jsonb_build_object(
                            'inst_id', consistency->'native_identity'->'inst_id',
                            'inst_type', consistency->'native_identity'->'inst_type',
                            'uly', consistency->'native_identity'->'uly',
                            'contract_spec_version',
                                consistency->'native_identity'->'contract_spec_version',
                            'tick_size', consistency->'native_identity'->'tick_size'
                        )),
                        'quote_reconciliation', jsonb_strip_nulls(jsonb_build_object(
                            'time_span_ms', consistency->'quote_reconciliation'->'time_span_ms',
                            'gap', consistency->'quote_reconciliation'->'gap',
                            'tolerance', consistency->'quote_reconciliation'->'tolerance',
                            'max_interval_spread',
                                consistency->'quote_reconciliation'->'max_interval_spread',
                            'source_skew_tolerance',
                                consistency->'quote_reconciliation'->'source_skew_tolerance',
                            'relative_tolerance_cap',
                                consistency->'quote_reconciliation'->'relative_tolerance_cap'
                        )),
                        'path', jsonb_strip_nulls(jsonb_build_object(
                            'source', consistency_path->'source',
                            'bar_count', consistency_path->'bar_count',
                            'path_low', consistency_path->'path_low',
                            'path_high', consistency_path->'path_high',
                            'observed_prices_within_path',
                                consistency_path->'observed_prices_within_path',
                            'continuity_accepted_via',
                                consistency_path->'continuity_accepted_via',
                            'missing_open_times_count',
                                CASE WHEN jsonb_typeof(
                                    consistency_path->'missing_open_times_ms'
                                ) = 'array' THEN to_jsonb(jsonb_array_length(
                                    consistency_path->'missing_open_times_ms'
                                )) ELSE consistency_path->'missing_open_times_count' END,
                            'missing_open_times_fingerprint',
                                CASE WHEN jsonb_typeof(
                                    consistency_path->'missing_open_times_ms'
                                ) = 'array' THEN to_jsonb(md5(
                                    (consistency_path->'missing_open_times_ms')::TEXT
                                )) ELSE consistency_path->'missing_open_times_fingerprint' END
                        )),
                        'reference_observations', jsonb_strip_nulls(jsonb_build_object(
                            'mark_price_available',
                                consistency->'reference_observations'->'mark_price_available',
                            'index_price_available',
                                consistency->'reference_observations'->'index_price_available'
                        )),
                        'ticker_sources', ticker_sources,
                        'executable_intervals', executable_intervals,
                        'provenance', jsonb_strip_nulls(jsonb_build_object(
                            'source', provenance->'source',
                            'strategy_version', provenance->'strategy_version',
                            'fallback_reason', CASE WHEN COALESCE(
                                provenance->>'fallback_reason', ''
                            ) <> '' THEN to_jsonb(left(
                                provenance->>'fallback_reason', 160
                            )) END,
                            'data_fingerprint', provenance->'data_fingerprint'
                        ))
                    )) END,
                'quality', jsonb_strip_nulls(jsonb_build_object(
                    'status', source->'quality'->'status',
                    'violation_count', source->'quality'->'violation_count',
                    'reasons', source->'quality'->'reasons'
                )),
                'storage_contract', jsonb_strip_nulls(jsonb_build_object(
                    'version', '2026-10-22.compact-market-fact.v1',
                    'full_fact_fingerprint', source->'fact_id',
                    'full_source_consistency_fingerprint',
                        provenance->'data_fingerprint'
                ))
            ));
            RETURN compact;
        END;
        $compact_market_fact$;
    """

    compact_sql = f"""
        CREATE OR REPLACE FUNCTION bb_compact_training_feature_snapshot(input_value JSONB)
        RETURNS JSONB
        LANGUAGE plpgsql IMMUTABLE
        AS $compact_features$
        DECLARE
            raw JSONB := CASE WHEN jsonb_typeof(input_value) = 'object'
                             THEN input_value ELSE '{{}}'::JSONB END;
            frozen JSONB;
            input_projection JSONB;
            compact JSONB;
            candidate JSONB;
            optional_item RECORD;
            frozen_keys TEXT[] := ARRAY[{frozen_keys}]::TEXT[];
            input_keys TEXT[] := ARRAY[{input_keys}]::TEXT[];
            raw_keys TEXT[] := ARRAY[{raw_keys}]::TEXT[];
        BEGIN
            SELECT COALESCE(jsonb_object_agg(item.key, item.value), '{{}}'::JSONB)
                INTO frozen
                FROM jsonb_each(raw) AS item(key, value)
                WHERE item.key = ANY(frozen_keys)
                  AND item.key <> 'feature_storage_compaction';
            IF frozen ? 'market_fact' THEN
                frozen := jsonb_set(
                    frozen,
                    '{{market_fact}}',
                    bb_compact_market_fact(frozen->'market_fact'),
                    true
                );
            END IF;
            SELECT COALESCE(jsonb_object_agg(item.key, item.value), '{{}}'::JSONB)
                INTO input_projection
                FROM jsonb_each(frozen) AS item(key, value)
                WHERE item.key = ANY(input_keys);
            input_projection := jsonb_strip_nulls(input_projection);
            IF raw ? 'feature_availability_contract' THEN
                frozen := frozen || jsonb_build_object(
                    'feature_storage_compaction',
                    jsonb_build_object(
                        'version', '2026-10-22.compact-feature-storage.v1',
                        'full_input_fingerprint',
                            raw->'feature_availability_contract'->>'input_fingerprint',
                        'compact_input_fingerprint',
                            md5(bb_training_feature_normalize_jsonb(input_projection)::TEXT)
                    )
                );
            END IF;
            IF bb_training_feature_storage_bytes(frozen) > {FROZEN_TRAINING_FEATURE_MAX_BYTES} THEN
                RETURN jsonb_build_object(
                    'training_feature_compaction_error', 'frozen_input_budget_exceeded'
                );
            END IF;
            compact := frozen;
            FOR optional_item IN
                SELECT entry.key, entry.value
                FROM jsonb_each(raw) AS entry(key, value)
                WHERE length(entry.key) <= 120
                  AND NOT (entry.key = ANY(frozen_keys))
                  AND NOT (entry.key = ANY(raw_keys))
                  AND (
                      jsonb_typeof(entry.value) IN ('number', 'boolean', 'null')
                      OR (
                          jsonb_typeof(entry.value) = 'string'
                          AND length(entry.value #>> '{{}}') <= {TRAINING_FEATURE_MAX_STRING_CHARS}
                      )
                      OR (
                          jsonb_typeof(entry.value) = 'object'
                          AND bb_training_feature_storage_bytes(entry.value)
                              <= {TRAINING_FEATURE_OBJECT_MAX_BYTES}
                      )
                  )
                ORDER BY entry.key
            LOOP
                candidate := compact || jsonb_build_object(
                    optional_item.key, optional_item.value
                );
                IF bb_training_feature_storage_bytes(candidate)
                    <= {TRAINING_FEATURE_SNAPSHOT_MAX_BYTES} THEN
                    compact := candidate;
                END IF;
            END LOOP;
            RETURN compact;
        END;
        $compact_features$;
    """
    return storage_sql, normalize_sql, market_fact_sql, compact_sql
