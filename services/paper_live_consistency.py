"""Deterministic parity contract for paper and live decision execution."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

PAPER_LIVE_PARITY_CONTRACT_VERSION = "2026-07-19.paper-live-decision-parity.v1"
PARITY_FIELDS = (
    "model_name",
    "action",
    "confidence",
    "position_size_pct",
    "suggested_leverage",
    "stop_loss_pct",
    "take_profit_pct",
)


def _value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _scalar(value: Any) -> Any:
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(f"{value:.15g}")
    return value


def decision_contract(decision: Any) -> dict[str, Any]:
    return {
        field: _scalar(_value(decision, field))
        for field in PARITY_FIELDS
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compare_paper_live_decisions(
    paper_decision: Any,
    live_decision: Any,
    *,
    paper_model_sha256: str | None,
    live_model_sha256: str | None,
    paper_strategy_fingerprint: str | None,
    live_strategy_fingerprint: str | None,
) -> dict[str, Any]:
    paper = decision_contract(paper_decision)
    live = decision_contract(live_decision)
    differences = {
        field: {"paper": paper[field], "live": live[field]}
        for field in PARITY_FIELDS
        if paper[field] != live[field]
    }
    model_match = bool(
        paper_model_sha256
        and live_model_sha256
        and paper_model_sha256 == live_model_sha256
    )
    strategy_match = bool(
        paper_strategy_fingerprint
        and live_strategy_fingerprint
        and paper_strategy_fingerprint == live_strategy_fingerprint
    )
    return {
        "contract_version": PAPER_LIVE_PARITY_CONTRACT_VERSION,
        "ok": not differences and model_match and strategy_match,
        "decision_fields_match": not differences,
        "model_sha256_match": model_match,
        "strategy_fingerprint_match": strategy_match,
        "differences": differences,
        "paper_contract": paper,
        "live_contract": live,
        "paper_decision_fingerprint": _fingerprint(paper),
        "live_decision_fingerprint": _fingerprint(live),
    }


def assert_paper_live_decision_parity(
    paper_decision: Any,
    live_decision: Any,
    *,
    paper_model_sha256: str | None,
    live_model_sha256: str | None,
    paper_strategy_fingerprint: str | None,
    live_strategy_fingerprint: str | None,
) -> dict[str, Any]:
    report = compare_paper_live_decisions(
        paper_decision,
        live_decision,
        paper_model_sha256=paper_model_sha256,
        live_model_sha256=live_model_sha256,
        paper_strategy_fingerprint=paper_strategy_fingerprint,
        live_strategy_fingerprint=live_strategy_fingerprint,
    )
    if not report["ok"]:
        reasons = []
        if report["differences"]:
            reasons.append("decision_fields_mismatch")
        if not report["model_sha256_match"]:
            reasons.append("model_sha256_mismatch_or_missing")
        if not report["strategy_fingerprint_match"]:
            reasons.append("strategy_fingerprint_mismatch_or_missing")
        raise ValueError("paper/live parity contract failed: " + ",".join(reasons))
    return report
