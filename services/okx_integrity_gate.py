"""Shared fail-closed classification for OKX integrity audit issues."""

from __future__ import annotations

import json
from typing import Any

OKX_TRADE_FACT_QUARANTINED_WARNING_KINDS = frozenset(
    {
        "contract_specification_evidence_missing",
        "manual_close_position_fact_not_exchange_backed",
        "okx_fill_not_linked_to_position",
        "order_position_missing",
        "orphan_position_quarantine_not_exchange_backed",
        "position_missing_close_order_link",
        "position_missing_entry_order_link",
        "position_order_link_missing_local_order",
        "superseded_position_residual",
    }
)

_EXPLICIT_INTEGRITY_EVIDENCE_KEYS = frozenset(
    {
        "issue_count",
        "warning_count",
        "critical_count",
        "severity_counts",
        "okx_authoritative_sync",
        "position_fact_link_repair",
    }
)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _issue_key(issue: dict[str, Any]) -> str:
    return json.dumps(issue, ensure_ascii=True, sort_keys=True, default=str)


def _dedupe_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for issue in issues:
        key = _issue_key(issue)
        if key in seen:
            continue
        seen.add(key)
        result.append(issue)
    return result


def partition_okx_integrity_issues(
    report: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return current blockers and explicitly quarantined historical warnings."""

    details = _safe_dict(report)
    authoritative = _safe_dict(details.get("okx_authoritative_sync"))
    sources = [details]
    if authoritative:
        sources.append(authoritative)

    issues = _dedupe_issues(
        [
            issue
            for source in sources
            for issue in _safe_list(source.get("issues"))
            if isinstance(issue, dict)
        ]
    )
    blocking: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    has_explicit_evidence = any(
        key in source for source in sources for key in _EXPLICIT_INTEGRITY_EVIDENCE_KEYS
    )
    if not issues and not has_explicit_evidence:
        blocking.append({"kind": "integrity_evidence_missing"})

    for issue in issues:
        severity = str(issue.get("severity") or "").strip().lower()
        kind = str(issue.get("kind") or "").strip()
        if severity == "critical":
            blocking.append(issue)
        elif severity == "warning":
            if kind in OKX_TRADE_FACT_QUARANTINED_WARNING_KINDS:
                quarantined.append(issue)
            else:
                blocking.append(issue)
        elif severity not in {"info", "observation"}:
            blocking.append(issue)

    for index, source in enumerate(sources):
        source_name = "authoritative" if index else "trade_fact"
        source_issues = [
            issue for issue in _safe_list(source.get("issues")) if isinstance(issue, dict)
        ]
        severity_counts = _safe_dict(source.get("severity_counts"))
        critical_count = max(
            _safe_int(source.get("critical_count")),
            _safe_int(severity_counts.get("critical")),
        )
        warning_count = max(
            _safe_int(source.get("warning_count")),
            _safe_int(severity_counts.get("warning")),
            _safe_int(source.get("manual_review_count")),
        )
        informational_count = max(
            _safe_int(source.get("info_count")),
            _safe_int(severity_counts.get("info")),
        ) + max(
            _safe_int(source.get("observation_count")),
            _safe_int(severity_counts.get("observation")),
        )
        if critical_count > 0 and not any(
            str(issue.get("severity") or "").lower() == "critical" for issue in source_issues
        ):
            blocking.append({"kind": f"unclassified_{source_name}_critical_integrity_issue"})
        if warning_count > 0 and not source_issues:
            blocking.append({"kind": f"unclassified_{source_name}_warning"})
        if _safe_int(source.get("repairable_count")) > 0:
            blocking.append({"kind": f"{source_name}_repairable_issue"})
        if (
            _safe_int(source.get("issue_count")) > 0
            and not source_issues
            and critical_count <= 0
            and warning_count <= 0
            and informational_count < _safe_int(source.get("issue_count"))
        ):
            blocking.append({"kind": f"unclassified_{source_name}_integrity_issue"})

    return _dedupe_issues(blocking), _dedupe_issues(quarantined)


def okx_integrity_has_current_blocking_issue(report: dict[str, Any]) -> bool:
    blocking, _quarantined = partition_okx_integrity_issues(report)
    return bool(blocking)
