"""OKX exchange error classification helpers.

The trading pipeline uses these helpers to keep execution, learning, and
UI wording aligned.  They deliberately classify exchange-side transient
failures separately from strategy mistakes and order-rule validation errors.
"""

from __future__ import annotations

import json
from typing import Any

OKX_TEMPORARY_SERVICE_CODE = "50001"
OKX_TEMPORARY_SERVICE_MARKERS = (
    "service temporarily unavailable",
    "temporarily unavailable",
    "max retries exceeded",
)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def extract_okx_error(value: Any) -> tuple[str | None, str | None]:
    """Return the most specific OKX error code/message found in a payload."""

    payload: dict[str, Any] | None
    if isinstance(value, dict):
        payload = value
    else:
        payload = _extract_json_payload(_stringify(value))
    if not payload:
        return None, None

    code = str(payload.get("code") or "").strip() or None
    message = str(payload.get("msg") or payload.get("message") or "").strip() or None
    rows = payload.get("data")
    if isinstance(rows, dict):
        rows = [rows]
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            code = str(row.get("sCode") or row.get("code") or code or "").strip() or code
            message = (
                str(
                    row.get("sMsg") or row.get("msg") or row.get("message") or message or ""
                ).strip()
                or message
            )
            if code or message:
                break
    return code, message


def is_okx_temporary_service_error(value: Any) -> bool:
    """True when OKX reports an exchange-side temporary service outage."""

    text = _stringify(value).lower()
    code, message = extract_okx_error(value)
    message_text = str(message or "").lower()
    combined = f"{text} {message_text}"
    if code == OKX_TEMPORARY_SERVICE_CODE:
        return True
    if OKX_TEMPORARY_SERVICE_CODE in combined and "okx" in combined:
        return True
    has_marker = any(marker in combined for marker in OKX_TEMPORARY_SERVICE_MARKERS)
    return bool(has_marker and ("okx" in combined or OKX_TEMPORARY_SERVICE_CODE in combined))


def okx_temporary_service_error_message(value: Any | None = None) -> str:
    """User-facing text for OKX temporary service failures."""

    _, message = extract_okx_error(value) if value is not None else (None, None)
    suffix = (
        f"OKX 原文：{message}"
        if message
        else "OKX 原文：Service temporarily unavailable. Please try again later."
    )
    return (
        "OKX 返回错误码 50001：交易所服务临时不可用，系统没有拿到订单成交确认。"
        "这不是下单前交易规则读取、最小张数、精度或仓位计算错误，也不计为策略质量失败；"
        "系统会临时跳过该币种，稍后自动重试。"
        f"{suffix}"
    )
