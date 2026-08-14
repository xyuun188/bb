"""Immutable strategy decision contracts shared by every execution mode."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any

STRATEGY_CONTEXT_VERSION = "bb.strategy-context.v1"
STRATEGY_DECISION_VERSION = "bb.strategy-decision.v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class StrategyContractError(ValueError):
    """Raised when strategy evidence is incomplete or internally inconsistent."""


class ExecutionMode(StrEnum):
    BACKTEST = "backtest"
    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"


class StrategyAction(StrEnum):
    ENTER = "enter"
    EXIT = "exit"
    HOLD = "hold"
    REDUCE = "reduce"


class PositionSide(StrEnum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"


def canonical_json_bytes(value: Any) -> bytes:
    normalized = thaw_json(freeze_json(value))
    try:
        text = json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise StrategyContractError(f"value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def freeze_json(value: Any) -> Any:
    """Deep-freeze JSON evidence while rejecting ambiguous/non-finite values."""

    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, datetime):
        return _utc_datetime(value, "snapshot datetime").isoformat()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StrategyContractError("snapshot contains a non-finite number")
        return float(value)
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise StrategyContractError("snapshot keys must be non-empty strings")
            frozen[raw_key] = freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(freeze_json(item) for item in value)
    raise StrategyContractError(
        f"snapshot contains unsupported value type: {type(value).__name__}"
    )


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Time-aligned pure strategy input, independent from order execution."""

    symbol: str
    market_snapshot: Mapping[str, Any]
    feature_snapshot: Mapping[str, Any]
    position_snapshot: Mapping[str, Any]
    account_constraints: Mapping[str, Any]
    decision_time: datetime
    execution_mode: ExecutionMode | str
    strategy_id: str
    strategy_version: str
    parameter_set_id: str
    parameter_version: str
    parameter_values: Mapping[str, Any]
    execution_assumptions: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = field(init=False, default=STRATEGY_CONTEXT_VERSION)
    strategy_input_sha256: str = field(init=False)
    context_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _required_text(self.symbol, "symbol"))
        object.__setattr__(
            self,
            "execution_mode",
            _enum_value(ExecutionMode, self.execution_mode, "execution_mode"),
        )
        object.__setattr__(
            self,
            "decision_time",
            _utc_datetime(self.decision_time, "decision_time"),
        )
        for field_name in (
            "strategy_id",
            "strategy_version",
            "parameter_set_id",
            "parameter_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )
        for field_name in (
            "market_snapshot",
            "feature_snapshot",
            "position_snapshot",
            "account_constraints",
            "parameter_values",
            "execution_assumptions",
        ):
            value = freeze_json(dict(getattr(self, field_name) or {}))
            object.__setattr__(self, field_name, value)
        input_hash = content_sha256(self.decision_input())
        object.__setattr__(self, "strategy_input_sha256", input_hash)
        object.__setattr__(self, "context_sha256", content_sha256(self.to_dict(False)))

    def decision_input(self) -> dict[str, Any]:
        """Return fields allowed to influence a pure strategy decision."""

        return {
            "contract_version": self.contract_version,
            "symbol": self.symbol,
            "market_snapshot": thaw_json(self.market_snapshot),
            "feature_snapshot": thaw_json(self.feature_snapshot),
            "position_snapshot": thaw_json(self.position_snapshot),
            "account_constraints": thaw_json(self.account_constraints),
            "decision_time": self.decision_time.isoformat(),
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "parameter_set_id": self.parameter_set_id,
            "parameter_version": self.parameter_version,
            "parameter_values": thaw_json(self.parameter_values),
        }

    def to_dict(self, include_hashes: bool = True) -> dict[str, Any]:
        payload = {
            **self.decision_input(),
            "execution_mode": self.execution_mode.value,
            "execution_assumptions": thaw_json(self.execution_assumptions),
        }
        if include_hashes:
            payload.update(
                {
                    "strategy_input_sha256": self.strategy_input_sha256,
                    "context_sha256": self.context_sha256,
                }
            )
        return payload


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    """Executor-neutral result of evaluating one :class:`StrategyContext`."""

    symbol: str
    action: StrategyAction | str
    side: PositionSide | str
    target_exposure: float
    confidence: float
    reason_codes: Sequence[str]
    protection_hints: Mapping[str, Any]
    strategy_version: str
    parameter_version: str
    decision_time: datetime
    strategy_input_sha256: str
    source: str
    contract_version: str = field(init=False, default=STRATEGY_DECISION_VERSION)
    decision_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _required_text(self.symbol, "symbol"))
        object.__setattr__(self, "action", _enum_value(StrategyAction, self.action, "action"))
        object.__setattr__(self, "side", _enum_value(PositionSide, self.side, "side"))
        object.__setattr__(
            self,
            "target_exposure",
            _finite_non_negative(self.target_exposure, "target_exposure"),
        )
        confidence = _finite_non_negative(self.confidence, "confidence")
        if confidence > 1:
            raise StrategyContractError("confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
        codes = tuple(_required_text(code, "reason_code") for code in self.reason_codes)
        if not codes:
            raise StrategyContractError("reason_codes must not be empty")
        object.__setattr__(self, "reason_codes", codes)
        object.__setattr__(self, "protection_hints", freeze_json(dict(self.protection_hints or {})))
        object.__setattr__(
            self,
            "strategy_version",
            _required_text(self.strategy_version, "strategy_version"),
        )
        object.__setattr__(
            self,
            "parameter_version",
            _required_text(self.parameter_version, "parameter_version"),
        )
        object.__setattr__(
            self,
            "decision_time",
            _utc_datetime(self.decision_time, "decision_time"),
        )
        object.__setattr__(
            self,
            "strategy_input_sha256",
            _required_sha256(self.strategy_input_sha256, "strategy_input_sha256"),
        )
        object.__setattr__(self, "source", _required_text(self.source, "source"))
        self._validate_action()
        object.__setattr__(self, "decision_sha256", content_sha256(self.to_dict(False)))

    def _validate_action(self) -> None:
        if self.action == StrategyAction.ENTER:
            if self.side == PositionSide.NONE or self.target_exposure <= 0:
                raise StrategyContractError("enter requires a side and positive target_exposure")
        elif self.action in {StrategyAction.EXIT, StrategyAction.REDUCE}:
            if self.side == PositionSide.NONE:
                raise StrategyContractError(f"{self.action.value} requires a position side")
            if self.action == StrategyAction.EXIT and self.target_exposure != 0:
                raise StrategyContractError("exit target_exposure must be zero")
        elif self.action == StrategyAction.HOLD and self.side == PositionSide.NONE:
            if self.target_exposure != 0:
                raise StrategyContractError("flat hold target_exposure must be zero")

    def semantics(self) -> dict[str, Any]:
        """Return execution-mode-neutral decision fields used for parity checks."""

        return {
            "contract_version": self.contract_version,
            "symbol": self.symbol,
            "action": self.action.value,
            "side": self.side.value,
            "target_exposure": self.target_exposure,
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "protection_hints": thaw_json(self.protection_hints),
            "strategy_version": self.strategy_version,
            "parameter_version": self.parameter_version,
            "decision_time": self.decision_time.isoformat(),
            "strategy_input_sha256": self.strategy_input_sha256,
            "source": self.source,
        }

    def to_dict(self, include_hash: bool = True) -> dict[str, Any]:
        payload = self.semantics()
        if include_hash:
            payload["decision_sha256"] = self.decision_sha256
        return payload


def assert_decision_matches_context(
    context: StrategyContext,
    decision: StrategyDecision,
) -> None:
    if decision.symbol != context.symbol:
        raise StrategyContractError("decision symbol does not match context")
    if decision.decision_time != context.decision_time:
        raise StrategyContractError("decision_time does not match context")
    if decision.strategy_version != context.strategy_version:
        raise StrategyContractError("strategy_version does not match context")
    if decision.parameter_version != context.parameter_version:
        raise StrategyContractError("parameter_version does not match context")
    if decision.strategy_input_sha256 != context.strategy_input_sha256:
        raise StrategyContractError("decision input fingerprint does not match context")


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise StrategyContractError(f"{field_name} must not be empty")
    if len(text) > 240:
        raise StrategyContractError(f"{field_name} is too long")
    return text


def _required_sha256(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name).lower()
    if not SHA256_PATTERN.fullmatch(text):
        raise StrategyContractError(f"{field_name} must be a SHA-256 hex digest")
    return text


def _utc_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise StrategyContractError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise StrategyContractError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _finite_non_negative(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StrategyContractError(f"{field_name} must be numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise StrategyContractError(f"{field_name} must be finite and non-negative")
    return number


def _enum_value(enum_type: type[StrEnum], value: Any, field_name: str) -> Any:
    try:
        return enum_type(str(getattr(value, "value", value)).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise StrategyContractError(f"{field_name} must be one of: {allowed}") from exc
