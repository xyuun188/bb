"""DTOs for entry and exit policy pipeline boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput


def _action_value(decision: DecisionOutput) -> str:
    return str(getattr(decision.action, "value", decision.action))


@dataclass(frozen=True, slots=True)
class EntryPipelineContext:
    decision: DecisionOutput
    model_name: str
    model_mode: str
    open_positions: tuple[dict[str, Any], ...]

    @classmethod
    def from_inputs(
        cls,
        *,
        decision: DecisionOutput,
        model_name: str,
        model_mode: str,
        open_positions: list[dict[str, Any]] | None,
    ) -> EntryPipelineContext:
        return cls(
            decision=decision,
            model_name=model_name,
            model_mode=model_mode,
            open_positions=tuple(open_positions or ()),
        )

    def public_data(self) -> dict[str, Any]:
        return {
            "pipeline": "entry",
            "model_name": self.model_name,
            "model_mode": self.model_mode,
            "symbol": self.decision.symbol,
            "action": _action_value(self.decision),
            "open_position_count": len(self.open_positions),
        }


@dataclass(frozen=True, slots=True)
class ExitPipelineContext:
    decision: DecisionOutput
    model_name: str
    open_positions: tuple[dict[str, Any], ...]
    refreshed_positions: tuple[dict[str, Any], ...] = ()
    arbitration: dict[str, Any] | None = None

    @classmethod
    def from_inputs(
        cls,
        *,
        decision: DecisionOutput,
        model_name: str,
        open_positions: list[dict[str, Any]] | None,
    ) -> ExitPipelineContext:
        return cls(
            decision=decision,
            model_name=model_name,
            open_positions=tuple(open_positions or ()),
        )

    def with_refreshed_positions(
        self,
        positions: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> ExitPipelineContext:
        return ExitPipelineContext(
            decision=self.decision,
            model_name=self.model_name,
            open_positions=self.open_positions,
            refreshed_positions=tuple(positions or ()),
            arbitration=self.arbitration,
        )

    def with_arbitration(self, arbitration: dict[str, Any]) -> ExitPipelineContext:
        return ExitPipelineContext(
            decision=self.decision,
            model_name=self.model_name,
            open_positions=self.open_positions,
            refreshed_positions=self.refreshed_positions,
            arbitration=arbitration,
        )

    def public_data(self) -> dict[str, Any]:
        return {
            "pipeline": "exit",
            "model_name": self.model_name,
            "symbol": self.decision.symbol,
            "action": _action_value(self.decision),
            "open_position_count": len(self.open_positions),
            "refreshed_position_count": len(self.refreshed_positions),
            "arbitration": self.arbitration or {},
        }
