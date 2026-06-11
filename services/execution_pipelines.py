"""Execution-facing entry and exit policy pipelines."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_brain.base_model import DecisionOutput
from services.exit_intent import classify_exit_intent
from services.trading_params import attach_trading_parameter_snapshot
from services.trading_policies import PolicyGateResult

PolicyProvider = Callable[[], Any]


def _policy_from_provider(provider: PolicyProvider, dependency_name: str) -> Any:
    policy = provider()
    if policy is None:
        raise RuntimeError(f"{dependency_name} is not initialized")
    return policy


def _with_strategy_parameters(
    result: PolicyGateResult,
    payload: dict[str, Any],
) -> PolicyGateResult:
    data = dict(result.data or {})
    data["strategy_parameters"] = payload
    result.data = data
    return result


class EntryExecutionPipeline:
    """Entry execution policy boundary used by ExecutionService."""

    def __init__(self, entry_policy_provider: PolicyProvider) -> None:
        self.entry_policy_provider = entry_policy_provider

    async def evaluate(
        self,
        decision: DecisionOutput,
        model_name: str,
        model_mode: str,
        open_positions: list[dict[str, Any]] | None,
    ) -> PolicyGateResult:
        payload = attach_trading_parameter_snapshot(
            decision,
            scope="entry_execution",
        )
        policy = _policy_from_provider(
            self.entry_policy_provider,
            "EntryExecutionPipeline.entry_policy",
        )
        result = await policy.evaluate(
            decision,
            model_name,
            model_mode,
            open_positions,
        )
        return _with_strategy_parameters(result, payload)


class ExitExecutionPipeline:
    """Exit execution policy boundary used by ExecutionService."""

    def __init__(self, exit_policy_provider: PolicyProvider) -> None:
        self.exit_policy_provider = exit_policy_provider

    async def evaluate(
        self,
        decision: DecisionOutput,
        model_name: str,
        open_positions: list[dict[str, Any]] | None,
        *,
        refresh_positions: bool = True,
    ) -> PolicyGateResult:
        intent = classify_exit_intent(decision) if decision.is_exit else None
        if intent is not None:
            raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
            raw["exit_pipeline"] = {
                "intent": intent.value,
                "stage": "pre_policy",
                "structured": True,
            }
            decision.raw_response = raw
        payload = attach_trading_parameter_snapshot(
            decision,
            scope="exit_execution",
        )
        policy = _policy_from_provider(
            self.exit_policy_provider,
            "ExitExecutionPipeline.exit_policy",
        )
        result = await policy.evaluate(
            decision,
            model_name,
            open_positions,
            refresh_positions=refresh_positions,
        )
        return _with_strategy_parameters(result, payload)
