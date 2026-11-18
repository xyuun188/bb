"""Dynamic pre-execution entry price validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from math import isfinite
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from core.symbols import okx_inst_id_from_symbol
from services.production_trade_gate import validate_production_trade_gate


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def _feature_snapshot(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is not None and hasattr(value, "to_dict"):
        snapshot = value.to_dict()
        return snapshot if isinstance(snapshot, dict) else {}
    return {}


def paper_entry_price_budget_reason(
    decision: DecisionOutput,
    model_mode: str,
) -> str | None:
    """Reject consumed paper signals only after final execution-cost sizing.

    The mean return is the paper entry authority. A negative downside quantile
    remains valid training evidence and must not become a new entry threshold.
    This check never creates the separate normal-paper trading permission.
    """

    if str(model_mode or "").lower() != "paper" or not decision.is_entry:
        return None
    raw = _safe_dict(decision.raw_response)
    if not _safe_dict(raw.get("normal_paper_trade")):
        return None
    sizing = _safe_dict(raw.get("profit_risk_sizing"))
    if sizing.get("production_eligible") is not True:
        return None  # Preserve the more specific sizing denial downstream.
    basis = _safe_dict(raw.get("pre_execution_analysis_basis"))
    check = _safe_dict(raw.get("pre_execution_price_check"))
    price = _safe_float(basis.get("snapshot_price"))
    latest = _safe_float(check.get("latest_price"))
    if not (
        basis.get("version") == "2026-08-19.entry-analysis-basis.v1"
        and basis.get("validated") is True
        and basis.get("symbol") == decision.symbol
        and basis.get("action") == decision.action.value
        and price > 0.0
        and latest > 0.0
        and _safe_float(check.get("snapshot_price")) == price
    ):
        return "模拟盘提交前原始分析价或最新价格证据不完整，未提交订单。"
    expected = _safe_float(sizing.get("expected_net_return_pct"))
    adverse = EntryPriceGuardPolicy._adverse_move(
        decision.action, (latest - price) / price,
    )
    remaining = expected - adverse * 100.0
    check.update({
        "return_budget_fraction": round(expected / 100.0, 8),
        "allowed_adverse_move_fraction": round(expected / 100.0, 8),
        "remaining_expected_net_return_pct": round(remaining, 8),
        "return_budget_basis": "final_sizing_expected_net_return_pct",
        "return_budget_verified": remaining > 0.0,
        "profitability_gate_applied": True,
        "safety_scope": "market_integrity_and_signal_validity",
    })
    provenance = dict(_safe_dict(check.get("policy_provenance")))
    provenance.update({
        "source": "normal_paper_final_expected_return",
        "strategy_version": "2026-11-16.paper-price-return-budget.v1",
    })
    check["policy_provenance"] = provenance
    raw["pre_execution_price_check"] = check
    decision.raw_response = raw
    if remaining <= 0.0:
        return (
            "下单前价格变化已耗尽本次信号的预期净收益，未提交旧信号订单；"
            "等待基于最新行情重新分析。"
        )
    return None


def paper_entry_execution_price_limit(
    decision: DecisionOutput,
    model_mode: str,
    *,
    latest_price: float,
    tick_size: float,
    protection: dict[str, Any] | None = None,
    price_limit: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Carry the approved paper signal budget through the actual venue fill."""
    raw = _safe_dict(decision.raw_response)
    if (
        str(model_mode or "").lower() != "paper"
        or not decision.is_entry
        or not _safe_dict(raw.get("normal_paper_trade"))
    ):
        return None
    basis = _safe_dict(raw.get("pre_execution_analysis_basis"))
    check = _safe_dict(raw.get("pre_execution_price_check"))
    sizing = _safe_dict(raw.get("profit_risk_sizing"))
    price = _safe_float(basis.get("snapshot_price"))
    latest = _safe_float(latest_price)
    tick = _safe_float(tick_size)
    expected = _safe_float(sizing.get("expected_net_return_pct"))
    result: dict[str, Any] = {
        "version": "2026-11-17.paper-execution-return-budget.v2",
        "eligible": False,
        "reason": "paper_execution_return_budget_evidence_incomplete",
        "side": "buy" if decision.action == Action.LONG else "sell",
        "snapshot_price": price,
        "latest_price": latest,
        "expected_net_return_pct": expected,
        "tick_size": tick,
        "fallback_order_type": "ioc",
    }
    if (
        sizing.get("production_eligible") is True
        and basis.get("version") == "2026-08-19.entry-analysis-basis.v1"
        and basis.get("validated") is True
        and basis.get("symbol") == decision.symbol
        and basis.get("action") == decision.action.value
        and _safe_float(check.get("snapshot_price")) == price
        and check.get("return_budget_verified") is True
        and min(price, latest, tick, expected) > 0.0
    ):
        # Strictly round inside the zero-return boundary, including when it
        # falls exactly on a tick. The same cap also binds delayed fallbacks.
        original = Decimal(str(price))
        step = Decimal(str(tick))
        budget = Decimal(str(expected)) / Decimal(100)
        long = decision.action == Action.LONG
        boundary = original * (1 + budget if long else 1 - budget)
        units = (boundary / step).to_integral_value(
            rounding=ROUND_CEILING if long else ROUND_FLOOR,
        )
        limit = float((units - 1 if long else units + 1) * step)
        result["return_budget_limit_price"] = limit
        venue_ready = True
        if price_limit is not None:
            buy_limit = _safe_float(price_limit.get("buy_price_limit"))
            sell_limit = _safe_float(price_limit.get("sell_price_limit"))
            venue_ready = bool(0.0 < sell_limit <= buy_limit)
            if venue_ready:
                venue_units = (
                    Decimal(str(buy_limit if long else sell_limit)) / step
                ).to_integral_value(rounding=ROUND_FLOOR if long else ROUND_CEILING)
                venue_limit = float(venue_units * step)
                limit = min(limit, venue_limit) if long else max(limit, venue_limit)
                venue_ready = sell_limit <= limit <= buy_limit
                result["venue_limit_price"] = venue_limit
            result["venue_price_limit_compatible"] = venue_ready
        protection_ready = True
        if protection is not None:
            stop = _safe_float(protection.get("stop_loss_price"))
            target = _safe_float(protection.get("take_profit_price"))
            protection_ready = bool(
                min(stop, target) > 0.0
                and (stop < latest < target if long else target < latest < stop)
            )
            if protection_ready:
                # OKX compares attached triggers to the parent order price,
                # including an IOC's worst fill price, not just the quote.
                target_units = (Decimal(str(target)) / step).to_integral_value(
                    rounding=ROUND_CEILING if long else ROUND_FLOOR,
                )
                protection_limit = float(
                    (target_units - 1 if long else target_units + 1) * step
                )
                limit = min(limit, protection_limit) if long else max(limit, protection_limit)
                protection_ready = bool(
                    stop < limit < target if long else target < limit < stop
                )
                result["protection_limit_price"] = protection_limit
            result.update({
                "stop_loss_price": stop,
                "take_profit_price": target,
                "protection_compatible": protection_ready,
            })
        if price_limit is not None and venue_ready:
            venue_ready = sell_limit <= limit <= buy_limit
            result["venue_price_limit_compatible"] = venue_ready
        adverse = EntryPriceGuardPolicy._adverse_move(
            decision.action, (latest - price) / price,
        )
        worst_adverse = EntryPriceGuardPolicy._adverse_move(
            decision.action, (limit - price) / price,
        )
        remaining = expected - adverse * 100.0
        quote_outside_venue = bool(
            price_limit is not None and venue_ready
            and (latest > buy_limit if long else latest < sell_limit)
        )
        eligible = (
            venue_ready and protection_ready and limit > 0.0
            and (latest <= limit if long else latest >= limit)
        )
        result.update({
            "eligible": eligible,
            "reason": (
                "paper_execution_venue_price_limit_incompatible" if not venue_ready
                else "paper_execution_protection_incompatible" if not protection_ready
                else "paper_execution_return_budget_ready" if eligible
                else "paper_execution_quote_outside_venue_price_limit" if quote_outside_venue
                else "paper_execution_quote_consumed_return_budget"
            ),
            "limit_price": limit,
            "remaining_expected_net_return_pct": round(remaining, 8),
            "worst_remaining_expected_net_return_pct": round(
                expected - worst_adverse * 100.0, 8,
            ),
        })
    raw["entry_execution_price_check"] = result
    decision.raw_response = raw
    return result


@dataclass(slots=True)
class EntryPriceGuardPolicy:
    """Require current adverse drift to fit inside authoritative return budget."""

    fresh_feature_provider: Callable[[str], Awaitable[Any]]
    market_data_quality_reason_provider: Callable[..., str | None]
    decision_age_seconds_provider: Callable[[DecisionOutput], float]
    pre_order_execution_facts_provider: (
        Callable[[str, DecisionOutput], Awaitable[dict[str, Any]]] | None
    ) = None

    async def guard_reason(
        self,
        decision: DecisionOutput,
        model_mode: str = "",
    ) -> str | None:
        if not decision.is_entry:
            return None

        snapshot = _safe_dict(decision.feature_snapshot)
        raw = _safe_dict(decision.raw_response)

        # The execution snapshot is the only market-data contract that can
        # authorize an order.  The analysis snapshot is historical context:
        # it remains an immutable price basis for drift measurement, but its
        # native fact must never veto a newer authoritative execution fact.
        execution_facts: dict[str, Any] = {}
        if self.pre_order_execution_facts_provider is not None:
            try:
                execution_facts = await self.pre_order_execution_facts_provider(
                    model_mode,
                    decision,
                )
            except Exception:
                return "Authoritative pre-order execution facts are unavailable; entry fails closed."
            if execution_facts.get("production_eligible") is not True:
                reasons = execution_facts.get("reasons") or [execution_facts.get("reason")]
                if "okx_entry_pair_compliance_restricted" in reasons:
                    return "OKX 曾因账户地区或合规限制拒绝该合约，当前仍在复查冷却期，本次未提交订单。"
                if "okx_pre_order_account_direction_capacity_zero" in reasons:
                    return (
                        "OKX 账户当前该合约方向最大可开数量为 0 张，未提交订单；"
                        "系统将自动复查额度，不影响其他合约或方向。"
                    )
                if "okx_pre_order_account_order_capacity_unavailable" in reasons:
                    return "OKX 账户方向可开额度暂时无法核验，未提交订单，等待额度查询恢复。"
                return (
                    "Authoritative pre-order execution facts are incomplete; entry fails closed: "
                    f"{execution_facts.get('reason') or 'unknown'}"
                )
            execution_snapshot = _safe_dict(execution_facts.get("feature_snapshot"))
            if not execution_snapshot:
                return "Authoritative pre-order execution snapshot is missing; entry fails closed."
            execution_inst_id = str(execution_facts.get("inst_id") or "").upper()
            expected_inst_id = okx_inst_id_from_symbol(decision.symbol).upper()
            if (
                not execution_inst_id
                or not expected_inst_id
                or execution_inst_id != expected_inst_id
            ):
                return "Pre-order market fact and execution fact instrument mismatch; entry fails closed."
            execution_quality_reason = self.market_data_quality_reason_provider(
                execution_snapshot,
                stage_label="pre-order execution snapshot",
            )
            if execution_quality_reason:
                return (
                    "Authoritative pre-order execution market fact is invalid; "
                    f"entry fails closed: {execution_quality_reason}"
                )
            fresh = execution_snapshot
        else:
            fresh = await self._fresh_valid_snapshot(decision.symbol)
            if not fresh:
                return "Fresh pre-order native market fact is incomplete; entry fails closed."

        stored_basis = _safe_dict(raw.get("pre_execution_analysis_basis"))
        stored_price = _safe_float(stored_basis.get("snapshot_price"))
        stored_basis_valid = bool(
            stored_basis.get("version") == "2026-08-19.entry-analysis-basis.v1"
            and stored_basis.get("validated") is True
            and str(stored_basis.get("symbol") or "") == str(decision.symbol or "")
            and str(stored_basis.get("action") or "") == decision.action.value
            and stored_price > 0
        )
        if stored_basis_valid:
            snapshot_price = stored_price
            analysis_fact = _safe_dict(stored_basis.get("market_fact"))
        else:
            snapshot_price = _safe_float(snapshot.get("current_price") or snapshot.get("close"))
            analysis_fact = _safe_dict(snapshot.get("market_fact"))
            stored_basis = {
                "version": "2026-08-19.entry-analysis-basis.v1",
                "validated": True,
                "symbol": decision.symbol,
                "action": decision.action.value,
                "snapshot_price": snapshot_price,
                "market_fact": analysis_fact,
            }

        if snapshot_price <= 0:
            return "Pre-order analysis price is missing; entry fails closed."
        latest_price = _safe_float(fresh.get("current_price") or fresh.get("close"))
        if latest_price <= 0:
            return "Fresh pre-order native price is unavailable; entry fails closed."

        paper_trade = str(model_mode or "").lower() == "paper"
        live_rules_canary = False
        if not paper_trade:
            gate_validation = validate_production_trade_gate(
                _safe_dict(decision.raw_response).get("production_trade_gate"),
                required_mode="live_rules_canary",
            )
            live_rules_canary = gate_validation.valid
        return_budget = (
            None
            if paper_trade or live_rules_canary
            else self._return_budget_fraction(decision)
        )
        if not paper_trade and not live_rules_canary and (
            return_budget is None or return_budget <= 0
        ):
            return "Authoritative fee-after return budget is missing; entry fails closed."

        move = (latest_price - snapshot_price) / snapshot_price
        adverse = self._adverse_move(decision.action, move)
        allowed = None if paper_trade or live_rules_canary else return_budget
        fresh_fact = _safe_dict(fresh.get("market_fact"))
        raw["pre_execution_analysis_basis"] = stored_basis
        raw["pre_execution_price_check"] = {
            "snapshot_price": snapshot_price,
            "latest_price": latest_price,
            "adverse_move_fraction": round(adverse, 8),
            "return_budget_fraction": (
                round(return_budget, 8) if return_budget is not None else None
            ),
            "allowed_adverse_move_fraction": (
                round(allowed, 8) if allowed is not None else None
            ),
            "decision_age_seconds": round(self.decision_age_seconds_provider(decision), 3),
            "contract_lifecycle": (
                "normal_paper_trade"
                if paper_trade
                else "live_rules_canary"
                if live_rules_canary
                else "live_ml"
            ),
            "production_permission": False if paper_trade else True,
            "profitability_gate_applied": not paper_trade and not live_rules_canary,
            "safety_scope": (
                "market_integrity_only"
                if paper_trade or live_rules_canary
                else "market_integrity_and_return_budget"
            ),
            "policy_provenance": {
                "source": (
                    "normal_paper_market_integrity_only"
                    if paper_trade
                    else "live_rules_canary_market_integrity"
                    if live_rules_canary
                    else "authoritative_fee_after_return_lcb"
                ),
                "observation_window": "current_pre_order_refresh",
                "sample_count": (
                    1
                    if paper_trade
                    else self._return_sample_count(decision)
                ),
                "generated_at": raw.get("generated_at") or "decision_runtime",
                "strategy_version": "2026-07-12.dynamic-price-budget.v1",
                "fallback_reason": "",
            },
            "native_market_fact_proof": {
                "analysis_fact_id": analysis_fact.get("fact_id"),
                "fresh_fact_id": fresh_fact.get("fact_id"),
                "analysis_inst_id": _safe_dict(
                    analysis_fact.get("native_identity")
                ).get("inst_id"),
                "fresh_inst_id": _safe_dict(fresh_fact.get("native_identity")).get(
                    "inst_id"
                )
                or execution_facts.get("inst_id"),
                "fresh_source_timestamp_ms": fresh_fact.get("source_timestamp_ms")
                or execution_facts.get("ticker_source_timestamp_ms"),
                "fresh_source_interface": fresh_fact.get("source_interface")
                or _safe_dict(execution_facts.get("policy_provenance")).get("source"),
            },
        }
        if paper_trade or live_rules_canary or (allowed is not None and adverse <= allowed):
            public_execution_facts = {
                key: value
                for key, value in execution_facts.items()
                if key not in {"feature_snapshot", "fee_snapshot"}
            }
            if execution_facts:
                fingerprint_payload = {
                    "facts": public_execution_facts,
                    "feature_snapshot": _safe_dict(execution_facts.get("feature_snapshot")),
                }
                public_execution_facts["input_fingerprint"] = hashlib.sha256(
                    json.dumps(
                        fingerprint_payload,
                        ensure_ascii=True,
                        sort_keys=True,
                        default=str,
                    ).encode("utf-8")
                ).hexdigest()
                raw["pre_order_execution_facts"] = public_execution_facts
                fresh["pre_order_execution_facts"] = public_execution_facts
            decision.feature_snapshot = {**snapshot, **fresh}
            decision.raw_response = raw
            return None

        decision.raw_response = raw

        return (
            "Current adverse price movement exceeds the authoritative fee-after return "
            "budget."
        )

    async def _fresh_valid_snapshot(self, symbol: str) -> dict[str, Any]:
        snapshot = _feature_snapshot(await self.fresh_feature_provider(symbol))
        if not snapshot:
            return {}
        reason = self.market_data_quality_reason_provider(
            snapshot,
            stage_label="pre-order refreshed market snapshot",
        )
        return {} if reason else snapshot

    @staticmethod
    def _adverse_move(action: Action, move: float) -> float:
        if action == Action.LONG:
            return max(move, 0.0)
        if action == Action.SHORT:
            return max(-move, 0.0)
        return 0.0

    @staticmethod
    def _side(decision: DecisionOutput) -> str:
        return "long" if decision.action == Action.LONG else "short"

    def _side_evidence(self, decision: DecisionOutput) -> dict[str, Any]:
        raw = _safe_dict(decision.raw_response)
        evidence = _safe_dict(raw.get("entry_candidate_evidence"))
        side_evidence = _safe_dict(evidence.get(self._side(decision)))
        if side_evidence:
            return side_evidence
        return _safe_dict(_safe_dict(raw.get("authoritative_return_candidate")).get("side_evidence"))

    def _return_budget_fraction(self, decision: DecisionOutput) -> float:
        evidence = self._side_evidence(decision)
        if evidence.get("production_eligible") is not True:
            return 0.0
        expected_net = _safe_float(evidence.get("expected_net_return_pct"))
        return_lcb = _safe_float(evidence.get("return_lcb_pct"))
        if expected_net <= 0 or return_lcb <= 0:
            return 0.0
        return min(expected_net, return_lcb) / 100.0

    def _return_sample_count(self, decision: DecisionOutput) -> int:
        return max(int(_safe_float(self._side_evidence(decision).get("production_source_count"))), 0)
