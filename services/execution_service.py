"""Execution service boundary.

ExecutionService owns serialized execution and the order-submit state machine.
TradingService remains the orchestrator and dependency provider, but the
submit/confirm/local-sync flow physically lives here.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from executor.base_executor import ExecutionResult
from services.decision_state import DecisionStage, DecisionStageStatus
from services.strategy_arbitration import arbitrate_decision
from services.trading_policies import PolicyGateResult


logger = structlog.get_logger(__name__)

AGENT_SKILLS_TRADING_EFFECTS_ENABLED = True


class ExecutionService:
    def __init__(self, orchestrator: Any) -> None:
        self.orchestrator = orchestrator

    async def execute_candidate(
        self,
        symbol: str,
        model_name: str,
        decision: Any,
        assessment: Any,
        decision_db_id: int | None,
        results: dict[str, Any],
        *,
        open_positions: list[dict[str, Any]] | None = None,
    ) -> ExecutionResult | None:
        async with self.orchestrator._execution_lock:
            return await self.execute_candidate_locked(
                symbol,
                model_name,
                decision,
                assessment,
                decision_db_id,
                results,
                open_positions=open_positions,
            )

    async def execute_candidate_locked(
        self,
        symbol: str,
        model_name: str,
        decision: DecisionOutput,
        assessment,
        decision_db_id: int | None,
        results: dict[str, Any],
        open_positions: list[dict] | None = None,
    ) -> ExecutionResult | None:
        ts = self.orchestrator
        for warning in assessment.warnings:
            results["warnings"].append({
                "model": model_name,
                "symbol": symbol,
                "warning": warning,
            })
            await ts._log_risk_event("warning", symbol, warning, model_name)

        execution_result = None
        model_mode = ts._get_model_execution_mode(model_name)

        async def mark_stage(
            stage: str,
            status: str,
            reason: str,
            data: dict[str, Any] | None = None,
        ) -> None:
            await ts._record_and_persist_decision_stage(
                decision_db_id,
                decision,
                stage,
                status,
                reason,
                data,
            )

        async def mark_blocked(reason: str, data: dict[str, Any] | None = None) -> None:
            await mark_stage(
                DecisionStage.RISK_CHECK,
                DecisionStageStatus.BLOCKED,
                reason,
                data,
            )

        async def block_before_submit(policy_result: PolicyGateResult) -> None:
            reason = str(policy_result.reason or "策略或风控检查未通过，未提交 OKX 订单。")
            blocker = str(policy_result.blocker or "policy_gate")
            data = {"blocker": blocker}
            if isinstance(policy_result.data, dict):
                data.update(policy_result.data)
            await mark_blocked(reason, data)
            if decision_db_id is not None:
                await ts._mark_decision_reason(decision_db_id, reason)
                await ts._mark_decision_raw_response(decision_db_id, decision.raw_response)
            await ts._log_risk_event(
                "warning",
                symbol,
                f"[{model_name}] {reason}",
                model_name,
            )
            if ts._position_review_alert_context(decision):
                await ts._log_position_review_risk_result(
                    decision,
                    model_name,
                    f"未执行：{reason}",
                )
            results["decisions"].append({
                "model": model_name,
                "symbol": symbol,
                "action": decision.action.value,
                "approved": True,
                "confidence": decision.confidence,
                "executed": False,
                "execution_status": "skipped",
                "reason": reason,
                "is_paper": (model_mode == "paper"),
            })

        arbitration = arbitrate_decision(decision)
        await mark_stage(
            DecisionStage.STRATEGY_ARBITRATION,
            arbitration.status,
            arbitration.reason,
            arbitration.data,
        )
        if decision.is_entry or decision.is_exit:
            await mark_stage(
                DecisionStage.RISK_CHECK,
                DecisionStageStatus.PENDING,
                "已进入执行前严重风险检查。",
                {"mode": model_mode},
            )
        if decision_db_id is not None:
            duplicate_reason = await ts._duplicate_decision_order_reason(decision_db_id, decision)
            if duplicate_reason:
                await block_before_submit(
                    PolicyGateResult.block(
                        "duplicate_decision_order",
                        duplicate_reason,
                    )
                )
                return None

        if decision.is_exit:
            exit_policy_result = await ts.exit_policy.evaluate(
                decision,
                model_name,
                open_positions,
            )
            if not exit_policy_result.passed:
                await block_before_submit(exit_policy_result)
                return None

        if decision.is_entry:
            entry_policy_result = await ts.entry_policy.evaluate(
                decision,
                model_name,
                model_mode,
                open_positions,
            )
            if not entry_policy_result.passed:
                await block_before_submit(entry_policy_result)
                return None
            if decision_db_id is not None:
                await ts._mark_decision_raw_response(decision_db_id, decision.raw_response)

        if decision.is_entry or decision.is_exit:
            await mark_stage(
                DecisionStage.RISK_CHECK,
                DecisionStageStatus.PASSED,
                "执行前严重风险检查通过，进入交易所提交阶段。",
                {"mode": model_mode},
            )
            await mark_stage(
                DecisionStage.EXCHANGE_SUBMIT,
                DecisionStageStatus.PENDING,
                "正在提交 OKX 订单并等待交易所返回结果。",
                {"mode": model_mode},
            )

        override_balance = None
        try:
            executor = await ts._get_okx_executor_for_mode(model_mode)
            ai_requested_leverage = float(decision.suggested_leverage or 1.0)
            override_balance = await ts._allocated_order_balance(model_mode, decision)
            execution_agent_skills = ts.agent_skills.execution_skills(
                decision=decision,
                model_mode=model_mode,
                override_balance=override_balance,
            )
            if execution_agent_skills:
                ts.agent_skills.attach(
                    decision,
                    phase="execution_precheck",
                    skills=execution_agent_skills,
                    note="提交 OKX 前的 Agent/Skills 执行守门。",
                )
                if decision_db_id is not None:
                    await ts._mark_decision_raw_response(decision_db_id, decision.raw_response)
            execution_guard_reason = (
                ts.agent_skills.block_reason(execution_agent_skills, for_entry=True)
                if AGENT_SKILLS_TRADING_EFFECTS_ENABLED
                else None
            )
            if decision.is_entry and execution_guard_reason:
                await mark_stage(
                    DecisionStage.RISK_CHECK,
                    DecisionStageStatus.BLOCKED,
                    execution_guard_reason,
                    {"blocker": "execution_agent_skills"},
                )
                await mark_stage(
                    DecisionStage.EXCHANGE_SUBMIT,
                    DecisionStageStatus.SKIPPED,
                    "执行前守门模块拦截，未向 OKX 提交订单。",
                    {"blocker": "execution_agent_skills"},
                )
                execution_result = ts._rejected_execution_result(
                    decision,
                    execution_guard_reason,
                )
            else:
                execution_timeout = 90.0 if decision.is_exit else 60.0
                execution_result = await asyncio.wait_for(
                    executor.place_order(
                        decision,
                        account_id=model_name,
                        override_balance=override_balance,
                    ),
                    timeout=execution_timeout,
                )
            if decision.is_entry and execution_result is not None:
                ts._attach_execution_leverage_summary(
                    decision,
                    execution_result,
                    ai_requested_leverage,
                )
        except asyncio.TimeoutError:
            logger.error(
                "decision execution timed out",
                model=model_name,
                symbol=symbol,
                action=decision.action.value,
                mode=model_mode,
            )
            execution_result = ts._rejected_execution_result(
                decision,
                (
                    "OKX 下单或确认超时，系统没有拿到最终订单结果；"
                    "本轮按未执行处理，下一轮会继续复盘该仓位。"
                ),
            )
            await mark_stage(
                DecisionStage.EXCHANGE_CONFIRM,
                DecisionStageStatus.FAILED,
                ts._execution_reason_from_result(execution_result),
                {"error_type": "timeout"},
            )
            await ts._log_risk_event(
                "warning",
                symbol,
                f"[{model_name}] OKX execution timed out",
                model_name,
            )
        except Exception as e:
            logger.error(
                "decision execution failed",
                model=model_name,
                symbol=symbol,
                action=decision.action.value,
                mode=model_mode,
                error=str(e),
            )
            execution_result = ts._rejected_execution_result(decision, e)
            await mark_stage(
                DecisionStage.EXCHANGE_SUBMIT,
                DecisionStageStatus.FAILED,
                ts._execution_reason_from_result(execution_result),
                {"error_type": "exception"},
            )
            await ts._log_risk_event(
                "warning",
                symbol,
                f"[{model_name}] OKX execution failed: {e}",
                model_name,
            )

        if execution_result is None and decision.is_exit:
            retry_intro = (
                "平仓裁决已生成，但第一次提交没有返回 OKX 订单结果；"
                "系统立即同步 OKX 仓位并重试一次平仓，避免错过平仓时机。"
            )
            await mark_stage(
                DecisionStage.EXCHANGE_CONFIRM,
                DecisionStageStatus.FAILED,
                retry_intro,
                {"retry": "exit_missing_execution_result"},
            )
            if decision_db_id is not None:
                await ts._mark_decision_pending_execution(decision_db_id, retry_intro)
            await ts._log_risk_event(
                "warning",
                symbol,
                f"[{model_name}] {retry_intro}",
                model_name,
            )
            await ts.okx_sync_service.reconcile_positions("exit missing execution result")
            exit_positions = await ts.okx_sync_service.get_open_positions_context()
            if open_positions is not None:
                open_positions[:] = exit_positions
            local_has_position = ts.exit_policy.has_matching_position(exit_positions, model_name, decision)
            exchange_has_position = await ts.okx_sync_service.has_matching_exchange_exit_position(
                model_name,
                decision,
            )
            if local_has_position or exchange_has_position:
                try:
                    retry_executor = await ts._get_okx_executor_for_mode(model_mode)
                    execution_result = await asyncio.wait_for(
                        retry_executor.place_order(
                            decision,
                            account_id=model_name,
                            override_balance=override_balance,
                        ),
                        timeout=45.0,
                    )
                    if execution_result is not None:
                        raw = (
                            execution_result.raw_response
                            if isinstance(execution_result.raw_response, dict)
                            else {}
                        )
                        raw["exit_missing_result_retry"] = True
                        raw["retry_reason"] = retry_intro
                        execution_result.raw_response = raw
                except asyncio.TimeoutError:
                    execution_result = ts._rejected_execution_result(
                        decision,
                        (
                            "平仓重试仍然超时：系统已同步 OKX 仓位并重新提交平仓，"
                            "但 45 秒内仍没有拿到订单结果。请以 OKX 当前仓位和委托状态为准；"
                            "下一轮持仓复盘会继续优先处理该仓位。"
                        ),
                    )
                except Exception as e:
                    execution_result = ts._rejected_execution_result(
                        decision,
                        (
                            "平仓重试失败：第一次提交没有返回订单结果，系统同步 OKX 仓位后已尝试重提，"
                            f"但交易接口返回错误：{e}"
                        ),
                    )
            else:
                execution_result = ts._rejected_execution_result(
                    decision,
                    (
                        "平仓裁决已生成，但第一次提交没有返回订单结果；系统随即同步 OKX 仓位，"
                        "发现本地和 OKX 都已经没有该方向可平仓位，因此没有重复提交平仓单。"
                    ),
                )

            if execution_result is None:
                execution_result = ts._rejected_execution_result(
                    decision,
                    (
                        "平仓重试后交易接口仍未返回执行结果。系统已避免把该状态继续标记为等待；"
                        "下一轮持仓复盘会再次检查 OKX 实际仓位并重新处理。"
                    ),
                )

        missing_result_reason = None
        if execution_result:
            result_text = " ".join(
                str(part or "")
                for part in (
                    execution_result.raw_response,
                    execution_result.exchange_order_id,
                    execution_result.status.value,
                )
            )
            if decision.is_entry and ts._is_untradable_exchange_error(result_text):
                ts._remember_untradable_symbol(symbol, result_text)
            elif decision.is_entry and ts._is_transient_entry_exchange_error(result_text):
                ts._remember_temporary_entry_block(
                    symbol,
                    result_text,
                    ts._transient_entry_block_minutes(result_text),
                )
            await ts._log_trade(execution_result, model_name, decision, decision_db_id)
            exchange_confirmed = ts._is_exchange_confirmed_execution(execution_result)
            exit_progress = ts._is_exit_progress_execution(execution_result)
            confirm_reason = ts._execution_reason_from_result(execution_result)
            if exchange_confirmed:
                await mark_stage(
                    DecisionStage.EXCHANGE_CONFIRM,
                    DecisionStageStatus.COMPLETED,
                    "OKX 已返回有效订单号并确认成交。",
                    {
                        "order_id": execution_result.order_id,
                        "exchange_order_id": execution_result.exchange_order_id,
                        "status": execution_result.status.value,
                        "price": execution_result.price,
                        "quantity": execution_result.quantity,
                    },
                )
            elif exit_progress:
                await mark_stage(
                    DecisionStage.EXCHANGE_CONFIRM,
                    DecisionStageStatus.PENDING,
                    confirm_reason,
                    {
                        "order_id": execution_result.order_id,
                        "exchange_order_id": execution_result.exchange_order_id,
                        "status": execution_result.status.value,
                    },
                )
            else:
                await mark_stage(
                    DecisionStage.EXCHANGE_CONFIRM,
                    DecisionStageStatus.FAILED,
                    confirm_reason,
                    {
                        "order_id": execution_result.order_id,
                        "exchange_order_id": execution_result.exchange_order_id,
                        "status": execution_result.status.value,
                    },
                )
            if (
                decision.is_exit
                and not exchange_confirmed
                and ts._result_has_no_exchange_position(execution_result)
            ):
                await ts.okx_sync_service.reconcile_positions("exit no-position result")
                if open_positions is not None:
                    open_positions[:] = await ts.okx_sync_service.get_open_positions_context()
            if exchange_confirmed or exit_progress:
                ts._trade_count += 1
                await ts._persist_position_from_execution(
                    model_name,
                    decision,
                    execution_result,
                    model_mode,
                )
                if open_positions is not None:
                    ts._apply_execution_to_open_positions(
                        open_positions,
                        model_name,
                        decision,
                        execution_result,
                    )
                if decision.is_exit:
                    ts._remember_recent_exit_group(model_name, decision)
                await mark_stage(
                    DecisionStage.LOCAL_SYNC,
                    DecisionStageStatus.COMPLETED,
                    "成交结果已写入本地订单/持仓记录。",
                    {"exit_progress": bool(exit_progress), "exchange_confirmed": bool(exchange_confirmed)},
                )
            else:
                await mark_stage(
                    DecisionStage.LOCAL_SYNC,
                    DecisionStageStatus.SKIPPED,
                    "交易所未确认成交，本地未改动持仓。",
                    {"exchange_confirmed": bool(exchange_confirmed), "exit_progress": bool(exit_progress)},
                )
            results["executions"].append({
                "model": model_name,
                "symbol": symbol,
                "action": decision.action.value,
                "order_id": execution_result.order_id,
                "status": execution_result.status.value,
                "quantity": execution_result.quantity,
                "price": execution_result.price,
                "is_paper": (model_mode == "paper"),
            })
            if exchange_confirmed:
                ts.risk_engine.circuit_breaker.record_trade(
                    execution_result.price * execution_result.quantity
                )
            if decision_db_id is not None and exchange_confirmed:
                await ts._mark_decision_executed(decision_db_id, execution_result.price)
                await ts._mark_decision_raw_response(decision_db_id, decision.raw_response)
                if decision.is_entry:
                    ts._clear_market_no_opportunity_symbol(symbol)
            elif decision_db_id is not None:
                await ts._mark_decision_reason(
                    decision_db_id,
                    ts._execution_reason_from_result(execution_result),
                )
                await ts._mark_decision_raw_response(decision_db_id, decision.raw_response)
            if model_mode != "paper" and decision.is_exit and execution_result.pnl != 0.0:
                await ts._persist_account_update(model_name, decision.model_name, execution_result)
                if decision_db_id is not None:
                    balance = await ts._get_account_balance(model_name)
                    pnl_pct = execution_result.pnl / balance if balance > 0 else 0.0
                    outcome = "profit" if execution_result.pnl > 0 else ("loss" if execution_result.pnl < 0 else "flat")
                    await ts._mark_decision_outcome(decision_db_id, outcome, pnl_pct)
            if ts._position_review_alert_context(decision):
                await ts._log_position_review_risk_result(
                    decision,
                    model_name,
                    execution_result=execution_result,
                )
        else:
            missing_result_reason = (
                "交易接口未返回执行结果，系统没有拿到 OKX 订单号，也没有生成本地订单；"
                "本次裁决已按未执行处理。"
            )
            await mark_stage(
                DecisionStage.EXCHANGE_CONFIRM,
                DecisionStageStatus.FAILED,
                missing_result_reason,
                {"error_type": "missing_execution_result"},
            )
            await mark_stage(
                DecisionStage.LOCAL_SYNC,
                DecisionStageStatus.SKIPPED,
                "没有成交结果，本地持仓未改动。",
                {"error_type": "missing_execution_result"},
            )
            if decision_db_id is not None:
                await ts._mark_decision_reason(decision_db_id, missing_result_reason)
                await ts._mark_decision_raw_response(decision_db_id, decision.raw_response)
            if ts._position_review_alert_context(decision):
                await ts._log_position_review_risk_result(
                    decision,
                    model_name,
                    f"未执行：{missing_result_reason}",
                )

        results["decisions"].append({
            "model": model_name,
            "symbol": symbol,
            "action": decision.action.value,
            "approved": True,
            "confidence": decision.confidence,
            "executed": ts._is_exchange_confirmed_execution(execution_result),
            "execution_status": execution_result.status.value if execution_result else None,
            "reason": missing_result_reason,
            "is_paper": (model_mode == "paper"),
        })
        return execution_result

