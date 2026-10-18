#!/usr/bin/env python3
"""Read-only check of full and compact AI-decision projections online."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

REMOTE_APP_DIR = "/data/bb/app"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-id", action="append", type=int, default=[])
    parser.add_argument("--backfill-downside-return", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--before-id", type=int)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print the execution-critical fields instead of the full raw payload.",
    )
    args = parser.parse_args()
    if not args.decision_id and not args.backfill_downside_return:
        parser.error("provide --decision-id or --backfill-downside-return")
    if args.apply and not args.backfill_downside_return:
        parser.error("--apply requires --backfill-downside-return")
    script = f"""
import asyncio, json
from pathlib import Path
from scripts.runtime_env_bootstrap import load_runtime_env_files, drop_privileges_to_runtime_user_if_needed
root = Path({REMOTE_APP_DIR!r})
load_runtime_env_files(project_root=root)
drop_privileges_to_runtime_user_if_needed(project_root=root)
from db.session import get_read_session_ctx
from models.decision import AIDecision
async def main():
    if {bool(args.backfill_downside_return)!r}:
        from services.decision_learning_projection import backfill_downside_return_projection
        result = await backfill_downside_return_projection(
            apply={bool(args.apply)!r},
            max_rows={max(1, min(args.max_rows, 100_000))!r},
            before_id={args.before_id!r},
        )
        print(json.dumps(result, ensure_ascii=False, default=str))
        return
    result = []
    async with get_read_session_ctx(statement_timeout_ms=30000) as session:
        for decision_id in {sorted(set(args.decision_id))!r}:
            row = await session.get(AIDecision, decision_id)
            if row is None:
                result.append({{"decision_id": decision_id, "found": False}})
                continue
            raw = row.raw_llm_response if isinstance(row.raw_llm_response, dict) else {{}}
            learning = row.decision_learning_snapshot if isinstance(row.decision_learning_snapshot, dict) else {{}}
            if {bool(args.compact)!r}:
                normal = raw.get("normal_paper_trade") if isinstance(raw.get("normal_paper_trade"), dict) else {{}}
                selection = raw.get("paper_trade_selection") if isinstance(raw.get("paper_trade_selection"), dict) else {{}}
                sizing = raw.get("profit_risk_sizing") if isinstance(raw.get("profit_risk_sizing"), dict) else {{}}
                facts = raw.get("pre_order_execution_facts") if isinstance(raw.get("pre_order_execution_facts"), dict) else {{}}
                exchange_facts = raw.get("exchange_risk_facts") if isinstance(raw.get("exchange_risk_facts"), dict) else {{}}
                opportunity = raw.get("opportunity_score") if isinstance(raw.get("opportunity_score"), dict) else {{}}
                stage_machine = raw.get("decision_state_machine") if isinstance(raw.get("decision_state_machine"), dict) else {{}}
                execution = raw.get("execution_result") if isinstance(raw.get("execution_result"), dict) else {{}}
                funnel_raw = raw.get("market_candidate_funnel") if isinstance(raw.get("market_candidate_funnel"), dict) else {{}}
                quality = raw.get("analysis_quality_contract") if isinstance(raw.get("analysis_quality_contract"), dict) else {{}}
                model_timings = raw.get("model_timings") if isinstance(raw.get("model_timings"), list) else []
                expert_failures = raw.get("expert_failures") if isinstance(raw.get("expert_failures"), list) else []
                market_progress = raw.get("market_analysis_progress") if isinstance(raw.get("market_analysis_progress"), dict) else {{}}
                entry_permission = raw.get("entry_permission") if isinstance(raw.get("entry_permission"), dict) else {{}}
                direction_support = raw.get("independent_direction_support") if isinstance(raw.get("independent_direction_support"), dict) else {{}}
                direction_competition_raw = raw.get("direction_competition") if isinstance(raw.get("direction_competition"), dict) else {{}}
                direction_competition = {{
                    side: {{
                        **{{
                            key: side_data.get(key)
                            for key in (
                                "score",
                                "raw_expected_return_pct",
                                "objective_expected_return_pct",
                                "expected_net_return_pct",
                                "decision_source_count",
                                "selected_horizon_minutes",
                                "loss_probability",
                            )
                        }},
                        "evidence": [
                            {{
                                key: item.get(key)
                                for key in (
                                    "source",
                                    "model_name",
                                    "provider_model",
                                    "decision_eligible",
                                    "training_aggregate_eligible",
                                    "raw_expected_return_pct",
                                    "objective_expected_return_pct",
                                    "expected_net_return_pct",
                                    "horizon_minutes",
                                    "loss_probability",
                                    "return_distribution_contract",
                                )
                                if key in item
                            }}
                            for item in (side_data.get("evidence") or [])
                            if isinstance(item, dict)
                        ],
                    }}
                    for side, side_data in direction_competition_raw.items()
                    if isinstance(side_data, dict) and side in {{"long", "short"}}
                }}
                funnel = {{
                    key: funnel_raw.get(key)
                    for key in (
                        "round_id",
                        "scan_symbol_count",
                        "feature_valid_count",
                        "feature_invalid_count",
                        "rank_selected_count",
                        "rank_tradable_candidates",
                        "rank_secondary_candidates",
                        "rank_filtered_out_reason_counts",
                        "analysis_only_selected_symbols",
                    )
                    if key in funnel_raw
                }}
                result.append({{
                    "decision_id": decision_id,
                    "found": True,
                    "symbol": row.symbol,
                    "action": row.action,
                    "was_executed": bool(row.was_executed),
                    "execution_reason": row.execution_reason,
                    "normal_paper_trade": {{
                        key: normal.get(key)
                        for key in (
                            "authorized",
                            "selection_reason",
                            "paper_training_only",
                            "paper_quality_mode",
                            "expected_net_return_pct",
                            "current_raw_expected_return_pct",
                            "objective_net_return_pct",
                            "current_edge_validated",
                        )
                    }},
                    "paper_trade_selection": {{
                        **{{
                            key: selection.get(key)
                            for key in (
                                "selection_reason",
                                "selected_side",
                                "expected_net_return_pct",
                                "current_raw_expected_return_pct",
                                "objective_net_return_pct",
                                "paper_training_only",
                                "paper_training_reasons",
                                "diagnostic_status",
                                "blocking_category",
                                "blocking_reasons",
                                "blocking_reasons_by_side",
                            )
                        }},
                        "by_side": {{
                            side: {{
                                key: side_support.get(key)
                                for key in (
                                    "eligible",
                                    "reason",
                                    "expected_net_return_pct",
                                    "raw_expected_return_pct",
                                    "objective_net_return_pct",
                                    "loss_probability",
                                    "prediction_horizon_minutes",
                                    "quant_evidence_families",
                                    "quant_family_summaries",
                                    "aligned_expert_count",
                                    "opposition_expert_count",
                                    "hold_expert_count",
                                    "blocking_reasons",
                                    "paper_quality_observation_only",
                                    "paper_quality_observation_reasons",
                                )
                            }}
                            for side, side_support in (selection.get("by_side") or {{}}).items()
                            if isinstance(side_support, dict)
                        }},
                    }},
                    "opportunity_score": {{
                        key: opportunity.get(key)
                        for key in (
                            "expected_net_return_pct",
                            "return_lcb_pct",
                            "execution_cost",
                            "return_distribution_contract",
                            "expected_net_breakdown",
                        )
                    }},
                    "analysis_diagnostics": {{
                        "analysis_quality_contract": quality,
                        "expert_failures": expert_failures,
                        "model_timings": model_timings,
                        "market_candidate_funnel": funnel,
                        "market_analysis_progress": market_progress,
                        "entry_permission": entry_permission,
                        "independent_direction_support": direction_support,
                        "direction_competition": direction_competition,
                    }},
                    "sizing": {{
                        key: sizing.get(key)
                        for key in (
                            "production_eligible",
                            "reason",
                            "final_notional_usdt",
                            "final_margin_usdt",
                            "final_leverage",
                            "available_margin_usdt",
                            "account_equity_usdt",
                            "policy_provenance",
                            "dynamic_leverage_decision",
                            "size_aware_profitability",
                            "execution_cost",
                            "minimum_order_evidence",
                            "risk_budget_usdt",
                            "fill_notional_ceiling_usdt",
                            "target_notional_usdt",
                            "contract_step_notional_usdt",
                            "expected_net_return_pct",
                            "return_lcb_pct",
                        )
                    }},
                    "pre_order_execution_facts": {{
                        key: facts.get(key)
                        for key in (
                            "production_eligible",
                            "reason",
                            "account_equity_usdt",
                            "available_margin_usdt",
                            "balance_snapshot",
                            "market_source_consistency",
                            "policy_provenance",
                        )
                    }},
                    "exchange_risk_facts": {{
                        key: exchange_facts.get(key)
                        for key in (
                            "production_eligible",
                            "account_equity_usdt",
                            "available_margin_usdt",
                            "reported_max_leverage",
                            "missing_contract_specs",
                            "entry_instrument_availability",
                            "balance_snapshot",
                            "policy_provenance",
                        )
                    }},
                    "execution_result": execution,
                    "decision_state_machine": stage_machine,
                }})
                continue
            result.append({{
                "decision_id": decision_id,
                "found": True,
                "raw_keys": sorted(str(key) for key in raw),
                "raw_retention_marker": raw.get("_retention"),
                "raw_decision_authority": (raw.get("normal_paper_trade") or {{}}).get("decision_authority"),
                "raw_trade_version": (raw.get("normal_paper_trade") or {{}}).get("version"),
                "learning_keys": sorted(str(key) for key in learning),
                "learning_decision_authority": (learning.get("normal_paper_trade") or {{}}).get("decision_authority"),
                "learning_trade_version": (learning.get("normal_paper_trade") or {{}}).get("version"),
                "learning_has_execution_result": isinstance(learning.get("execution_result"), dict),
            }})
    print(json.dumps(result, ensure_ascii=False, default=str))
asyncio.run(main())
"""
    command = (
        f"cd {shlex.quote(REMOTE_APP_DIR)} && runuser -u bb -- /bin/bash -lc "
        + shlex.quote(".venv/bin/python - <<'PY'\n" + script + "\nPY")
    )
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        safe_print(run_remote_text(ssh, command, timeout=180, max_output_chars=30_000))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
