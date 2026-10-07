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
    parser.add_argument("--decision-id", action="append", type=int, required=True)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print the execution-critical fields instead of the full raw payload.",
    )
    args = parser.parse_args()
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
                stage_machine = raw.get("decision_state_machine") if isinstance(raw.get("decision_state_machine"), dict) else {{}}
                execution = raw.get("execution_result") if isinstance(raw.get("execution_result"), dict) else {{}}
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
                        key: selection.get(key)
                        for key in (
                            "selection_reason",
                            "selected_side",
                            "expected_net_return_pct",
                            "current_raw_expected_return_pct",
                            "objective_net_return_pct",
                            "paper_training_only",
                            "paper_training_reasons",
                        )
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
