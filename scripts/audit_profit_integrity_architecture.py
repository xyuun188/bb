"""Static ownership and legacy-path audit for the production profit loop."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    "ai_brain",
    "core",
    "db",
    "executor",
    "models",
    "risk_manager",
    "scripts",
    "services",
    "web_dashboard/api",
)
DELETION_LEDGER_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "profit_integrity"
    / "2026-07-15-phase10-deletion-ledger.json"
)

OWNER_CONTRACTS = {
    "symbol_identity": {
        "owner": "core/symbols.py",
        "calls": {
            "normalize_trading_symbol",
            "okx_inst_id_from_payload",
            "okx_inst_id_from_symbol",
            "symbol_from_okx_inst_id",
            "symbol_from_okx_market",
            "symbol_from_okx_payload",
        },
    },
    "market_price_facts": {
        "owner": "core/market_facts.py",
        "calls": {"build_market_fact", "market_fact_contract_reasons", "market_fact_reasons"},
    },
    "execution_cost": {
        "owner": "services/execution_cost_model.py",
        "calls": {"attach_execution_cost_facts", "execution_cost_estimate"},
    },
    "return_distribution": {
        "owner": "services/return_objective.py",
        "calls": {
            "combine_production_return_distribution",
            "profit_factor",
            "standardized_return_distribution",
            "validate_return_distribution_contract",
        },
    },
    "production_entry_eligibility": {
        "owner": "services/return_execution_policy.py",
        "calls": {"apply_production_entry_policy", "assess_production_entry"},
    },
    "persisted_entry_contract_audit": {
        "owner": "services/trade_execution_contract.py",
        "calls": {"validate_production_entry_contract"},
    },
    "risk_budget_and_sizing": {
        "owner": "services/entry_profit_risk_sizing.py",
        "calls": {"EntryProfitRiskSizingPolicy", "reconcile_profit_risk_sizing"},
    },
    "exchange_execution_and_protection": {
        "owner": "executor/okx_executor.py",
        "calls": {"OKXExecutor"},
    },
    "training_quality": {
        "owner": "services/training_data_quality.py",
        "calls": {
            "annotate_sample",
            "annotate_training_payload",
            "governance_report",
            "quality_report",
            "trainable_samples",
        },
    },
    "authoritative_reflection": {
        "owner": "services/authoritative_trade_outcome.py",
        "calls": {"AuthoritativeTradeOutcomeService", "build_authoritative_trade_outcome"},
    },
}

REMOVED_MEMORY_POLICY_FIELDS = {
    "confidence_adjustment",
    "position_size_multiplier",
}
REMOVED_RUNTIME_TOKENS = {
    "active_profile",
    "active_strategy_profile_id",
    "legacy_policy",
    "recovery_observation_eligible",
    "runtime_recovery",
    "strategy_profile_id",
    "strategy_profile_version",
}
AUDIT_TOKEN_ALLOWLIST = {
    "scripts/audit_profit_integrity_architecture.py",
    "scripts/audit_strategy_scheduler_contract.py",
    "scripts/inspect_online_strategy_health.py",
}
PRODUCTION_DECISION_FILES = {
    "services/dynamic_leverage_allocator.py",
    "services/entry_opportunity_scoring.py",
    "services/entry_profit_risk_sizing.py",
    "services/model_promotion_policy.py",
    "services/return_execution_policy.py",
    "services/strategy_learning.py",
}
RISK_OWNER_FILES = {
    "risk_manager/engine.py",
    "risk_manager/position_limits.py",
    "services/dynamic_position_capacity.py",
    "services/entry_profit_risk_sizing.py",
}
DIAGNOSTIC_DECISION_NAMES = {
    "accuracy",
    "auc",
    "pr_auc",
    "win_rate",
    "winrate",
}
SELF_REFERENTIAL_RISK_INPUTS = {
    "final_notional",
    "final_notional_usdt",
    "position_size",
    "position_size_pct",
    "stop_loss_pct",
    "stressed_loss_fraction",
    "target_notional",
    "target_notional_usdt",
}


def _source_files() -> list[Path]:
    files: list[Path] = []
    for root_name in SOURCE_ROOTS:
        root = PROJECT_ROOT / root_name
        if root.exists():
            files.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    return sorted(set(files))


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _target_names(node: ast.AST) -> set[str]:
    return {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
    }


def _referenced_names(node: ast.AST) -> set[str]:
    names = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
    names.update(item.attr for item in ast.walk(node) if isinstance(item, ast.Attribute))
    names.update(
        str(item.value)
        for item in ast.walk(node)
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    )
    return names


def _contains_infinite_value(node: ast.AST) -> bool:
    for item in ast.walk(node):
        if isinstance(item, ast.Attribute) and item.attr.lower() in {"inf", "infinity"}:
            return True
        if (
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id == "float"
            and item.args
            and isinstance(item.args[0], ast.Constant)
            and str(item.args[0].value).lower() in {"inf", "+inf", "infinity", "+infinity"}
        ):
            return True
    return False


def _legacy_token_allowed(relative: str, token: str) -> bool:
    if relative in AUDIT_TOKEN_ALLOWLIST:
        return True
    return relative == "db/session.py" and token in REMOVED_MEMORY_POLICY_FIELDS


def load_deletion_ledger() -> dict[str, Any]:
    return json.loads(DELETION_LEDGER_PATH.read_text(encoding="utf-8"))


def audit() -> dict[str, Any]:
    consumers = {name: [] for name in OWNER_CONTRACTS}
    violations: list[dict[str, Any]] = []
    for path in _source_files():
        relative = _relative(path)
        source = path.read_text(encoding="utf-8-sig")
        try:
            tree = ast.parse(source, filename=relative)
        except SyntaxError as exc:
            violations.append(
                {"code": "python_syntax_error", "file": relative, "line": exc.lineno or 0}
            )
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _call_name(node)
                for semantic, contract in OWNER_CONTRACTS.items():
                    if name in contract["calls"]:
                        consumers[semantic].append({"file": relative, "line": node.lineno, "call": name})
            if isinstance(node, (ast.Name, ast.Attribute)):
                identifier = node.id if isinstance(node, ast.Name) else node.attr
                if identifier in REMOVED_MEMORY_POLICY_FIELDS:
                    violations.append(
                        {
                            "code": "removed_memory_policy_field_runtime_reference",
                            "file": relative,
                            "line": node.lineno,
                            "token": identifier,
                        }
                    )
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for token in REMOVED_MEMORY_POLICY_FIELDS | REMOVED_RUNTIME_TOKENS:
                    if token in node.value and not _legacy_token_allowed(relative, token):
                        violations.append(
                            {
                                "code": "removed_runtime_token_present",
                                "file": relative,
                                "line": node.lineno,
                                "token": token,
                            }
                        )
            if isinstance(node, ast.Compare) and relative in PRODUCTION_DECISION_FILES:
                segment = ast.get_source_segment(source, node) or ""
                lowered = segment.lower()
                matched = sorted(name for name in DIAGNOSTIC_DECISION_NAMES if name in lowered)
                if matched:
                    violations.append(
                        {
                            "code": "diagnostic_metric_used_in_production_comparison",
                            "file": relative,
                            "line": node.lineno,
                            "metrics": matched,
                        }
                    )
            if relative in RISK_OWNER_FILES and _contains_infinite_value(node):
                violations.append(
                    {"code": "infinite_risk_limit", "file": relative, "line": node.lineno}
                )
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and relative in RISK_OWNER_FILES:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                target_names = set().union(*(_target_names(target) for target in targets))
                if any("risk_budget" in name or name == "max_loss" for name in target_names):
                    referenced = _referenced_names(node.value)
                    overlap = sorted(SELF_REFERENTIAL_RISK_INPUTS.intersection(referenced))
                    if overlap:
                        violations.append(
                            {
                                "code": "self_referential_risk_budget",
                                "file": relative,
                                "line": node.lineno,
                                "inputs": overlap,
                            }
                        )

    for rows in consumers.values():
        rows.sort(key=lambda item: (item["file"], item["line"], item["call"]))
    ledger = load_deletion_ledger()
    required_ledger_fields = {
        "legacy_path",
        "production_consumers",
        "test_consumers",
        "data_migration",
        "replacement_owner",
        "deletion_commit",
    }
    for index, row in enumerate(ledger.get("deletions") or []):
        missing = sorted(required_ledger_fields.difference(row))
        if missing:
            violations.append(
                {
                    "code": "deletion_ledger_incomplete",
                    "file": _relative(DELETION_LEDGER_PATH),
                    "line": index + 1,
                    "missing": missing,
                }
            )
    return {
        "status": "ok" if not violations else "blocked",
        "optimization_target": "maximize_authoritative_fee_after_return_rate",
        "source_file_count": len(_source_files()),
        "owners": {
            semantic: {
                "owner": contract["owner"],
                "consumer_count": len(consumers[semantic]),
                "consumers": consumers[semantic],
            }
            for semantic, contract in OWNER_CONTRACTS.items()
        },
        "deletion_ledger_version": ledger.get("version"),
        "violations": violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit()
    print(json.dumps(report, ensure_ascii=False, indent=None if args.json else 2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
