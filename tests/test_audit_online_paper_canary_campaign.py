from __future__ import annotations

from scripts import audit_online_paper_canary_campaign as audit


def test_remote_script_compiles_against_current_normal_paper_contract() -> None:
    script = audit._remote_script(remote_app_dir="/data/bb/app", hours=24, limit=1000)

    compile(script, "<online-normal-paper-audit>", "exec")
    assert "NORMAL_PAPER_TRADE_VERSION" in script
    assert "normal_paper_trade_contract_reasons" in script
    assert "load_authoritative_trade_training_samples" in script
    assert 'raw.get("normal_paper_trade")' in script
    assert "PAPER_BOOTSTRAP_" not in script
    assert '"production_permission": False' in script
    assert ".limit(LIMIT)" not in script
    assert '"latest_candidates": rows[:LIMIT]' in script
