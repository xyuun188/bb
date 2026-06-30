from __future__ import annotations

from scripts import sync_phase3_model_inventory as sync


def test_phase3_inventory_sync_updates_reports_from_canonical_manifests() -> None:
    script = sync.render_remote_inventory_sync()

    assert sync.REPORT_DOWNLOAD_MANIFEST in script
    assert sync.REPORT_VALIDATION_MANIFEST in script
    assert "'decision_maker': 'Qwen/Qwen3-32B-AWQ'" in script
    assert "'expert_pool': 'BB-FinQuant-Expert-14B'" in script
    assert "llm_decision_maker/Qwen--Qwen3-32B-AWQ" in script
    assert "llm_decision_maker/Qwen--Qwen3-14B-AWQ" not in script
    assert "'served_model_name': 'BB-FinQuant-Expert-14B'" in script
    assert "'specialization_required': True" in script
    assert "'specialization_status': 'pending'" in script
    assert "http://127.0.0.1:8101/health" in script
    assert "write_json(REPORT_DOWNLOAD_MANIFEST, download)" in script
    assert "write_json(REPORT_VALIDATION_MANIFEST, validation)" in script


def test_phase3_inventory_sync_dry_run_does_not_connect(monkeypatch, capsys) -> None:
    def fail_connect(*_args, **_kwargs):
        raise AssertionError("dry-run must not connect to the model server")

    monkeypatch.setattr(sync, "connect_remote_ssh", fail_connect)

    assert sync.main(["--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "phase3_model_download_manifest_latest.json" in output
    assert "Qwen/Qwen3-32B-AWQ" in output
