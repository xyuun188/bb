from __future__ import annotations

from scripts import sync_phase3_model_inventory as sync


def _rendered_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {"__name__": "phase3_inventory_test"}
    exec(  # noqa: S102 - execute only the repository-owned generated script under test.
        compile(sync.render_remote_inventory_sync(), "<inventory-sync>", "exec"),
        namespace,
    )
    return namespace


def test_phase3_inventory_sync_updates_reports_from_canonical_manifests() -> None:
    script = sync.render_remote_inventory_sync()

    assert sync.REPORT_DOWNLOAD_MANIFEST in script
    assert sync.REPORT_VALIDATION_MANIFEST in script
    assert "'decision_maker': 'Qwen/Qwen3-14B-AWQ'" in script
    assert "'expert_pool': 'BB-FinQuant-Expert-14B'" in script
    assert "/data/trade_models/Qwen/Qwen3-14B-AWQ" in script
    assert "'served_model_name': 'BB-FinQuant-Expert-14B'" in script
    assert "'specialization_required': True" in script
    assert "'specialization_status': 'pending'" in script
    assert "http://127.0.0.1:8101/health" in script
    assert "HEALTH_RESPONSE_MAX_BYTES + 1" in script
    assert '"quant_api_artifact_model_id": health.get("artifact_model_id")' in script
    assert "write_json(REPORT_DOWNLOAD_MANIFEST, download)" in script
    assert "write_json(REPORT_VALIDATION_MANIFEST, validation)" in script


def test_phase3_inventory_sync_assigns_identity_to_new_llm_rows() -> None:
    namespace = _rendered_namespace()
    update_llm_rows = namespace["update_llm_rows"]
    expected_slots = set(namespace["EXPECTED_LLM_SLOTS"])

    updated = update_llm_rows({"models": []}, validation=True)

    observed_slots = {
        row.get("slot")
        for row in updated["models"]
        if isinstance(row, dict)
    }
    assert observed_slots == expected_slots


def test_phase3_inventory_sync_rejects_missing_llm_identity() -> None:
    namespace = _rendered_namespace()
    assert_expected_llm_slots = namespace["assert_expected_llm_slots"]

    try:
        assert_expected_llm_slots(
            {"models": [{"slot": "llm_expert_pool"}]},
            manifest_name="validation manifest",
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("missing LLM identities must fail inventory synchronization")

    assert "llm_decision_maker" in message
    assert "llm_high_risk_review" in message


def test_phase3_inventory_sync_dry_run_does_not_connect(monkeypatch, capsys) -> None:
    def fail_connect(*_args, **_kwargs):
        raise AssertionError("dry-run must not connect to the model server")

    monkeypatch.setattr(sync, "connect_remote_ssh", fail_connect)

    assert sync.main(["--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "phase3_model_download_manifest_latest.json" in output
    assert "Qwen/Qwen3-14B-AWQ" in output
