from __future__ import annotations

import json
from pathlib import Path

import pytest

import core.finquant_adapter_registry_remote as registry


def _configure_registry(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(registry, "ROOT", root)
    monkeypatch.setattr(registry, "CURRENT", root / "current.json")
    monkeypatch.setattr(registry, "ROLLBACK", root / "rollback.json")
    monkeypatch.setattr(registry, "RETIRED", root / "retired")


def test_activate_shadow_keeps_adapter_non_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_registry(monkeypatch, tmp_path)
    adapter = tmp_path / "versions" / "v1"
    adapter.mkdir(parents=True)
    manifest = adapter / "specialization_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    pointer = {
        "registry_version": registry.REGISTRY_VERSION,
        "adapter_version": "v1",
        "adapter_path": str(adapter),
        "manifest_path": str(manifest),
        "manifest_sha256": registry.sha256_file(manifest),
        "adapter_sha256": None,
        "routing_state": "shadow_only",
        "can_influence_live": False,
    }
    shadow = tmp_path / "shadow" / "v1.json"
    shadow.parent.mkdir()
    shadow.write_text(json.dumps(pointer), encoding="utf-8")
    monkeypatch.setattr(registry, "validate_pointer", lambda value: ({}, adapter))

    result = registry.activate_shadow(shadow)

    current = json.loads((tmp_path / "current.json").read_text(encoding="utf-8"))
    assert current["adapter_path"] == str(adapter)
    assert current["routing_state"] == "shadow_only"
    assert current["can_influence_live"] is False
    assert result["live_routing_enabled"] is False
    assert not (tmp_path / "rollback.json").exists()


@pytest.mark.parametrize(
    ("routing_state", "can_influence_live"),
    [("live", False), ("shadow_only", True)],
)
def test_activate_shadow_rejects_live_capable_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    routing_state: str,
    can_influence_live: bool,
) -> None:
    _configure_registry(monkeypatch, tmp_path)
    shadow = tmp_path / "shadow" / "v1.json"
    shadow.parent.mkdir()
    shadow.write_text(
        json.dumps(
            {
                "adapter_path": str(tmp_path / "versions" / "v1"),
                "routing_state": routing_state,
                "can_influence_live": can_influence_live,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(registry, "validate_pointer", lambda value: ({}, Path(value["adapter_path"])))

    with pytest.raises(ValueError):
        registry.activate_shadow(shadow)


def test_activate_shadow_rejects_pointer_outside_shadow_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_registry(monkeypatch, tmp_path / "registry")
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="inside the registry shadow directory"):
        registry.activate_shadow(outside)


def test_verified_registry_evidence_survives_inventory_refresh(tmp_path, monkeypatch) -> None:
    from scripts.sync_phase3_model_inventory import render_target_inventory_sync

    pointer = {
        "adapter_version": "verified-adapter",
        "adapter_path": "/data/BB/models/finquant_target_27b/versions/verified-adapter",
        "manifest_path": "/data/BB/models/finquant_target_27b/versions/verified-adapter/manifest.json",
        "specialization_id": "verified-specialization",
    }
    monkeypatch.setattr(registry, "validate_current", lambda **_kwargs: pointer)
    monkeypatch.setattr(
        registry,
        "validate_pointer",
        lambda _pointer: ({"specialization_status": "trained_shadow_not_live"}, tmp_path),
    )
    monkeypatch.setattr(registry, "DOWNLOAD_MANIFEST", tmp_path / "download.json")
    monkeypatch.setattr(registry, "VALIDATION_MANIFEST", tmp_path / "validation.json")

    result = registry.sync_evidence()
    namespace = {"__name__": "inventory_contract_test"}
    exec(  # noqa: S102 - exercise the repository-owned generated consumer.
        compile(render_target_inventory_sync(), "<inventory-sync>", "exec"),
        namespace,
    )
    refreshed = namespace["update"](
        json.loads((tmp_path / "validation.json").read_text()),
        {
            "slot": "llm_decision_and_expert_carrier",
            "served_model_name": registry.MODEL_NAME,
        },
    )
    row = refreshed["models"][0]
    assert row["specialization_evidence_verified"] is True
    assert row["specialization_evidence"] == result["evidence"]["specialization_evidence"]
    assert row["adapter_path"] == pointer["adapter_path"]
