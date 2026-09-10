from core.model_topology import (
    activate_verified_candidate,
    legacy_14b_topology,
    qwen27_candidate_topology,
)


def test_legacy_topology_is_audit_only_and_matches_observed_ports():
    topology = legacy_14b_topology()
    assert topology.live_routing_enabled is False
    assert {model.port for model in topology.models} == {8000, 8002, 8003}
    assert all(model.stage == "legacy_shadow" for model in topology.models)


def test_unverified_27b_candidate_cannot_enter_paper_with_missing_identity():
    topology = qwen27_candidate_topology(stage="candidate_not_configured")
    model = topology.by_role("decision_and_expert_carrier")
    assert model is not None
    assert model.identity_complete is False
    assert topology.validate() == ()


def test_verified_candidate_is_single_local_model_and_never_live_by_accident():
    topology = activate_verified_candidate(
        qwen27_candidate_topology(),
        model_id="qwen3.8-27b-awq",
        repo_id="verified/repo",
        revision="sha256:revision",
        path="/data/trade_models/verified/qwen3.8-27b-awq",
        stage="paper",
    )
    assert len(topology.models) == 1
    assert topology.models[0].identity_complete is True
    assert topology.models[0].stage == "paper"
    assert topology.live_routing_enabled is False
    assert topology.validate() == ()


def test_topology_serialization_exposes_identity_and_validation_without_secrets():
    topology = qwen27_candidate_topology()
    payload = topology.to_dict()
    assert payload["local_model_count_target"] == 1
    assert payload["models"][0]["model_id"] == "qwen3.8-27b-unverified"
    assert payload["models"][0]["identity_complete"] is False
    assert payload["validation_errors"] == []
    assert "api_key" not in str(payload).lower()
