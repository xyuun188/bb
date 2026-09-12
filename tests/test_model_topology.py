from core.model_topology import (
    TARGET_SINGLE_MODEL_PROFILE,
    activate_verified_candidate,
    normalize_topology_profile,
    qwen27_candidate_topology,
    target_topology_ready,
)


def test_only_single_model_profile_is_supported():
    assert normalize_topology_profile(None) == TARGET_SINGLE_MODEL_PROFILE
    assert normalize_topology_profile(TARGET_SINGLE_MODEL_PROFILE) == TARGET_SINGLE_MODEL_PROFILE


def test_unverified_27b_candidate_cannot_enter_paper_with_missing_identity():
    topology = qwen27_candidate_topology(stage="candidate_not_configured")
    model = topology.by_role("decision_and_expert_carrier")
    assert model is not None
    assert model.identity_complete is False
    assert topology.validate() == ()
    assert target_topology_ready(topology) is False


def test_verified_candidate_is_single_local_model_and_never_live_by_accident():
    topology = activate_verified_candidate(
        qwen27_candidate_topology(),
        model_id="qwen3.8-27b",
        repo_id="Qwen/Qwen3.8-27B",
        revision="a" * 40,
        path="/data/BB/models/qwen3.8-27b",
        stage="paper",
    )
    assert len(topology.models) == 1
    assert topology.models[0].identity_complete is True
    assert topology.models[0].stage == "paper"
    assert topology.live_routing_enabled is False
    assert topology.validate() == ()
    assert target_topology_ready(topology) is True


def test_topology_serialization_exposes_identity_and_validation_without_secrets():
    topology = qwen27_candidate_topology()
    payload = topology.to_dict()
    assert payload["profile"] == TARGET_SINGLE_MODEL_PROFILE
    assert payload["local_model_count_target"] == 1
    assert payload["models"][0]["model_id"] == "qwen3.8-27b-unverified"
    assert payload["models"][0]["identity_complete"] is False
    assert payload["validation_errors"] == []
    assert "api_key" not in str(payload).lower()
