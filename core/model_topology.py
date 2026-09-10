"""Declarative model-service topology and promotion safety contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

TopologyStage = Literal[
    "legacy_shadow",
    "candidate_not_configured",
    "candidate_validated",
    "paper",
    "live",
]
TopologyProfile = Literal["legacy_shadow", "target_single_model"]

LEGACY_SHADOW_PROFILE: TopologyProfile = "legacy_shadow"
TARGET_SINGLE_MODEL_PROFILE: TopologyProfile = "target_single_model"


def normalize_topology_profile(profile: str | None) -> TopologyProfile:
    selected = LEGACY_SHADOW_PROFILE if profile is None else profile.strip().lower()
    if selected not in {LEGACY_SHADOW_PROFILE, TARGET_SINGLE_MODEL_PROFILE}:
        raise ValueError(f"unsupported model topology profile: {profile!r}")
    return selected


@dataclass(frozen=True)
class ModelTunnelRoute:
    name: str
    local_port: int
    remote_port: int
    health_path: str


def model_tunnel_routes(profile: str | None = None) -> tuple[ModelTunnelRoute, ...]:
    """Describe connectivity only; reachable ports never authorize trading."""

    selected = normalize_topology_profile(profile)
    quant = ModelTunnelRoute("phase3-quant-api", 18001, 8101, "/health/live")
    if selected == TARGET_SINGLE_MODEL_PROFILE:
        return (ModelTunnelRoute("target-single-model", 18000, 8000, "/v1/models"), quant)
    legacy = legacy_14b_topology()
    models = sorted(legacy.models, key=lambda model: model.port)
    routes = tuple(
        ModelTunnelRoute(model.model_id, model.port + 10000, model.port, "/v1/models")
        for model in models
    )
    return (routes[0], quant, *routes[1:])


@dataclass(frozen=True)
class ModelServiceSpec:
    """Identity and resource contract for one model endpoint."""

    model_id: str
    role: str
    repo_id: str | None
    revision: str | None
    path: str | None
    endpoint: str
    port: int
    tokenizer: str | None = None
    context_length: int = 4096
    max_concurrency: int = 1
    gpu_memory_budget_gib: float | None = None
    stage: TopologyStage = "candidate_not_configured"
    live_routing_enabled: bool = False

    @property
    def identity_complete(self) -> bool:
        return all(
            isinstance(value, str)
            and bool(value.strip())
            and value.strip().lower() not in {"unknown", "unverified", "pending"}
            and not any(char in value for char in ("\n", "\r", "\x00"))
            for value in (self.model_id, self.repo_id, self.revision, self.path)
        ) and "unverified" not in self.model_id.lower()

    def to_dict(self) -> dict[str, Any]:
        """Return a secret-free, stable representation for audits and APIs."""

        return {
            "model_id": self.model_id,
            "role": self.role,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "path": self.path,
            "endpoint": self.endpoint,
            "port": self.port,
            "tokenizer": self.tokenizer,
            "context_length": self.context_length,
            "max_concurrency": self.max_concurrency,
            "gpu_memory_budget_gib": self.gpu_memory_budget_gib,
            "stage": self.stage,
            "identity_complete": self.identity_complete,
            "live_routing_enabled": self.live_routing_enabled,
        }


@dataclass(frozen=True)
class ModelTopology:
    """Complete local/remote model topology, including deterministic services."""

    models: tuple[ModelServiceSpec, ...]
    quant_api_port: int = 8101
    cloud_reviewer_enabled: bool = True
    cloud_reviewer_required_for_high_risk_entry: bool = True
    local_model_count_target: int = 1
    live_routing_enabled: bool = False

    @property
    def profile(self) -> TopologyProfile:
        """Return the deployment profile represented by this topology."""

        if len(self.models) == 1 and self.models[0].role == "decision_and_expert_carrier":
            return TARGET_SINGLE_MODEL_PROFILE
        return LEGACY_SHADOW_PROFILE

    def by_role(self, role: str) -> ModelServiceSpec | None:
        return next((model for model in self.models if model.role == role), None)

    def to_dict(self) -> dict[str, Any]:
        """Return the topology contract without credentials or runtime secrets."""

        return {
            "profile": self.profile,
            "models": [model.to_dict() for model in self.models],
            "quant_api_port": self.quant_api_port,
            "cloud_reviewer_enabled": self.cloud_reviewer_enabled,
            "cloud_reviewer_required_for_high_risk_entry": (
                self.cloud_reviewer_required_for_high_risk_entry
            ),
            "local_model_count_target": self.local_model_count_target,
            "live_routing_enabled": self.live_routing_enabled,
            "validation_errors": list(self.validate()),
        }

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        ids = [model.model_id for model in self.models]
        ports = [model.port for model in self.models]
        if len(ids) != len(set(ids)):
            errors.append("duplicate_model_id")
        if len(ports) != len(set(ports)):
            errors.append("duplicate_model_port")
        if self.local_model_count_target < 1:
            errors.append("invalid_local_model_count_target")
        if any(model.live_routing_enabled for model in self.models) and not self.live_routing_enabled:
            errors.append("model_live_routing_without_topology_live_flag")
        for model in self.models:
            if model.stage in {"paper", "live"} and not model.identity_complete:
                errors.append(f"incomplete_identity:{model.model_id}")
            if model.stage == "live" and not self.live_routing_enabled:
                errors.append(f"live_stage_without_topology_live_flag:{model.model_id}")
            if model.max_concurrency < 1:
                errors.append(f"invalid_max_concurrency:{model.model_id}")
            if not 1 <= model.port <= 65535:
                errors.append(f"invalid_model_port:{model.model_id}")
            if model.context_length < 1:
                errors.append(f"invalid_context_length:{model.model_id}")
        return tuple(dict.fromkeys(errors))


def legacy_14b_topology() -> ModelTopology:
    """Describe the currently observed 14B shadow topology for audit only."""

    return ModelTopology(
        models=(
            ModelServiceSpec(
                model_id="qwen3-14b-trade",
                role="decision_maker",
                repo_id="Qwen/Qwen3-14B-AWQ",
                revision="unknown",
                path="/data/trade_models/Qwen/Qwen3-14B-AWQ",
                endpoint="http://127.0.0.1:18000/v1",
                port=8000,
                stage="legacy_shadow",
            ),
            ModelServiceSpec(
                model_id="BB-FinQuant-Expert-14B",
                role="expert_pool",
                repo_id="Qwen/Qwen3-14B-AWQ",
                revision="20260712T094555Z-4f40bc0974e6",
                path="/data/BB/models/finquant_lora/versions/20260712T094555Z-4f40bc0974e6",
                endpoint="http://127.0.0.1:18003/v1",
                port=8003,
                stage="legacy_shadow",
            ),
            ModelServiceSpec(
                model_id="deepseek-r1-14b-risk",
                role="high_risk_review",
                repo_id="casperhansen/deepseek-r1-distill-qwen-14b-awq",
                revision="unknown",
                path="/data/trade_models/DeepSeek/deepseek-r1-distill-qwen-14b-awq",
                endpoint="http://127.0.0.1:18002/v1",
                port=8002,
                stage="legacy_shadow",
            ),
        ),
        live_routing_enabled=False,
    )


def qwen27_candidate_topology(
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    path: str | None = None,
    stage: TopologyStage = "candidate_not_configured",
) -> ModelTopology:
    """Build the one-local-model target without guessing its identity."""

    model = ModelServiceSpec(
        model_id="qwen3.8-27b-unverified",
        role="decision_and_expert_carrier",
        repo_id=repo_id,
        revision=revision,
        path=path,
        endpoint="http://127.0.0.1:18000/v1",
        port=8000,
        tokenizer=path,
        context_length=4096,
        max_concurrency=1,
        gpu_memory_budget_gib=34.0,
        stage=stage,
        live_routing_enabled=False,
    )
    return ModelTopology(models=(model,))


def target_topology_ready(topology: ModelTopology) -> bool:
    """Check the requested route contract, not deployment or promotion evidence."""

    model = topology.by_role("decision_and_expert_carrier")
    return bool(
        topology.profile == TARGET_SINGLE_MODEL_PROFILE
        and topology.local_model_count_target == 1
        and model is not None
        and model.identity_complete
        and model.stage in {"candidate_validated", "paper"}
        and topology.cloud_reviewer_enabled
        and topology.cloud_reviewer_required_for_high_risk_entry
        and model.model_id.lower() not in {
            legacy.model_id.lower() for legacy in legacy_14b_topology().models
        }
        and not topology.live_routing_enabled
        and not model.live_routing_enabled
        and not topology.validate()
    )


def topology_for_profile(
    profile: str | None = None,
    *,
    model_id: str | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
    path: str | None = None,
    stage: TopologyStage = "candidate_not_configured",
) -> ModelTopology:
    """Build a deployment topology without silently guessing model identity.

    ``legacy_shadow`` is retained only for audit/rollback compatibility. The
    target profile requires all identity fields and remains non-live by
    construction.
    """

    selected = normalize_topology_profile(profile)
    if selected == LEGACY_SHADOW_PROFILE:
        return legacy_14b_topology()
    if selected != TARGET_SINGLE_MODEL_PROFILE:
        raise ValueError(f"unsupported model topology profile: {profile!r}")
    topology = qwen27_candidate_topology(stage=stage)
    values = (model_id, repo_id, revision, path)
    if all(isinstance(value, str) and value.strip() for value in values):
        if stage not in {"candidate_not_configured", "candidate_validated", "paper"}:
            raise ValueError("candidate configuration cannot enable live routing")
        model = replace(
            topology.models[0],
            model_id=str(model_id).strip(),
            repo_id=str(repo_id).strip(),
            revision=str(revision).strip(),
            path=str(path).strip(),
            tokenizer=str(path).strip(),
            stage=stage,
        )
        return replace(topology, models=(model,))
    return topology


def activate_verified_candidate(
    topology: ModelTopology,
    *,
    model_id: str,
    repo_id: str,
    revision: str,
    path: str,
    stage: TopologyStage = "candidate_validated",
) -> ModelTopology:
    """Return a copy with a verified candidate identity and no live routing."""

    if stage not in {"candidate_validated", "paper"}:
        raise ValueError("candidate activation cannot enable live routing")
    candidate = topology.by_role("decision_and_expert_carrier")
    if candidate is None:
        raise ValueError("decision_and_expert_carrier is missing")
    replacement = replace(
        candidate,
        model_id=model_id,
        repo_id=repo_id,
        revision=revision,
        path=path,
        tokenizer=path,
        stage=stage,
        live_routing_enabled=False,
    )
    return replace(topology, models=(replacement,), live_routing_enabled=False)
