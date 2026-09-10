"""Declarative model-service topology and promotion safety contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

TopologyStage = Literal[
    "legacy_shadow",
    "candidate_not_configured",
    "candidate_validated",
    "paper",
    "live",
]


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
            for value in (self.model_id, self.repo_id, self.revision, self.path)
        )


@dataclass(frozen=True)
class ModelTopology:
    """Complete local/remote model topology, including deterministic services."""

    models: tuple[ModelServiceSpec, ...]
    quant_api_port: int = 8101
    cloud_reviewer_enabled: bool = True
    cloud_reviewer_required_for_high_risk_entry: bool = True
    local_model_count_target: int = 1
    live_routing_enabled: bool = False

    def by_role(self, role: str) -> ModelServiceSpec | None:
        return next((model for model in self.models if model.role == role), None)

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
