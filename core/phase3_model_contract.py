"""Canonical Phase 3 model-service identities and endpoints."""

from __future__ import annotations

PHASE3_DECISION_MODEL_ID = "qwen3-14b-trade"
PHASE3_RISK_MODEL_ID = "deepseek-r1-14b-risk"
PHASE3_EXPERT_MODEL_ID = "BB-FinQuant-Expert-14B"
PHASE3_QUANT_API_ID = "phase3_quant_api"
PHASE3_DECISION_REPO_ID = "Qwen/Qwen3-14B-AWQ"
PHASE3_RISK_REPO_ID = "casperhansen/deepseek-r1-distill-qwen-14b-awq"

PHASE3_PLATFORM_ENDPOINTS = {
    PHASE3_DECISION_MODEL_ID: "http://127.0.0.1:18000/v1",
    PHASE3_QUANT_API_ID: "http://127.0.0.1:18001",
    PHASE3_RISK_MODEL_ID: "http://127.0.0.1:18002/v1",
    PHASE3_EXPERT_MODEL_ID: "http://127.0.0.1:18003/v1",
}

PHASE3_MODEL_SERVER_SERVICES = (
    ("bb-phase3-llm-decision.service", PHASE3_DECISION_MODEL_ID, 8000),
    ("bb-phase3-llm-risk-review.service", PHASE3_RISK_MODEL_ID, 8002),
    ("bb-phase3-llm-expert.service", PHASE3_EXPERT_MODEL_ID, 8003),
)

PHASE3_REQUIRED_LLM_MODEL_IDS = frozenset(
    {
        PHASE3_DECISION_MODEL_ID.lower(),
        PHASE3_RISK_MODEL_ID.lower(),
        PHASE3_EXPERT_MODEL_ID.lower(),
    }
)

PHASE3_APPROVED_RUNTIME_MODEL_PATHS = (
    "/data/trade_models/Qwen/Qwen3-14B-AWQ",
    "/data/trade_models/DeepSeek/deepseek-r1-distill-qwen-14b-awq",
)
