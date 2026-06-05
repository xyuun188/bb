"""
Model factory — creates AI model instances from settings.ai_models config.

Supports the multi-model configuration pattern: each entry in
settings.ai_models becomes an independently-configured LLMAgent.
Falls back to the legacy 3-model set when ai_models is empty.
"""

from __future__ import annotations

from ai_brain.base_model import AbstractAIModel
from ai_brain.llm_agent import LLMAgent
from config.settings import settings


def create_models_from_config() -> list[AbstractAIModel]:
    """Build model instances from settings.ai_models configuration.

    Each config entry creates an LLMAgent with its own name, api_base,
    api_key, and model. When settings.ai_models is empty, falls back to
    the legacy default set (LLM Agent + FinBERT + XGBoost).
    """
    fixed_models = settings.get_fixed_ai_models(include_empty=False)
    if fixed_models:
        models: list[AbstractAIModel] = []
        for cfg in fixed_models:
            if cfg.get("enabled") is False:
                continue
            name = cfg.get("name", "llm_agent")
            api_config = {
                "api_base": cfg.get("api_base", ""),
                "api_key": cfg.get("api_key", ""),
                "model": cfg.get("model", "gpt-4"),
                "role": cfg.get("role", ""),
                "label": cfg.get("label", name),
                "weight": cfg.get("weight", 1.0),
            }
            agent = LLMAgent(name=name, api_config=api_config)
            models.append(agent)

            # Register balance into model_initial_balances if specified
            if "balance" in cfg and name not in settings.model_initial_balances:
                settings.model_initial_balances[name] = float(cfg["balance"])

        return models

    # Legacy fallback: only create model when legacy api_key is configured
    if settings.ai_api_key:
        return [LLMAgent()]
    return []
