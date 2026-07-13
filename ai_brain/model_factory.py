"""
Model factory — creates AI model instances from settings.ai_models config.

Supports the multi-model configuration pattern: each configured fixed slot
becomes an independently configured LLMAgent.
"""

from __future__ import annotations

from ai_brain.base_model import AbstractAIModel
from ai_brain.llm_agent import LLMAgent
from config.settings import settings


def create_models_from_config() -> list[AbstractAIModel]:
    """Build model instances from settings.ai_models configuration.

    Each config entry creates an LLMAgent with its own name, api_base,
    api_key, and model. Missing fixed-slot configuration fails closed.
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

        return models

    return []
