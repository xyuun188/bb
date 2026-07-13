from ai_brain.model_factory import create_models_from_config
from config.settings import settings


def test_model_factory_does_not_restore_legacy_single_model_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        type(settings),
        "get_fixed_ai_models",
        lambda _self, **_kwargs: [],
    )
    assert create_models_from_config() == []
    assert not hasattr(settings, "ai_api_key")
