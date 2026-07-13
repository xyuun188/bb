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


def test_model_factory_builds_keyless_loopback_fixed_slot(monkeypatch) -> None:
    monkeypatch.setattr(
        type(settings),
        "get_fixed_ai_models",
        lambda _self, **_kwargs: [
            {
                "name": "trend_expert",
                "role": "trend_direction",
                "label": "Trend",
                "weight": 1.0,
                "api_base": "http://127.0.0.1:18003/v1",
                "api_key": "",
                "model": "BB-FinQuant-Expert-14B",
                "enabled": True,
            }
        ],
    )

    models = create_models_from_config()

    assert [model.name for model in models] == ["trend_expert"]
