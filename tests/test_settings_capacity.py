from config.settings import DEFAULT_MAX_OPEN_POSITIONS_PER_MODEL, Settings


def test_max_open_positions_zero_uses_safe_default() -> None:
    cfg = Settings(_env_file=None, max_open_positions_per_model=0)  # type: ignore[call-arg]

    assert cfg.max_open_positions_per_model == DEFAULT_MAX_OPEN_POSITIONS_PER_MODEL
