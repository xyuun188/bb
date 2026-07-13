from config.settings import Settings


def test_removed_fixed_risk_and_capacity_settings_cannot_be_loaded() -> None:
    cfg = Settings(
        _env_file=None,
        max_open_positions_per_model=20,
        max_same_symbol_positions_per_side=2,
        max_position_pct=0.25,
        max_daily_loss_pct=0.05,
        hard_stop_loss_pct=0.05,
        max_leverage=20,
    )  # type: ignore[call-arg]

    for name in (
        "max_open_positions_per_model",
        "max_same_symbol_positions_per_side",
        "max_position_pct",
        "max_daily_loss_pct",
        "hard_stop_loss_pct",
        "max_leverage",
    ):
        assert not hasattr(cfg, name)
