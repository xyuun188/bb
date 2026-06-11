from services.position_margin import PositionMarginCalculator


def test_position_margin_calculator_uses_minimum_one_leverage():
    calculator = PositionMarginCalculator()

    assert calculator.margin(100.0, 5.0) == 20.0
    assert calculator.margin(100.0, 0.0) == 100.0
    assert calculator.margin(100.0, -2.0) == 100.0


def test_position_margin_calculator_handles_bad_values_defensively():
    calculator = PositionMarginCalculator()

    assert calculator.margin("bad", 5.0) == 0.0
    assert calculator.margin(100.0, "bad") == 100.0
    assert calculator.margin(None, None) == 0.0
