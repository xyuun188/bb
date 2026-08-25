from core.contract_math import persisted_product_isclose


def test_persisted_product_accepts_eight_decimal_rounding_drift() -> None:
    assert persisted_product_isclose(
        1.28138362,
        45.96792255,
        0.0278756,
    )


def test_persisted_product_rejects_material_contract_mutation() -> None:
    assert not persisted_product_isclose(
        1.0,
        90.0,
        0.01,
    )


def test_persisted_product_rejects_non_numeric_values() -> None:
    assert not persisted_product_isclose("not-a-number", 1.0, 1.0)
