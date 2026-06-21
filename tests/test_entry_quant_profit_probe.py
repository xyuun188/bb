from types import SimpleNamespace

from ai_brain.base_model import Action, DecisionOutput
from services.entry_probe_market_quality import EntryProbeMarketQualityPolicy
from services.entry_quant_profit_probe import EntryQuantProfitProbePolicy


def _u(escaped: str) -> str:
    return escaped.encode("ascii").decode("unicode_escape")


def _hold_decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.HOLD,
        confidence=0.55,
        reasoning="hold",
        raw_response={},
        feature_snapshot={"close": 100.0},
    )


def _tools(**profit_overrides):
    profit = {
        "available": True,
        "best_side": "long",
        "adjusted_long_return_pct": 0.80,
        "adjusted_short_return_pct": 0.10,
        "long_loss_probability": 0.35,
    }
    profit.update(profit_overrides)
    return {"profit_prediction": profit}


def _fv(**values):
    defaults = {
        "current_price": 100.0,
        "close": 100.0,
        "returns_20": 0.0,
        "volume_ratio": 1.0,
        "price_vs_sma20": 0.0,
        "price_vs_sma50": 0.0,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def _policy(score_payload: dict) -> EntryQuantProfitProbePolicy:
    def score_candidate(decision: DecisionOutput, strategy: dict) -> float:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw["opportunity_score"] = score_payload
        decision.raw_response = raw
        return float(score_payload.get("score", 1.0))

    return EntryQuantProfitProbePolicy(EntryProbeMarketQualityPolicy(), score_candidate)


def test_quant_profit_probe_creates_candidate_after_post_score_passes() -> None:
    candidate = _policy(
        {
            "score": 1.4,
            "expected_net_return_pct": 0.6,
            "profit_quality_ratio": 0.8,
            "tail_risk_score": 0.4,
        }
    ).create(_hold_decision(), _fv(), {}, None, _tools(), None)

    assert candidate is not None
    assert candidate.action == Action.LONG
    assert candidate.raw_response["quant_profit_probe"]["triggered"] is True
    assert candidate.raw_response["opportunity_score"]["expected_net_return_pct"] == 0.6


def test_quant_profit_probe_records_market_quality_block() -> None:
    original = _hold_decision()

    candidate = _policy(
        {
            "expected_net_return_pct": 0.6,
            "profit_quality_ratio": 0.8,
            "tail_risk_score": 0.4,
        }
    ).create(
        original,
        _fv(returns_20=-0.06, price_vs_sma20=-0.2, price_vs_sma50=-0.3),
        {},
        None,
        _tools(adjusted_long_return_pct=0.30, adjusted_short_return_pct=0.01),
        None,
    )

    assert candidate is None
    assert original.raw_response["quant_profit_probe_blocked"]["blocked"] is True


def test_quant_profit_probe_rejects_after_post_score_fails() -> None:
    original = _hold_decision()

    candidate = _policy(
        {
            "score": 0.3,
            "expected_net_return_pct": -0.1,
            "profit_quality_ratio": 0.0,
            "tail_risk_score": 0.4,
        }
    ).create(original, _fv(), {}, None, _tools(), None)

    assert candidate is None
    block = original.raw_response["quant_profit_probe_blocked"]
    assert block["blocked"] is True
    assert block["expected_net_return_pct"] == -0.1
    assert "服务端盈利模型" in block["reason"]
    assert _u("\\u93c8") not in block["reason"]


def test_quant_profit_probe_reads_wrapped_profit_payload() -> None:
    candidate = _policy(
        {
            "score": 1.2,
            "expected_net_return_pct": 0.5,
            "profit_quality_ratio": 0.7,
            "tail_risk_score": 0.4,
        }
    ).create(
        _hold_decision(),
        _fv(),
        {},
        None,
        {
            "profit_prediction": {
                "ok": True,
                "data": {
                    "prediction": {
                        "best_side": "long",
                        "adjusted_long_return_pct": 0.80,
                        "adjusted_short_return_pct": 0.10,
                        "long_loss_probability": 0.35,
                    }
                },
            }
        },
        None,
    )

    assert candidate is not None
    assert candidate.action == Action.LONG
    assert "服务端盈利模型" in candidate.reasoning
    assert _u("\\u93c8") not in candidate.reasoning
