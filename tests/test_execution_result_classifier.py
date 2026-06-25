from executor.base_executor import ExecutionResult, OrderStatus
from services.entry_symbol_blocklist import EntrySymbolBlocklistPolicy
from services.execution_result_classifier import ExecutionResultClassifier


def _result(
    status: OrderStatus,
    *,
    raw_response: dict | None = None,
    order_id: str = "local-1",
    exchange_order_id: str | None = "okx-1",
    quantity: float = 1.0,
) -> ExecutionResult:
    return ExecutionResult(
        order_id=order_id,
        exchange_order_id=exchange_order_id,
        symbol="BTC/USDT",
        side="long",
        order_type="market",
        quantity=quantity,
        price=100.0,
        status=status,
        raw_response=raw_response or {},
    )


def test_execution_reason_handles_empty_result() -> None:
    policy = ExecutionResultClassifier()

    assert policy.reason_from_result(None) == "交易接口未返回执行结果。"


def test_execution_reason_describes_entry_tracking_partial_fill() -> None:
    policy = ExecutionResultClassifier()
    result = _result(
        OrderStatus.PARTIAL,
        raw_response={
            "entry_tracking": True,
            "filled_contracts": 2,
            "remaining_contracts": 3,
        },
    )

    reason = policy.reason_from_result(result)

    assert "OKX 开仓委托已部分成交" in reason
    assert "已成交约 2 张" in reason
    assert "剩余约 3 张" in reason


def test_execution_reason_describes_exit_tracking_pending_progress() -> None:
    policy = ExecutionResultClassifier()
    result = _result(
        OrderStatus.OPEN,
        raw_response={
            "exit_tracking": True,
            "filled_contracts": 1,
            "remaining_contracts": 2,
        },
    )

    reason = policy.reason_from_result(result)

    assert "OKX 平仓订单正在追单中" in reason
    assert "不会重复提交" in reason


def test_execution_reason_translates_known_okx_errors() -> None:
    policy = ExecutionResultClassifier()
    result = _result(
        OrderStatus.REJECTED,
        raw_response={"error": "51008 Insufficient USDT margin"},
        exchange_order_id=None,
    )

    assert "账户可用 USDT 保证金不足" in policy.reason_from_result(result)


def test_execution_reason_translates_okx_50001_as_transient_exchange_error() -> None:
    policy = ExecutionResultClassifier()
    result = _result(
        OrderStatus.REJECTED,
        raw_response={
            "error": (
                'Max retries exceeded: okx {"code":"50001","data":[],'
                '"msg":"Service temporarily unavailable. Please try again later."}'
            )
        },
        exchange_order_id=None,
    )

    reason = policy.reason_from_result(result)

    assert "交易所服务临时不可用" in reason
    assert "不计为策略质量失败" in reason
    assert "最小张数" in reason
    assert "仓位计算错误" in reason


def test_execution_reason_translates_okx_json_parameter_errors() -> None:
    policy = ExecutionResultClassifier()
    result = _result(
        OrderStatus.REJECTED,
        raw_response={
            "error": (
                'okx {"code":"1","data":[{"clOrdId":"local-1",'
                '"ordId":"","sCode":"51000","sMsg":"Parameter tpTriggerPx error"}]}'
            )
        },
        exchange_order_id=None,
    )

    reason = policy.reason_from_result(result)

    assert "保护止盈触发价 tpTriggerPx 无效" in reason
    assert "clOrdId" not in reason


def test_execution_reason_translates_okx_contract_delivery_error() -> None:
    policy = ExecutionResultClassifier()
    result = _result(
        OrderStatus.REJECTED,
        raw_response={
            "error": (
                'okx {"code":"1","data":[{"ordId":"","sCode":"51028",'
                '"sMsg":"Contract under delivery."}],"msg":"All operations failed"}'
            )
        },
        exchange_order_id=None,
    )

    reason = policy.reason_from_result(result)

    assert "OKX 51028" in reason
    assert "Contract under delivery" in reason
    assert "暂停重复提交平仓" in reason


def test_execution_reason_uses_untradable_checker() -> None:
    policy = ExecutionResultClassifier(
        untradable_exchange_error_checker=lambda text: "instrument suspended" in text
    )
    result = _result(
        OrderStatus.REJECTED,
        raw_response={"error": "instrument suspended"},
        exchange_order_id=None,
    )

    reason = policy.reason_from_result(result)

    assert "交易对当前不可交易" in reason


def test_execution_reason_uses_default_untradable_policy_terms() -> None:
    blocklist = EntrySymbolBlocklistPolicy(lambda symbol: symbol)
    policy = ExecutionResultClassifier(
        untradable_exchange_error_checker=blocklist.is_untradable_exchange_error
    )
    result = _result(
        OrderStatus.REJECTED,
        raw_response={"error": "OKX 提示该交易对当前不可交易"},
        exchange_order_id=None,
    )

    reason = policy.reason_from_result(result)

    assert "交易对当前不可交易" in reason
    assert "系统已暂时跳过该交易对" in reason


def test_execution_result_detects_no_exchange_position() -> None:
    policy = ExecutionResultClassifier()
    result = _result(
        OrderStatus.REJECTED,
        raw_response={"error": "51169 don't have any positions in this direction"},
        exchange_order_id=None,
    )

    assert policy.result_has_no_exchange_position(result) is True
    assert "没有对应方向的可平仓位" in policy.reason_from_result(result)


def test_execution_confirmation_requires_real_exchange_order_id() -> None:
    policy = ExecutionResultClassifier()

    assert policy.is_exchange_confirmed_execution(_result(OrderStatus.FILLED)) is True
    assert (
        policy.is_exchange_confirmed_execution(
            _result(OrderStatus.FILLED, exchange_order_id="rejected")
        )
        is False
    )
    assert (
        policy.is_exchange_confirmed_execution(_result(OrderStatus.FILLED, quantity=0.0)) is False
    )
    assert (
        policy.is_exchange_confirmed_execution(
            _result(OrderStatus.FILLED, quantity=1.0, exchange_order_id="")
        )
        is False
    )
    assert policy.is_exchange_confirmed_execution(_result(OrderStatus.PARTIAL)) is False


def test_exit_progress_requires_tracking_partial_and_order_id() -> None:
    policy = ExecutionResultClassifier()

    assert (
        policy.is_exit_progress_execution(
            _result(OrderStatus.PARTIAL, raw_response={"exit_tracking": True})
        )
        is True
    )
    assert (
        policy.is_exit_progress_execution(
            _result(
                OrderStatus.PARTIAL,
                raw_response={"exit_tracking": True},
                exchange_order_id=None,
            )
        )
        is False
    )
    assert (
        policy.is_exit_progress_execution(
            _result(OrderStatus.PARTIAL, raw_response={"entry_tracking": True})
        )
        is False
    )
