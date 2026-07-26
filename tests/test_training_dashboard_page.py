from pathlib import Path

from fastapi.testclient import TestClient

from web_dashboard.app import app


def test_training_page_is_independent_from_main_dashboard() -> None:
    static_dir = Path(__file__).resolve().parents[1] / "web_dashboard" / "static"
    training = (static_dir / "training.html").read_text(encoding="utf-8")
    script = (static_dir / "js" / "training.js").read_text(encoding="utf-8")

    assert "模型训练与决策" in training
    assert "/api/model-training/registry" in script
    assert "/api/ml-signal/status" in script
    assert "返回主面板" in training
    assert "就绪与晋升证据" in training
    assert "ml.readiness" in script
    assert "readiness.blocking_reasons" in script
    assert "ml.leave_one_symbol_out_report" in script
    assert "ml.authoritative_trade_return_evidence" in script
    assert "training.governance" in script
    assert "model.display_name" in script
    assert "model.blocking_reasons" in script
    assert "本地 ML 费后收益质量" in script
    assert "线上 DeepSeek 最终决策" in script
    assert "model.model_family" not in script
    assert "new AbortController()" in script
    assert "registry: 45_000" in script
    assert "strategy: 20_000" in script
    assert "请求超过 ${Math.round(timeoutMs / 1000)} 秒未完成" in script
    assert "renderEndpoint(key)" in script
    assert "scheduler.models || scheduler.schedulers" in script
    assert "renderEndpointError(key, requestErrors[key])" in script
    assert "数据不可用" in script
    assert "缺少来自不同决策组的市场机会与执行成本监督样本" in script
    assert "当前退出压力为零，继续持有" in script
    assert "未登记中文说明的系统原因" not in script


def test_training_page_route_exists_without_changing_root_route() -> None:
    client = TestClient(app)
    root = client.get("/", follow_redirects=False)
    training = client.get("/training", follow_redirects=False)

    assert root.status_code in {200, 401, 302}
    assert training.status_code in {200, 401, 302}
    if training.status_code == 200:
        assert "模型训练与决策" in training.text


def test_training_page_static_bundle_uses_valid_chinese_and_cache_version() -> None:
    static_dir = Path(__file__).resolve().parents[1] / "web_dashboard" / "static"
    training = (static_dir / "training.html").read_text(encoding="utf-8")
    script = (static_dir / "js" / "training.js").read_text(encoding="utf-8")

    assert "training.js?v=20260726-training-evidence" in training
    assert "当前 Artifact 与运行时收益监督合同不兼容" in script
    assert "可用，证据未达标" in script
    assert "真实晋升阻断" in script
    assert "鏈煡" not in script
    assert "璇佹嵁" not in script
