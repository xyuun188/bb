from pathlib import Path

from fastapi.testclient import TestClient

from web_dashboard.app import app


def test_training_page_is_independent_from_main_dashboard() -> None:
    static_dir = Path(__file__).resolve().parents[1] / "web_dashboard" / "static"
    training = (static_dir / "training.html").read_text(encoding="utf-8")
    script = (static_dir / "js" / "training.js").read_text(encoding="utf-8")
    assert "模型训练与决策" in training
    assert "/api/model-training/registry" in script
    assert "主面板" in training


def test_training_page_route_exists_without_changing_root_route() -> None:
    client = TestClient(app)
    root = client.get("/", follow_redirects=False)
    training = client.get("/training", follow_redirects=False)

    assert root.status_code in {200, 401, 302}
    assert training.status_code in {200, 401, 302}
    if training.status_code == 200:
        assert "模型训练与决策" in training.text
