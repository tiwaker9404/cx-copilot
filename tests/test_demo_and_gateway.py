"""The try-it page, the demo info route and the Vercel AI Gateway adapter."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from cx_copilot import models
from cx_copilot.api import create_app
from cx_copilot.config import Settings


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CX_DB_PATH", str(tmp_path / "cx.db"))
    monkeypatch.setenv("CX_CHECKPOINT_PATH", str(tmp_path / "cp.db"))
    monkeypatch.setenv("CX_MODEL", "fake")
    return TestClient(create_app(Settings.from_env()))


def test_demo_page_and_info(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "cx-copilot" in page.text and "/v1/tickets" in page.text
    info = client.get("/v1/demo/info").json()
    assert info == {"model": "fake", "live": False, "version": info["version"]}


def test_demo_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setenv("CX_DB_PATH", str(tmp_path / "cx.db"))
    monkeypatch.setenv("CX_CHECKPOINT_PATH", str(tmp_path / "cp.db"))
    monkeypatch.setenv("CX_DEMO", "0")
    c = TestClient(create_app(Settings.from_env()))
    assert c.get("/").status_code == 404


def test_demo_flow_escalates_danger(client):
    r = client.post("/v1/tickets", json={"text": "The charger started smoking and burned my desk."})
    assert r.status_code == 200 and r.json()["escalated"] is True


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        msg = SimpleNamespace(content='{"category": "billing"}')
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


def test_gateway_adapter_uses_json_mode_for_triage():
    comp = _FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    m = models.get_model("gateway:deepseek/deepseek-v4.1-flash-1017", client=fake_client)
    assert m.name == "gateway:deepseek/deepseek-v4.1-flash-1017"
    m.complete("triage", "sys", "user")
    assert comp.calls[0]["model"] == "deepseek/deepseek-v4.1-flash-1017"
    assert comp.calls[0]["response_format"] == {"type": "json_object"}


def test_gateway_needs_a_key(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("VERCEL_OIDC_TOKEN", raising=False)
    assert models.missing_key("gateway:x/y") == "AI_GATEWAY_API_KEY"
    with pytest.raises(ValueError):
        models.get_model("gateway:x/y")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "k" * 20)
    assert models.missing_key("gateway:x/y") is None
