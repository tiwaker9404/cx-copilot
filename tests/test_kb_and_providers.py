import json
from types import SimpleNamespace

import pytest

from cx_copilot import kb
from cx_copilot.models import OLLAMA_DEFAULT_URL, OpenAIModel, get_model, missing_key
from cx_copilot.service import TicketService
from cx_copilot.config import Settings
from cx_copilot.store import Store


def test_example_kb_file_matches_the_built_in_policies():
    loaded = kb.load_policies("examples/kb.northwind.json")
    assert [p.id for p in loaded] == [p.id for p in kb.POLICIES]


def test_service_uses_a_custom_kb(tmp_path):
    path = tmp_path / "kb.json"
    path.write_text(json.dumps([{"id": "refunds", "category": "returns",
                                 "keywords": ["refund"], "text": "Refunds take 10 days."}]))
    svc = TicketService(Settings(db_path=":memory:", kb_path=str(path)), store=Store(":memory:"))
    out = svc.handle("I want a refund please")
    assert out["policy_ids"] == ["refunds"] and "10 days" in out["reply"]


@pytest.mark.parametrize("content,match", [
    ("{}", "non-empty JSON list"),
    ('[{"id": "a"}]', "needs id, category"),
    ('[{"id": "a", "category": "nope", "keywords": ["x"], "text": "t"}]', "unknown category"),
    ('[{"id": "a", "category": "billing", "keywords": [], "text": "t"}]', "at least one"),
    ('[{"id": "a", "category": "billing", "keywords": ["x"], "text": "t"},'
     ' {"id": "a", "category": "billing", "keywords": ["y"], "text": "t"}]', "duplicate"),
])
def test_bad_kb_files_fail_loudly(tmp_path, content, match):
    path = tmp_path / "kb.json"
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        kb.load_policies(path)


def test_ollama_needs_no_key_and_points_at_localhost(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert missing_key("ollama:llama3.2") is None
    m = get_model("ollama:llama3.2")
    assert m.name == "ollama:llama3.2"
    assert str(m._client.base_url).rstrip("/") == OLLAMA_DEFAULT_URL


def test_ollama_base_url_is_configurable(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434/v1")
    assert "ollama:11434" in str(get_model("ollama:llama3.2")._client.base_url)


def test_json_mode_is_requested_for_triage_only():
    seen = []

    def create(**kw):
        seen.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])

    stub = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    m = OpenAIModel("x", stub, provider="ollama", json_mode=True)
    m.complete("triage", "s", "u")
    m.complete("reply", "s", "u")
    assert seen[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in seen[1]


def test_openai_compat_requires_a_base_url(monkeypatch):
    monkeypatch.delenv("CX_OPENAI_COMPAT_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="CX_OPENAI_COMPAT_BASE_URL"):
        get_model("openai_compat:any")
    monkeypatch.setenv("CX_OPENAI_COMPAT_BASE_URL", "http://localhost:1234/v1")
    assert get_model("openai_compat:any").name == "openai_compat:any"
