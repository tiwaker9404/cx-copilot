from types import SimpleNamespace

import pytest

from cx_copilot import kb
from cx_copilot.retrieval_eval import evaluate, main as retrieval_main
from cx_copilot.dataset import load_tickets


@pytest.mark.parametrize("method", ["keyword", "bm25", "hybrid"])
def test_obvious_tickets_find_their_policy(method):
    r = kb.Retriever(method=method)
    assert r.search("How do I cancel my subscription?", "billing")[0].id == "billing_subscription"
    assert r.search("I was charged twice for order 4471", "billing")[0].id == "billing_duplicate"


def test_keywords_match_whole_words_not_fragments():
    r = kb.Retriever(method="keyword")
    assert r.search("I'm happy with it, I appreciate the help", "other") == []  # not "app"
    assert [p.id for p in r.search("the app freezes", "technical")] == ["app_crash"]


def test_redaction_placeholders_never_match():
    assert all(p.id != "email_change" for p in kb.retrieve("My address is [EMAIL]", "account",
                                                           method="hybrid"))


def test_bm25_ignores_weak_matches():
    assert kb.Retriever(method="bm25").search("Are you hiring for the warehouse?", "other") == []


def test_embed_method_needs_an_embedder_and_works_with_one():
    with pytest.raises(ValueError, match="embedder"):
        kb.Retriever(method="embed")
    r = kb.Retriever(method="embed", embedder=kb.HashEmbedder())
    assert r.search("cancel any time under settings billing", "billing")[0].id == \
        "billing_subscription"
    assert kb.Retriever(method="hybrid", embedder=kb.HashEmbedder()).name == "hybrid+fake"


def test_openai_embedder_caches_and_uses_nomic_prefixes(tmp_path):
    calls = []

    def create(model, input):
        calls.append(list(input))
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(len(t)), 1.0])
                                     for t in input])

    stub = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    e = kb.OpenAIEmbedder("nomic-embed-text", client=stub, cache_dir=tmp_path)
    e.embed(["a", "b"], "document")
    e.embed(["a", "b"], "document")
    assert calls == [["search_document: a", "search_document: b"]]  # second call cached


def test_default_method_beats_keyword_on_the_labelled_set():
    rows = load_tickets()
    best = evaluate(kb.Retriever(), rows)
    old = evaluate(kb.Retriever(method="keyword"), rows)
    assert best["recall@2"] >= old["recall@2"] and best["mrr"] >= old["mrr"]
    assert best["recall@2"] >= 0.9


def test_retrieval_cli_writes_reports(tmp_path):
    out = tmp_path / "r"
    assert retrieval_main(["--out", str(out), "--embed", "fake"]) == 0
    md = out.with_suffix(".md").read_text()
    assert "`hybrid+fake`" in md and "recall@2" in md
