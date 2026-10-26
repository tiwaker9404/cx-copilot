from cx_copilot import kb
from cx_copilot.dataset import content_hash, load_tickets, validate
from cx_copilot.graph import REPEAT_CONTACT_LIMIT, customer_contacts
from cx_copilot.models import CATEGORIES

import pytest


def test_dataset_is_well_formed():
    rows = load_tickets()
    assert len(rows) >= 120
    for r in rows:
        assert r["gold"]["category"] in CATEGORIES
        assert isinstance(r["gold"]["escalate"], bool)


def test_every_category_and_policy_is_covered():
    rows = load_tickets()
    assert {r["gold"]["category"] for r in rows} == set(CATEGORIES)
    labelled = {p for r in rows for p in r["gold"]["relevant_policies"]}
    assert labelled == {p.id for p in kb.POLICIES}, "every policy needs at least one ticket"


def test_every_required_fact_is_reachable_from_retrieval():
    """Ceiling check: if the default retriever can't surface a fact, no prompt could ever be
    graded on it."""
    retriever = kb.Retriever()
    for r in load_tickets():
        g = r["gold"]
        if g["escalate"] or not g["required_facts"]:
            continue
        query = " ".join([r["ticket"]] + [m["text"] for m in r.get("history") or []
                                          if m["role"] == "customer"])
        text = " ".join(p.text for p in retriever.search(query, g["category"])).lower()
        for fact in g["required_facts"]:
            assert fact.lower() in text, f"{r['id']}: {fact!r} not retrievable"


def test_required_facts_come_from_the_labelled_policies():
    texts = {p.id: p.text.lower() for p in kb.POLICIES}
    for r in load_tickets():
        for fact in r["gold"]["required_facts"]:
            assert any(fact.lower() in texts[p] for p in r["gold"]["relevant_policies"]), r["id"]


def test_dataset_mixes_escalations_and_non_escalations():
    flags = [r["gold"]["escalate"] for r in load_tickets()]
    assert 10 <= sum(flags) <= len(flags) // 2


def test_threads_cover_follow_ups_and_repeat_contacts():
    threads = [r for r in load_tickets() if r.get("history")]
    assert len(threads) >= 20
    repeat = [r for r in threads if customer_contacts(r["history"]) >= REPEAT_CONTACT_LIMIT]
    assert repeat and all(r["gold"]["escalate"] for r in repeat)
    assert any(not r["gold"]["escalate"] for r in threads)
    for r in threads:  # alternate customer / agent, starting with the customer
        roles = [m["role"] for m in r["history"]]
        assert roles == ["customer", "agent"] * (len(roles) // 2)


def test_validation_rejects_bad_rows():
    good = load_tickets()[0]
    with pytest.raises(ValueError, match="duplicate"):
        validate([good, good])
    with pytest.raises(ValueError, match="unknown policies"):
        validate([{**good, "gold": {**good["gold"], "relevant_policies": ["nope"]}}])
    with pytest.raises(ValueError, match="history"):
        validate([{**good, "history": [{"role": "bot", "text": "x"}]}])


def test_content_hash_changes_with_the_data():
    rows = load_tickets()
    assert content_hash(rows) != content_hash(rows[:-1])
