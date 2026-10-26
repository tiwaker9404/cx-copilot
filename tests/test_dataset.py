from cx_copilot import kb
from cx_copilot.dataset import load_tickets
from cx_copilot.models import CATEGORIES


def test_dataset_is_well_formed():
    rows = load_tickets()
    assert len(rows) >= 30
    for r in rows:
        assert r["gold"]["category"] in CATEGORIES
        assert isinstance(r["gold"]["escalate"], bool)


def test_every_required_fact_is_reachable_from_retrieval():
    """Ceiling check: if retrieval can't surface a fact, no prompt could ever be graded on it."""
    for r in load_tickets():
        g = r["gold"]
        if g["escalate"] or not g["required_facts"]:
            continue
        text = " ".join(p.text for p in kb.retrieve(r["ticket"], g["category"])).lower()
        for fact in g["required_facts"]:
            assert fact.lower() in text, f"{r['id']}: {fact!r} not retrievable"


def test_dataset_mixes_escalations_and_non_escalations():
    flags = [r["gold"]["escalate"] for r in load_tickets()]
    assert 3 <= sum(flags) <= len(flags) // 2
