"""Policy knowledge base for the fictional shop "Northwind Goods", and retrieval over it.

Four retrieval methods, all behind `Retriever.search(query, category, k)`:

    keyword   hand-written keyword hits + a category tie-break (the original, v0.1)
    bm25      BM25 over each policy's text, keywords and id; no hand-written rules needed
    embed     cosine similarity of embeddings from a local model (Ollama nomic-embed-text)
    hybrid    reciprocal-rank fusion of keyword + bm25 (+ embed when an embedder is given)

`cx-retrieval` measures recall@k and MRR of each one against the labelled
`relevant_policies` in the dataset; `DEFAULT_METHOD` is the winner of that eval
(see evals/retrieval.md).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Policy:
    id: str
    category: str
    keywords: tuple[str, ...]
    text: str


def _p(pid: str, category: str, keywords: tuple[str, ...], text: str) -> Policy:
    return Policy(pid, category, keywords, text)


POLICIES: tuple[Policy, ...] = (
    # ------------------------------------------------------------- returns
    _p("refund_window", "returns",
       ("refund", "return", "money back", "exchange", "send back", "send it back"),
       "Refunds are available within 30 days of delivery for unused items in original "
       "packaging. Approved refunds go back to the original payment method in "
       "5-7 business days."),
    _p("warranty", "returns",
       ("broken", "defective", "cracked", "warranty", "stopped working", "doesn't work",
        "faulty", "split", "ripped", "torn", "fell apart", "came apart", "peeling"),
       "Items are covered by a 12-month warranty against defects. We send a free "
       "replacement once you share a photo of the defect."),
    _p("damaged_in_transit", "returns",
       ("arrived damaged", "damaged", "smashed", "dented", "box was crushed", "shattered",
        "arrived broken"),
       "If an item arrives damaged, send a photo within 48 hours of delivery and we ship a "
       "free replacement. You don't need to return the damaged item."),
    _p("wrong_item", "returns",
       ("wrong item", "wrong size", "wrong color", "not what i ordered", "different item",
        "sent me the wrong"),
       "If you received the wrong item, we email a prepaid return label and ship the correct "
       "item as soon as the carrier scans the return."),
    _p("return_shipping_cost", "returns",
       ("return shipping", "return label", "pay for the return", "cost to return",
        "label fee"),
       "Return shipping is free for defective or wrong items. For other returns, a $6 label "
       "fee is deducted from the refund."),
    _p("gift_returns", "returns",
       ("return a gift", "returning a gift", "was a gift", "was a present",
        "received as a gift", "gift receipt"),
       "Gifts can be returned within 30 days of delivery and are refunded as store credit to "
       "the recipient. The buyer is not notified."),
    # ------------------------------------------------------------ shipping
    _p("shipping_times", "shipping",
       ("shipping", "delivery", "arrive", "tracking", "how long", "express", "late",
        "hasn't arrived", "where is my order"),
       "Standard shipping takes 3-5 business days and express takes 1-2 business days. "
       "A tracking email is sent when your order ships."),
    _p("lost_package", "shipping",
       ("lost", "missing", "never arrived", "stuck", "hasn't moved", "hasn't updated",
        "no movement"),
       "If tracking shows no movement for 7 days, we reissue the order or refund it in "
       "full. Please contact us within 30 days of the ship date."),
    _p("delivered_not_received", "shipping",
       ("delivered", "never got", "didn't receive", "not received", "porch"),
       "If tracking says delivered but nothing arrived, check with neighbors and "
       "household members for 24 hours, then contact us and we will open a carrier "
       "investigation."),
    _p("order_changes", "shipping",
       ("cancel my order", "cancel the order", "cancel an order", "change my order",
        "modify", "add an item", "ordered by mistake", "change the size"),
       "Orders can be changed or cancelled within 1 hour of placing them under "
       "Account > Orders. After that the order ships and can be returned once it arrives."),
    _p("address_change", "shipping",
       ("shipping address", "delivery address", "wrong address", "change the address",
        "change my address", "moved", "ship to a different", "redirect"),
       "We can update the shipping address until the order ships. Once it has shipped, the "
       "carrier cannot redirect it, so please contact us before then."),
    _p("international_shipping", "shipping",
       ("international", "internationally", "canada", "overseas", "abroad", "uk",
        "united kingdom", "europe", "customs", "duties", "outside the us"),
       "We ship to the US and Canada only. Canadian orders take 5-10 business days and may "
       "owe import duties on delivery."),
    # ------------------------------------------------------------- billing
    _p("billing_duplicate", "billing",
       ("charged twice", "double charge", "double charged", "duplicate", "twice", "pending",
        "two charges", "charged two times", "charged three times"),
       "Confirmed duplicate charges are reversed within 3-5 business days. A pending "
       "authorization drops off automatically within 5 days."),
    _p("billing_subscription", "billing",
       ("cancel my subscription", "subscription", "renew", "membership", "auto-renew"),
       "You can cancel any time under Settings > Billing. Access continues until the "
       "end of the paid period."),
    _p("price_adjustment", "billing",
       ("price drop", "went on sale", "price match", "cheaper now", "lower price",
        "price difference", "price adjustment"),
       "If an item's price drops within 14 days of your purchase, we refund the difference "
       "once per order. Price matching with other stores is not offered."),
    _p("promo_codes", "billing",
       ("promo", "coupon", "discount code", "code didn't work", "code not working",
        "voucher"),
       "Promo codes must be entered at checkout and can't be combined or added to an order "
       "after it is placed."),
    _p("payment_declined", "billing",
       ("declined", "payment failed", "card was rejected", "card not accepted",
        "payment didn't go through"),
       "If your card is declined, check that the billing address matches the card and try "
       "again, or pay with PayPal. We never see your full card number."),
    _p("invoices", "billing",
       ("invoice", "receipt", "vat", "tax receipt", "proof of purchase"),
       "Invoices are under Account > Orders: open the order and choose Download invoice."),
    # ------------------------------------------------------------- account
    _p("password_reset", "account",
       ("password", "log in", "login", "sign in", "reset link"),
       "Use Forgot password on the sign-in page. The reset link expires after "
       "60 minutes."),
    _p("account_lockout", "account",
       ("locked out", "locked", "too many attempts", "account is locked"),
       "After 5 failed sign-in attempts an account locks for 30 minutes, then you can try "
       "again or use Forgot password."),
    _p("email_change", "account",
       ("change the email", "change my email", "change email", "update my email",
        "update email", "new email", "old email"),
       "Changing the email on an account requires a verification code sent to the "
       "current email address."),
    _p("two_factor", "account",
       ("two-factor", "2fa", "authenticator", "verification app", "lost my phone",
        "security code", "backup code"),
       "If you lost your two-factor device, sign in with one of your backup codes. Without "
       "backup codes, identity verification takes 2-3 business days."),
    _p("delete_account", "account",
       ("delete my account", "delete account", "close my account", "remove my data",
        "delete my data", "all my data"),
       "You can delete your account under Settings > Privacy > Delete account. Deletion "
       "becomes permanent after 30 days; signing in before then cancels it."),
    # ----------------------------------------------------------- technical
    _p("app_crash", "technical",
       ("crash", "freez*", "won't open", "not loading", "app"),
       "Update to the latest app version and restart. If the problem continues, send "
       "your device model and OS version so engineering can investigate."),
    _p("site_error", "technical",
       ("website", "check out", "checkout", "error", "browser"),
       "If the website shows an error, clear your browser cache and try again in a "
       "private window. If it persists, send us the page link and a screenshot."),
    _p("notifications", "technical",
       ("notification", "too many emails", "unsubscribe", "push", "alerts", "spam"),
       "Turn order and marketing notifications on or off under Settings > Notifications. "
       "Marketing emails also have an unsubscribe link at the bottom."),
    _p("guest_orders", "technical",
       ("guest", "checked out as a guest", "order isn't showing", "can't see my order",
        "not in my order history", "missing from my account"),
       "Orders placed as a guest don't appear in your account automatically. Add one under "
       "Account > Orders > Find an order, using the order number and email."),
    # --------------------------------------------------------------- other
    _p("rewards", "other",
       ("points", "rewards", "loyalty", "member perks"),
       "Northwind Rewards points post 14 days after delivery and expire after 12 months "
       "without a purchase."),
    _p("gift_cards", "other",
       ("gift card", "gift certificate", "e-gift"),
       "Gift cards never expire, can be used on any order and can't be exchanged for cash."),
    _p("stores", "other",
       ("physical store", "store near", "visit", "in person", "showroom", "pick up",
        "pickup", "location"),
       "Northwind Goods is online only. We have no physical stores or pickup locations."),
)


def load_policies(path: str | Path) -> tuple[Policy, ...]:
    """Load your own policies from JSON: a list of {id, category, keywords, text}.

    Swap the fictional Northwind policies for real ones without touching code
    (set CX_KB_PATH). Invalid files fail loudly at startup, not at request time.
    """
    from .models import CATEGORIES

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty JSON list of policies")
    out, seen = [], set()
    for i, item in enumerate(raw):
        try:
            pid, cat, kws, text = item["id"], item["category"], item["keywords"], item["text"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path}: policy #{i} needs id, category, keywords, text") from exc
        if cat not in CATEGORIES:
            raise ValueError(f"{path}: policy {pid!r} has unknown category {cat!r}")
        if not kws or not all(isinstance(k, str) and k.strip() for k in kws):
            raise ValueError(f"{path}: policy {pid!r} needs at least one keyword")
        if pid in seen:
            raise ValueError(f"{path}: duplicate policy id {pid!r}")
        seen.add(pid)
        out.append(Policy(pid, cat, tuple(k.lower() for k in kws), str(text)))
    return tuple(out)


# ------------------------------------------------------------------ text helpers
_PLACEHOLDER = re.compile(r"\[[A-Z]+\]")
_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has have hi hello "
    "how i i'm if in is it it's its just me my of on or our please so that the their them "
    "then there this to too us was we were what when where which who why will with would you "
    "your yours thanks thank im ive dont cant still also get got any some about after "
    "again all am been before being both each few more most other own same should than very "
    "s t".split())


def clean_query(text: str) -> str:
    """Lower-case and drop redaction placeholders, so "[EMAIL]" never matches the email policy."""
    return _PLACEHOLDER.sub(" ", text).lower()


def keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Whole words or phrases, allowing a plural or -ed/-ing ending ("refund" matches
    "refunds", not "refundable"). A trailing "*" makes it a prefix: "freez*" matches "freezes"."""
    kw = keyword.lower().strip()
    if kw.endswith("*"):
        return re.compile(r"\b" + re.escape(kw[:-1]))
    return re.compile(r"\b" + re.escape(kw) + r"(?:s|es|ed|d|ing)?\b")


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokens(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(clean_query(text)) if t not in STOPWORDS]


# ------------------------------------------------------------------ embeddings
class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        """`kind` is "query" or "document" (some models want different prefixes)."""
        ...


class HashEmbedder:
    """Offline stand-in for tests: a hashed bag of stems. Not a quality benchmark."""

    name = "fake"

    def __init__(self, dims: int = 256) -> None:
        self.dims = dims

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dims
            for tok in tokens(text):
                vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dims] += 1.0
            out.append(vec)
        return out


class OpenAIEmbedder:
    """Embeddings from any OpenAI-compatible server. With Ollama it is free and local:
    `ollama pull nomic-embed-text`, then use `ollama:nomic-embed-text`."""

    PREFIX = {"nomic-embed-text": {"query": "search_query: ", "document": "search_document: "}}

    def __init__(self, model: str, *, client: Any = None, base_url: str | None = None,
                 api_key: str | None = None, provider: str = "ollama",
                 cache_dir: str | Path | None = None) -> None:
        if client is None:
            import openai

            client = openai.OpenAI(base_url=base_url, api_key=api_key or "ollama",
                                   max_retries=4, timeout=60.0)
        self._client = client
        self._model = model
        self._cache = Path(cache_dir) / "embeddings" if cache_dir else None
        self.name = f"{provider}:{model}"

    def _key(self, text: str) -> Path | None:
        if self._cache is None:
            return None
        digest = hashlib.sha256(f"{self.name}\x1f{text}".encode()).hexdigest()
        return self._cache / f"{digest}.json"

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        prefix = self.PREFIX.get(self._model.split(":")[0], {}).get(kind, "")
        texts = [prefix + t for t in texts]
        out: list[list[float] | None] = []
        todo: list[int] = []
        for i, text in enumerate(texts):
            path = self._key(text)
            if path is not None and path.exists():
                out.append(json.loads(path.read_text()))
            else:
                out.append(None)
                todo.append(i)
        if todo:
            resp = self._client.embeddings.create(model=self._model,
                                                  input=[texts[i] for i in todo])
            for i, item in zip(todo, resp.data):
                out[i] = list(item.embedding)
                path = self._key(texts[i])
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(out[i]))
        return [v for v in out if v is not None]


def get_embedder(spec: str | None, *, cache_dir: str | Path | None = None,
                 client: Any = None) -> Embedder | None:
    """None | "fake" | "ollama:<model>" | "openai_compat:<model>"."""
    if not spec:
        return None
    provider, _, model = spec.partition(":")
    if provider == "fake":
        return HashEmbedder()
    if provider == "ollama" and model:
        from .models import OLLAMA_DEFAULT_URL

        return OpenAIEmbedder(model, client=client, provider="ollama", cache_dir=cache_dir,
                              base_url=os.environ.get("OLLAMA_BASE_URL", OLLAMA_DEFAULT_URL))
    if provider == "openai_compat" and model:
        return OpenAIEmbedder(model, client=client, provider="openai_compat",
                              cache_dir=cache_dir,
                              base_url=os.environ.get("CX_OPENAI_COMPAT_BASE_URL"),
                              api_key=os.environ.get("CX_OPENAI_COMPAT_API_KEY", "none"))
    raise ValueError(f"Bad embedder spec {spec!r}. Try 'ollama:nomic-embed-text' or 'fake'.")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ------------------------------------------------------------------ retrieval
METHODS = ("keyword", "bm25", "embed", "hybrid")
# Picked from `cx-retrieval` on the labelled set; see evals/retrieval.md.
DEFAULT_METHOD = "hybrid"

BM25_K1, BM25_B = 1.2, 0.75
# Thresholds were read off the score distributions on the labelled set (relevant matches:
# 10th percentile 4.6, median 10.8; irrelevant: median 3.0, 90th percentile 5.4). That makes
# the set a development set for retrieval, not a held-out test; see the README.
BM25_MIN_SCORE = 6.0     # below this, a BM25 match is a coincidence of common words
RELATIVE_CUTOFF = 0.5    # a lower-ranked hit must score at least this share of the top hit
EMBED_MIN_SIM = 0.55     # below this, an embedding match does not qualify on its own
RRF_K = 60
CATEGORY_BOOST = 0.1     # same category as the triage label: a small tie-breaking nudge


class Retriever:
    def __init__(self, policies: tuple[Policy, ...] | None = None,
                 method: str = DEFAULT_METHOD, embedder: Embedder | None = None) -> None:
        if method not in METHODS:
            raise ValueError(f"retrieval method must be one of {METHODS}, got {method!r}")
        if method == "embed" and embedder is None:
            raise ValueError("retrieval method 'embed' needs an embedder (e.g. "
                             "ollama:nomic-embed-text)")
        self.policies = POLICIES if policies is None else policies
        self.method = method
        self.embedder = embedder
        self.name = method + (f"+{embedder.name}" if embedder and method in ("embed", "hybrid")
                              else "")
        self._docs = [tokens(" ".join((p.id.replace("_", " "), p.text, " ".join(p.keywords))))
                      for p in self.policies]
        self._avgdl = sum(map(len, self._docs)) / max(1, len(self._docs))
        n = len(self._docs)
        df = Counter(t for doc in self._docs for t in set(doc))
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._tf = [Counter(doc) for doc in self._docs]
        self._vectors: list[list[float]] | None = None
        self._kw = [[keyword_pattern(k) for k in p.keywords] for p in self.policies]

    # each scorer returns {policy index: score} for the policies that qualify
    def _keyword(self, query: str, category: str) -> dict[int, float]:
        text = clean_query(query)
        out = {}
        for i, p in enumerate(self.policies):
            hits = sum(1 for rx in self._kw[i] if rx.search(text))
            if hits:
                out[i] = hits * 2 + (1 if p.category == category else 0)
        return out

    def _bm25(self, query: str, category: str) -> dict[int, float]:
        q = tokens(query)
        out = {}
        for i, tf in enumerate(self._tf):
            dl = len(self._docs[i])
            score = sum(self._idf.get(t, 0.0) * tf[t] * (BM25_K1 + 1)
                        / (tf[t] + BM25_K1 * (1 - BM25_B + BM25_B * dl / self._avgdl))
                        for t in q if t in tf)
            if score >= BM25_MIN_SCORE:
                out[i] = score * (1 + CATEGORY_BOOST if self.policies[i].category == category
                                  else 1)
        return out

    def _embed(self, query: str, category: str) -> dict[int, float]:
        assert self.embedder is not None
        if self._vectors is None:
            self._vectors = self.embedder.embed([p.text for p in self.policies], "document")
        qv = self.embedder.embed([clean_query(query)], "query")[0]
        out = {}
        for i, v in enumerate(self._vectors):
            sim = _cosine(qv, v)
            if sim >= EMBED_MIN_SIM:
                out[i] = sim + (CATEGORY_BOOST / 10 if self.policies[i].category == category
                                else 0)
        return out

    def ranked(self, query: str, category: str) -> list[tuple[Policy, float]]:
        """Every qualifying policy, best first."""
        if self.method == "hybrid":
            lists = [self._keyword(query, category), self._bm25(query, category)]
            if self.embedder is not None:
                lists.append(self._embed(query, category))
            fused: dict[int, float] = {}
            for scores in lists:
                for rank, i in enumerate(sorted(scores, key=lambda j: (-scores[j], j)), 1):
                    fused[i] = fused.get(i, 0.0) + 1 / (RRF_K + rank)
            scores = fused
        else:
            scores = {"keyword": self._keyword, "bm25": self._bm25,
                      "embed": self._embed}[self.method](query, category)
        order = sorted(scores, key=lambda i: (-scores[i], self.policies[i].id))
        if self.method in ("bm25", "embed") and order:
            top = scores[order[0]]
            order = [i for i in order if scores[i] >= RELATIVE_CUTOFF * top]
        return [(self.policies[i], scores[i]) for i in order]

    def search(self, query: str, category: str, k: int = 2) -> list[Policy]:
        return [p for p, _ in self.ranked(query, category)[:k]]


def retrieve(ticket: str, category: str, k: int = 2,
             policies: tuple[Policy, ...] | None = None, method: str = "keyword") -> list[Policy]:
    """Convenience wrapper (and the v0.1 behaviour when method="keyword")."""
    return Retriever(policies, method).search(ticket, category, k)
