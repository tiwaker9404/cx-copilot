"""Vercel entrypoint: serves the cx-copilot API and the try-it page as one Python function.

Free on the Hobby plan. Settings come from environment variables as usual; on Vercel the only
differences are that SQLite lives in /tmp (the deployment filesystem is read-only) and that the
model defaults to Vercel AI Gateway when AI_GATEWAY_API_KEY is set, else the offline stand-in.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ.setdefault("CX_DB_PATH", "/tmp/cx.db")
os.environ.setdefault("CX_CHECKPOINT_PATH", "/tmp/checkpoints.db")
os.environ.setdefault("CX_RATE_LIMIT_PER_MIN", "30")
if "CX_MODEL" not in os.environ:
    gateway_model = os.environ.get("CX_GATEWAY_MODEL", "deepseek/deepseek-v4.1-flash-1017")
    os.environ["CX_MODEL"] = f"gateway:{gateway_model}" if os.environ.get("AI_GATEWAY_API_KEY") else "fake"

from cx_copilot.api import configure_logging, create_app  # noqa: E402

configure_logging(os.environ.get("CX_LOG_LEVEL", "INFO"))
_app = create_app()

from urllib.parse import parse_qsl, urlencode  # noqa: E402

_PREFIX = "/api/index"


def _original_path(scope) -> tuple[str, bytes] | None:
    """vercel.json rewrites every request to /api/index?__path=/<original path>. Recover the
    path the visitor asked for and drop the helper parameter from the query string."""
    pairs = parse_qsl(scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)
    path = next((v for k, v in pairs if k == "__path"), None)
    if path is None:
        p = scope.get("path", "")
        if p == _PREFIX or p.startswith(_PREFIX + "/"):
            return (p[len(_PREFIX):] or "/"), scope.get("query_string", b"")
        return None
    rest = urlencode([(k, v) for k, v in pairs if k != "__path"]).encode("latin-1")
    return (path if path.startswith("/") else "/" + path), rest


async def app(scope, receive, send):
    if scope.get("type") in ("http", "websocket"):
        found = _original_path(scope)
        if found:
            path, query = found
            scope = dict(scope, path=path, raw_path=path.encode(), query_string=query)
    await _app(scope, receive, send)
