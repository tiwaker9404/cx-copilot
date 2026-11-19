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

_PREFIX = "/api/index"


async def app(scope, receive, send):
    """vercel.json rewrites every path to /api/index. Some runtimes hand the function that
    rewritten path instead of the one the visitor asked for, so map it back before routing."""
    if scope.get("type") in ("http", "websocket"):
        path = scope.get("path", "")
        if path == _PREFIX or path.startswith(_PREFIX + "/"):
            new_path = path[len(_PREFIX):] or "/"
            scope = dict(scope, path=new_path, raw_path=new_path.encode())
    await _app(scope, receive, send)
