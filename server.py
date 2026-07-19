"""
Standalone KB-RAG MCP server (streamable HTTP, stateless) for internet deployment.

Exposes 4 tools for agentic RAG over a pgvector-backed knowledge base:
  kb_search, kb_get_article, kb_list_categories, kb_health

Auth: every /mcp request must carry "Authorization: Bearer <key>" where <key> is one
of the comma-separated values in MCP_API_KEYS. No keys configured = fail closed.
/health is unauthenticated (for Render health checks).

Run locally:   uvicorn server:app --port 8080
Render runs:   uvicorn server:app --host 0.0.0.0 --port $PORT
"""
import hmac
import json
import logging

from dotenv import load_dotenv

load_dotenv()

from mcp.server.fastmcp import FastMCP  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402

from rag import tools  # noqa: E402
from rag.config import get_settings  # noqa: E402

logging.basicConfig(level=get_settings().app_log_level.upper())
logger = logging.getLogger("kb_rag_mcp")

mcp = FastMCP("kb-rag-mcp")
# Stateless: every request is self-contained; no Mcp-Session-Id bookkeeping needed by clients.
mcp.settings.stateless_http = True


@mcp.tool(
    description=(
        "Semantic search over KB articles using hybrid retrieval (vector + full-text) with "
        "diversity-aware selection (MMR) so near-duplicate articles don't crowd out a genuinely "
        "different match. Pass the user's issue in their own words as `query` -- do not extract "
        "keywords first. Returns cited chunks plus a confidence score; if `low_confidence` is true, "
        "consider refining the query or asking the user a clarifying question."
    )
)
async def kb_search(query: str, k: int = 8, category: str = "", published_only: bool = True):
    return await tools.kb_search(query=query, k=k, category=category or None, published_only=published_only)


@mcp.tool(
    description=(
        "Fetch the FULL text of one KB article by its kb_number (e.g. after kb_search returned a "
        "promising but truncated chunk). Returns title, category, and the complete body."
    )
)
async def kb_get_article(kb_number: str):
    return await tools.kb_get_article(kb_number=kb_number)


@mcp.tool(
    description=(
        "List all KB categories with article counts. Useful to scope a follow-up kb_search "
        "with the `category` filter when a query is ambiguous across domains."
    )
)
async def kb_list_categories():
    return await tools.kb_list_categories()


@mcp.tool(
    description=(
        "KB store diagnostics: how many articles/chunks are indexed and when they were last "
        "ingested. Call this if searches keep returning nothing, to distinguish an empty "
        "knowledge base from a genuinely unmatched query."
    )
)
async def kb_health():
    return await tools.kb_health()


class BearerAuthMiddleware:
    """Pure ASGI middleware: constant-time bearer-token check on everything except /health."""

    def __init__(self, app, exempt_paths: frozenset[str]):
        self._app = app
        self._exempt_paths = exempt_paths

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        if scope.get("path") in self._exempt_paths:
            await self._app(scope, receive, send)
            return

        api_keys = get_settings().api_key_set
        auth_header = ""
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                auth_header = value.decode("latin-1")
                break

        token = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else ""
        authorized = bool(token) and any(hmac.compare_digest(token, key) for key in api_keys)

        if not authorized:
            body = json.dumps(
                {"error": "unauthorized", "message": "Provide a valid 'Authorization: Bearer <key>' header."}
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"application/json"), (b"www-authenticate", b"Bearer")],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        await self._app(scope, receive, send)


async def health(_request):
    return JSONResponse({"status": "ok", "service": "kb-rag-mcp"})


_inner = mcp.streamable_http_app()
_inner.routes.insert(0, Route("/health", health, methods=["GET"]))

app = BearerAuthMiddleware(_inner, exempt_paths=frozenset({"/health"}))
