# kb-rag-mcp

Standalone MCP server exposing agentic-RAG tools over a pgvector-backed knowledge base.
Designed for public internet deployment (Render) with bearer-token auth; embeddings via
Azure OpenAI; LLM/agent side lives in Azure AI Foundry (or any MCP-capable client).

## Tools

| Tool | Purpose |
|---|---|
| `kb_search` | Hybrid retrieval (vector + full-text, RRF-fused, MMR-diversified). Returns chunks + citations + confidence. |
| `kb_get_article` | Full article body by `kb_number` — for drilling into a promising hit. |
| `kb_list_categories` | Categories + article counts — for scoping a refined search. |
| `kb_health` | Chunk/article counts + last ingest time — distinguishes "empty KB" from "no match". |

The agentic loop lives in the *client agent*: search → check `low_confidence` → refine query /
scope by category / fetch full article → answer with citations.

## Setup

1. `cp .env.example .env` and fill in every value (see comments in that file).
2. Create the schema: paste `scripts/schema.sql` into the Neon SQL editor (or `psql "$DATABASE_URL" -f scripts/schema.sql`).
3. Ingest articles: `python scripts/ingest.py scripts/data/kb_dummy_200.json`
4. Run locally: `uvicorn server:app --port 8080`

## Auth

Every `/mcp` request needs `Authorization: Bearer <key>` where `<key>` is listed in
`MCP_API_KEYS` (comma-separated; one key per client, rotate by adding/removing).
No keys configured = all requests rejected. `/health` is open (used by Render health checks).

## Deploy (Render)

1. Push this directory to a GitHub repo.
2. Render → New → Blueprint → select the repo (`render.yaml` is picked up automatically).
3. Fill the `sync: false` env vars in the Render dashboard (same values as your local `.env`).
4. Your MCP endpoint: `https://<service>.onrender.com/mcp`

Free-tier note: Render spins the service down after idle; first request after idle takes
~30-60s. Fine for testing; use a paid instance for a client-facing demo.

## Smoke test

```bash
curl https://<service>.onrender.com/health

curl -X POST https://<service>.onrender.com/mcp \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```
