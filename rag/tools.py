from typing import Any, Dict, List, Optional

from rag import store
from rag.config import get_settings
from rag.embeddings import get_embedding_service


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


async def kb_search(
    query: str,
    k: Any = 8,
    category: Optional[str] = None,
    published_only: Any = True,
) -> Dict[str, Any]:
    """Hybrid semantic + keyword search over KB chunks with diversity-aware selection."""
    settings = get_settings()
    q = (query or "").strip()
    if not q:
        return {"ok": False, "error": "missing_query", "message": "query is required."}

    top_k = max(1, min(_coerce_int(k, 8), 20))
    is_published_only = True if published_only is None else bool(published_only)

    embed_svc = get_embedding_service(settings)
    [query_embedding] = await embed_svc.embed([q])

    candidates = await store.hybrid_candidates(
        settings,
        query_embedding=query_embedding,
        query_text=q,
        published_only=is_published_only,
        category=category or None,
    )

    if not candidates:
        return {
            "ok": True,
            "query": q,
            "chunks": [],
            "citations": [],
            "confidence": 0.0,
            "low_confidence": True,
            "message": "No relevant KB content found for this query.",
            "retrieval": {"fused_candidates": 0, "selected": 0},
        }

    selected = store.mmr_select(candidates, query_embedding, k=top_k)
    confidence, low_confidence = store.top_similarity_confidence(selected)

    chunks = [
        {
            "kb_number": c.get("kb_number"),
            "title": c.get("title"),
            "chunk_index": c.get("chunk_index"),
            "chunk_text": c.get("chunk_text"),
            "category": c.get("category"),
            "workflow_state": c.get("workflow_state"),
            "similarity": round(float(c.get("_query_similarity", 0.0)), 4),
        }
        for c in selected
    ]

    seen_kb: set = set()
    citations: List[Dict[str, Any]] = []
    for c in selected:
        kb_number = c.get("kb_number")
        if kb_number in seen_kb:
            continue
        seen_kb.add(kb_number)
        citations.append(
            {"kb_number": kb_number, "title": c.get("title"), "workflow_state": c.get("workflow_state")}
        )

    return {
        "ok": True,
        "query": q,
        "chunks": chunks,
        "citations": citations,
        "confidence": confidence,
        "low_confidence": low_confidence,
        "retrieval": {"fused_candidates": len(candidates), "selected": len(selected)},
    }


async def kb_get_article(kb_number: str) -> Dict[str, Any]:
    """Fetch a full article (all its chunks, in order) by KB number."""
    settings = get_settings()
    number = (kb_number or "").strip()
    if not number:
        return {"ok": False, "error": "missing_kb_number", "message": "kb_number is required."}

    rows = await store.fetch_article(settings, number)
    if not rows:
        return {"ok": False, "error": "not_found", "message": f"No article found with kb_number {number}."}

    first = rows[0]
    return {
        "ok": True,
        "kb_number": first["kb_number"],
        "title": first["title"],
        "category": first["category"],
        "workflow_state": first["workflow_state"],
        "chunk_count": len(rows),
        "body": "\n\n".join(str(r["chunk_text"]) for r in rows),
    }


async def kb_list_categories() -> Dict[str, Any]:
    """List KB categories with article counts, so searches can be scoped."""
    settings = get_settings()
    rows = await store.list_categories(settings)
    return {
        "ok": True,
        "categories": rows,
        "total_articles": sum(int(r["article_count"]) for r in rows),
    }


async def kb_health() -> Dict[str, Any]:
    """Basic KB store diagnostics: article/chunk counts and last ingest time."""
    settings = get_settings()
    stats = await store.health(settings)
    return {
        "ok": True,
        "chunk_count": int(stats.get("chunk_count") or 0),
        "article_count": int(stats.get("article_count") or 0),
        "last_ingested_at": str(stats.get("last_ingested_at") or ""),
    }
