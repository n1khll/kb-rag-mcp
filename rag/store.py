import asyncio
import math
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras

from rag.config import Settings

# Standard reciprocal-rank-fusion constant; lower weights rank-1 more heavily.
_RRF_K = 60

# Cosine-similarity floor below which even the best hit is flagged low-confidence.
# Tuned against the 200-article dummy eval; revisit with real data.
_LOW_CONFIDENCE_THRESHOLD = 0.35


def _connect(settings: Settings):
    return psycopg2.connect(settings.database_url)


def _vector_literal(embedding: List[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def _parse_vector(raw: Any) -> List[float]:
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    text = str(raw).strip().strip("[]")
    if not text:
        return []
    return [float(x) for x in text.split(",")]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _vector_search_sync(
    settings: Settings,
    query_embedding: List[float],
    limit: int,
    published_only: bool,
    category: Optional[str],
) -> List[Dict[str, Any]]:
    conn = _connect(settings)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, kb_number, kb_sys_id, title, chunk_index, chunk_text,
                       category, workflow_state, sys_updated_on, embedding
                FROM kb_chunks
                WHERE (%(published_only)s = false OR workflow_state = 'published')
                  AND (%(category)s IS NULL OR category = %(category)s)
                ORDER BY embedding <=> %(query_vec)s::vector
                LIMIT %(limit)s
                """,
                {
                    "published_only": published_only,
                    "category": category,
                    "query_vec": _vector_literal(query_embedding),
                    "limit": limit,
                },
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _fulltext_search_sync(
    settings: Settings,
    query_text: str,
    limit: int,
    published_only: bool,
    category: Optional[str],
) -> List[Dict[str, Any]]:
    conn = _connect(settings)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, kb_number, kb_sys_id, title, chunk_index, chunk_text,
                       category, workflow_state, sys_updated_on, embedding,
                       ts_rank_cd(tsv, plainto_tsquery('english', %(query_text)s)) AS rank
                FROM kb_chunks
                WHERE tsv @@ plainto_tsquery('english', %(query_text)s)
                  AND (%(published_only)s = false OR workflow_state = 'published')
                  AND (%(category)s IS NULL OR category = %(category)s)
                ORDER BY rank DESC
                LIMIT %(limit)s
                """,
                {
                    "query_text": query_text,
                    "published_only": published_only,
                    "category": category,
                    "limit": limit,
                },
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _reciprocal_rank_fusion(
    vector_rows: List[Dict[str, Any]],
    fulltext_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    scores: Dict[int, float] = {}
    rows_by_id: Dict[int, Dict[str, Any]] = {}
    for rank, row in enumerate(vector_rows, start=1):
        scores[row["id"]] = scores.get(row["id"], 0.0) + 1.0 / (_RRF_K + rank)
        rows_by_id[row["id"]] = row
    for rank, row in enumerate(fulltext_rows, start=1):
        scores[row["id"]] = scores.get(row["id"], 0.0) + 1.0 / (_RRF_K + rank)
        rows_by_id.setdefault(row["id"], row)

    fused = [{**rows_by_id[chunk_id], "fused_score": score} for chunk_id, score in scores.items()]
    fused.sort(key=lambda r: r["fused_score"], reverse=True)
    return fused


async def hybrid_candidates(
    settings: Settings,
    query_embedding: List[float],
    query_text: str,
    published_only: bool = True,
    category: Optional[str] = None,
    vector_k: int = 40,
    fulltext_k: int = 40,
) -> List[Dict[str, Any]]:
    """Vector + full-text search concurrently, fused with Reciprocal Rank Fusion."""
    vector_rows, fulltext_rows = await asyncio.gather(
        asyncio.to_thread(_vector_search_sync, settings, query_embedding, vector_k, published_only, category),
        asyncio.to_thread(_fulltext_search_sync, settings, query_text, fulltext_k, published_only, category),
    )
    return _reciprocal_rank_fusion(vector_rows, fulltext_rows)


def mmr_select(
    candidates: List[Dict[str, Any]],
    query_embedding: List[float],
    k: int = 8,
    lambda_mult: float = 0.7,
    pool_size: int = 50,
) -> List[Dict[str, Any]]:
    """Pick k results that are relevant AND mutually diverse (Maximal Marginal Relevance).

    Prevents many near-duplicate articles on one topic from crowding out a genuinely
    different-but-relevant match.
    """
    pool = [dict(c) for c in candidates[:pool_size]]
    for c in pool:
        c["_embedding_vec"] = _parse_vector(c["embedding"])
        c["_query_similarity"] = _cosine_similarity(query_embedding, c["_embedding_vec"])

    remaining = list(pool)
    selected: List[Dict[str, Any]] = []
    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda c: c["_query_similarity"])
        else:
            def _mmr_score(c: Dict[str, Any]) -> float:
                diversity_penalty = max(
                    _cosine_similarity(c["_embedding_vec"], s["_embedding_vec"]) for s in selected
                )
                return lambda_mult * c["_query_similarity"] - (1 - lambda_mult) * diversity_penalty

            best = max(remaining, key=_mmr_score)
        selected.append(best)
        remaining.remove(best)

    for c in selected:
        c.pop("_embedding_vec", None)
        c.pop("embedding", None)
    return selected


def top_similarity_confidence(selected: List[Dict[str, Any]]) -> tuple[float, bool]:
    if not selected:
        return 0.0, True
    top = round(float(selected[0].get("_query_similarity", 0.0)), 3)
    return top, top < _LOW_CONFIDENCE_THRESHOLD


def _rows_sync(settings: Settings, sql: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    conn = _connect(settings)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or {})
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


async def fetch_article(settings: Settings, kb_number: str) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(
        _rows_sync,
        settings,
        """
        SELECT kb_number, title, category, workflow_state, chunk_index, chunk_text
        FROM kb_chunks
        WHERE kb_number = %(kb_number)s
        ORDER BY chunk_index
        """,
        {"kb_number": kb_number},
    )


async def list_categories(settings: Settings) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(
        _rows_sync,
        settings,
        """
        SELECT COALESCE(category, 'uncategorized') AS category,
               COUNT(DISTINCT kb_number) AS article_count
        FROM kb_chunks
        GROUP BY 1
        ORDER BY article_count DESC
        """,
    )


async def health(settings: Settings) -> Dict[str, Any]:
    rows = await asyncio.to_thread(
        _rows_sync,
        settings,
        """
        SELECT COUNT(*) AS chunk_count,
               COUNT(DISTINCT kb_number) AS article_count,
               MAX(created_at) AS last_ingested_at
        FROM kb_chunks
        """,
    )
    return rows[0] if rows else {"chunk_count": 0, "article_count": 0, "last_ingested_at": None}
