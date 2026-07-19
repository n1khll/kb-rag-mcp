"""
Chunk, embed, and upsert KB articles into kb_chunks.

Input JSON: a list of article objects:
  kb_number (required), kb_sys_id (optional, defaults to kb_number), title,
  category, workflow_state (defaults to "published"), body (required)

Usage (from the kb_rag_mcp directory, with .env populated):
  python scripts/ingest.py scripts/data/kb_dummy_200.json
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import psycopg2
import tiktoken
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from rag.config import get_settings  # noqa: E402
from rag.embeddings import get_embedding_service  # noqa: E402

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 100
EMBED_BATCH_SIZE = 64

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _chunk_text(text: str) -> List[str]:
    tokens = _ENCODING.encode(text or "")
    if not tokens:
        return []
    if len(tokens) <= CHUNK_SIZE_TOKENS:
        return [text.strip()]
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE_TOKENS, len(tokens))
        chunks.append(_ENCODING.decode(tokens[start:end]).strip())
        if end >= len(tokens):
            break
        start = max(0, end - CHUNK_OVERLAP_TOKENS)
    return [c for c in chunks if c]


def _vector_literal(embedding: List[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


async def main(path: Path) -> None:
    settings = get_settings()
    embed_svc = get_embedding_service(settings)

    articles = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(articles, list):
        raise ValueError("Input JSON must be a list of article objects.")

    rows: List[Dict[str, Any]] = []
    for article in articles:
        kb_number = str(article["kb_number"]).strip()
        kb_sys_id = str(article.get("kb_sys_id") or kb_number).strip()
        body = str(article.get("body") or "").strip()
        if not body:
            print(f"SKIP {kb_number}: empty body")
            continue
        for idx, chunk in enumerate(_chunk_text(body)):
            rows.append(
                {
                    "kb_number": kb_number,
                    "kb_sys_id": kb_sys_id,
                    "title": article.get("title"),
                    "category": article.get("category"),
                    "workflow_state": article.get("workflow_state") or "published",
                    "chunk_index": idx,
                    "chunk_text": chunk,
                }
            )

    if not rows:
        print("Nothing to ingest.")
        return

    max_index_by_article: Dict[str, int] = {}
    for row in rows:
        sid = row["kb_sys_id"]
        max_index_by_article[sid] = max(max_index_by_article.get(sid, -1), row["chunk_index"])

    print(f"Chunked {len(articles)} articles into {len(rows)} chunks. Embedding...")
    vectors: List[List[float]] = []
    for i in range(0, len(rows), EMBED_BATCH_SIZE):
        batch = [r["chunk_text"] for r in rows[i : i + EMBED_BATCH_SIZE]]
        vectors.extend(await embed_svc.embed(batch))

    conn = psycopg2.connect(settings.database_url)
    try:
        with conn.cursor() as cur:
            for row, vec in zip(rows, vectors):
                cur.execute(
                    """
                    INSERT INTO kb_chunks
                        (kb_number, kb_sys_id, title, chunk_index, chunk_text, category, workflow_state, embedding)
                    VALUES (%(kb_number)s, %(kb_sys_id)s, %(title)s, %(chunk_index)s, %(chunk_text)s,
                            %(category)s, %(workflow_state)s, %(embedding)s::vector)
                    ON CONFLICT (kb_sys_id, chunk_index) DO UPDATE
                    SET kb_number = EXCLUDED.kb_number,
                        title = EXCLUDED.title,
                        chunk_text = EXCLUDED.chunk_text,
                        category = EXCLUDED.category,
                        workflow_state = EXCLUDED.workflow_state,
                        embedding = EXCLUDED.embedding
                    """,
                    {**row, "embedding": _vector_literal(vec)},
                )
            # Drop stale trailing chunks if a re-ingested article got shorter.
            for kb_sys_id, max_index in max_index_by_article.items():
                cur.execute(
                    "DELETE FROM kb_chunks WHERE kb_sys_id = %s AND chunk_index > %s",
                    (kb_sys_id, max_index),
                )
        conn.commit()
    finally:
        conn.close()

    print(f"Ingested {len(rows)} chunks across {len(max_index_by_article)} articles.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk, embed, and upsert KB articles.")
    parser.add_argument("input_json", type=Path)
    args = parser.parse_args()
    asyncio.run(main(args.input_json))
