from typing import Any

from openai import OpenAI

from src.week10.pipeline.settings import Settings
from src.week10.rag.retrieval import retrieve
from src.week10.rag import cache

CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

_client = None


def _openai() -> OpenAI:
    """Lazy-init the OpenAI client (unit tests can import without a key)."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


SYSTEM = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context. If the context does not contain the answer, say so "
    "plainly. Cite the source_id in square brackets after any fact you use."
)


def ask_rag(
    question: str,
    settings: Settings | None = None,
    k: int = 3,
    cache_threshold: float = 0.95,
) -> dict[str, Any]:

    client = _openai()

    # ── Layer 1: semantic cache ──
    q_vec = (
        client.embeddings.create(model="text-embedding-3-small", input=[question])
        .data[0]
        .embedding
    )

    hit = cache.lookup(q_vec, threshold=cache_threshold)
    if hit:
        return {
            "answer": hit["answer"],
            "cache_hit": True,
            "similarity": hit["similarity"],
            "matched": hit["matched_question"],
        }

    # ── Miss: full RAG pipeline ──
    """Full pipeline: embed → retrieve → prompt → generate. Returns dict."""
    retrieved = retrieve(question, k=k)
    docs = [hit["doc"] for hit in retrieved]

    context = "\n\n".join(
        f"[{hit['chunk_id']}] (source: {hit['source_id']})\n{hit['text']}"
        for hit in docs
    )

    resp = client.chat.completions.create(
        model=settings.model if settings is not None else CHAT_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\n---\n\nQuestion: {question}",
            },
        ],
    )

    answer = resp.choices[0].message.content

    # Store in cache for next time
    cache.put(question, q_vec, answer)

    return {
        "answer": answer,
        "cache_hit": False,
        "sources": [doc["source_id"] for doc in docs],
        "chunk_ids": [doc["chunk_id"] for doc in docs],
        "tokens_in": resp.usage.prompt_tokens,
        "tokens_cached": getattr(
            getattr(resp.usage, "prompt_tokens_details", None), "cached_tokens", 0
        ),
        "tokens_out": resp.usage.completion_tokens,
    }
