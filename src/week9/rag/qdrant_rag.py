from typing import Any

from openai import OpenAI

from src.week9.pipeline.settings import Settings
from src.week9.rag.retrieval import retrieve

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
) -> dict[str, Any]:
    """Full pipeline: embed → retrieve → prompt → generate. Returns dict."""
    retrieved = retrieve(question, k=k)
    docs = [hit["doc"] for hit in retrieved]

    context = "\n\n".join(
        f"[{hit['chunk_id']}] (source: {hit['source_id']})\n{hit['text']}"
        for hit in docs
    )

    resp = _openai().chat.completions.create(
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

    return {
        "answer": resp.choices[0].message.content,
        "sources": [doc["source_id"] for doc in docs],
        "chunk_ids": [doc["chunk_id"] for doc in docs],
        "tokens_in": resp.usage.prompt_tokens,
        "tokens_out": resp.usage.completion_tokens,
    }
