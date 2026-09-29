from typing import Any

import numpy as np


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if va.ndim != 1 or vb.ndim != 1 or va.shape != vb.shape:
        raise ValueError(
            f"Embedding dimension mismatch: query={va.shape}, chunk={vb.shape}"
        )

    denominator = np.linalg.norm(va) * np.linalg.norm(vb)
    if denominator == 0:
        return 0.0
    return float(np.dot(va, vb) / denominator)


def retrieve(
    query_embedding: list[float], chunks: list[dict[str, Any]], top_k: int = 3
) -> list[dict[str, Any]]:
    """Return the top-k chunks ranked by cosine similarity."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    ranked = sorted(
        (
            {
                "chunk_id": chunk["chunk_id"],
                "source_id": chunk["source_id"],
                "text": chunk["text"],
                "score": cosine(query_embedding, chunk["embedding"]),
            }
            for chunk in chunks
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    return ranked[:top_k]
