import os
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

COLLECTION_NAME = "capstone_chunks"
EMBEDDING_DIM = 1536  # text-embedding-3-small


@dataclass
class QdrantStore:
    """Wraps a Qdrant client + a collection name.

    Convention: one QdrantStore instance per running app.
    Instantiate via load_store() below.
    """

    client: QdrantClient
    collection: str
    dim: int = EMBEDDING_DIM


def _get_client() -> QdrantClient:
    """Build a QdrantClient using QDRANT_URL + QDRANT_API_KEY env vars.

    Resolution order (matches the W7 lesson plan):
      1. QDRANT_URL + QDRANT_API_KEY → Qdrant Cloud
      2. QDRANT_URL only → local Docker
      3. Default → http://localhost:6333
    """
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY") or None
    return QdrantClient(url=url, api_key=api_key, timeout=60)


def load_store(
    collection_name: str = COLLECTION_NAME, dim: int = EMBEDDING_DIM
) -> QdrantStore:
    """Connect to a collection; defaults preserve the OpenAI small pipeline."""
    return QdrantStore(client=_get_client(), collection=collection_name, dim=dim)


def ensure_collection(store: QdrantStore) -> None:
    """Create a collection or check its existing vector configuration.

    Idempotent — safe to call on every app start.
    """
    existing = [c.name for c in store.client.get_collections().collections]
    if store.collection in existing:
        config = store.client.get_collection(store.collection).config.params.vectors
        if (
            not isinstance(config, VectorParams)
            or config.size != store.dim
            or config.distance != Distance.COSINE
        ):
            raise ValueError(
                f"Collection '{store.collection}' must use {store.dim}-dimensional "
                "Cosine vectors. Choose a separate collection for this cache."
            )
        return
    store.client.create_collection(
        collection_name=store.collection,
        vectors_config=VectorParams(size=store.dim, distance=Distance.COSINE),
    )


def upsert_chunks(
    store: QdrantStore, chunks: list[dict], vectors: list[list[float]]
) -> None:
    """Push (chunk, vector) pairs into the store.

    Each chunk is a dict with 'chunk_id', 'source_id', and 'text' keys
    (matches the shape produced by W6's naive_rag.load_corpus).
    """
    points = [
        PointStruct(
            id=idx,
            vector=vec,
            payload={
                "chunk_id": chunk["chunk_id"],
                "source_id": chunk["source_id"],
                "text": chunk["text"],
            },
        )
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]
    batch_size = 32
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        store.client.upsert(
            collection_name=store.collection, points=batch, wait=True
        )
        print(f"  Uploaded {start + len(batch)}/{len(points)} chunks.")


def collection_size(store: QdrantStore) -> int:
    """Return the number of points in the collection. Used for smoke tests."""
    info = store.client.get_collection(store.collection)
    return info.points_count
