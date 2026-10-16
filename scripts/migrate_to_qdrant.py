"""Upload a saved embedding cache to Qdrant without re-embedding.

Run: uv run --env-file .env python scripts/migrate_to_qdrant.py
Local: add --index data/embeddings_local.json --collection capstone_chunks_local
"""

import argparse
import json
import sys
from pathlib import Path

# Keep this before project imports when running the script directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.week7.rag.qdrant_store import (  # noqa: E402 — requires project path above
    collection_size,
    ensure_collection,
    load_store,
    upsert_chunks,
)

INDEX_PATH = PROJECT_ROOT / "data/embeddings_large.json"


def read_cache(path: Path) -> tuple[dict, list[dict], list[list[float]]]:
    """Read model metadata, chunks, and their matching vectors."""
    index = json.loads(path.read_text(encoding="utf-8"))
    metadata = index["metadata"]
    entries = index["chunks"]
    if not entries:
        raise ValueError("The embedding cache is empty.")
    chunks = [
        {key: entry[key] for key in ("chunk_id", "source_id", "text")}
        for entry in entries
    ]
    vectors = [entry["embedding"] for entry in entries]
    return metadata, chunks, vectors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--collection", default="capstone_chunks")
    args = parser.parse_args(argv)

    print(f"Loading embeddings from {args.index}...")
    metadata, chunks, vectors = read_cache(args.index)
    dim = metadata["embedding_dimension"]
    print(
        f"Loaded {len(chunks)} chunks: {metadata['embedding_model']}, {dim} dimensions."
    )

    store = load_store(collection_name=args.collection, dim=dim)
    try:
        print(f"Connecting to Qdrant collection '{store.collection}'...")
        ensure_collection(store)
        print(f"Upserting {len(chunks)} points into '{store.collection}'...")
        upsert_chunks(store, chunks, vectors)
        count = collection_size(store)
        if count != len(chunks):
            raise ValueError(
                f"Expected {len(chunks)} points after upload, found {count}."
            )
        print(
            f"Done. Collection now has {count} points. No embedding API calls were made."
        )
    finally:
        store.client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
