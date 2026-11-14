"""Update one clinical case in the Week 10 knowledge base.

Usage:
    uv run --env-file .env python -m scripts.update_document
"""

from pathlib import Path

from qdrant_client.models import PayloadSchemaType

from src.week10.ingest.pipeline import ingest_or_update_source, read_corpus
from src.week10.rag.qdrant_store import load_store


def main():
    source = "case_2481"
    body = next(
        text
        for _, source_id, text in read_corpus(Path("data/corpus/cases.jsonl"))
        if source_id == source
    )

    store = load_store()
    try:
        store.client.create_payload_index(
            collection_name=store.collection,
            field_name="source_id",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )
        store.client.create_payload_index(
            collection_name=store.collection,
            field_name="deleted_at",
            field_schema=PayloadSchemaType.FLOAT,
            wait=True,
        )
        result = ingest_or_update_source(store, source, body)
    finally:
        store.client.close()

    print(f"Update complete for {source}:")
    print(f"  Tombstoned: {result['tombstoned']} old chunks")
    print(f"  Ingested:   {result['ingested']} new chunks")
    print(f"  Cache cleared: {result['cache_cleared']} entries")


if __name__ == "__main__":
    main()
